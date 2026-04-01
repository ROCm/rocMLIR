// **Contraint**
// 1. You must use arith.constant instead of linalg.fill
// 2. You cannot emit scf inside the linalg.generic body loop
// 3. YOu can only use the tensor, arith and linalg dialect entirely 

func.func @causal_mask_attention(%q: tensor<10x512xf32>, %kt: tensor<512x10xf32>, %v: tensor<10x512xf32>) -> tensor<10x512xf32> {
    %output_init = arith.constant dense<0.0> : tensor<10x512xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(i, j) -> (i, j)>],
        iterator_types = ["parallel", "parallel"]
    } outs(%output_init : tensor<10x512xf32>) {
        ^bb0(%out: f32):
            // Step 1: Q @ K^T -> tensor<10x10xf32>
            %qkt_init = arith.constant dense<0.0> : tensor<10x10xf32>
            %qkt = linalg.matmul ins(%q, %kt : tensor<10x512xf32>, tensor<512x10xf32>) outs(%qkt_init : tensor<10x10xf32>) -> tensor<10x10xf32>

            // Step 2a: Extract column 0 from qkt as initial max values (avoids -inf)
            %max_empty = tensor.empty() : tensor<10xf32>
            %col0 = linalg.generic {
                indexing_maps = [affine_map<(d0) -> (d0, 0)>, affine_map<(d0) -> (d0)>],
                iterator_types = ["parallel"]
            } ins(%qkt : tensor<10x10xf32>) outs(%max_empty : tensor<10xf32>) {
                ^bb0(%val: f32, %out_2a: f32):
                    linalg.yield %val : f32
            } -> tensor<10xf32>

            // Step 2b: Row-wise causal max reduction
            %row_max = linalg.generic {
                indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>],
                iterator_types = ["parallel", "reduction"]
            } ins(%qkt : tensor<10x10xf32>) outs(%col0 : tensor<10xf32>) {
                ^bb0(%val: f32, %acc: f32):
                    %m = linalg.index 0 : index
                    %n = linalg.index 1 : index
                    %causal = arith.cmpi ule, %n, %m : index
                    %candidate = arith.maximumf %acc, %val : f32
                    %res = arith.select %causal, %candidate, %acc : f32
                    linalg.yield %res : f32
            } -> tensor<10xf32>

            // Step 3: exp(qkt - row_max) with causal zeroing
            %exp_empty = tensor.empty() : tensor<10x10xf32>
            %exp_result = linalg.generic {
                indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>, affine_map<(m, n) -> (m, n)>],
                iterator_types = ["parallel", "parallel"]
            } ins(%qkt, %row_max : tensor<10x10xf32>, tensor<10xf32>) outs(%exp_empty : tensor<10x10xf32>) {
                ^bb0(%qkt_val: f32, %max_val: f32, %out_3: f32):
                    %m = linalg.index 0 : index
                    %n = linalg.index 1 : index
                    %causal = arith.cmpi ule, %n, %m : index
                    %diff = arith.subf %qkt_val, %max_val : f32
                    %exp_val = math.exp %diff : f32
                    %zero_s = arith.constant 0.0 : f32
                    %masked = arith.select %causal, %exp_val, %zero_s : f32
                    linalg.yield %masked : f32
            } -> tensor<10x10xf32>

            // Step 4: Row-wise sum of exp values
            %sum_init = arith.constant dense<0.0> : tensor<10xf32>
            %row_sum = linalg.generic {
                indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>],
                iterator_types = ["parallel", "reduction"]
            } ins(%exp_result : tensor<10x10xf32>) outs(%sum_init : tensor<10xf32>) {
                ^bb0(%val: f32, %acc: f32):
                    %sum = arith.addf %acc, %val : f32
                    linalg.yield %sum : f32
            } -> tensor<10xf32>

            // Step 5: Normalize exp values by row sum
            %softmax_empty = tensor.empty() : tensor<10x10xf32>
            %softmax = linalg.generic {
                indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m)>, affine_map<(m, n) -> (m, n)>],
                iterator_types = ["parallel", "parallel"]
            } ins(%exp_result, %row_sum : tensor<10x10xf32>, tensor<10xf32>) outs(%softmax_empty : tensor<10x10xf32>) {
                ^bb0(%exp_val: f32, %sum_val: f32, %out_5: f32):
                    %normalized = arith.divf %exp_val, %sum_val : f32
                    linalg.yield %normalized : f32
            } -> tensor<10x10xf32>

            // Step 6: output[i,j] = sum_k(softmax[i,k] * V[k,j])
            %i = linalg.index 0 : index
            %j = linalg.index 1 : index
            %acc_scalar = arith.constant dense<0.0> : tensor<f32>
            %k_range = tensor.empty() : tensor<10xf32>
            %dot = linalg.generic {
                indexing_maps = [affine_map<(k) -> (k)>, affine_map<(k) -> ()>],
                iterator_types = ["reduction"]
            } ins(%k_range : tensor<10xf32>) outs(%acc_scalar : tensor<f32>) {
                ^bb0(%dummy: f32, %acc: f32):
                    %k = linalg.index 0 : index
                    %s_ik = tensor.extract %softmax[%i, %k] : tensor<10x10xf32>
                    %v_kj = tensor.extract %v[%k, %j] : tensor<10x512xf32>
                    %prod = arith.mulf %s_ik, %v_kj : f32
                    %new_acc = arith.addf %acc, %prod : f32
                    linalg.yield %new_acc : f32
            } -> tensor<f32>
            %result_scalar = tensor.extract %dot[] : tensor<f32>

            linalg.yield %result_scalar : f32
    } -> tensor<10x512xf32>

    func.return %result : tensor<10x512xf32>
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)

// command: 
//  ./bin/rocmlir-opt ../attention-linalg.mlir --canonicalize --canonicalize  --one-shot-bufferize="bufferize-function-boundaries" \
//      --linalg-generalize-named-ops --convert-linalg-to-affine-loops --canonicalize --cse \
//      --lower-affine -convert-math-to-llvm -convert-scf-to-cf -convert-cf-to-llvm -convert-func-to-llvm\
//      --convert-math-to-llvm --reconcile-unrealized-casts -finalize-memref-to-llvm \
//      --convert-arith-to-llvm --reconcile-unrealized-casts |\
// 	external/llvm-project/llvm/bin/mlir-runner -e main -entry-point-result=void\
//	    --shared-libs=external/llvm-project/llvm/lib/libmlir_rocm_runtime.so,lib/libconv-validation-wrappers.so,external/llvm-project/llvm/lib/libmlir_runner_utils.so,external/llvm-project/llvm/lib/libmlir_c_runner_utils.so,external/llvm-project/llvm/lib/libmlir_async_runtime.so

func.func @main() {
    %q  = arith.constant dense<1.0> : tensor<10x512xf32>
    %kt = arith.constant dense<2.0> : tensor<512x10xf32>
    %v  = arith.constant dense<3.0> : tensor<10x512xf32>

    %result = func.call @causal_mask_attention(%q, %kt, %v)
        : (tensor<10x512xf32>, tensor<512x10xf32>, tensor<10x512xf32>) -> tensor<10x512xf32>

    %unranked = tensor.cast %result : tensor<10x512xf32> to tensor<*xf32>
    call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()
    return
}