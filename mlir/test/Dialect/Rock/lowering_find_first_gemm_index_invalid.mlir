// RUN: rocmlir-opt --rock-find-first-gemm-index %s -verify-diagnostics

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

func.func @error_no_gemm_input_trace(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>) attributes {kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  // expected-error @below {{Cannot trace first gemm index for linalg.generic op}}
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %fresh_alloc = memref.alloc() : memref<16x16xf32>
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      
      // This linalg op doesn't use the GEMM result at all
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} 
        ins(%fresh_alloc : memref<16x16xf32>) 
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.constant 1.0 : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, firstGemmIndices = array<i64: 0>, softmaxType = f32}
  
  memref.copy %alloc, %arg4 : memref<16x16xf32> to memref<16x16xf32>
  return
}

func.func @error_multiple_gemm_inputs_trace(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>) attributes {kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
   // expected-error @below {{Multiple inputs trace back to first gemm argument}}
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %t1 = rock.transform %gemm0_out by <affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["dim0", "dim1"] at [0, 1] -> ["dim0", "dim1"] at [0, 1]>] bounds = [16, 16] -> [16, 16]> : memref<16x16xf32> to memref<16x16xf32>
      %t2 = rock.transform %gemm0_out by <affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim0", "dim1"] at [0, 1]>] bounds = [16, 16] -> [16, 16]> : memref<16x16xf32> to memref<16x16xf32>
      
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      // Both inputs trace back to gemm0_out
      linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} 
        ins(%t1, %t2 : memref<16x16xf32>, memref<16x16xf32>) 
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %add = arith.addf %in, %in_1 : f32
        linalg.yield %add : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, firstGemmIndices = array<i64: 0>, softmaxType = f32}
  
  memref.copy %alloc, %arg4 : memref<16x16xf32> to memref<16x16xf32>
  return
}

func.func @error_invalid_firstGemmIndex(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>) attributes {kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  // expected-error @below {{First gemm index out of bounds for preSecondGemmRegion}}
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise otherIns(%arg3 : memref<16x16xf32>) {
    ^bb0(%gemm0_out: memref<16x16xf32>, %extra_arg: memref<16x16xf32>, %output: memref<16x16xf32>):
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} 
        ins(%gemm0_out, %extra_arg : memref<16x16xf32>, memref<16x16xf32>) 
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %add = arith.addf %in, %in_1 : f32
        linalg.yield %add : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, firstGemmIndices = array<i64: 3>, softmaxType = f32}
  
  memref.copy %alloc, %arg4 : memref<16x16xf32> to memref<16x16xf32>
  return
}


// expected-error @+1 {{More than one gemm+gemm like operation found, expected only one.}}
func.func @error_multiple_attention_ops(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>,
                                         %arg3: memref<16x16xf32>, %arg4: memref<16x16xf32>, %arg5: memref<16x16xf32>) attributes {kernel} {
  %alloc0 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%gemm0_out : memref<16x16xf32>)
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.maximumf %in, %in : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc0 = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, firstGemmIndices = array<i64: 0>, softmaxType = f32}

  // Second attention operation - will trigger the error
  %alloc1 = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
  rock.attention{
    qk = %arg3 * %arg4 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%gemm0_out : memref<16x16xf32>)
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.maximumf %in, %in : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc1 = softmax(qk) * %arg5 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, firstGemmIndices = array<i64: 0>, softmaxType = f32}

  return
}

func.func @error_multiple_firstGemmIndices_values(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) attributes {kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>

// expected-error @+1 {{Expected exactly one first gemm index, found: 2}}
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%gemm0_out : memref<16x16xf32>)
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.maximumf %in, %in : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>,
     // Error: Multiple indices provided in firstGemmIndices
     firstGemmIndices = array<i64: 0, 1>, softmaxType = f32}

  memref.copy %alloc, %arg3 : memref<16x16xf32> to memref<16x16xf32>
  return
}

func.func @error_empty_firstGemmIndices(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>, %arg3: memref<16x16xf32>) attributes {kernel} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x16xf32>
// expected-error @+1 {{Expected exactly one first gemm index, found: 0}}
  rock.attention{
    qk = %arg0 * %arg1 : memref<16x16xf32>, memref<16x16xf32>
    qk = elementwise {
    ^bb0(%gemm0_out: memref<16x16xf32>, %output: memref<16x16xf32>):
      %alloc_0 = memref.alloc() : memref<16x16xf32>
      linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]}
        ins(%gemm0_out : memref<16x16xf32>)
        outs(%alloc_0 : memref<16x16xf32>) {
      ^bb0(%in: f32, %out: f32):
        %0 = arith.maximumf %in, %in : f32
        linalg.yield %0 : f32
      }
      memref.copy %alloc_0, %output : memref<16x16xf32> to memref<16x16xf32>
      rock.yield
    }
    %alloc = softmax(qk) * %arg2 : memref<16x16xf32> -> memref<16x16xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>,
     // Error: Empty firstGemmIndices array
     firstGemmIndices = array<i64>, softmaxType = f32}

  memref.copy %alloc, %arg3 : memref<16x16xf32> to memref<16x16xf32>
  return
}
