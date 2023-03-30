//./bin/mlir-opt --convert-linalg-to-affine-loops --lower-affine --convert-linalg-to-llvm --convert-std-to-llvm ../mlir/examples/prototypes/threadwise_gemm.mlir | ./bin/mlir-cpu-runner -e main --entry-point-result=void -shared-libs=lib/%prefix_mlir_runner_utils.so
//Unranked Memref rank = 2 descriptor@ = 0x7ffe573badf0
//Memref base@ = 0x4ff5260 rank = 2 offset = 0 sizes = [4, 7] strides = [7, 1] data =
//[[5,   5,   5,   5,   5,   5,   5],
// [5,   5,   5,   5,   5,   5,   5],
// [5,   5,   5,   5,   5,   5,   5],
// [5,   5,   5,   5,   5,   5,   5]]
   
module{
func @main() -> (){
  %A = alloc() : memref<5x4xf32>
  %B = alloc() : memref<5x7xf32>
  %C = alloc() : memref<4x7xf32>
  %cf1 = constant 1.00000e+00 : f32
  linalg.fill(%A, %cf1) : memref<5x4xf32>, f32 
  linalg.fill(%B, %cf1) : memref<5x7xf32>, f32 
  %cf2 = constant 0.00000e+00 : f32
  linalg.fill(%C, %cf2) : memref<4x7xf32>, f32
  call @threadwise_gemm_affine(%A, %B, %C) : (memref<5x4xf32>, memref<5x7xf32>, memref<4x7xf32>) -> ()  
  %U = memref_cast %C : memref<4x7xf32, 0> to memref<*xf32>
  call @print_memref_f32(%U):(memref<*xf32>) -> ()
  return 
}

func @threadwise_gemm_affine(%arg0:memref<5x4xf32>, %arg1: memref<5x7xf32>, %arg2: memref<4x7xf32>) -> (){
  affine.for %k = 0 to 5 { 
    affine.for %m = 0 to 4 {
      affine.for %n = 0 to 7 {
        %1 = affine.load %arg0[%k, %m] : memref<5x4xf32>
        %2 = affine.load %arg1[%k, %n] : memref<5x7xf32>
        %3 = mulf %2, %1 : f32
        %4 = affine.load %arg2[%m, %n] : memref<4x7xf32>
        %5 = addf %4, %3 : f32
        affine.store %5, %arg2[%m, %n] : memref<4x7xf32>
      }
    }
  }
  return
}

func @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

}
