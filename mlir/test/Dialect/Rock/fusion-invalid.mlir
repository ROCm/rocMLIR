// RUN: rocmlir-opt -rock-linalg-align %s -verify-diagnostics -split-input-file
func.func @transpose_in_kernel(%arg0: memref<8x4xf32>, %arg1: memref<32xf32, 5>) attributes {kernel = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2.0 : f32
  // expected-error@+1 {{'memref.alloc' op could not use fusion to eliminate this intermediate buffer. Kernel compilation canot proceed}}
  %0 = memref.alloc() : memref<4x8xf32>
  rock.transforming_for (%arg2, %arg3) = [](%c0, %c0) (%arg4) = validity bounds [1, 8] strides [1, 1] {
    rock.global_store %arg1[%arg3] -> %0[%arg2, %arg3] if %arg4
      storeMethod(set) {length = 1 : index}
      : memref<32xf32, 5> -> memref<4x8xf32>, index, index
  }
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]} ins(%0 : memref<4x8xf32>) outs(%arg0 : memref<8x4xf32>) {
  ^bb0(%arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg4, %c2 : f32
    linalg.yield %1 : f32
  }
  func.return
}
