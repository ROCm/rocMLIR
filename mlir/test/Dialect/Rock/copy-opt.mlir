// RUN: rocmlir-opt -rock-copy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @rock_keep_copy
#map0 = affine_map<(d0) -> (d0)>
func.func @rock_keep_copy(%arg0 : memref<32xf32>,
                          %arg1 : memref<32xf32>) {
  %c0 = arith.constant 0.000000e+00 : f32
  // CHECK: memref.copy
  %0 = memref.alloc() : memref<32xf32>
  linalg.fill ins(%c0 : f32) outs(%0 : memref<32xf32>)
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg0 : memref<32xf32>) outs(%0 : memref<32xf32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      %9 = arith.addf %arg4, %c0 : f32
      linalg.yield %9 : f32
  }
  memref.copy %0, %arg1 : memref<32xf32> to memref<32xf32>
  func.return
}

// CHECK-LABEL: func.func @rock_copy_opt
func.func @rock_copy_opt(%arg0 : memref<32xf32>,
                         %arg1 : memref<32xf32>) {
  %c0 = arith.constant 0.000000e+00 : f32
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%arg1 :
  // CHECK-NOT: memref.copy
  %0 = memref.alloc() : memref<32xf32>
  linalg.generic {indexing_maps = [#map0, #map0], iterator_types = ["parallel"]} ins(%arg0 : memref<32xf32>) outs(%0 : memref<32xf32>) {
    ^bb0(%arg4: f32, %arg5: f32):
      %9 = arith.addf %arg4, %c0 : f32
      linalg.yield %9 : f32
  }
  memref.copy %0, %arg1 : memref<32xf32> to memref<32xf32>
  func.return
}

