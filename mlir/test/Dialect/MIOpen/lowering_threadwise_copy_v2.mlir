// RUN: miopen-opt -miopen-lowering-step3 %s | FileCheck %s

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func.func @miopen_threadwise_copy_v2(%source : memref<32xf32, 5>,
                                %dest2D : memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[slice:.*]] = miopen.in_bounds_load{{.*}}: memref<32xf32, 5>, index -> vector<4xf32>
  // CHECK: miopen.buffer_store %[[slice]]{{.*}}: vector<4xf32> -> memref<32x32xf32>
  miopen.threadwise_copy_v2 {
      length = 4 : index, leftOobDims = [], rightOobDims = [],
      storeMethod = 0 : i32}
    %source[%c0] -> %dest2D[%c0, %c0]
    : memref<32xf32, 5> -> memref<32x32xf32>, index, index
  func.return
}

