// RUN: miopen-opt -miopen-blockwise-gemm-to-threadwise %s | FileCheck %s

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func.func @miopen_threadwise_copy_v2(%source : memref<32xf32, 5>,
                                %dest2D : memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[slice:.*]] = miopen.in_bounds_load{{.*}}: memref<32xf32, 5>, index -> vector<4xf32>
  // CHECK: miopen.buffer_store set %[[slice]]{{.*}}: vector<4xf32> -> memref<32x32xf32>
  miopen.threadwise_copy_v2 %source[%c0] -> %dest2D[%c0, %c0]
      storeMethod(set) {
      length = 4 : index, leftOobDims = [], rightOobDims = [] }
    : memref<32xf32, 5> -> memref<32x32xf32>, index, index
  func.return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2_long
func.func @miopen_threadwise_copy_v2_long(%source : memref<32xf32, 5>,
                                %dest2D : memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}[%[[coords:[^{]*]]{{.*}}: vector<4xf32> -> memref<32x32xf32>
  // CHECK: miopen.buffer_store set {{.*}}%[[coords]]{{.*}} offset = 4 : index{{.*}}: vector<4xf32> -> memref<32x32xf32>
  miopen.threadwise_copy_v2 %source[%c0] -> %dest2D[%c0, %c0]
      storeMethod(set) {
      length = 8 : index, leftOobDims = [], rightOobDims = [] }
    : memref<32xf32, 5> -> memref<32x32xf32>, index, index
  func.return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2_8xf16
func.func @miopen_threadwise_copy_v2_8xf16(%source : memref<32xf16, 5>,
                                           %dest : memref<64xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_store
  // CHECK-NOT: miopen.buffer_store
  miopen.threadwise_copy_v2 %source[%c0] -> %dest[%c0]
      storeMethod(set) {
      length = 8 : index, leftOobDims = [], rightOobDims = [] }
    : memref<32xf16, 5> -> memref<64xf16>, index
  func.return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2_7xf16
func.func @miopen_threadwise_copy_v2_7xf16(%source : memref<32xf16, 5>,
                                           %dest : memref<64xf16>) {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_store {{.*}} : vector<4xf16>
  // CHECK: miopen.buffer_store {{.*}} : vector<2xf16>
  // CHECK: miopen.buffer_store {{.*}} : f16
  // CHECK-NOT: miopen.buffer_store
  miopen.threadwise_copy_v2 %source[%c0] -> %dest[%c0]
      storeMethod(set) {
      length = 7 : index, leftOobDims = [], rightOobDims = [] }
    : memref<32xf16, 5> -> memref<64xf16>, index
  func.return
}
