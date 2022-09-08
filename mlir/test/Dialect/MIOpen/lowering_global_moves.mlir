// RUN: miopen-opt -miopen-blockwise-gemm-to-threadwise %s | FileCheck %s

// CHECK-LABEL: func.func @miopen_threadwise_copy_v2
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

// CHECK-LABEL: func.func @miopen_global_load
func.func @miopen_global_load(%source2D : memref<32x32xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[loaded:.*]] = miopen.buffer_load {{.*}}: memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret:.*]] = miopen.insert_slice %[[loaded]]
  %loaded = miopen.global_load %source2D[%c0, %c0]
    { leftOobDims = [], rightOobDims = [] }
    : memref<32x32xf32> -> vector<4xf32>
  // CHECK: return %[[ret]]
  func.return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @miopen_global_load_long
func.func @miopen_global_load_long(%source2D : memref<32x32xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ret_0:.*]] = arith.constant {{.*}} : vector<8xf32>
  // CHECK: %[[loaded_0:.*]] = miopen.buffer_load {{.*}}: memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret_1:.*]] = miopen.insert_slice %[[loaded_0]] -> %[[ret_0]]
  // CHECK: %[[loaded_1:.*]] = miopen.buffer_load {{.*}} offset = 4 {{.*}}: memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret:.*]] = miopen.insert_slice %[[loaded_1]] -> %[[ret_1]]
  %loaded = miopen.global_load %source2D[%c0, %c0]
    { leftOobDims = [], rightOobDims = [] }
    : memref<32x32xf32> -> vector<8xf32>
  // CHECK: return %[[ret]]
  func.return %loaded : vector<8xf32>
}

// CHECK-LABEL: func.func @miopen_global_load_8xf16
func.func @miopen_global_load_8xf16(%source2D : memref<32x32xf16>) -> vector<8xf16> {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_load
  // CHECK-NOT: miopen.buffer_load
  %loaded = miopen.global_load %source2D[%c0, %c0]
    { leftOobDims = [], rightOobDims = [] }
    : memref<32x32xf16> -> vector<8xf16>
  func.return %loaded : vector<8xf16>
}

// CHECK-LABEL: func.func @miopen_global_load_7xf16
func.func @miopen_global_load_7xf16(%source2D : memref<32x32xf16>) -> vector<7xf16> {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_load
  // CHECK: miopen.buffer_load
  // CHECK: miopen.buffer_load
  // CHECK-NOT: miopen.buffer_load
  %loaded = miopen.global_load %source2D[%c0, %c0]
    { leftOobDims = [], rightOobDims = [] }
    : memref<32x32xf16> -> vector<7xf16>
  func.return %loaded : vector<7xf16>
}

// CHECK-LABEL: func.func @miopen_global_load_7xi8
func.func @miopen_global_load_7xi8(%source2D : memref<32x32xi8>) -> vector<7xi8> {
  %c0 = arith.constant 0 : index
  // CHECK: miopen.buffer_load {{.*}} -> vector<4xi8>
  // CHECK: miopen.buffer_load {{.*}} -> i8
  // CHECK: miopen.buffer_load {{.*}} -> i8
  // CHECK: miopen.buffer_load {{.*}} -> i8
  // CHECK-NOT: miopen.buffer_load
  %loaded = miopen.global_load %source2D[%c0, %c0]
    { leftOobDims = [], rightOobDims = [] }
    : memref<32x32xi8> -> vector<7xi8>
  func.return %loaded : vector<7xi8>
}

