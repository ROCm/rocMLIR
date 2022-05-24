// RUN: miopen-opt -miopen-lowering-step3 %s | FileCheck %s

// CHECK-LABEL: func @miopen_threadwise_copy_v2
func.func @miopen_threadwise_copy_v2(%source : vector<32xf16>,
                                %dest1D : memref<32xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[slice:.*]] = miopen.extract_slice{{.*}}: vector<32xf16> -> vector<4xf16>
  // CHECK: %[[ext:.*]] = arith.extf %[[slice]]
  // CHECK: miopen.buffer_store %[[ext]]{{.*}}: vector<4xf32> -> memref<32xf32>
  miopen.threadwise_copy_v2 {
      length = 4 : index, leftOobDims = [], rightOobDims = [],
      storeMethod = 0 : i32}
    %source[%c0] -> %dest1D[%c0]
    : vector<32xf16> -> memref<32xf32>, index
  func.return
}

