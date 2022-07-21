// RUN: miopen-opt -miopen-vectorization-inference-test \
// RUN: -allow-unregistered-dialect --mlir-print-ir-after-all %s 2>&1 \
// RUN: | FileCheck %s

#transform_map0 = #miopen.transform_map<affine_map<(d0, d1) -> (d1, d0)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [1, 0]>]
  bounds = [4, 8] -> [8, 4]>
#transform_map1 = #miopen.transform_map<affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>
  by [<Merge{4, 8} ["x"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [32] -> [4, 8]>
#transform_map2 = #miopen.transform_map<affine_map<(d0, d1) -> (d0 + d1 * 8)>
  by [<Embed{8,1} ["a", "b"] at [0, 1] -> ["x"] at [0]>]
  bounds = [4,8] -> [32]>

// CHECK-LABEL: func.func @test
func.func @test_vectorization() {
  // CHECK-NEXT: result = 4
  %0 = "get_length"() {transforms = [], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<4x8xf32>)
  // CHECK-NEXT: result = 1
  %1 = "get_length"() {transforms = [], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<4x8xf32>)

  // CHECK-NEXT: result = 4
  %2 = "get_length"() {transforms = [#transform_map0], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<8x4xf32>)
  // CHECK-NEXT: result = 1
  %3 = "get_length"() {transforms = [#transform_map0], in_dim = 1 : index, max_len = 8 : index} : () -> (memref<8x4xf32>)

  // CHECK-NEXT: result = 8
  %4 = "get_length"() {transforms = [#transform_map1], in_dim = 0 : index, max_len = 8 : index} : () -> (memref<4x8xf32>)
  // CHECK-NEXT: result = 1
  %5 = "get_length"() {transforms = [#transform_map1], in_dim = 0 : index, max_len = 3 : index} : () -> (memref<4x8xf32>)
  // CHECK-NEXT: result = 2
  %6 = "get_length"() {transforms = [#transform_map1], in_dim = 0 : index, max_len = 6 : index} : () -> (memref<4x8xf32>)

  // CHECK-NEXT: result = 8
  %7 = "get_length"() {transforms = [#transform_map2], in_dim = 1 : index, max_len = 8 : index} : () -> (memref<32xf32>)
  // CHECK-NEXT: result = 1
  %8 = "get_length"() {transforms = [#transform_map2], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<32xf32>)

  // CHECK-NEXT: result = 32
  %9 = "get_length"() {transforms = [#transform_map1, #transform_map2], in_dim = 0 : index, max_len = 32 : index} : () -> (memref<32xf32>)
  // CHECK-NEXT: result = 16
  %10 = "get_length"() {transforms = [#transform_map1, #transform_map2], in_dim = 0 : index, max_len = 16 : index} : () -> (memref<32xf32>)

  // Swapping around dimensions between merge and embed
  // CHECK-NEXT: result = 1
  %11 = "get_length"() {transforms = [#transform_map1, #transform_map0], in_dim = 0 : index, max_len = 32 : index} : () -> (memref<8x4xf32>)
  func.return
}
