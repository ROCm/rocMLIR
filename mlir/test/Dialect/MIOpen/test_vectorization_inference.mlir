// RUN: miopen-opt -miopen-vectorization-inference-test \
// RUN: -allow-unregistered-dialect %s \
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

#transform_map3 = #miopen.transform_map<affine_map<(d0, d1) -> (d0, 0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <Broadcast{1} ["a"] at [1] -> ["a"] at [1]>]
  bounds = [8,4] -> [8,1]>

#transform_map4 = #miopen.transform_map<affine_map<(d0, d1) -> (d0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <AddDim{1} ["a"] at [1] -> [] at []>]
  bounds = [8,1] -> [8]>

#transform_map5 = #miopen.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 4} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

#transform_map6 = #miopen.transform_map<affine_map<(d0) -> (d0 - 2)>
  by [<Pad{2, 2} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

#transform_map7 = #miopen.transform_map<affine_map<(d0, d1) -> (d0 floordiv 8, d1, d0 mod 8)>
  by [<Merge{4, 8} ["x"] at [0] -> ["a", "b"] at [0, 2]>,
  <PassThrough ["y"] at [1] -> ["y"] at [1]>]
  bounds = [32, 5] -> [4, 5, 8]>

#transform_map8 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{5} ["y"] at [1] -> [] at []>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 1]>]
  bounds = [4, 5, 8] -> [4, 8]>

#transform_map9 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["y"] at [1] -> ["y"] at [1]>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 2]>]
  bounds = [4, 5, 8] -> [4, 1, 8]>

// Test for the alignment bug
#transform_map_align_e11 = #miopen.transform_map<affine_map<(d0, d1) -> (d0 + d1)>
  by [<Embed{1, 1} ["a", "b"] at [0, 1] -> ["xpad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_e41 = #miopen.transform_map<affine_map<(d0, d1) -> (d0 * 2 + d1)>
  by [<Embed{4, 1} ["a", "b"] at [0, 1] -> ["xpad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_pad = #miopen.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 2} ["xpad"] at [0] -> ["x"] at [0]>]
  bounds = [16] -> [14]>

#transform_map_align_nopad = #miopen.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 0} ["xpad"] at [0] -> ["x"] at [0]>]
  bounds = [16] -> [16]>

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

  // CHECK-NEXT: result = 1
  %12 = "get_length"() {transforms = [#transform_map3], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<8x1xf32>)
  // Except a unit broadcast at the end is no impediment
  // CHECK-NEXT: result = 8
  %13 = "get_length"() {transforms = [#transform_map3], in_dim = 0 : index, max_len = 8 : index} : () -> (memref<8x1xf32>)
  // Barring this all being a merge and switch around result
  // CHECK-NEXT: result = 1
  %14 = "get_length"() {transforms = [#transform_map1, #transform_map0, #transform_map3], in_dim = 0 : index, max_len = 8 : index} : () -> (memref<8x1xf32>)
  // Unless we drop the dimension
  // CHECK-NEXT: result = 8
  %15 = "get_length"() {transforms = [#transform_map1, #transform_map0, #transform_map3, #transform_map4],
    in_dim = 0 : index, max_len = 8 : index} : () -> (memref<8xf32>)

  // Padding by vector lengths behaves
  // CHECK-NEXT: result = 4
  %16 = "get_length"() {transforms = [#transform_map5], in_dim = 0 : index, max_len = 8 : index} : () -> (memref<4xf32>)
  // Left and right padding apply separately
  // CHECK-NEXT: result = 2
  %17 = "get_length"() {transforms = [#transform_map6], in_dim = 0 : index, max_len = 8 : index} : () -> (memref<4xf32>)

  // Intervening dimensions prevent merge
  // CHECK-NEXT: result = 8
  %18 = "get_length"() {transforms = [#transform_map7], in_dim = 0 : index, max_len = 32 : index} : () -> (memref<4x5x8xf32>)
  // Unless it was an illusion
  // CHECK-NEXT: result = 32
  %19 = "get_length"() {transforms = [#transform_map7, #transform_map8], in_dim = 0 : index, max_len = 32 : index} : () -> (memref<4x8xf32>)
  // or a broadcast
  // CHECK-NEXT: result = 32
  %20 = "get_length"() {transforms = [#transform_map7, #transform_map9], in_dim = 0 : index, max_len = 32 : index} : () -> (memref<4x1x8xf32>)

  // Regression test for padding and alignment
  // CHECK-NEXT: result = 1
  %21 = "get_length"() {transforms = [#transform_map_align_e11, #transform_map_align_pad], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<14xf32>)
  // CHECK-NEXT: result = 2
  %22 = "get_length"() {transforms = [#transform_map_align_e41, #transform_map_align_pad], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<14xf32>)
  // CHECK-NEXT: result = 4
  %23 = "get_length"() {transforms = [#transform_map_align_e11, #transform_map_align_nopad], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<16xf32>)
  // CHECK-NEXT: result = 4
  %24 = "get_length"() {transforms = [#transform_map_align_e41, #transform_map_align_nopad], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<16xf32>)

  func.return
}


// CHECK-LABEL: func @test_vectorization_align_pad
func.func @test_vectorization_align_pad() {
  func.return
}
