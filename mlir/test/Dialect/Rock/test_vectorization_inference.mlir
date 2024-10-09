// RUN: rocmlir-opt -rock-vectorization-inference-test \
// RUN: -allow-unregistered-dialect %s \
// RUN: | FileCheck %s

// CHECK-LABEL: @test_raw
func.func @test_raw(%buf: memref<4x8xf32>) {
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%buf) {in_dim = 1 : index, in_dim_len = 4 : index} : (memref<4x8xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_raw_wrong_dim
func.func @test_raw_wrong_dim(%buf: memref<4x8xf32>) {
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%buf) {in_dim = 0 : index} : (memref<4x8xf32>) -> ()
  func.return
}

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [1, 0]>]
  bounds = [4, 8] -> [8, 4]>

// CHECK-LABEL: @test_tr
func.func @test_tr(%buf: memref<8x4xf32>) {
  %0 = rock.transform %buf by #transform_map0 : memref<8x4xf32> to memref<4x8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%0) {in_dim = 0 : index} : (memref<4x8xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_tr_wrong_dim
func.func @test_tr_wrong_dim(%buf: memref<8x4xf32>) {
  %0 = rock.transform %buf by #transform_map0 : memref<8x4xf32> to memref<4x8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%0) {in_dim = 1 : index} : (memref<4x8xf32>) -> ()
  func.return
}

#transform_map1 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>
  by [<Merge{4, 8} ["1"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [32] -> [4, 8]>

// CHECK-LABEL: @test_merge_dim_len
func.func @test_merge_dim_len(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%0) {in_dim = 0 : index, in_dim_len = 8 : index} : (memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_merge
func.func @test_merge(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 32
  "get_length"(%0) {in_dim = 0 : index} : (memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_merge_limit_for_datatype
func.func @test_merge_limit_for_datatype(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%0) {in_dim = 0 : index, limitForDataType} : (memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_merge_misaligned
func.func @test_merge_misaligned(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%0) {in_dim = 0 : index, in_dim_len = 3 : index} : (memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_merge_partially_aligned
func.func @test_merge_partially_aligned(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 2
  "get_length"(%0) {in_dim = 0 : index, in_dim_len = 6 : index} : (memref<32xf32>) -> ()
  func.return
}

#transform_map2 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + d0 * 8)>
  by [<Embed{8,1} ["a", "b"] at [0, 1] -> ["1"] at [0]>]
  bounds = [4,8] -> [32]>

// CHECK-LABEL: @test_embed
func.func @test_embed(%buf: memref<32xf32>) {
  %0 = rock.transform %buf by #transform_map2 : memref<32xf32> to memref<4x8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%0) {in_dim = 1 : index} : (memref<4x8xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_embed_wrong_dim
func.func @test_embed_wrong_dim(%buf: memref<32xf32>) {
  %0 = rock.transform %buf by #transform_map2 : memref<32xf32> to memref<4x8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%0) {in_dim = 0 : index} : (memref<4x8xf32>) -> ()
  func.return
}


// CHECK-LABEL: @test_merge_embed
func.func @test_merge_embed(%buf: memref<32xf32>) {
  %0 = rock.transform %buf by #transform_map2 : memref<32xf32> to memref<4x8xf32>
  %1 = rock.transform %0 by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 32
  "get_length"(%1) {in_dim = 0 : index} : (memref<32xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_merge_embed_clamped
func.func @test_merge_embed_clamped(%buf: memref<32xf32>) {
  %0 = rock.transform %buf by #transform_map2 : memref<32xf32> to memref<4x8xf32>
  %1 = rock.transform %0 by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 16
  "get_length"(%1) {in_dim = 0 : index, in_dim_len = 16 : index} : (memref<32xf32>) -> ()
  func.return
}


// Swapping around dimensions between merge and embed
// CHECK-LABEL: @test_merge_tr
func.func @test_merge_tr(%buf: memref<8x4xf32>) {
  %0 = rock.transform %buf by #transform_map0 : memref<8x4xf32> to memref<4x8xf32>
  %1 = rock.transform %0 by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%1) {in_dim = 0 : index} : (memref<32xf32>) -> ()
  func.return
}


#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <Broadcast{1} ["a"] at [1] -> ["a"] at [1]>]
  bounds = [8,4] -> [8,1]>

// CHECK-LABEL: @test_broadcast
func.func @test_broadcast(%buf: memref<8x1xf32>) {
  %0 = rock.transform %buf by #transform_map3 : memref<8x1xf32> to memref<8x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%0) {in_dim = 1 : index} : (memref<8x4xf32>) -> ()
  func.return
}

// Except a unit broadcast at the end is no impediment
// CHECK-LABEL: @test_unit_broadcast_vectorizes_subsequent
func.func @test_unit_broadcast_vectorizes_subsequent(%buf: memref<8x1xf32>) {
  %0 = rock.transform %buf by #transform_map3 : memref<8x1xf32> to memref<8x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%0) {in_dim = 0 : index} : (memref<8x4xf32>) -> ()
  func.return
}

// Barring this all being a merge and switch around result
// CHECK-LABEL: @test_merge_tr_broadcast
func.func @test_merge_tr_broadcast(%buf: memref<8x1xf32>) {
  %0 = rock.transform %buf by #transform_map3 : memref<8x1xf32> to memref<8x4xf32>
  %1 = rock.transform %0 by #transform_map0 : memref<8x4xf32> to memref<4x8xf32>
  %2 = rock.transform %1 by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%2) {in_dim = 0 : index} : (memref<32xf32>) -> ()
  func.return
}

#transform_map4 = #rock.transform_map<affine_map<(d0, d1) -> (d0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <AddDim{1} ["a"] at [1] -> [] at []>]
  bounds = [8,1] -> [8]>

// Unless we drop the dimension
// CHECK-LABEL: @test_merge_tr_broadcast_adddim
func.func @test_merge_tr_broadcast_adddim(%buf: memref<8xf32>) {
  %0 = rock.transform %buf by #transform_map4 : memref<8xf32> to memref<8x1xf32>
  %1 = rock.transform %0 by #transform_map3 : memref<8x1xf32> to memref<8x4xf32>
  %2 = rock.transform %1 by #transform_map0 : memref<8x4xf32> to memref<4x8xf32>
  %3 = rock.transform %2 by #transform_map1 : memref<4x8xf32> to memref<32xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%3) {in_dim = 0 : index} : (memref<32xf32>) -> ()
  func.return
}

#transform_map5 = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 4} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

#transform_map6 = #rock.transform_map<affine_map<(d0) -> (d0 - 2)>
  by [<Pad{2, 2} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

// Padding by vector lengths behaves
// CHECK-LABEL: @test_pad_vector_len
func.func @test_pad_vector_len(%buf: memref<4xf32>) {
  %0 = rock.transform %buf by #transform_map5 : memref<4xf32> to memref<8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%0) {in_dim = 0 : index} : (memref<8xf32>) -> ()
  func.return
}

// Left and right padding apply separately
// CHECK-LABEL: @test_left_right_pad
func.func @test_left_right_pad(%buf: memref<4xf32>) {
  %0 = rock.transform %buf by #transform_map6 : memref<4xf32> to memref<8xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 2
  "get_length"(%0) {in_dim = 0 : index} : (memref<8xf32>) -> ()
  func.return
}

#transform_map7 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 8, d1, d0 mod 8)>
  by [<Merge{4, 8} ["1"] at [0] -> ["a", "b"] at [0, 2]>,
  <PassThrough ["0"] at [1] -> ["0"] at [1]>]
  bounds = [32, 5] -> [4, 5, 8]>

// Intervening dimensions prevent merge
// CHECK-LABEL: @test_partial_merge
func.func @test_partial_merge(%buf: memref<4x5x8xf32>) {
  %0 = rock.transform %buf by #transform_map7 : memref<4x5x8xf32> to memref<32x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%0) {in_dim = 0 : index} : (memref<32x5xf32>) -> ()
  func.return
}

#transform_map8 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{5} ["0"] at [1] -> [] at []>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 1]>]
  bounds = [4, 5, 8] -> [4, 8]>

// Unless it was an illusion
// CHECK-LABEL: @test_partial_merge_adddim
func.func @test_partial_merge_adddim(%buf: memref<4x8xf32>) {
  %0 = rock.transform %buf by #transform_map8 : memref<4x8xf32> to memref<4x5x8xf32>
  %1 = rock.transform %0 by #transform_map7 : memref<4x5x8xf32> to memref<32x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 32
  "get_length"(%1) {in_dim = 0 : index} : (memref<32x5xf32>) -> ()
  func.return
}

#transform_map9 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["0"] at [1] -> ["0"] at [1]>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 2]>]
  bounds = [4, 5, 8] -> [4, 1, 8]>

// or a broadcast
// CHECK-LABEL: @test_partial_merge_broadcast
func.func @test_partial_merge_broadcast(%buf: memref<4x1x8xf32>) {
  %0 = rock.transform %buf by #transform_map9 : memref<4x1x8xf32> to memref<4x5x8xf32>
  %1 = rock.transform %0 by #transform_map7 : memref<4x5x8xf32> to memref<32x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 32
  "get_length"(%1) {in_dim = 0 : index} : (memref<32x5xf32>) -> ()
  func.return
}

#transform_map_align_e11 = #rock.transform_map<affine_map<(d0, d1) -> (d0 + d1)>
  by [<Embed{1, 1} ["a", "b"] at [0, 1] -> ["1pad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_e41 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 2 + d1)>
  by [<Embed{4, 1} ["a", "b"] at [0, 1] -> ["1pad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_pad = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 2} ["1pad"] at [0] -> ["1"] at [0]>]
  bounds = [16] -> [14]>

#transform_map_align_nopad = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 0} ["1pad"] at [0] -> ["1"] at [0]>]
  bounds = [16] -> [16]>

// Regression tests for padding and alignment - namely, you can't vectorize
// through an otherwive vectorizable pad without alignment guarantees
// CHECK-LABEL: @test_embed_1_1_pad
func.func @test_embed_1_1_pad(%buf: memref<14xf32>) {
  %0 = rock.transform %buf by #transform_map_align_pad : memref<14xf32> to memref<16xf32>
  %1 = rock.transform %0 by #transform_map_align_e11 : memref<16xf32> to memref<4x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%1) {in_dim = 1 : index} : (memref<4x4xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_embed_4_1_pad
func.func @test_embed_4_1_pad(%buf: memref<14xf32>) {
  %0 = rock.transform %buf by #transform_map_align_pad : memref<14xf32> to memref<16xf32>
  %1 = rock.transform %0 by #transform_map_align_e41 : memref<16xf32> to memref<4x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 2
  "get_length"(%1) {in_dim = 1 : index} : (memref<4x4xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_embed_1_1_trivial_pad
func.func @test_embed_1_1_trivial_pad(%buf: memref<16xf32>) {
  %0 = rock.transform %buf by #transform_map_align_nopad : memref<16xf32> to memref<16xf32>
  %1 = rock.transform %0 by #transform_map_align_e11 : memref<16xf32> to memref<4x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 1 : index} : (memref<4x4xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_embed_4_1_trivial_pad
func.func @test_embed_4_1_trivial_pad(%buf: memref<16xf32>) {
  %0 = rock.transform %buf by #transform_map_align_nopad : memref<16xf32> to memref<16xf32>
  %1 = rock.transform %0 by #transform_map_align_e41 : memref<16xf32> to memref<4x4xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 1 : index} : (memref<4x4xf32>) -> ()
  func.return
}

// Unfold detection transforms
#transform_merge = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, d0 mod 3)>
  by [<Merge{8, 3} ["1"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [24] -> [8, 3]>

#transform_merge_2 = #rock.transform_map<affine_map<(d0, d1) -> (d1 floordiv 4, d0, (d1 mod 4) floordiv 2, d1 mod 2)>
  by [<Merge{128, 2, 2} ["gemmN"] at [1] -> ["N", "Ho", "Wo"] at [0, 2, 3]>,
      <PassThrough ["gemmM"] at [0] -> ["k"] at [1]>]
  bounds = [32, 512] -> [128, 32, 2, 2]>

#transform_merge_4 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 9, (d0 mod 3) floordiv 3, d0 mod 3)>
  by [<Merge{8, 3, 3} ["1"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [72] -> [8, 3, 3]>

#transform_transpose = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
   by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [1, 0]>]
   bounds = [8, 3] -> [3, 8]>

#transform_shuffle = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>
  by [<PassThrough ["N", "K", "Ho", "Wo"] at [0, 1, 2, 3] -> ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0]>]
  bounds = [128, 32, 2, 2] -> [2, 2, 32, 128]>

#transform_shuffle_4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  by [<PassThrough ["a", "b", "c"] at [0, 1, 2] -> ["a", "c", "b"] at [0, 2, 1]>]
  bounds = [8, 3, 3] -> [8, 3, 3]>

#transform_unmerge = #rock.transform_map<affine_map<(d0, d1) -> (3*d1 + d0)>
  by [<Unmerge{8, 3} ["a", "b"] at [1, 0] -> ["1"] at [0]>]
  bounds = [3, 8] -> [24]>

#transform_unmerge_2 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 + 2*d1 + 4*d2 + 128*d3)>
  by [<Unmerge{128, 32, 2, 2} ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0] -> ["1"] at [0]>]
  bounds = [2, 2, 32, 128] -> [16384]>

// Tests for unfold detection
// CHECK-LABEL: @test_merge_tr_unmerge
func.func @test_merge_tr_unmerge(%buf: memref<24xf32>) {
  %0 = rock.transform %buf by #transform_unmerge : memref<24xf32> to memref<3x8xf32>
  %1 = rock.transform %0 by #transform_transpose : memref<3x8xf32> to memref<8x3xf32>
  %2 = rock.transform %1 by #transform_merge : memref<8x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24xf32>) -> ()
  func.return
}


// CHECK-LABEL: @test_implicit_unmerge
func.func @test_implicit_unmerge(%buf: memref<8x3xf32>) {
  %0 = rock.transform %buf by #transform_merge : memref<8x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%0) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24xf32>) -> ()
  func.return
}


// CHECK-LABEL: @test_merge_shuffle_unmerge
func.func @test_merge_shuffle_unmerge(%buf: memref<16384xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_2 : memref<16384xf32> to memref<2x2x32x128xf32>
  %1 = rock.transform %0 by #transform_shuffle : memref<2x2x32x128xf32> to memref<128x32x2x2xf32>
  %2 = rock.transform %1 by #transform_merge_2 : memref<128x32x2x2xf32> to memref<32x512xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 1 : index} : (memref<32x512xf32>) -> ()
  func.return
}

// CHECK-LABEL: @test_twisted_dimensions_dont_collapse
func.func @test_twisted_dimensions_dont_collapse(%buf: memref<8x3x3xf32>) {
  %0 = rock.transform %buf by #transform_shuffle_4 : memref<8x3x3xf32> to memref<8x3x3xf32>
  %1 = rock.transform %0 by #transform_merge_4 : memref<8x3x3xf32> to memref<72xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%1) {in_dim = 0 : index} : (memref<72xf32>) -> ()
  func.return
}

#transform_merge_3 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 3, d1, d0 mod 3)>
  by [<Merge{8, 3} ["1"] at [0] -> ["a", "b"] at [0, 2]>,
  <PassThrough ["0"] at [1] -> ["0"] at [1]>]
  bounds = [24, 5] -> [8, 5, 3]>

// Intervening dimensions still prevent merge
// CHECK-LABEL @test_no_contiguous_intervening_dim
func.func @test_no_contiguous_intervening_dim(%buf: memref<8x5x3xf32>) {
  %0 = rock.transform %buf by #transform_merge_3 : memref<8x5x3xf32> to memref<24x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%0) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24x5xf32>) -> ()
  func.return
}

#transform_shuffle_2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{5} ["0"] at [1] -> [] at []>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 1]>]
  bounds = [8, 5, 3] -> [8, 3]>

#transform_shuffle_3 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["0"] at [1] -> ["0"] at [1]>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 2]>]
  bounds = [8, 5, 3] -> [8, 1, 3]>

// Unless it was an illusion
// CHECK-LABEL: @test_intervening_adddim_allows_contiguous
func.func @test_intervening_adddim_allows_contiguous(%buf: memref<8x3xf32>) {
  %0 = rock.transform %buf by #transform_shuffle_2 : memref<8x3xf32> to memref<8x5x3xf32>
  %1 = rock.transform %0 by #transform_merge_3 : memref<8x5x3xf32> to memref<24x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24x5xf32>) -> ()
  func.return
}

// or a broadcast
// CHECK-LABEL: @test_intervening_broadcast_allows_contiguous
func.func @test_intervening_broadcast_allows_contiguous(%buf: memref<8x1x3xf32>) {
  %0 = rock.transform %buf by #transform_shuffle_3 : memref<8x1x3xf32> to memref<8x5x3xf32>
  %1 = rock.transform %0 by #transform_merge_3 : memref<8x5x3xf32> to memref<24x5xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24x5xf32>) -> ()
  func.return
}

#transform_unmerge_3 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (128*d3 + 4*d2+ 2*d1+ d0)>
  by [<Embed{128, 4, 2, 1} ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0] -> ["1"] at [0]>]
  bounds = [2, 2, 32, 128] -> [16384]>

// Unfold detection with embed
// CHECK-LABEL @test_merge_shuffle_embed
func.func @test_merge_shuffle_embed(%buf: memref<16384xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_3 : memref<16384xf32> to memref<2x2x32x128xf32>
  %1 = rock.transform %0 by #transform_shuffle : memref<2x2x32x128xf32> to memref<128x32x2x2xf32>
  %2 = rock.transform %1 by #transform_merge_2 : memref<128x32x2x2xf32> to memref<32x512xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 1 : index} : (memref<32x512xf32>) -> ()
  func.return
}

#transform_merge_5 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, d0 mod 3)>
  by [<Merge{4, 3} ["1"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [12] -> [4, 3]>

#transform_unmerge_5 = #rock.transform_map<affine_map<(d0,d1) -> (d0*3+d1)>
  by [<Embed{3, 1} ["a", "b"] at [0,1] -> ["1"] at [0]>]
  bounds = [4,3] -> [12]>

// CHECK-LABEL: @test_merge_embed_full_continuity
func.func @test_merge_embed_full_continuity(%buf: memref<12xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_5 : memref<12xf32> to memref<4x3xf32>
  %1 = rock.transform %0 by #transform_merge_5 : memref<4x3xf32> to memref<12xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 12
  "get_length"(%1) {in_dim = 0 : index} : (memref<12xf32>) -> ()
  func.return
}

#transform_unmerge_9 = #rock.transform_map<affine_map<(d0, d1) -> (d0 + d1*3)>
  by [<Embed{1, 3} ["a", "c"] at [0, 1] -> ["1"] at [0]>]
  bounds = [3, 8] -> [24]>

// CHECK-LABEL: @test_merge_transpose_transposing_embed
func.func @test_merge_transpose_transposing_embed(%buf: memref<12xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_9 : memref<12xf32> to memref<3x8xf32>
  %1 = rock.transform %0 by #transform_transpose : memref<3x8xf32> to memref<8x3xf32>
  %2 = rock.transform %1 by #transform_merge : memref<8x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 0 : index, in_dim_len = 4 : index} : (memref<24xf32>) -> ()
  func.return
}

#transform_merge_6 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 3, d0 mod 3, d1 floordiv 7, d1 mod 7)>
  by [<Merge{4, 3} ["1"] at [0] -> ["a", "b"] at [0, 1]>, <Merge{8,7} ["0"] at [1] -> ["c", "d"] at [2, 3]>]
  bounds = [12, 56] -> [4, 3, 8, 7]>

#transform_unmerge_6 = #rock.transform_map<affine_map<(d0,d1,d2,d3) -> (d0*3+d1,d2*7+d3)>
  by [<Embed{3, 1} ["a", "b"] at [0,1] -> ["1"] at [0]>, <Embed{7,1} ["b","c"] at [2,3] -> ["0"] at [1]>]
  bounds = [4,3,8,7] -> [12,56]>

// Partial unfold detection with embed
// CHECK-LABEL: @test_merge_embed_collapse
func.func @test_merge_embed_collapse(%buf: memref<12x56xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_6 : memref<12x56xf32> to memref<4x3x8x7xf32>
  %1 = rock.transform %0 by #transform_merge_6 : memref<4x3x8x7xf32> to memref<12x56xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 1 : index, in_dim_len = 4 : index} : (memref<12x56xf32>) -> ()
  func.return
}

#transform_unmerge_7 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0*3 + d1 + d0)>
  by [<Unmerge{8, 1, 3} ["a", "b", "c"] at [0, 1, 2] -> ["1"] at [0]>]
  bounds = [8,1,3] -> [24]>

#transform_merge_7 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, 0, d0 mod 3)>
  by [<Merge{8, 1, 3} ["1"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [24] -> [8, 1, 3]>

// More broadcast examples
// CHECK-LABEL: @test_unmerge_merge_extra_1
func.func @test_unmerge_merge_extra_1(%buf: memref<24xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_7 : memref<24xf32> to memref<8x1x3xf32>
  %1 = rock.transform %0 by #transform_merge_7 : memref<8x1x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 0 : index, limitForDataType} : (memref<24xf32>) -> ()
  func.return
}

#transform_unmerge_8 = #rock.transform_map<affine_map<(d0, d1) -> (d0*3 + d1)>
  by [<Unmerge{8, 3} ["a", "c"] at [0, 1] -> ["1"] at [0]>]
  bounds = [8,3] -> [24]>

#transform_shuffle_5 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{1} ["b"] at [1] -> [] at []>, <PassThrough ["a", "c"] at [0, 2] -> ["a", "b"] at [0,1]>]
  bounds = [8, 1, 3] -> [8, 3]>

// CHECK-LABEL: @test_merge_adddim_unmerge
func.func @test_merge_adddim_unmerge(%buf: memref<24xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_8 : memref<24xf32> to memref<8x3xf32>
  %1 = rock.transform %0 by #transform_shuffle_5 : memref<8x3xf32> to memref<8x1x3xf32>
  %2 = rock.transform %1 by #transform_merge_7 : memref<8x1x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 0 : index, limitForDataType} : (memref<24xf32>) -> ()
  func.return
}

#transform_merge_9 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, 0, d0 mod 3)>
  by [<Merge{8, 2, 3} ["1"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [48] -> [8, 2, 3]>

#transform_shuffle_6 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["b"] at [1] -> ["b"] at [1]>, <PassThrough ["a", "c"] at [0, 2] -> ["a", "c"] at [0,2]>]
  bounds = [8, 2, 3] -> [8, 1, 3]>

// Note: this isn't 4 because making sure the 8 part is divided by the
// broadcast length is important.
// CHECK-LABEL: @test_merge_broadcast_middle_unmerge
func.func @test_merge_broadcast_middle_unmerge(%buf: memref<24xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_7 : memref<24xf32> to memref<8x1x3xf32>
  %1 = rock.transform %0 by #transform_shuffle_6 : memref<8x1x3xf32> to memref<8x2x3xf32>
  %2 = rock.transform %1 by #transform_merge_9 : memref<8x2x3xf32> to memref<48xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 3
  // TODO: currently, the data type limiter is a gcd, but, really, this buffer
  // should vectorize by 3 without problems.
  "get_length"(%2) {in_dim = 0 : index} : (memref<48xf32>) -> ()
  func.return
}

#transform_inject_unit_const = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0, d1)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "c"] at [0, 2]>,
    <ConstDim{0, 1} [] at [] -> ["b"] at [1]>]
  bounds = [8, 3] -> [8, 1, 3]>

#transform_inject_non_unit_const = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0, d1)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "c"] at [0, 2]>,
    <ConstDim{0, 2} [] at [] -> ["b"] at [1]>]
  bounds = [8, 3] -> [8, 2, 3]>

#transform_unmerge_injected_non_unit = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2 + 3 * (d1 + d0 * 2))>
  by [<Unmerge{8, 2, 3} ["a", "b", "c"] at [0, 1, 2] -> ["1"] at [0]>]
  bounds = [8, 2, 3] -> [48]>


// CHECK-LABEL: @test_inject_unit_const
func.func @test_inject_unit_const(%buf: memref<24xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_7 : memref<24xf32> to memref<8x1x3xf32>
  %1 = rock.transform %0 by #transform_inject_unit_const : memref<8x1x3xf32> to memref<8x3xf32>
  %2 = rock.transform %1 by #transform_merge : memref<8x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%2) {in_dim = 0 : index, limitForDataType} : (memref<24xf32>) -> ()
  func.return
}

// Even though we injected a 0, it _could_ have been a 1.
// CHECK-LABEL: @test_inject_non_unit_const
func.func @test_inject_non_unit_const(%buf: memref<48xf32>) {
  %0 = rock.transform %buf by #transform_unmerge_injected_non_unit : memref<48xf32> to memref<8x2x3xf32>
  %1 = rock.transform %0 by #transform_inject_non_unit_const : memref<8x2x3xf32> to memref<8x3xf32>
  %2 = rock.transform %1 by #transform_merge : memref<8x3xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%2) {in_dim = 0 : index, limitForDataType} : (memref<24xf32>) -> ()
  func.return
}

#transform_map_embed_tiebreak1 = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0, d1)>
  by [<PassThrough ["1"] at [0] -> ["1"] at [0]>,
    <Merge{2, 8} ["0"] at [1] -> ["a", "b"] at [1, 2]>]
  bounds = [4, 16] -> [4, 2, 8]>

#transform_map_embed_tiebreak2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>
  by [<Embed{1, 1, 1} ["1", "a", "b"] at [0, 1, 2] -> ["d"] at [0]>]
  bounds = [4, 2, 8] -> [13]>

// CHECK-LABEL: @test_embed_tiebreak_1
func.func @test_embed_tiebreak_1(%buf: memref<13xf32>) {
  %0 = rock.transform %buf by #transform_map_embed_tiebreak2 : memref<13xf32> to memref<4x2x8xf32>
  %1 = rock.transform %0 by #transform_map_embed_tiebreak1 : memref<4x2x8xf32> to memref<4x16xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 8
  "get_length"(%1) {in_dim = 1 : index} : (memref<4x16xf32>) -> ()
  func.return
}

// iso-coefficient embed alignment checks
#transform_map_over_vec_bottom1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 45, d0, d1 floordiv 4, (d1 mod 4) floordiv 2, (d2 mod 45) floordiv 5, d1 mod 2, d2 mod 5)>
  by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>,
    <Merge{64, 2, 2} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>,
    <Merge{64, 9, 5} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>]
  bounds = [1, 256, 2880] -> [64, 1, 64, 2, 9, 2, 5]>
#transform_map_over_vec_bottom2 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)>
  by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>,
    <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>,
    <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>]
  bounds = [64, 1, 64, 2, 9, 2, 5] -> [64, 1, 64, 10, 6]>
#transform_map_over_vec_bottom3 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 3, d4)>
  by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>,
    <PassThrough ["gi"] at [1] -> ["gi"] at [1]>,
    <PassThrough ["ci"] at [2] -> ["ci"] at [2]>,
    <Pad{3, 3, 0, 2} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>]
  bounds = [64, 1, 64, 10, 6] -> [64, 1, 64, 4, 4]>

// CHECK-LABEL: @test_padded_conv2gemm_input
func.func @test_padded_conv2gemm_input(%buf: memref<64x1x64x4x4xf32>) {
  %0 = rock.transform %buf by #transform_map_over_vec_bottom3 : memref<64x1x64x4x4xf32> to memref<64x1x64x10x6xf32>
  %1 = rock.transform %0 by #transform_map_over_vec_bottom2 : memref<64x1x64x10x6xf32> to memref<64x1x64x2x9x2x5xf32>
  %2 = rock.transform %1 by #transform_map_over_vec_bottom1 : memref<64x1x64x2x9x2x5xf32> to memref<1x256x2880xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%2) {in_dim = 1 : index, in_dim_len = 16 : index} : (memref<1x256x2880xf32>) -> ()
  func.return
}

#transform_map_over_vec_top1 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 4, d4 mod 4, d5 floordiv 4, d5 mod 4)>
  by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>,
    <Merge{16, 4} ["tid"] at [4] -> ["n_thread", "k_thread"] at [4, 5]>,
    <Merge{4, 4} ["iter"] at [5] -> ["n_iter", "k_iter"] at [6, 7]>]
  bounds = [16, 1, 2, 45, 64, 16] -> [16, 1, 2, 45, 16, 4, 4, 4]>
#transform_map_over_vec_top2 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 4 + d5) * 4 + d7, (d3 * 16 + d4) * 4 + d6)>
  by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>,
    <Unmerge{16, 4, 4} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>,
    <Unmerge{45, 16, 4} ["n_block", "n_thread", "n_iter"] at [3, 4, 6] -> ["n"] at [2]>,
    <AddDim{2} ["m_block"] at [2] -> [] at []>]
  bounds = [16, 1, 2, 45, 16, 4, 4, 4] -> [1, 256, 2880]>

// CHECK-LABEL: @test_padded_conv2gemm_input_with_mfmas
func.func @test_padded_conv2gemm_input_with_mfmas(%buf: memref<64x1x64x4x4xf32>) {
  %0 = rock.transform %buf by #transform_map_over_vec_bottom3 : memref<64x1x64x4x4xf32> to memref<64x1x64x10x6xf32>
  %1 = rock.transform %0 by #transform_map_over_vec_bottom2 : memref<64x1x64x10x6xf32> to memref<64x1x64x2x9x2x5xf32>
  %2 = rock.transform %1 by #transform_map_over_vec_bottom1 : memref<64x1x64x2x9x2x5xf32> to memref<1x256x2880xf32>
  %3 = rock.transform %2 by #transform_map_over_vec_top2 : memref<1x256x2880xf32> to memref<16x1x2x45x16x4x4x4xf32>
  %4 = rock.transform %3 by #transform_map_over_vec_top1 : memref<16x1x2x45x16x4x4x4xf32> to memref<16x1x2x45x64x16xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 1
  "get_length"(%4) {in_dim = 5 : index} : (memref<16x1x2x45x64x16xf32>) -> ()
  func.return
}


#transform_map_merge_gemm = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 16, d2 mod 16, 0)>
by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>,
    <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>,
    <Merge{48, 16, 1} ["gemmN"] at [2] -> ["n_block", "n_iter", "n_tid"] at [2, 3, 4]>]
bounds = [1, 16, 768] -> [1, 16, 48, 16, 1]>

#transform_map_unmerge_gemm =  #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, (d2 + d4) * 16 + d3)>
by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>,
     <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>,
     <Unmerge{48, 1, 16} ["n_block", "n_tid", "n_iter"] at [2, 4, 3] -> ["gemmN"] at [2]>]
bounds = [1, 16, 48, 16, 1] -> [1, 16, 768]>

// Unit dimension gets swapped during Merge/Unmerge
// TODO: we should make sure that this test passes for the right reason. Indeed, the reason why we can vectorize this
// transformation is because the singleton dimension, for vectorization purposes, can be ignored.
// However, we need to make sure that the engine does not signal that `{d0, d1, d2}` are contiguous
// because it would lead to a flattening of the Merge dimension space when `collapseContiguousMerges` is called.
// CHECK-LABEL: @test_merge_unmerge_twisting
func.func @test_merge_unmerge_twisting(%buf: memref<1x16x768xf32>) {
  %0 = rock.transform %buf by #transform_map_unmerge_gemm : memref<1x16x768xf32> to memref<1x16x48x16x1xf32>
  %1 = rock.transform %0 by #transform_map_merge_gemm : memref<1x16x48x16x1xf32> to memref<1x16x768xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%1) {in_dim = 2 : index, in_dim_len = 4 : index} : (memref<1x16x768xf32>) -> ()
  func.return
}

#id2 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @test_fusions_ignored_by_default
func.func @test_fusions_ignored_by_default(%buf: memref<4x8xi8>) {
  %alloc1 = memref.alloc() : memref<8x4xi8>
  // This is incorrect, since the eventual memory layout is transposed
  // CHECK: get_length
  // CHECK-SAME: result = 4
  "get_length"(%alloc1) {in_dim = 1 : index} : (memref<8x4xi8>) -> ()
  %alloc2 = memref.alloc() : memref<4x8xi8>
  %0 = rock.transform %alloc2 by #transform_map0 : memref<4x8xi8> to memref<8x4xi8>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
    ins(%alloc1 : memref<8x4xi8>)
    outs(%0 : memref<8x4xi8>) {
    ^bb0(%i : i8, %o : i8):
      linalg.yield %i : i8
  }
  memref.copy %alloc2, %buf : memref<4x8xi8> to memref<4x8xi8>
  func.return
}

// CHECK-LABEL: func @test_fusion_traversal
func.func @test_fusion_traversal(%buf: memref<4x8xi8>) {
  %alloc1 = memref.alloc() : memref<8x4xi8>
  // But here we see the transpose.
  // CHECK: get_length
  // CHECK-Same: fusionTraversalStatus = true
  // CHECK-SAME: result = 1
  "get_length"(%alloc1) {in_dim = 1 : index, traverseFusions} : (memref<8x4xi8>) -> ()
  %alloc2 = memref.alloc() : memref<4x8xi8>
  %0 = rock.transform %alloc2 by #transform_map0 : memref<4x8xi8> to memref<8x4xi8>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
    ins(%alloc1 : memref<8x4xi8>)
    outs(%0 : memref<8x4xi8>) {
    ^bb0(%i : i8, %o : i8):
      linalg.yield %i : i8
  }
  memref.copy %alloc2, %buf : memref<4x8xi8> to memref<4x8xi8>
  func.return
}

// CHECK-LABEL: func @test_fusion_traversal_fail
func.func @test_fusion_traversal_fail(%buf: memref<4x8xi8>, %buf2: memref<8x4xi8>) {
  %alloc1 = memref.alloc() : memref<8x4xi8>
  // But here we see the transpose.
  // CHECK: get_length
  // CHECK-Same: fusionTraversalStatus = false
  // CHECK-SAME: result = 4
  "get_length"(%alloc1) {in_dim = 1 : index, traverseFusions} : (memref<8x4xi8>) -> ()
  memref.copy %alloc1, %buf2 : memref<8x4xi8> to memref<8x4xi8>
  %alloc2 = memref.alloc() : memref<4x8xi8>
  %0 = rock.transform %alloc2 by #transform_map0 : memref<4x8xi8> to memref<8x4xi8>
  linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
    ins(%alloc1 : memref<8x4xi8>)
    outs(%0 : memref<8x4xi8>) {
    ^bb0(%i : i8, %o : i8):
      linalg.yield %i : i8
  }
  memref.copy %alloc2, %buf : memref<4x8xi8> to memref<4x8xi8>
  func.return
}

// CHECK-LABEL: func @test_vectorization_align_pad
func.func @test_vectorization_align_pad() {
  func.return
}

// CHECK-LABEL: func @test_input_fusion_traversal_good
func.func @test_input_fusion_traversal_good(%buf: memref<4x8xi8>) {
  %0 = rock.transform %buf by #transform_map0 : memref<4x8xi8> to memref<8x4xi8>
  %alloc1 = memref.alloc() : memref<8x4xi8>
  linalg.generic {rock.majorTensorNumber = 0 : index, indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
    ins(%0 : memref<8x4xi8>)
    outs(%alloc1 : memref<8x4xi8>) {
    ^bb0(%i : i8, %o : i8):
      linalg.yield %i : i8
  }
  // Here we see the transpose so we get to vectorize
  // CHECK: get_length
  // CHECK-Same: fusionTraversalStatus = true
  // CHECK-SAME: result = 8
  "get_length"(%alloc1) {in_dim = 0 : index, traverseFusions} : (memref<8x4xi8>) -> ()
  func.return
}

// CHECK-LABEL: func @test_input_fusion_traversal_bad
func.func @test_input_fusion_traversal_bad(%buf: memref<4x8xi8>) {
  %0 = rock.transform %buf by #transform_map0 : memref<4x8xi8> to memref<8x4xi8>
  %alloc1 = memref.alloc() : memref<8x4xi8>
  linalg.generic {rock.majorTensorNumber = 0 : index, indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
    ins(%0 : memref<8x4xi8>)
    outs(%alloc1 : memref<8x4xi8>) {
    ^bb0(%i : i8, %o : i8):
      linalg.yield %i : i8
  }
  // And here the transpose tells us we can't vectorize
  // CHECK: get_length
  // CHECK-Same: fusionTraversalStatus = true
  // CHECK-SAME: result = 1
  "get_length"(%alloc1) {in_dim = 1 : index, traverseFusions} : (memref<8x4xi8>) -> ()
  func.return
}

#id1 = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @input_fusion_true_max_length
func.func @input_fusion_true_max_length(%buf: memref<8xf16>) {
  %alloc1 = memref.alloc() : memref<8xf32>
  linalg.generic {rock.majorTensorNumber = 0 : index, indexing_maps = [#id1, #id1], iterator_types = ["parallel"]}
    ins(%buf : memref<8xf16>)
    outs(%alloc1 : memref<8xf32>) {
    ^bb0(%i : f16, %o : f32):
      %e = arith.extf %i : f16 to f32
      linalg.yield %e : f32
  }
  // Confirm that we vectorize by f16's, not f32s
  // CHECK: get_length
  // CHECK-Same: fusionTraversalStatus = true
  // CHECK-SAME: result = 8
  "get_length"(%alloc1) {in_dim = 0 : index, traverseFusions} : (memref<8xf32>) -> ()
  func.return
}

// CHECK-LABEL: @issue_1643_embed_alignment
#issue_1643_split_iter = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
by [<Merge{3, 8} ["iter"] at [0] -> ["k", "n"] at [0, 1]>]
bounds = [24] -> [3, 8]>
#issue_1643_embed = #rock.transform_map<affine_map<(d0, d1) -> (2 * d0 + d1)>
by [<Embed{2, 1} ["k", "n"] at [0, 1] -> ["ipad"] at [0]>]
bounds = [3, 8] -> [12]>
#issue_1643_pad = #rock.transform_map<affine_map<(d0) -> (d0 - 4)>
by [<Pad{4, 4} ["ipad"] at [0] -> ["i"] at [0]>]
bounds = [12] -> [4]>
func.func @issue_1643_embed_alignment(%buf: memref<4xf32>) {
  %0 = rock.transform %buf by #issue_1643_pad : memref<4xf32> to memref<12xf32>
  %1 = rock.transform %0 by #issue_1643_embed : memref<12xf32> to memref<3x8xf32>
  %2 = rock.transform %1 by #issue_1643_split_iter : memref<3x8xf32> to memref<24xf32>
  // CHECK: get_length
  // CHECK-SAME: result = 2
  "get_length"(%2) {in_dim = 0 : index} : (memref<24xf32>) -> ()
  func.return
}
