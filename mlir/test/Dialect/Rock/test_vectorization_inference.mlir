// RUN: rocmlir-opt -rock-vectorization-inference-test \
// RUN: -allow-unregistered-dialect %s \
// RUN: | FileCheck %s

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [1, 0]>]
  bounds = [4, 8] -> [8, 4]>

#transform_map1 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 8, d0 mod 8)>
  by [<Merge{4, 8} ["x"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [32] -> [4, 8]>

#transform_map2 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + d0 * 8)>
  by [<Embed{8,1} ["a", "b"] at [0, 1] -> ["x"] at [0]>]
  bounds = [4,8] -> [32]>

#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <Broadcast{1} ["a"] at [1] -> ["a"] at [1]>]
  bounds = [8,4] -> [8,1]>

#transform_map4 = #rock.transform_map<affine_map<(d0, d1) -> (d0)>
  by [<PassThrough ["b"] at [0] -> ["b"] at [0]>,
    <AddDim{1} ["a"] at [1] -> [] at []>]
  bounds = [8,1] -> [8]>

#transform_map5 = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 4} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

#transform_map6 = #rock.transform_map<affine_map<(d0) -> (d0 - 2)>
  by [<Pad{2, 2} ["b"] at [0] -> ["b"] at [0]>]
  bounds = [8] -> [4]>

#transform_map7 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 8, d1, d0 mod 8)>
  by [<Merge{4, 8} ["x"] at [0] -> ["a", "b"] at [0, 2]>,
  <PassThrough ["y"] at [1] -> ["y"] at [1]>]
  bounds = [32, 5] -> [4, 5, 8]>

#transform_map8 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{5} ["y"] at [1] -> [] at []>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 1]>]
  bounds = [4, 5, 8] -> [4, 8]>

#transform_map9 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["y"] at [1] -> ["y"] at [1]>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 2]>]
  bounds = [4, 5, 8] -> [4, 1, 8]>

// Test for the alignment bug
#transform_map_align_e11 = #rock.transform_map<affine_map<(d0, d1) -> (d0 + d1)>
  by [<Embed{1, 1} ["a", "b"] at [0, 1] -> ["xpad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_e41 = #rock.transform_map<affine_map<(d0, d1) -> (d0 * 2 + d1)>
  by [<Embed{4, 1} ["a", "b"] at [0, 1] -> ["xpad"] at [0]>]
  bounds = [4, 4] -> [16]>

#transform_map_align_pad = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 2} ["xpad"] at [0] -> ["x"] at [0]>]
  bounds = [16] -> [14]>

#transform_map_align_nopad = #rock.transform_map<affine_map<(d0) -> (d0)>
  by [<Pad{0, 0} ["xpad"] at [0] -> ["x"] at [0]>]
  bounds = [16] -> [16]>

// Unfold detection transforms
#transform_merge = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, d0 mod 3)>
  by [<Merge{8, 3} ["x"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [24] -> [8, 3]>

#transform_merge_2 = #rock.transform_map<affine_map<(d0, d1) -> (d1 floordiv 4, d0, (d1 mod 4) floordiv 2, d1 mod 2)>
  by [<Merge{128, 2, 2} ["gemmN"] at [1] -> ["N", "Ho", "Wo"] at [0, 2, 3]>,
      <PassThrough ["gemmM"] at [0] -> ["k"] at [1]>]
  bounds = [32, 512] -> [128, 32, 2, 2]>

#transform_merge_3 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 3, d1, d0 mod 3)>
  by [<Merge{8, 3} ["x"] at [0] -> ["a", "b"] at [0, 2]>,
  <PassThrough ["y"] at [1] -> ["y"] at [1]>]
  bounds = [24, 5] -> [8, 5, 3]>

#transform_merge_4 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 9, (d0 mod 3) floordiv 3, d0 mod 3)>
  by [<Merge{8, 3, 3} ["x"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [72] -> [8, 3, 3]>

#transform_transpose = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)>
   by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "b"] at [1, 0]>]
   bounds = [8, 3] -> [3, 8]>

#transform_shuffle = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d3, d2, d1, d0)>
  by [<PassThrough ["N", "K", "Ho", "Wo"] at [0, 1, 2, 3] -> ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0]>]
  bounds = [128, 32, 2, 2] -> [2, 2, 32, 128]>

#transform_shuffle_2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{5} ["y"] at [1] -> [] at []>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 1]>]
  bounds = [8, 5, 3] -> [8, 3]>

#transform_shuffle_3 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["y"] at [1] -> ["y"] at [1]>,
  <PassThrough ["a", "b"] at [0, 2] -> ["a", "b"] at [0, 2]>]
  bounds = [8, 5, 3] -> [8, 1, 3]>

#transform_shuffle_4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)>
  by [<PassThrough ["a", "b", "c"] at [0, 1, 2] -> ["a", "c", "b"] at [0, 2, 1]>]
  bounds = [8, 3, 3] -> [8, 3, 3]>

#transform_unmerge = #rock.transform_map<affine_map<(d0, d1) -> (3*d1 + d0)>
  by [<Unmerge{8, 3} ["a", "b"] at [1, 0] -> ["x"] at [0]>]
  bounds = [8, 3] -> [24]>

#transform_unmerge_2 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 + 2*d1 + 4*d2 + 128*d3)>
  by [<Unmerge{128, 32, 2, 2} ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0] -> ["x"] at [0]>]
  bounds = [128, 32, 2, 2] -> [16384]>

#transform_unmerge_3 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (128*d3 + 4*d2+ 2*d1+ d0)>
  by [<Embed{128, 4, 2, 1} ["Wo", "Ho", "K", "N"] at [3, 2, 1, 0] -> ["x"] at [0]>]
  bounds = [128, 32, 2, 2] -> [16384]>

#transform_merge_5 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, d0 mod 3)>
  by [<Merge{4, 3} ["x"] at [0] -> ["a", "b"] at [0, 1]>]
  bounds = [12] -> [4, 3]>

#transform_unmerge_5 = #rock.transform_map<affine_map<(d0,d1) -> (d0*3+d1)>
  by [<Embed{3, 1} ["a", "b"] at [0,1] -> ["x"] at [0]>]
  bounds = [4,3] -> [12]>

#transform_merge_6 = #rock.transform_map<affine_map<(d0, d1) -> (d0 floordiv 3, d0 mod 3, d1 floordiv 7, d1 mod 7)>
  by [<Merge{4, 3} ["x"] at [0] -> ["a", "b"] at [0, 1]>, <Merge{8,7} ["y"] at [1] -> ["c", "d"] at [2, 3]>]
  bounds = [12, 56] -> [4, 3, 8, 7]>

#transform_unmerge_6 = #rock.transform_map<affine_map<(d0,d1,d2,d3) -> (d0*3+d1,d2*7+d3)>
  by [<Embed{3, 1} ["a", "b"] at [0,1] -> ["x"] at [0]>, <Embed{7,1} ["b","c"] at [2,3] -> ["y"] at [1]>]
  bounds = [4,3,8,7] -> [12,56]>

#transform_merge_7 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, 0, d0 mod 3)>
  by [<Merge{8, 1, 3} ["x"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [24] -> [8, 1, 3]>

#transform_shuffle_5 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2)>
  by [<AddDim{1} ["b"] at [1] -> [] at []>, <PassThrough ["a", "c"] at [0, 2] -> ["a", "b"] at [0,1]>]
  bounds = [8, 1, 3] -> [8, 3]>

#transform_unmerge_7 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0*3 + d1 + d0)>
  by [<Unmerge{8, 1, 3} ["a", "b", "c"] at [0, 1, 2] -> ["x"] at [0]>]
  bounds = [8,1,3] -> [24]>

#transform_unmerge_8 = #rock.transform_map<affine_map<(d0, d1) -> (d0*3 + d1)>
  by [<Unmerge{8, 3} ["a", "c"] at [0, 1] -> ["x"] at [0]>]
  bounds = [8,3] -> [24]>

#transform_merge_9 = #rock.transform_map<affine_map<(d0) -> (d0 floordiv 3, 0, d0 mod 3)>
  by [<Merge{8, 2, 3} ["x"] at [0] -> ["a", "b", "c"] at [0, 1, 2]>]
  bounds = [48] -> [8, 2, 3]>

#transform_shuffle_6 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, 0, d2)>
  by [<Broadcast{1} ["b"] at [1] -> ["b"] at [1]>, <PassThrough ["a", "c"] at [0, 2] -> ["a", "c"] at [0,2]>]
  bounds = [8, 2, 3] -> [8, 1, 3]>

#transform_unmerge_9 = #rock.transform_map<affine_map<(d0, d1) -> (d0 + d1*3)>
  by [<Embed{1, 3} ["a", "c"] at [0, 1] -> ["x"] at [0]>]
  bounds = [3, 8] -> [24]>

#transform_inject_unit_const = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0, d1)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "c"] at [0, 2]>,
    <ConstDim{0, 1} [] at [] -> ["b"] at [1]>]
  bounds = [8, 3] -> [8, 1, 3]>

#transform_inject_non_unit_const = #rock.transform_map<affine_map<(d0, d1) -> (d0, 0, d1)>
  by [<PassThrough ["a", "b"] at [0, 1] -> ["a", "c"] at [0, 2]>,
    <ConstDim{0, 2} [] at [] -> ["b"] at [1]>]
  bounds = [8, 3] -> [8, 2, 3]>

#transform_unmerge_injected_non_unit = #rock.transform_map<affine_map<(d0, d1, d2) -> (d2 + 3 * (d1 + d0 * 2))>
  by [<Unmerge{8, 2, 3} ["a", "b", "c"] at [0, 1, 2] -> ["x"] at [0]>]
  bounds = [8, 2, 3] -> [48]>

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

  // Tests for unfold detection
  // CHECK-NEXT: result = 4
  %25 = "get_length"() {transforms = [#transform_merge, #transform_transpose, #transform_unmerge], in_dim = 0 : index, max_len = 4: index} : () -> (memref<24xf32>)

  // CHECK-NEXT: result = 4
  %26 = "get_length"() {transforms = [#transform_merge], in_dim = 0 : index, max_len = 4: index} : () -> (memref<8x3xf32>)
  // CHECK-NEXT: result = 4
  %27 = "get_length"() {transforms = [#transform_merge_2, #transform_shuffle, #transform_unmerge_2], in_dim = 1 : index, max_len = 4: index} : () -> (memref<16384xf32>)
  // CHECK-NEXT: result = 1
  %28 = "get_length"() {transforms = [#transform_merge_4, #transform_shuffle_4], in_dim = 0 : index, max_len = 4: index} : () -> (memref<8x3x3xf32>)

  // Intervening dimensions still prevent merge
  // CHECK-NEXT: result = 1
  %29 = "get_length"() {transforms = [#transform_merge_3], in_dim = 0 : index, max_len =  4: index} : () -> (memref<8x5x3xf32>)
  // Unless it was an illusion
  // CHECK-NEXT: result = 4
  %30 = "get_length"() {transforms = [#transform_merge_3, #transform_shuffle_2], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<8x3xf32>)
  // or a broadcast
  // CHECK-NEXT: result = 4
  %31 = "get_length"() {transforms = [#transform_merge_3, #transform_shuffle_3], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<8x1x3xf32>)

  // Unfold detection with embed
  // CHECK-NEXT: result = 4
  %32 = "get_length"() {transforms = [#transform_merge_2, #transform_shuffle, #transform_unmerge_3], in_dim = 1 : index, max_len = 4: index} : () -> (memref<16384xf32>)

  // CHECK-NEXT: result = 4
  %33 = "get_length"() {transforms = [#transform_merge_5, #transform_unmerge_5], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<12xf32>)

  // CHECK-NEXT: result = 4
  %34 = "get_length"() {transforms = [#transform_merge, #transform_transpose, #transform_unmerge_9], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<12xf32>)

  // Partial unfold detection with embed
  // CHECK-NEXT: result = 4
  %35 = "get_length"() {transforms = [#transform_merge_6, #transform_unmerge_6], in_dim = 1 : index, max_len = 4 : index} : () -> (memref<12x56xf32>)

  // More broadcast examples
  // CHECK-NEXT: result = 4
  %36 = "get_length"() {transforms = [#transform_merge_7, #transform_unmerge_7], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<24xf32>)
  // CHECK-NEXT: result = 4
  %37 = "get_length"() {transforms = [#transform_merge_7, #transform_shuffle_5, #transform_unmerge_8], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<24xf32>)
  // CHECK-NEXT: result = 4
  %38 = "get_length"() {transforms = [#transform_merge_9, #transform_shuffle_6, #transform_unmerge_7], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<24xf32>)
  // CHECK-NEXT: result = 4
  %39 = "get_length"() {transforms = [#transform_merge, #transform_inject_unit_const, #transform_unmerge_7], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<24xf32>)
  // Even though we injected a 0, it _could_ have been a 1.
  // CHECK-NEXT: result = 1
  %40 = "get_length"() {transforms = [#transform_merge, #transform_inject_non_unit_const, #transform_unmerge_injected_non_unit], in_dim = 0 : index, max_len = 4 : index} : () -> (memref<48xf32>)
 func.return
}


// CHECK-LABEL: func @test_vectorization_align_pad
func.func @test_vectorization_align_pad() {
  func.return
}
