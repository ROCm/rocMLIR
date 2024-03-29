// RUN: rocmlir-opt -rock-collapse-contiguous-merges-test \
// RUN: -allow-unregistered-dialect -split-input-file %s \
// RUN: | FileCheck --enable-var-scope %s

#unmerge_trmap = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>

// CHECK: [[MAP:#.+]] = affine_map<(d0, d1) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 2, d1)>
// CHECK: [[TRMAP:#.+]] = #rock.transform_map<[[MAP]] by [<PassThrough ["a"] at [1] -> ["a"] at [3]>, <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>] bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: func @test_wrong_shape([[ARG0:%.+]]: memref<5x4x3x5xf32>)
// CHECK-NEXT: [[MERGED:%.+]] = rock.transform [[ARG0]] by [[TRMAP]]
// CHECK-NEXT: "collapse_merges"([[MERGED]])
func.func @test_wrong_shape(%arg0: memref<5x4x3x5xf32>) {
  %merged = rock.transform %arg0 by #unmerge_trmap : memref<5x4x3x5xf32> to memref<24x5xf32>
  "collapse_merges"(%merged) : (memref<24x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_basic_unmerge"
"pre-split-mark-test_basic_unmerge"() : () -> ()
// -----

// CHECK: [[MAP:#.+]] = affine_map<(d0, d1) -> (0, 0, d0, d1)>
// CHECK: [[TRMAP:#.+]] = #rock.transform_map<[[MAP]] by [<PassThrough ["a"] at [1] -> ["a"] at [3]>, <Merge{1, 1, 24} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>] bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: func @test_basic_unmerge([[ARG0:%.+]]: memref<4x3x2x5xf32>)
// CHECK-NEXT: [[MERGED:%.+]] = rock.transform [[ARG0]] by [[TRMAP]]
// CHECK-NEXT: "collapse_merges"([[MERGED]])

#unmerge_trmap = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>

func.func @test_basic_unmerge(%arg0: memref<4x3x2x5xf32>) {
  %0 = rock.transform %arg0 by #unmerge_trmap : memref<4x3x2x5xf32> to memref<24x5xf32>
  "collapse_merges"(%0) : (memref<24x5xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_partial_merge_conv2gemm"
"pre-split-mark-test_partial_merge_conv2gemm"() : () -> ()
// -----

// CHECK: [[MAP:#.+]] = affine_map<(d0, d1) -> (d1 floordiv 6, d0, 0, d1 mod 6)>
// CHECK: [[TRMAP:#.+]] = #rock.transform_map<[[MAP]] by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>, <Merge{4, 1, 6} ["gemmN"] at [1] -> ["n", "0", "1"] at [0, 2, 3]>] bounds = [5, 24] -> [4, 5, 3, 2]>
// CHECK: func @test_partial_merge_conv2gemm([[ARG0:%.+]]: memref<4x5x3x2xf32>)
// CHECK-NEXT: [[MERGED:%.+]] = rock.transform [[ARG0]] by [[TRMAP]]
// CHECK-NEXT: "collapse_merges"([[MERGED]])
#conv2gemm_trmap = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, d1, (d0 mod 6) floordiv 3, d0 mod 2)>
  by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>,
    <Merge{4, 3, 2} ["gemmN"] at [1] -> ["n", "0", "1"] at [0, 2, 3]>]
  bounds = [5, 24] -> [4, 5, 3, 2]>

func.func @test_partial_merge_conv2gemm(%arg0: memref<4x5x3x2xf32>) {
  %0 = rock.transform %arg0 by #conv2gemm_trmap : memref<4x5x3x2xf32> to memref<5x24xf32>
  "collapse_merges"(%0) : (memref<5x24xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_batch_transpose_bug_1407"
"pre-split-mark-test_batch_transpose_bug_1407"() : () -> ()
// -----

// CHECK: [[PERM_MAP:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
// CHECK: [[CONV2GEMM_MAP:#.+]] = affine_map<(d0, d1) -> (0, d0, d1, 0)>
// CHECK: [[PERM_TRMAP:#.+]] = #rock.transform_map<[[PERM_MAP]]
// CHECK: [[CONV2GEMM_TRMAP:#.+]] = #rock.transform_map<[[CONV2GEMM_MAP]] by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>, <Merge{1, 65536, 1} ["gemmN"] at [1] -> ["0", "1", "n"] at [0, 2, 3]>] bounds = [512, 65536] -> [256, 512, 256, 1]>
// CHECK: func @test_batch_transpose_bug_1407([[ARG0:%.+]]: memref<1x512x256x256xf32>)
// CHECK-NEXT: [[PERMED:%.+]] = rock.transform [[ARG0]] by [[PERM_TRMAP]]
// CHECK-NEXT: [[MERGED:%.+]] = rock.transform [[PERMED]] by [[CONV2GEMM_TRMAP]]
// CHECK-NEXT: "collapse_merges"([[MERGED]])
#perm_trmap = #rock.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d3, d1, d0, d2)>
  by [<PassThrough ["n", "k", "0", "1"] at [3, 1, 0, 2]-> ["n", "k", "0", "1"] at [0, 1, 2, 3]>]
  bounds = [256, 512, 256, 1] -> [1, 512, 256, 256]>
#conv2gemm_trmap = #rock.transform_map<
  affine_map<(d0, d1) ->(d0 floordiv 6, d1, (d0 mod 6) floordiv 3, d0 mod 2)>
  by [<PassThrough ["gemmM"] at [0] -> ["k"] at [1]>,
    <Merge{256, 256, 1} ["gemmN"] at [1] -> ["0", "1", "n"] at [0, 2, 3]>]
  bounds = [512, 65536] -> [256, 512, 256, 1]>

func.func @test_batch_transpose_bug_1407(%arg0: memref<1x512x256x256xf32>) {
  %0 = rock.transform %arg0 by #perm_trmap : memref<1x512x256x256xf32> to memref<256x512x256x1xf32>
  %1 = rock.transform %0 by #conv2gemm_trmap : memref<256x512x256x1xf32> to memref<512x65536xf32>
  "collapse_merges"(%1) : (memref<512x65536xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-test_need_for_clone"
"pre-split-mark-test_need_for_clone"() : () -> ()
// -----
// CHECK: [[OLD_MAP:#.+]] = affine_map<(d0, d1) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 2, d1)>
// CHECK: [[NEW_MAP:#.+]] = affine_map<(d0, d1) -> (0, 0, d0, d1)>
// CHECK: [[PERM_MAP:#.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: [[OLD_TRMAP:#.+]] = #rock.transform_map<[[OLD_MAP]] by [<PassThrough ["a"] at [1] -> ["a"] at [3]>, <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>] bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: [[NEW_TRMAP:#.+]] = #rock.transform_map<[[NEW_MAP]] by [<PassThrough ["a"] at [1] -> ["a"] at [3]>, <Merge{1, 1, 24} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>] bounds = [24, 5] -> [4, 3, 2, 5]>
// CHECK: [[PERM_TRMAP:#.+]] = #rock.transform_map<[[PERM_MAP]] by [<PassThrough ["1", "a"] at [1, 0] -> ["1", "a"] at [0, 1]>] bounds = [5, 24] -> [24, 5]>
// CHECK: func @test_need_for_clone([[ARG0:%.+]]: memref<4x3x2x5xf32>)
// CHECK-NEXT: [[OLD_MERGED:%.+]] = rock.transform [[ARG0]] by [[OLD_TRMAP]]
// CHECK-NEXT: [[NEW_MERGED:%.+]] = rock.transform [[ARG0]] by [[NEW_TRMAP]]
// CHECK-NEXT: [[OLD_PERMED:%.+]] = rock.transform [[OLD_MERGED]] by [[PERM_TRMAP]]
// CHECK-NEXT: [[NEW_PERMED:%.+]] = rock.transform [[NEW_MERGED]] by [[PERM_TRMAP]]
// CHECK-NEXT: "collapse_merges"([[NEW_PERMED]])
// CHECK-NEXT: "use"([[OLD_PERMED]])

#unmerge_trmap = #rock.transform_map<
  affine_map<(d0, d1) -> (d0 floordiv 6, (d0 mod 6) floordiv 3, d0 mod 2, d1)>
  by [<PassThrough ["a"] at [1] -> ["a"] at [3]>,
    <Merge{4, 3, 2} ["1"] at [0] -> ["b", "c", "d"] at [0, 1, 2]>]
  bounds = [24, 5] -> [4, 3, 2, 5]>
#perm_trmap = #rock.transform_map<
  affine_map<(d0, d1) -> (d1, d0)>
  by [<PassThrough ["1", "a"] at [1, 0] -> ["1", "a"] at [0, 1]>]
  bounds = [5, 24] -> [24, 5]>

func.func @test_need_for_clone(%arg0: memref<4x3x2x5xf32>) {
  %0 = rock.transform %arg0 by #unmerge_trmap : memref<4x3x2x5xf32> to memref<24x5xf32>
  %1 = rock.transform %0 by #perm_trmap : memref<24x5xf32> to memref<5x24xf32>
  "collapse_merges"(%1) : (memref<5x24xf32>) -> ()
  "use"(%1) : (memref<5x24xf32>) -> ()
  return
}
// CHECK-LABEL: "pre-split-mark-negative-test_reuse_dim"
"pre-split-mark-negative-test_reuse_dim"() : () -> ()
// Please note that in the following test we should find two groups in the
// unmerge {3,5} and {4,6}.  The group {3,5} has a singleton dimension in the
// middle. If we add that dimension we are creating a single {3,4,5,6} group,
// which is wrong. So once the dimension 4 is used (either as singleton in
// the {3,4,5} group or in the {4,6} group) it must not be reused
// -----
// CHECK: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, ((d1 * 4 + d7 + d3) * 2 + d5) * 8 + d9, (d2 * 4 + d8 + d4) * 16 + d6)>
// CHECK: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, 0, d3 floordiv 16, d3 mod 16, d4 floordiv 32, (d4 mod 32) floordiv 8, d4 mod 8)>
#merge_map = #rock.transform_map<
      affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, 0, d3 floordiv 16, d3 mod 16, d4 floordiv 32, (d4 mod 32) floordiv 8, d4 mod 8)>
         by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>,
             <Merge{1, 1, 2, 16} ["tid"] at [3] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [3, 4, 5, 6]>,
             <Merge{4, 4, 8} ["item"] at [4] -> ["rep_i", "rep_j", "item_i"] at [7, 8, 9]>] bounds = [1, 2, 2, 32, 128] -> [1, 2, 2, 1, 1, 2, 16, 4, 4, 8]>

#unmerge_map = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, ((d1 * 4 + d7 + d3) * 2 + d5) * 8 + d9, (d2 * 4 + d8 + d4) * 16 + d6)>
               by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>,
                   <Unmerge{2, 4, 1, 2, 8} ["m_block", "rep_i", "wave_m", "m_tid", "item_i"] at [1, 7, 3, 5, 9] -> ["gemmM"] at [1]>,
                   <Unmerge{2, 4, 1, 16} ["n_block", "rep_j", "wave_n", "n_tid"] at [2, 8, 4, 6] -> ["gemmN"] at [2]>] bounds = [1, 2, 2, 1, 1, 2, 16, 4, 4, 8] -> [1, 128, 128]>

func.func @negative_test_reuse_dim(%arg0: memref<1x128x128xf32>) {
  %0 = rock.transform %arg0 by #unmerge_map: memref<1x128x128xf32> to  memref<1x2x2x1x1x2x16x4x4x8xf32>
  %1 = rock.transform %0 by #merge_map: memref<1x2x2x1x1x2x16x4x4x8xf32> to memref<1x2x2x32x128xf32>
  "collapse_merges"(%1) : (memref<1x2x2x32x128xf32>) -> ()
  "use"(%1) : (memref<1x2x2x32x128xf32>) -> ()
  return
}

// CHECK-LABEL: "pre-split-mark-no_test_yet"
"pre-split-mark-no_test_yet"() : () -> ()
// -----
