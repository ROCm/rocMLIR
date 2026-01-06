// RUN: rocmlir-opt -rock-linalg-align %s | FileCheck %s

// A Pad transform should be created to handle the subview (padding 24 on dim 1)
// CHECK-DAG: #{{.*}} = #rock.transform_map<{{.*}}<Pad{0, 0, 0, 24, 0, 0}

#map = affine_map<(d0, d1, d2) -> ((d0 * 16 + d1) * 24 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 24 + d1) * 16 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 8 + d5) * 8 + d7, d2 * 16 + d4 + d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, 0, d5)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 8 + d4) * 8 + d6, (d3 * 16 + d5) * 2 + d7)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 16, d4 mod 16, d5 floordiv 2, d5 mod 2)>
#map8 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map9 = affine_map<(d0, d1) -> (0, d0)>
#map10 = affine_map<(d0, d1, d2) -> ((d0 + d1) * 8 + d2)>
#map11 = affine_map<(d0, d1) -> (0, 0, d0)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0 * 16 + d1 + d2)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1 mod 16, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d0 + d1, d2, d3)>
#map15 = affine_map<(d0, d1) -> (d0 floordiv 8, d1, 0, d0 mod 8)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> ((d1 + d2) * 8 + d4, d0 + d3)>
#map17 = affine_map<(d0, d1) -> (d0 floordiv 8, d0 mod 8, 0, 0, d1)>
#map18 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map19 = affine_map<(d0, d1) -> (d0, d1)>
#map20 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 8 + d2)>
#map21 = affine_map<(d0, d1) -> (0, d1, d0)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2)>
#map23 = affine_map<(d0, d1, d2, d3, d4) -> ((d0 + d2) * 8 + d4, d3 * 16 + d1)>
#map24 = affine_map<(d0, d1) -> (d0 floordiv 16, d0 mod 16, 0, d1 floordiv 8, d1 mod 8)>
#map25 = affine_map<(d0) -> (d0 * 4)>
#map26 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 64, (d3 mod 64) floordiv 16, d3 mod 16, 0, 0, 0, d4)>
#map27 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, 0, d3, d4, d5, 0, 0, 0, 0, d8, d9)>
#map28 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (d0, d1, d2, ((d3 + d7 + d9 + d11) * 4 + d5) * 4 + d12, (d8 * 2 + d4 + d10) * 16 + d6)>
#map29 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4 floordiv 16, d4 mod 16)>
#map30 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5 * 2 + d4)>
#map31 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d3, d2 * 32 + d4)>
#map32 = affine_map<(d0, d1, d2) -> ((d0 * 48 + d1) * 24 + d2)>
#mfma_gemm_params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 16, nPerBlock = 32, kpack = 8, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 16, 24} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 16, 24] -> [1536]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{4, 24, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 24, 16] -> [1536]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [4, 16, 24] -> [4, 24, 16]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 48} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 8} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>] bounds = [4, 64, 32] -> [4, 16, 24]>
#transform_map4 = #rock.transform_map<#map3 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 48} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 8} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [4, 64, 32] -> [4, 16, 24]>
#transform_map5 = #rock.transform_map<#map3 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 8} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 8} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [4, 32, 32] -> [4, 24, 24]>
#transform_map6 = #rock.transform_map<#map4 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{1, 8, 8} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{2, 16, 1} ["m_block", "m_thread", "m_iter"] at [2, 4, 6] -> ["m"] at [2]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [1, 4, 2, 1, 16, 8, 1, 8] -> [4, 64, 32]>
#transform_map7 = #rock.transform_map<#map5 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{16, 8} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{1, 8} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [1, 4, 2, 1, 128, 8] -> [1, 4, 2, 1, 16, 8, 1, 8]>
#transform_map8 = #rock.transform_map<#map6 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{1, 8, 8} ["k_loop", "k_thread", "k_iter"] at [0, 4, 6] -> ["k"] at [1]>, <Unmerge{1, 16, 2} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{2} ["m_block"] at [2] -> [] at []>] bounds = [1, 4, 2, 1, 8, 16, 8, 2] -> [4, 64, 32]>
#transform_map9 = #rock.transform_map<#map7 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{8, 16} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{8, 2} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [1, 4, 2, 1, 128, 16] -> [1, 4, 2, 1, 8, 16, 8, 2]>
#transform_map10 = #rock.transform_map<#map8 by [<Unmerge{1, 8} ["m_iter", "k_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [1, 8] -> [8]>
#transform_map11 = #rock.transform_map<#map9 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [1]>, <Merge{1} ["m"] at [1] -> ["m_iter"] at [0]>] bounds = [8, 1] -> [1, 8]>
#transform_map12 = #rock.transform_map<#map10 by [<Unmerge{1, 1, 8} ["kouterPerThread", "m_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 1, 8] -> [8]>
#transform_map13 = #rock.transform_map<#map11 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{1} ["m"] at [1] -> ["m_iter"] at [1]>] bounds = [8, 1] -> [1, 1, 8]>
#transform_map14 = #rock.transform_map<#map12 by [<Unmerge{8, 16, 1} ["k_outer", "m", "kpack_idx"] at [0, 1, 2] -> ["raw"] at [0]>, <AddDim{8} ["kpack_vec"] at [3] -> [] at []>] bounds = [8, 16, 1, 8] -> [128]>
#transform_map15 = #rock.transform_map<#map13 by [<PassThrough ["k_outer"] at [0] -> ["k_outer"] at [0]>, <Broadcast{16} ["m"] at [1] -> ["m"] at [1]>, <PassThrough ["kpack_idx", "kpack_vec"] at [2, 3] -> ["kpack_idx", "kpack_vec"] at [2, 3]>] bounds = [8, 128, 1, 8] -> [8, 16, 1, 8]>
#transform_map16 = #rock.transform_map<#map14 by [<PassThrough ["k_outer"] at [0] -> ["k_outer"] at [0]>, <Embed{1, 1} ["k_outer", "m"] at [0, 1] -> ["m"] at [1]>, <PassThrough ["kpack_idx", "kpack_vec"] at [2, 3] -> ["kpack_idx", "kpack_vec"] at [2, 3]>] bounds = [8, 16, 1, 8] -> [8, 128, 1, 8]>
#transform_map17 = #rock.transform_map<#map15 by [<Merge{8, 1, 8} ["k"] at [0] -> ["k_outer", "kpack_idx", "kpack_vec"] at [0, 2, 3]>, <Merge{16} ["d"] at [1] -> ["m"] at [1]>] bounds = [64, 16] -> [8, 16, 1, 8]>
#transform_map18 = #rock.transform_map<#map16 by [<Unmerge{8, 1, 8} ["k_thread", "kouterPerThread", "kpackPerThread"] at [1, 2, 4] -> ["k"] at [0]>, <Unmerge{16, 1} ["m_thread", "m_iter"] at [0, 3] -> ["m"] at [1]>] bounds = [16, 8, 1, 1, 8] -> [64, 16]>
#transform_map19 = #rock.transform_map<#map17 by [<Merge{16, 8} ["tid"] at [0] -> ["m_thread", "k_thread"] at [0, 1]>, <Merge{1, 1, 8} ["iter"] at [1] -> ["kouterPerThread", "m_iter", "kpackPerThread"] at [2, 3, 4]>] bounds = [128, 8] -> [16, 8, 1, 1, 8]>
#transform_map20 = #rock.transform_map<#map18 by [<Unmerge{8, 2} ["k_iter", "n_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [8, 2] -> [16]>
#transform_map21 = #rock.transform_map<#map19 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [0]>, <Merge{2} ["n"] at [1] -> ["n_iter"] at [1]>] bounds = [8, 2] -> [8, 2]>
#transform_map22 = #rock.transform_map<#map20 by [<Unmerge{1, 2, 8} ["kouterPerThread", "n_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
#transform_map23 = #rock.transform_map<#map21 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{2} ["n"] at [1] -> ["n_iter"] at [1]>] bounds = [8, 2] -> [1, 2, 8]>
#transform_map24 = #rock.transform_map<#map22 by [<Unmerge{8, 32, 1} ["k_outer", "n", "kpack_idx"] at [0, 1, 2] -> ["raw"] at [0]>, <AddDim{8} ["kpack_vec"] at [3] -> [] at []>] bounds = [8, 32, 1, 8] -> [256]>
#transform_map25 = #rock.transform_map<#map15 by [<Merge{8, 1, 8} ["k"] at [0] -> ["k_outer", "kpack_idx", "kpack_vec"] at [0, 2, 3]>, <Merge{32} ["d"] at [1] -> ["n"] at [1]>] bounds = [64, 32] -> [8, 32, 1, 8]>
#transform_map26 = #rock.transform_map<#map23 by [<Unmerge{8, 1, 8} ["k_thread", "kouterPerThread", "kpackPerThread"] at [0, 2, 4] -> ["k"] at [0]>, <Unmerge{2, 16} ["n_iter", "n_thread"] at [3, 1] -> ["n"] at [1]>] bounds = [8, 16, 1, 2, 8] -> [64, 32]>
#transform_map27 = #rock.transform_map<#map24 by [<Merge{8, 16} ["tid"] at [0] -> ["k_thread", "n_thread"] at [0, 1]>, <Merge{1, 2, 8} ["iter"] at [1] -> ["kouterPerThread", "n_iter", "kpackPerThread"] at [2, 3, 4]>] bounds = [128, 16] -> [8, 16, 1, 2, 8]>
#transform_map28 = #rock.transform_map<#map25 by [<Embed{4} ["vector"] at [0] -> ["scalar"] at [0]>] bounds = [1] -> [4]>
#transform_map29 = #rock.transform_map<#map26 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 4, 16} ["tid"] at [3] -> ["wave", "m_tid", "n_tid"] at [3, 4, 5]>, <Merge{1, 1, 1, 4} ["item"] at [4] -> ["i", "j", "vec_group", "vec_item"] at [6, 7, 8, 9]>] bounds = [4, 2, 1, 128, 4] -> [4, 2, 1, 2, 4, 16, 1, 1, 1, 4]>
#transform_map30 = #rock.transform_map<#map27 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{1, 2} ["wave"] at [3] -> ["wave_m", "wave_n"] at [3, 4]>, <PassThrough ["m_tid", "n_tid"] at [4, 5] -> ["m_tid", "n_tid"] at [5, 6]>, <Merge{1, 1} ["i"] at [6] -> ["m_i", "n_i"] at [7, 8]>, <Merge{1, 1} ["j"] at [7] -> ["blk_row", "blk_col"] at [9, 10]>, <PassThrough ["vec_group", "vec_item"] at [8, 9] -> ["vec_group", "vec_item"] at [11, 12]>] bounds = [4, 2, 1, 2, 4, 16, 1, 1, 1, 4] -> [4, 2, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 4]>
#transform_map31 = #rock.transform_map<#map28 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Unmerge{1, 1, 1, 1, 4, 4} ["m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [7, 3, 9, 11, 5, 12] -> ["gemmBlockM"] at [3]>, <Unmerge{1, 2, 1, 16} ["n_i", "wave_n", "blk_col", "n_tid"] at [8, 4, 10, 6] -> ["gemmBlockN"] at [4]>] bounds = [4, 2, 1, 1, 2, 4, 16, 1, 1, 1, 1, 1, 4] -> [4, 2, 1, 16, 32]>
#transform_map32 = #rock.transform_map<#map29 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <PassThrough ["gemmBlockM"] at [3] -> ["gemmBlockM"] at [3]>, <Merge{2, 16} ["gemmBlockN"] at [4] -> ["n_iter", "n_tid"] at [4, 5]>] bounds = [4, 2, 1, 16, 32] -> [4, 2, 1, 16, 2, 16]>
#transform_map33 = #rock.transform_map<#map30 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <PassThrough ["gemmBlockM"] at [3] -> ["gemmBlockM"] at [3]>, <Unmerge{16, 2} ["n_tid", "n_iter"] at [5, 4] -> ["gemmBlockN"] at [4]>] bounds = [4, 2, 1, 16, 2, 16] -> [4, 2, 1, 16, 32]>
#transform_map34 = #rock.transform_map<#map31 by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{2, 16} ["m_block", "gemmBlockM"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{1, 32} ["n_block", "gemmBlockN"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [4, 2, 1, 16, 32] -> [4, 32, 32]>
#transform_map35 = #rock.transform_map<#map32 by [<Unmerge{4, 48, 24} ["col0", "col1", "col2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 48, 24] -> [4608]>
module {
  // CHECK-LABEL: func.func @mlir_dot_sigmoid
  func.func @mlir_dot_sigmoid(%arg0: memref<1536xf16>, %arg1: memref<1536xf16>, %arg2: memref<4608xf16>) attributes {arch = "gfx950", block_size = 128 : i32, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 8 : i32, kernel = "mixr", output_swizzle = 2 : i64, waves_per_eu = 0 : i64} {
    %cst = arith.constant 1.000000e+00 : f16
    %0 = rock.transform %arg1 by #transform_map : memref<1536xf16> to memref<4x16x24xf16>
    %1 = rock.transform %arg0 by #transform_map1 : memref<1536xf16> to memref<4x24x16xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x24x24xf16>
    %2 = rock.transform %1 by #transform_map2 : memref<4x24x16xf16> to memref<4x16x24xf16>
    %3 = rock.transform %2 by #transform_map3 : memref<4x16x24xf16> to memref<4x64x32xf16>
    %4 = rock.transform %0 by #transform_map4 : memref<4x16x24xf16> to memref<4x64x32xf16>
    %5 = rock.transform %alloc by #transform_map5 : memref<4x24x24xf16> to memref<4x32x32xf16>
    %alloc_0 = memref.alloc() : memref<4x32x32xf32>
    %6 = rock.workgroup_id : index
    %7 = rock.workitem_id : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %8 = arith.remui %6, %c4 : index
    %9 = arith.divui %6, %c4 : index
    %10 = arith.muli %8, %c2 : index
    %11 = arith.addi %9, %10 : index
    %c7 = arith.constant 7 : index
    %12 = arith.cmpi sgt, %6, %c7 : index
    %13 = arith.select %12, %6, %11 : index
    %c32 = arith.constant 32 : index
    %c32_1 = arith.constant 32 : index
    %c2_2 = arith.constant 2 : index
    %c2_3 = arith.constant 2 : index
    %14 = arith.divui %13, %c2_3 : index
    %15 = arith.remui %13, %c2_3 : index
    %16 = arith.divui %15, %c32_1 : index
    %17 = arith.muli %16, %c32 : index
    %18 = arith.subi %c2_2, %17 : index
    %19 = arith.minui %18, %c32 : index
    %20 = arith.remui %15, %19 : index
    %21 = arith.addi %17, %20 : index
    %22 = arith.remui %15, %c32_1 : index
    %23 = arith.divui %22, %19 : index
    %24 = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
    %25 = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    %c0 = arith.constant 0 : index
    %view = memref.view %24[%c0][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<128xvector<8xf16>, #gpu.address_space<workgroup>>
    %c0_4 = arith.constant 0 : index
    %view_5 = memref.view %25[%c0_4][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
    %26 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %27 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %28 = rock.alloc() : memref<1xvector<4xf32>, #gpu.address_space<private>>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<4xf32>
    rock.fill(%28, %cst_6) : memref<1xvector<4xf32>, #gpu.address_space<private>>, vector<4xf32>
    %c1 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %c0_8 = arith.constant 0 : index
    %29 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %30 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %31 = rock.alloc() : memref<8xf16, #gpu.address_space<private>>
    %32 = rock.alloc() : memref<8xf16, #gpu.address_space<private>>
    scf.for %arg3 = %c0_8 to %c1 step %c1_7 {
      rock.stage {
        %35 = rock.transform %3 by #transform_map6 : memref<4x64x32xf16> to memref<1x4x2x1x16x8x1x8xf16>
        %36 = rock.transform %35 by #transform_map7 : memref<1x4x2x1x16x8x1x8xf16> to memref<1x4x2x1x128x8xf16>
        %37 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%36) [%arg3, %14, %21, %23, %7] -> %31 : memref<1x4x2x1x128x8xf16> -> memref<8xf16, #gpu.address_space<private>>, vector<8xi1>
        %38 = rock.transform %4 by #transform_map8 : memref<4x64x32xf16> to memref<1x4x2x1x8x16x8x2xf16>
        %39 = rock.transform %38 by #transform_map9 : memref<1x4x2x1x8x16x8x2xf16> to memref<1x4x2x1x128x16xf16>
        %40 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%39) [%arg3, %14, %21, %23, %7] -> %29 : memref<1x4x2x1x128x16xf16> -> memref<16xf16, #gpu.address_space<private>>, vector<16xi1>
        rock.yield
      } {name = "GlobalRead"}
      rock.stage {
        %35 = rock.workitem_id : index
        %36 = rock.transform %31 by #transform_map10 : memref<8xf16, #gpu.address_space<private>> to memref<1x8xf16, #gpu.address_space<private>>
        %37 = rock.transform %36 by #transform_map11 : memref<1x8xf16, #gpu.address_space<private>> to memref<8x1xf16, #gpu.address_space<private>>
        %38 = rock.transform %32 by #transform_map12 : memref<8xf16, #gpu.address_space<private>> to memref<1x1x8xf16, #gpu.address_space<private>>
        %39 = rock.transform %38 by #transform_map13 : memref<1x1x8xf16, #gpu.address_space<private>> to memref<8x1xf16, #gpu.address_space<private>>
        %c0_12 = arith.constant 0 : index
        %view_13 = memref.view %24[%c0_12][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<128xvector<8xf16>, #gpu.address_space<workgroup>>
        %40 = rock.transform %view_13 by #transform_map14 : memref<128xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x16x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %41 = rock.transform %40 by #transform_map15 : memref<8x16x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x128x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %42 = rock.transform %41 by #transform_map16 : memref<8x128x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x16x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %43 = rock.transform %42 by #transform_map17 : memref<8x16x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<64x16xvector<8xf16>, #gpu.address_space<workgroup>>
        %44 = rock.transform %43 by #transform_map18 : memref<64x16xvector<8xf16>, #gpu.address_space<workgroup>> to memref<16x8x1x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %45 = rock.transform %44 by #transform_map19 : memref<16x8x1x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<128x8xvector<8xf16>, #gpu.address_space<workgroup>>
        rock.threadwise_copy %37 -> %39 : memref<8x1xf16, #gpu.address_space<private>> -> memref<8x1xf16, #gpu.address_space<private>>
        rock.threadwise_write_all {forceUnroll, useIndexDiffs} %32 -> [](%45) [%35] by  set : memref<8xf16, #gpu.address_space<private>> -> memref<128x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %46 = rock.workitem_id : index
        %47 = rock.transform %29 by #transform_map20 : memref<16xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
        %48 = rock.transform %47 by #transform_map21 : memref<8x2xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
        %49 = rock.transform %30 by #transform_map22 : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
        %50 = rock.transform %49 by #transform_map23 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
        %c0_14 = arith.constant 0 : index
        %view_15 = memref.view %25[%c0_14][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
        %51 = rock.transform %view_15 by #transform_map24 : memref<256xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x32x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %52 = rock.transform %51 by #transform_map25 : memref<8x32x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<64x32xvector<8xf16>, #gpu.address_space<workgroup>>
        %53 = rock.transform %52 by #transform_map26 : memref<64x32xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x16x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>>
        %54 = rock.transform %53 by #transform_map27 : memref<8x16x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<128x16xvector<8xf16>, #gpu.address_space<workgroup>>
        rock.threadwise_copy %48 -> %50 : memref<8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>
        rock.threadwise_write_all {forceUnroll, useIndexDiffs} %30 -> [](%54) [%46] by  set : memref<16xf16, #gpu.address_space<private>> -> memref<128x16xvector<8xf16>, #gpu.address_space<workgroup>>
        rock.yield
      } {name = "LDSWrite"}
      rock.lds_barrier
      rock.stage {
        rock.blockwise_gemm_accel %28 += %26 from %view * %27 from %view_5 {blockSize = 128 : i32, matrixParamsA = #rock.blockwise_matrix_params<elementType = f16, elementTypeLoad = f16, rotateDWithK = true, swapThreadIterSubDims = false, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 4, d = 32, inDPerThread = 1>, matrixParamsB = #rock.blockwise_matrix_params<elementType = f16, elementTypeLoad = f16, rotateDWithK = false, swapThreadIterSubDims = true, LDSLayoutDxK = false, directToLDS = false, splitKAcrossThreadsFirst = false, g = 4, d = 32, inDPerThread = 2>, params = #mfma_gemm_params} : memref<1xvector<4xf32>, #gpu.address_space<private>> += memref<2xvector<8xf16>, #gpu.address_space<private>> from memref<128xvector<8xf16>, #gpu.address_space<workgroup>> * memref<2xvector<8xf16>, #gpu.address_space<private>> from memref<256xvector<8xf16>, #gpu.address_space<workgroup>>
        rock.yield
      } {name = "MMA"}
      rock.lds_barrier
    } {pipeline = #rock.pipeline<2>}
    %33 = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
    %c0_9 = arith.constant 0 : index
    rock.transforming_for {forceUnroll, useIndexDiffs} (%arg3) = [](%c0_9), (%arg4) = [#transform_map28](%c0_9) (%arg5, %arg6) = validity bounds [1] strides [1] {
      %35 = memref.load %28[%arg3] : memref<1xvector<4xf32>, #gpu.address_space<private>>
      rock.in_bounds_store %35 -> %33[%arg4] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index
      rock.yield
    }
    rock.threadwise_write_all {forceUnroll, useIndexDiffs} %33 -> [#transform_map29, #transform_map30, #transform_map31, #transform_map32, #transform_map33, #transform_map34](%alloc_0) [%14, %21, %23, %7] by  set : memref<4xf32, #gpu.address_space<private>> -> memref<4x32x32xf32>
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<4x32x32xf32>) outs(%5 : memref<4x32x32xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: f32, %out: f16):
      %35 = arith.truncf %in : f32 to f16
      linalg.yield %35 : f16
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<4x24x24xf16>
    // The sigmoid linalg.generic should be fused to operate on GPU private memory
    // CHECK: arith.negf
    // CHECK: math.exp
    // CHECK: arith.addf
    // CHECK: arith.divf
    linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<4x24x24xf16>) outs(%alloc_10 : memref<4x24x24xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: f16, %out: f16):
      %35 = arith.negf %in : f16
      %36 = math.exp %35 : f16
      %37 = arith.addf %36, %cst : f16
      %38 = arith.divf %cst, %37 : f16
      linalg.yield %38 : f16
    }
    %alloc_11 = memref.alloc() : memref<4608xf16>
    %34 = rock.transform %alloc_11 by #transform_map35 : memref<4608xf16> to memref<4x48x24xf16>
    // The subview should still exist in the IR
    // CHECK: memref.subview
    // CHECK-SAME: memref<4x48x24xf16> to memref<4x24x24xf16, strided<[1152, 24, 1]>>
    %subview = memref.subview %34[0, 0, 0] [4, 24, 24] [1, 1, 1] : memref<4x48x24xf16> to memref<4x24x24xf16, strided<[1152, 24, 1]>>
    // Verify the transform chain is applied to %arg2 (the output)
    // CHECK: rock.transform %arg2
    // The threadwise_write_all should write to the output with the view chain
    // CHECK: rock.threadwise_write_all
    // CHECK-SAME: memref<4xf16, #gpu.address_space<private>>
    // CHECK-SAME: memref<4x32x32xf16>
    // Verify that the memref.copy operations have been eliminated
    // CHECK-NOT: memref.copy
    memref.copy %alloc_10, %subview : memref<4x24x24xf16> to memref<4x24x24xf16, strided<[1152, 24, 1]>>
    memref.copy %alloc_11, %arg2 : memref<4608xf16> to memref<4608xf16>
    return
  }
}

