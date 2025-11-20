// RUN: rocmlir-opt --rock-add-async-wait --split-input-file --verify-diagnostics %s | FileCheck %s
func.func @gemm_pipelining(%arg0: memref<2359296xbf16>, %arg1: memref<2359296xbf16>, %arg2: memref<3145728xbf16>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 768 : i32, kernel, arch = "gfx950:sramecc+:xnack-", num_cu = 256 : i64} {
  %c11 = arith.constant 11 : index
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %c767 = arith.constant 767 : index
  %c192 = arith.constant 192 : index
  %c4 = arith.constant 4 : index
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 1024 + d2)> by [<Unmerge{3, 768, 1024} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 1024] -> [2359296]> : memref<2359296xbf16> to memref<3x768x1024xbf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 1024 + d2)> by [<Unmerge{3, 768, 1024} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 1024] -> [2359296]> : memref<2359296xbf16> to memref<3x768x1024xbf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 1024 + d2)> by [<Unmerge{3, 1024, 1024} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 1024] -> [3145728]> : memref<3145728xbf16> to memref<3x1024x1024xbf16>
  %3 = rock.workgroup_id : index
  %4 = rock.workitem_id : index
  %5 = arith.remui %3, %c4 : index
  %6 = arith.divui %3, %c4 : index
  %7 = arith.muli %5, %c192 : index
  %8 = arith.addi %6, %7 : index
  %9 = arith.cmpi sgt, %3, %c767 : index
  %10 = arith.select %9, %3, %8 : index
  %11 = arith.divui %10, %c256 : index
  %12 = arith.remui %10, %c256 : index
  %13 = arith.divui %12, %c512 : index
  %14 = arith.muli %13, %c32 : index
  %15 = arith.subi %c16, %14 : index
  %16 = arith.minui %15, %c32 : index
  %17 = arith.remui %12, %16 : index
  %18 = arith.addi %14, %17 : index
  %19 = arith.remui %12, %c512 : index
  %20 = arith.divui %19, %16 : index
  %21 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %22 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %23 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %24 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %25 = rock.alloc() : memref<64xi8, #gpu.address_space<private>>
  %view = memref.view %25[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<32xbf16, #gpu.address_space<private>>
  %26 = rock.alloc() : memref<64xi8, #gpu.address_space<private>>
  %view_0 = memref.view %26[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<32xbf16, #gpu.address_space<private>>
  %27 = rock.alloc() : memref<1xvector<16xf32>, #gpu.address_space<private>>
  affine.for %arg3 = 0 to 1 {
    memref.store %cst, %27[%arg3] : memref<1xvector<16xf32>, #gpu.address_space<private>>
  }
  %view_1 = memref.view %23[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_2 = memref.view %24[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_3 = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_4 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %28 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d2 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["m_block", "m_thread", "m_iter"] at [2, 5, 7] -> ["m"] at [2]>, <AddDim{16} ["n_block"] at [3] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
  %29 = rock.transform %28 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "m_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "m_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
  %30 = rock.extract_multibuffer(%view_3, %view_4) [%c0](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %31 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%29) [%c0, %11, %18, %20, %4] -> %30 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
  %32 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d3 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
  %33 = rock.transform %32 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
  %34 = rock.extract_multibuffer(%view_1, %view_2) [%c0](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %35 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%33) [%c0, %11, %18, %20, %4] -> %34 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
  rock.lds_barrier {barrier_stage = #rock<BarrierStage forward>}
  %36 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d2 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["m_block", "m_thread", "m_iter"] at [2, 5, 7] -> ["m"] at [2]>, <AddDim{16} ["n_block"] at [3] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
  %37 = rock.transform %36 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "m_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "m_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
  %38 = rock.extract_multibuffer(%view_3, %view_4) [%c1](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %39 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%37) [%c1, %11, %18, %20, %4] -> %38 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
  %40 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d3 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
  %41 = rock.transform %40 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
  %42 = rock.extract_multibuffer(%view_1, %view_2) [%c1](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %43 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%41) [%c1, %11, %18, %20, %4] -> %42 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
  %44 = rock.workitem_id : index
  %view_5 = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_6 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %45 = rock.extract_multibuffer(%view_5, %view_6) [%c0](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %46 = rock.transform %45 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
  %47 = rock.transform %46 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d0) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_m", "blk_td"] at [4, 0, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_n"] at [1] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %48 = rock.transform %47 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %49 = rock.transform %48 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
  %50 = rock.transform %49 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["mk"] at [1] -> ["m", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
  %51 = rock.extract_multibuffer(%view) [%c0](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
  // This will wait for the first 2 rock.threadwise_read_into
  // CHECK: rock.async_wait {num_inst = 6 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%50) [%44] -> %51 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
  %52 = rock.workitem_id : index
  %view_7 = memref.view %23[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_8 = memref.view %24[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %53 = rock.extract_multibuffer(%view_7, %view_8) [%c0](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %54 = rock.transform %53 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
  %55 = rock.transform %54 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d1) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_n", "blk_td"] at [4, 1, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_m"] at [0] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %56 = rock.transform %55 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %57 = rock.transform %56 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
  %58 = rock.transform %57 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["nk"] at [1] -> ["n", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
  %59 = rock.extract_multibuffer(%view_0) [%c0](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
  // This will wait for the last 4 rock.threadwise_read_into
  // CHECK: rock.async_wait {num_inst = 4 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%58) [%52] -> %59 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
  scf.for %arg3 = %c0 to %c10 step %c1 {
    rock.lds_barrier {barrier_stage = #rock<BarrierStage forward>}
    %78 = arith.addi %arg3, %c2 : index
    %79 = arith.addi %arg3, %c2 : index
    %80 = arith.addi %arg3, %c2 : index
    %81 = arith.addi %arg3, %c2 : index
    %82 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d2 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["m_block", "m_thread", "m_iter"] at [2, 5, 7] -> ["m"] at [2]>, <AddDim{16} ["n_block"] at [3] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
    %83 = rock.transform %82 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "m_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "m_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
    %84 = rock.extract_multibuffer(%view_3, %view_4) [%78](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %85 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%83) [%79, %11, %18, %20, %4] -> %84 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
    %86 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d3 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
    %87 = rock.transform %86 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
    %88 = rock.extract_multibuffer(%view_1, %view_2) [%80](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %89 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%87) [%81, %11, %18, %20, %4] -> %88 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
    affine.for %arg4 = 0 to 1 {
      %view_17 = memref.view %25[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %110 = rock.extract_multibuffer(%view_17) [%arg3](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %111 = rock.transform %110 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["iidx", "k"] at [0, 1] -> ["mk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      %subview = memref.subview %111[%arg4, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
      %112 = rock.transform %subview by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      affine.for %arg5 = 0 to 1 {
        %view_18 = memref.view %26[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
        %113 = rock.extract_multibuffer(%view_18) [%arg3](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
        %114 = rock.transform %113 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["jidx", "k"] at [0, 1] -> ["nk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
        %subview_19 = memref.subview %114[%arg5, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
        %115 = rock.transform %subview_19 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
        affine.for %arg6 = 0 to 4 {
          %116 = rock.transform %27 by <affine_map<(d0, d1) -> (d0 + d1)> by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]> : memref<1xvector<16xf32>, #gpu.address_space<private>> to memref<1x1xvector<16xf32>, #gpu.address_space<private>>
          rock.threadwise_gemm_accel %116 += %112 * %115 at[%arg4, %arg5, %arg6] features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 4, outputSwizzle = 2, forceUnroll = true>} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x4xvector<8xbf16>, #gpu.address_space<private>> * memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
        }
      }
    }
    %90 = arith.addi %arg3, %c1 : index
    %91 = arith.addi %arg3, %c1 : index
    %92 = arith.addi %arg3, %c1 : index
    %93 = arith.addi %arg3, %c1 : index
    %94 = rock.workitem_id : index
    %view_13 = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
    %view_14 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
    %95 = rock.extract_multibuffer(%view_13, %view_14) [%90](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %96 = rock.transform %95 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
    %97 = rock.transform %96 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d0) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_m", "blk_td"] at [4, 0, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_n"] at [1] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
    %98 = rock.transform %97 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
    %99 = rock.transform %98 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
    %100 = rock.transform %99 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["mk"] at [1] -> ["m", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
    %101 = rock.extract_multibuffer(%view) [%91](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
    // CHECK: rock.async_wait {num_inst = 6 : i32}
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%100) [%94] -> %101 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
    %102 = rock.workitem_id : index
    %view_15 = memref.view %23[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
    %view_16 = memref.view %24[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
    %103 = rock.extract_multibuffer(%view_15, %view_16) [%92](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %104 = rock.transform %103 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
    %105 = rock.transform %104 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d1) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_n", "blk_td"] at [4, 1, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_m"] at [0] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
    %106 = rock.transform %105 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
    %107 = rock.transform %106 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
    %108 = rock.transform %107 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["nk"] at [1] -> ["n", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
    %109 = rock.extract_multibuffer(%view_0) [%93](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
    // rock.async_wait {num_inst = 4 : i32}
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%108) [%102] -> %109 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
  }
  rock.lds_barrier {barrier_stage = #rock<BarrierStage forward>}
  affine.for %arg3 = 0 to 1 {
    %view_13 = memref.view %25[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
    %78 = rock.extract_multibuffer(%view_13) [%c10](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
    %79 = rock.transform %78 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["iidx", "k"] at [0, 1] -> ["mk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
    %subview = memref.subview %79[%arg3, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
    %80 = rock.transform %subview by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
    affine.for %arg4 = 0 to 1 {
      %view_14 = memref.view %26[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %81 = rock.extract_multibuffer(%view_14) [%c10](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %82 = rock.transform %81 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["jidx", "k"] at [0, 1] -> ["nk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      %subview_15 = memref.subview %82[%arg4, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
      %83 = rock.transform %subview_15 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      affine.for %arg5 = 0 to 4 {
        %84 = rock.transform %27 by <affine_map<(d0, d1) -> (d0 + d1)> by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]> : memref<1xvector<16xf32>, #gpu.address_space<private>> to memref<1x1xvector<16xf32>, #gpu.address_space<private>>
        rock.threadwise_gemm_accel %84 += %80 * %83 at[%arg3, %arg4, %arg5] features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 4, outputSwizzle = 2, forceUnroll = true>} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x4xvector<8xbf16>, #gpu.address_space<private>> * memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      }
    }
  }
  %60 = rock.workitem_id : index
  %view_9 = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_10 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %61 = rock.extract_multibuffer(%view_9, %view_10) [%c11](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %62 = rock.transform %61 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
  %63 = rock.transform %62 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d0) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_m", "blk_td"] at [4, 0, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_n"] at [1] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %64 = rock.transform %63 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %65 = rock.transform %64 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
  %66 = rock.transform %65 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["mk"] at [1] -> ["m", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
  %67 = rock.extract_multibuffer(%view) [%c11](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
  // CHECK: rock.async_wait {num_inst = 0 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%66) [%60] -> %67 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
  %68 = rock.workitem_id : index
  %view_11 = memref.view %23[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_12 = memref.view %24[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %69 = rock.extract_multibuffer(%view_11, %view_12) [%c11](memref<4096xbf16, #gpu.address_space<workgroup>>, memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
  %70 = rock.transform %69 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
  %71 = rock.transform %70 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d1) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_n", "blk_td"] at [4, 1, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_m"] at [0] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %72 = rock.transform %71 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
  %73 = rock.transform %72 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
  %74 = rock.transform %73 by <affine_map<(d0, d1) -> (d0, 0, d1)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{1, 32} ["nk"] at [1] -> ["n", "k"] at [1, 2]>] bounds = [256, 32] -> [256, 1, 32]> : memref<256x1x32xbf16, #gpu.address_space<workgroup>> to memref<256x32xbf16, #gpu.address_space<workgroup>>
  %75 = rock.extract_multibuffer(%view_0) [%c11](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
  // CHECK: rock.async_wait {num_inst = 0 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%74) [%68] -> %75 : memref<256x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
  affine.for %arg3 = 0 to 1 {
    %view_13 = memref.view %25[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
    %78 = rock.extract_multibuffer(%view_13) [%c11](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
    %79 = rock.transform %78 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["iidx", "k"] at [0, 1] -> ["mk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
    %subview = memref.subview %79[%arg3, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
    %80 = rock.transform %subview by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
    affine.for %arg4 = 0 to 1 {
      %view_14 = memref.view %26[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %81 = rock.extract_multibuffer(%view_14) [%c11](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %82 = rock.transform %81 by <affine_map<(d0, d1) -> (d0 * 4 + d1)> by [<Unmerge{1, 4} ["jidx", "k"] at [0, 1] -> ["nk"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      %subview_15 = memref.subview %82[%arg4, 0] [1, 4] [1, 1] : memref<1x4xvector<8xbf16>, #gpu.address_space<private>> to memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>>
      %83 = rock.transform %subview_15 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, strided<[1], offset: ?>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      affine.for %arg5 = 0 to 4 {
        %84 = rock.transform %27 by <affine_map<(d0, d1) -> (d0 + d1)> by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]> : memref<1xvector<16xf32>, #gpu.address_space<private>> to memref<1x1xvector<16xf32>, #gpu.address_space<private>>
        rock.threadwise_gemm_accel %84 += %80 * %83 at[%arg3, %arg4, %arg5] features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 4, outputSwizzle = 2, forceUnroll = true>} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x4xvector<8xbf16>, #gpu.address_space<private>> * memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
      }
    }
  }  
  // Omitted rest of the IR here for simplicity
  return
}

// -----

func.func @gemm_no_pipelining(%arg0: memref<2359296xbf16>, %arg1: memref<2359296xbf16>, %arg2: memref<3145728xbf16>) attributes {block_size = 256 : i32, enable_splitk_for_tuning, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 768 : i32, kernel, arch = "gfx950:sramecc+:xnack-", num_cu = 256 : i64} {
  %c1 = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c16 = arith.constant 16 : index
  %c512 = arith.constant 512 : index
  %c32 = arith.constant 32 : index
  %c767 = arith.constant 767 : index
  %c192 = arith.constant 192 : index
  %c4 = arith.constant 4 : index
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 1024 + d2)> by [<Unmerge{3, 768, 1024} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 1024] -> [2359296]> : memref<2359296xbf16> to memref<3x768x1024xbf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 1024 + d2)> by [<Unmerge{3, 768, 1024} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 1024] -> [2359296]> : memref<2359296xbf16> to memref<3x768x1024xbf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 1024 + d2)> by [<Unmerge{3, 1024, 1024} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 1024] -> [3145728]> : memref<3145728xbf16> to memref<3x1024x1024xbf16>
  %3 = rock.workgroup_id : index
  %4 = rock.workitem_id : index
  %5 = arith.remui %3, %c4 : index
  %6 = arith.divui %3, %c4 : index
  %7 = arith.muli %5, %c192 : index
  %8 = arith.addi %6, %7 : index
  %9 = arith.cmpi sgt, %3, %c767 : index
  %10 = arith.select %9, %3, %8 : index
  %11 = arith.divui %10, %c256 : index
  %12 = arith.remui %10, %c256 : index
  %13 = arith.divui %12, %c512 : index
  %14 = arith.muli %13, %c32 : index
  %15 = arith.subi %c16, %14 : index
  %16 = arith.minui %15, %c32 : index
  %17 = arith.remui %12, %16 : index
  %18 = arith.addi %14, %17 : index
  %19 = arith.remui %12, %c512 : index
  %20 = arith.divui %19, %16 : index
  %21 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %22 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %view = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_0 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %23 = rock.alloc() : memref<64xi8, #gpu.address_space<private>>
  %24 = rock.alloc() : memref<64xi8, #gpu.address_space<private>>
  %25 = rock.alloc() : memref<1xvector<16xf32>, #gpu.address_space<private>>
  affine.for %arg3 = 0 to 1 {
    memref.store %cst, %25[%arg3] : memref<1xvector<16xf32>, #gpu.address_space<private>>
  }
  %view_1 = memref.view %22[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  %view_2 = memref.view %21[%c0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<4096xbf16, #gpu.address_space<workgroup>>
  scf.for %arg3 = %c0 to %c12 step %c1 {
    rock.lds_barrier {barrier_stage = #rock<BarrierStage backward>}
    %28 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d2 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["m_block", "m_thread", "m_iter"] at [2, 5, 7] -> ["m"] at [2]>, <AddDim{16} ["n_block"] at [3] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
    %29 = rock.transform %28 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "m_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "m_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
    %30 = rock.extract_multibuffer(%view_2) [%arg3](memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %31 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%29) [%arg3, %11, %18, %20, %4] -> %30 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
    %32 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 2 + d6) * 32 + d4, (d3 * 8 + d5) * 8 + d7)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{12, 2, 32} ["k_loop", "k_iter", "k_thread"] at [0, 6, 4] -> ["k"] at [1]>, <Unmerge{16, 8, 8} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [12, 3, 16, 16, 32, 8, 2, 8] -> [3, 768, 1024]> : memref<3x768x1024xbf16> to memref<12x3x16x16x32x8x2x8xbf16>
    %33 = rock.transform %32 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [12, 3, 16, 16, 256, 16] -> [12, 3, 16, 16, 32, 8, 2, 8]> : memref<12x3x16x16x32x8x2x8xbf16> to memref<12x3x16x16x256x16xbf16>
    %34 = rock.extract_multibuffer(%view_1) [%arg3](memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
    %35 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%33) [%arg3, %11, %18, %20, %4] -> %34 : memref<12x3x16x16x256x16xbf16> -> memref<4096xbf16, #gpu.address_space<workgroup>>, vector<4096xi1>
    rock.lds_barrier {barrier_stage = #rock<BarrierStage forward>}
    // CHECK: rock.async_wait {num_inst = 0 : i32}
    %36 = rock.workitem_id : index
    affine.for %arg4 = 0 to 1 {
      %view_3 = memref.view %23[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
      %view_4 = memref.view %23[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<32xbf16, #gpu.address_space<private>>
      %37 = rock.extract_multibuffer(%view) [%arg3](memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
      %38 = rock.transform %37 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
      %39 = rock.transform %38 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d0) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_m", "blk_td"] at [4, 0, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_n"] at [1] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
      %40 = rock.transform %39 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
      %41 = rock.transform %40 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
      %42 = rock.extract_multibuffer(%view_4) [%arg4](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%41) [%36, %arg4] -> %42 : memref<256x1x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
      affine.for %arg5 = 0 to 1 {
        %view_5 = memref.view %24[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<4xvector<8xbf16>, #gpu.address_space<private>>
        %view_6 = memref.view %24[%c0][] : memref<64xi8, #gpu.address_space<private>> to memref<32xbf16, #gpu.address_space<private>>
        %43 = rock.extract_multibuffer(%view_0) [%arg3](memref<4096xbf16, #gpu.address_space<workgroup>>) : memref<4096xbf16, #gpu.address_space<workgroup>>
        %44 = rock.transform %43 by <affine_map<(d0, d1) -> (d1 * 64 + d0)> by [<Unmerge{64, 64} ["k", "d"] at [1, 0] -> ["source_offset"] at [0]>] bounds = [64, 64] -> [4096]> : memref<4096xbf16, #gpu.address_space<workgroup>> to memref<64x64xbf16, #gpu.address_space<workgroup>>
        %45 = rock.transform %44 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((d4 * 2 + d1) * 32 + d3, (d2 * 4 + d5) * 8 + d6)> by [<Unmerge{1, 2, 32} ["d_iter", "wave_n", "blk_td"] at [4, 1, 3] -> ["d"] at [0]>, <Unmerge{2, 4, 8} ["blk_id", "k_iter", "k_vec"] at [2, 5, 6] -> ["k"] at [1]>, <AddDim{2} ["wave_m"] at [0] -> [] at []>] bounds = [2, 2, 2, 32, 1, 4, 8] -> [64, 64]> : memref<64x64xbf16, #gpu.address_space<workgroup>> to memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
        %46 = rock.transform %45 by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 floordiv 2, d0 mod 2, d1, d2, d3, d4, d5)> by [<Merge{2, 2} ["wave_id"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [1, 2, 3, 4, 5] -> ["blk_id", "blk_td", "d_iter", "k_iter", "k_vec"] at [2, 3, 4, 5, 6]>] bounds = [4, 2, 32, 1, 4, 8] -> [2, 2, 2, 32, 1, 4, 8]> : memref<2x2x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>>
        %47 = rock.transform %46 by <affine_map<(d0, d1, d2) -> (d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1, d2 floordiv 8, d2 mod 8)> by [<Merge{4, 2, 32} ["tid"] at [0] -> ["wave_id", "blk_id", "blk_td"] at [0, 1, 2]>, <Merge{4, 8} ["k_iter"] at [2] -> ["k_iter", "k_vec"] at [4, 5]>, <PassThrough ["d_iter"] at [1] -> ["d_iter"] at [3]>] bounds = [256, 1, 32] -> [4, 2, 32, 1, 4, 8]> : memref<4x2x32x1x4x8xbf16, #gpu.address_space<workgroup>> to memref<256x1x32xbf16, #gpu.address_space<workgroup>>
        %48 = rock.extract_multibuffer(%view_6) [%arg5](memref<32xbf16, #gpu.address_space<private>>) : memref<32xbf16, #gpu.address_space<private>>
        rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%47) [%36, %arg5] -> %48 : memref<256x1x32xbf16, #gpu.address_space<workgroup>> -> memref<32xbf16, #gpu.address_space<private>>
        affine.for %arg6 = 0 to 4 {
          %49 = rock.transform %25 by <affine_map<(d0, d1) -> (d0 + d1)> by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]> : memref<1xvector<16xf32>, #gpu.address_space<private>> to memref<1x1xvector<16xf32>, #gpu.address_space<private>>
          %50 = rock.extract_multibuffer(%view_3) [%arg4](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
          %51 = rock.transform %50 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
          %52 = rock.extract_multibuffer(%view_5) [%arg5](memref<4xvector<8xbf16>, #gpu.address_space<private>>) : memref<4xvector<8xbf16>, #gpu.address_space<private>>
          %53 = rock.transform %52 by <affine_map<(d0, d1) -> (d1)> by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 4] -> [4]> : memref<4xvector<8xbf16>, #gpu.address_space<private>> to memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
          rock.threadwise_gemm_accel %49 += %51 * %53 at[%arg4, %arg5, %arg6] features =  mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b {params = #rock.mfma_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 3, outputSwizzle = 2, forceUnroll = true>} : memref<1x1xvector<16xf32>, #gpu.address_space<private>> += memref<1x4xvector<8xbf16>, #gpu.address_space<private>> * memref<1x4xvector<8xbf16>, #gpu.address_space<private>>
        }
      }
    }
  }
  // Omitted rest of the IR here for simplicity
  return
}

// -----

func.func @async_wait_simple_test(%arg0: memref<4x256xf32>, %arg1: memref<4x256xf32>, %arg2: memref<4x256xf32>, %arg3: memref<4x256xf32>) attributes {arch = "gfx950:sramecc+:xnack-", block_size = 256 : i32, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 1 : i32, kernel, num_cu = 256 : i64} {
  %tid = rock.workitem_id : index
  %1 = rock.alloc() : memref<1x256xf32, #gpu.address_space<workgroup>>
  %2 = rock.alloc() : memref<1x256xf32, #gpu.address_space<workgroup>>
  %3 = rock.alloc() : memref<1x256xf32, #gpu.address_space<workgroup>>
  %4 = rock.alloc() : memref<1x256xf32, #gpu.address_space<workgroup>>
  %5 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  %6 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  %7 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  %8 = rock.alloc() : memref<64xf32, #gpu.address_space<private>>
  
  // Global loads (Global memory -> LDS)
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%arg0) [%tid] -> %1 : memref<4x256xf32> -> memref<1x256xf32, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%arg1) [%tid] -> %2 : memref<4x256xf32> -> memref<1x256xf32, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%arg2) [%tid] -> %3 : memref<4x256xf32> -> memref<1x256xf32, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%arg3) [%tid] -> %4 : memref<4x256xf32> -> memref<1x256xf32, #gpu.address_space<workgroup>>
  
  // Local loads (LDS -> registers)
  // 
  // CHECK: rock.async_wait {num_inst = 3 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%1) [%tid] -> %5 : memref<1x256xf32, #gpu.address_space<workgroup>> -> memref<64xf32, #gpu.address_space<private>>
  // CHECK: rock.async_wait {num_inst = 2 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%2) [%tid] -> %6 : memref<1x256xf32, #gpu.address_space<workgroup>> -> memref<64xf32, #gpu.address_space<private>>
  // CHECK: rock.async_wait {num_inst = 1 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%3) [%tid] -> %7 : memref<1x256xf32, #gpu.address_space<workgroup>> -> memref<64xf32, #gpu.address_space<private>>
  // CHECK: rock.async_wait {num_inst = 0 : i32}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%tid] -> %8 : memref<1x256xf32, #gpu.address_space<workgroup>> -> memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

func.func @async_wait_error(%arg0: memref<4x256xf32>, %arg1: memref<4x256xf32>, %arg2: memref<4x256xf32>, %arg3: memref<4x256xf32>) attributes {arch = "gfx950:sramecc+:xnack-", block_size = 256 : i32, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 1 : i32, kernel, num_cu = 256 : i64} {
  %tid = rock.workitem_id : index
  %1 = rock.alloc() : memref<1x256xf32, #gpu.address_space<workgroup>>
  
  // expected-error @+1 {{No use found for ThreadwiseReadIntoOp. Is there a ThreadwiseReadIntoOp that writes to LDS but none is reading from it?}}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%arg0) [%tid] -> %1 : memref<4x256xf32> -> memref<1x256xf32, #gpu.address_space<workgroup>>  
  return
}
