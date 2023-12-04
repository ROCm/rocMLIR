// RUN: rocmlir-opt %s --rock-multibuffer-test | FileCheck %s
// RUN: rocmlir-opt %s --rock-multibuffer-test --rock-blockwise-gemm-to-threadwise --rock-threadwise-gemm-lowering | FileCheck %s --check-prefix=LOWERING
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 8 + d5) * 8 + d7, (d2 * 32 + d4) * 2 + d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 8, d4 mod 8, d5 floordiv 8, d5 mod 8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 8 + d4) * 8 + d6, (d3 * 32 + d5) * 2 + d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 32, d4 mod 32, d5 floordiv 2, d5 mod 2)>
#map5 = affine_map<(d0, d1) -> (d0 * 8 + d1)>
#map6 = affine_map<(d0, d1) -> (d1, d0)>
#map7 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 8 + d2)>
#map8 = affine_map<(d0, d1) -> (0, d1, d0)>
#map9 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map10 = affine_map<(d0, d1) -> (d0, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1 + d2)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 64, d1 mod 64, d2, d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d0 + d1, d2, d3)>
#map15 = affine_map<(d0, d1) -> (d0 floordiv 8, d1, 0, d0 mod 8)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> ((d1 + d2) * 8 + d4, d0 * 2 + d3)>
#map17 = affine_map<(d0, d1) -> (d0 floordiv 8, d0 mod 8, 0, d1 floordiv 8, d1 mod 8)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> ((d0 + d2) * 8 + d4, d3 * 32 + d1)>
#map19 = affine_map<(d0, d1) -> (d0 floordiv 32, d0 mod 32, 0, d1 floordiv 8, d1 mod 8)>
#xldops_gemm_params = #rock.xdlops_gemm_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, forceUnroll = true>
#transform_map = #rock.transform_map<#map by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 1024, 384] -> [1, 384, 1024]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{16, 8, 8} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{6, 32, 2} ["m_block", "m_thread", "m_iter"] at [2, 4, 6] -> ["m"] at [2]>, <AddDim{16} ["n_block"] at [3] -> [] at []>] bounds = [16, 1, 6, 16, 32, 8, 2, 8] -> [1, 1024, 384]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 8} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{2, 8} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [16, 1, 6, 16, 256, 16] -> [16, 1, 6, 16, 32, 8, 2, 8]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{16, 8, 8} ["k_loop", "k_thread", "k_iter"] at [0, 4, 6] -> ["k"] at [1]>, <Unmerge{16, 32, 2} ["n_block", "n_thread", "n_iter"] at [3, 5, 7] -> ["n"] at [2]>, <AddDim{6} ["m_block"] at [2] -> [] at []>] bounds = [16, 1, 6, 16, 8, 32, 8, 2] -> [1, 1024, 1024]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{8, 32} ["tid"] at [4] -> ["k_thread", "n_thread"] at [4, 5]>, <Merge{8, 2} ["iter"] at [5] -> ["k_iter", "n_iter"] at [6, 7]>] bounds = [16, 1, 6, 16, 256, 16] -> [16, 1, 6, 16, 8, 32, 8, 2]>
#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{2, 8} ["m_iter", "k_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [2, 8] -> [16]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [1]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [0]>] bounds = [8, 2] -> [2, 8]>
#transform_map7 = #rock.transform_map<#map7 by [<Unmerge{1, 2, 8} ["kouterPerThread", "m_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
#transform_map8 = #rock.transform_map<#map8 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{2} ["m"] at [1] -> ["m_iter"] at [1]>] bounds = [8, 2] -> [1, 2, 8]>
#transform_map9 = #rock.transform_map<#map9 by [<Unmerge{8, 2} ["k_iter", "n_iter"] at [0, 1] -> ["iter"] at [0]>] bounds = [8, 2] -> [16]>
#transform_map10 = #rock.transform_map<#map10 by [<Merge{8} ["k"] at [0] -> ["k_iter"] at [0]>, <Merge{2} ["n"] at [1] -> ["n_iter"] at [1]>] bounds = [8, 2] -> [8, 2]>
#transform_map11 = #rock.transform_map<#map7 by [<Unmerge{1, 2, 8} ["kouterPerThread", "n_iter", "kpackPerThread"] at [0, 1, 2] -> ["iter"] at [0]>] bounds = [1, 2, 8] -> [16]>
#transform_map12 = #rock.transform_map<#map8 by [<Merge{1, 8} ["k"] at [0] -> ["kouterPerThread", "kpackPerThread"] at [0, 2]>, <Merge{2} ["n"] at [1] -> ["n_iter"] at [1]>] bounds = [8, 2] -> [1, 2, 8]>
#transform_map13 = #rock.transform_map<#map11 by [<Unmerge{8, 64, 1} ["k_outer", "m", "kpack_idx"] at [0, 1, 2] -> ["raw"] at [0]>, <AddDim{8} ["kpack_vec"] at [3] -> [] at []>] bounds = [8, 64, 1, 8] -> [512]>
#transform_map14 = #rock.transform_map<#map12 by [<PassThrough ["k_outer"] at [0] -> ["k_outer"] at [0]>, <AddDim{8} ["to_discard"] at [1] -> [] at []>, <PassThrough ["m"] at [2] -> ["m"] at [1]>, <PassThrough ["kpack_idx"] at [3] -> ["kpack_idx"] at [2]>, <PassThrough ["kpack_vec"] at [4] -> ["kpack_vec"] at [3]>] bounds = [8, 8, 64, 1, 8] -> [8, 64, 1, 8]>
#transform_map15 = #rock.transform_map<#map13 by [<PassThrough ["k_outer"] at [0] -> ["k_outer"] at [0]>, <Merge{8, 64} ["m"] at [1] -> ["to_discard", "m"] at [1, 2]>, <PassThrough ["kpack_idx"] at [2] -> ["kpack_idx"] at [3]>, <PassThrough ["kpack_vec"] at [3] -> ["kpack_vec"] at [4]>] bounds = [8, 512, 1, 8] -> [8, 8, 64, 1, 8]>
#transform_map16 = #rock.transform_map<#map14 by [<PassThrough ["k_outer"] at [0] -> ["k_outer"] at [0]>, <Embed{1, 1} ["k_outer", "m"] at [0, 1] -> ["m"] at [1]>, <PassThrough ["kpack_idx", "kpack_vec"] at [2, 3] -> ["kpack_idx", "kpack_vec"] at [2, 3]>] bounds = [8, 64, 1, 8] -> [8, 512, 1, 8]>
#transform_map17 = #rock.transform_map<#map15 by [<Merge{8, 1, 8} ["k"] at [0] -> ["k_outer", "kpack_idx", "kpack_vec"] at [0, 2, 3]>, <Merge{64} ["d"] at [1] -> ["m"] at [1]>] bounds = [64, 64] -> [8, 64, 1, 8]>
#transform_map18 = #rock.transform_map<#map16 by [<Unmerge{8, 1, 8} ["k_thread", "kouterPerThread", "kpackPerThread"] at [1, 2, 4] -> ["k"] at [0]>, <Unmerge{32, 2} ["m_thread", "m_iter"] at [0, 3] -> ["m"] at [1]>] bounds = [32, 8, 1, 2, 8] -> [64, 64]>
#transform_map19 = #rock.transform_map<#map17 by [<Merge{32, 8} ["tid"] at [0] -> ["m_thread", "k_thread"] at [0, 1]>, <Merge{1, 2, 8} ["iter"] at [1] -> ["kouterPerThread", "m_iter", "kpackPerThread"] at [2, 3, 4]>] bounds = [256, 16] -> [32, 8, 1, 2, 8]>
#transform_map20 = #rock.transform_map<#map11 by [<Unmerge{8, 64, 1} ["k_outer", "n", "kpack_idx"] at [0, 1, 2] -> ["raw"] at [0]>, <AddDim{8} ["kpack_vec"] at [3] -> [] at []>] bounds = [8, 64, 1, 8] -> [512]>
#transform_map21 = #rock.transform_map<#map15 by [<Merge{8, 1, 8} ["k"] at [0] -> ["k_outer", "kpack_idx", "kpack_vec"] at [0, 2, 3]>, <Merge{64} ["d"] at [1] -> ["n"] at [1]>] bounds = [64, 64] -> [8, 64, 1, 8]>
#transform_map22 = #rock.transform_map<#map18 by [<Unmerge{8, 1, 8} ["k_thread", "kouterPerThread", "kpackPerThread"] at [0, 2, 4] -> ["k"] at [0]>, <Unmerge{2, 32} ["n_iter", "n_thread"] at [3, 1] -> ["n"] at [1]>] bounds = [8, 32, 1, 2, 8] -> [64, 64]>
#transform_map23 = #rock.transform_map<#map19 by [<Merge{8, 32} ["tid"] at [0] -> ["k_thread", "n_thread"] at [0, 1]>, <Merge{1, 2, 8} ["iter"] at [1] -> ["kouterPerThread", "n_iter", "kpackPerThread"] at [2, 3, 4]>] bounds = [256, 16] -> [8, 32, 1, 2, 8]>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a"} {
  func.func @rock_gemm(%arg0: memref<1x384x1024xf16>, %arg1: memref<1x1024x1024xf16>, %arg2: memref<1x384x1024xf16>) attributes {block_size = 256 : i32, grid_size = 96 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a", wave_size = 64 : i32} {
    %0 = rock.transform %arg0 by #transform_map : memref<1x384x1024xf16> to memref<1x1024x384xf16>
    %alloc = memref.alloc() : memref<1x384x1024xf32>
    %1 = rock.transform %0 by #transform_map1 : memref<1x1024x384xf16> to memref<16x1x6x16x32x8x2x8xf16>
    %2 = rock.transform %1 by #transform_map2 : memref<16x1x6x16x32x8x2x8xf16> to memref<16x1x6x16x256x16xf16>
    %3 = rock.transform %arg1 by #transform_map3 : memref<1x1024x1024xf16> to memref<16x1x6x16x8x32x8x2xf16>
    %4 = rock.transform %3 by #transform_map4 : memref<16x1x6x16x8x32x8x2xf16> to memref<16x1x6x16x256x16xf16>
    %5 = rock.workgroup_id : index
    %6 = rock.workitem_id : index
    %7 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %8 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %c0 = arith.constant 0 : index
    %c22 = arith.constant 22 : index
    %c352 = arith.constant 352 : index
    %c6 = arith.constant 6 : index
    %c96 = arith.constant 96 : index
    %9 = arith.divui %5, %c96 : index
    %10 = arith.remui %5, %c96 : index
    %11 = arith.divui %10, %c352 : index
    %12 = arith.muli %11, %c22 : index
    %13 = arith.subi %c6, %12 : index
    %14 = arith.minui %13, %c22 : index
    %15 = arith.remui %10, %14 : index
    %16 = arith.addi %12, %15 : index
    %17 = arith.remui %10, %c352 : index
    %18 = arith.divui %17, %14 : index
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%2) [%c0, %9, %16, %18, %6] -> %7 : memref<16x1x6x16x256x16xf16> -> memref<16xf16, #gpu.address_space<private>>
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%c0, %9, %16, %18, %6] -> %8 : memref<16x1x6x16x256x16xf16> -> memref<16xf16, #gpu.address_space<private>>
    %c0_0 = arith.constant 0 : index
    // LOWERING: %[[multibuf0:.*]] = rock.alloc() : memref<64xi8, #gpu.address_space<private>>
    %raw = rock.alloc(){__multibuffer__=2} : memref<32xi8, #gpu.address_space<private>>
    // LOWERING: %[[multibuf0_view:.*]] = rock.reinterpret_multibuffer %[[multibuf0]] {{.*}} : memref<64xi8, #gpu.address_space<private>> to memref<2x16xf16, #gpu.address_space<private>>
    %19 = memref.view %raw[%c0_0][] : memref<32xi8, #gpu.address_space<private>> to memref<16xf16, #gpu.address_space<private>>
    %20 = rock.alloc() : memref<16xf16, #gpu.address_space<private>>
    %21 = rock.transform %7 by #transform_map5 : memref<16xf16, #gpu.address_space<private>> to memref<2x8xf16, #gpu.address_space<private>>
    %22 = rock.transform %21 by #transform_map6 : memref<2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
    %23 = rock.transform %19 by #transform_map7 : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
    %24 = rock.transform %23 by #transform_map8 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
    %25 = rock.transform %8 by #transform_map9 : memref<16xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
    %26 = rock.transform %25 by #transform_map10 : memref<8x2xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
    %27 = rock.transform %20 by #transform_map11 : memref<16xf16, #gpu.address_space<private>> to memref<1x2x8xf16, #gpu.address_space<private>>
    %28 = rock.transform %27 by #transform_map12 : memref<1x2x8xf16, #gpu.address_space<private>> to memref<8x2xf16, #gpu.address_space<private>>
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    // CHECK: %[[multibuf1:.*]] = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>
    %29 = rock.alloc(){__multibuffer__=2} : memref<8192xi8, #gpu.address_space<workgroup>>
    %30 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
    // CHECK: %[[multibuf_view:.*]] = rock.reinterpret_multibuffer %[[multibuf1]] {{.*}} : memref<16384xi8, #gpu.address_space<workgroup>> to memref<2x512xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t1:.*]] = rock.transform %[[multibuf_view]] by {{.*}} : memref<2x512xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t2:.*]] = rock.transform %[[t1]] by {{.*}} : memref<2x8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t3:.*]] = rock.transform %[[t2]] by {{.*}} : memref<2x8x8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t4:.*]] = rock.transform %[[t3]] by {{.*}} : memref<2x8x512x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t5:.*]] = rock.transform %[[t4]] by {{.*}} : memref<2x8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t6:.*]] = rock.transform %[[t5]] by {{.*}} : memref<2x64x64xvector<8xf16>, #gpu.address_space<workgroup>>
    // CHECK: %[[t7:.*]] = rock.transform %[[t6]] by {{.*}} : memref<2x32x8x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %view = memref.view %29[%c0_0][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<512xvector<8xf16>, #gpu.address_space<workgroup>>
    %31 = rock.transform %view by #transform_map13 : memref<512xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %32 = rock.transform %31 by #transform_map14 : memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %33 = rock.transform %32 by #transform_map15 : memref<8x8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x512x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %34 = rock.transform %33 by #transform_map16 : memref<8x512x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %35 = rock.transform %34 by #transform_map17 : memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<64x64xvector<8xf16>, #gpu.address_space<workgroup>>
    %36 = rock.transform %35 by #transform_map18 : memref<64x64xvector<8xf16>, #gpu.address_space<workgroup>> to memref<32x8x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %37 = rock.transform %36 by #transform_map19 : memref<32x8x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    %c0_1 = arith.constant 0 : index
    %view_2 = memref.view %30[%c0_1][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<512xvector<8xf16>, #gpu.address_space<workgroup>>
    %38 = rock.transform %view_2 by #transform_map20 : memref<512xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %39 = rock.transform %38 by #transform_map21 : memref<8x64x1x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<64x64xvector<8xf16>, #gpu.address_space<workgroup>>
    %40 = rock.transform %39 by #transform_map22 : memref<64x64xvector<8xf16>, #gpu.address_space<workgroup>> to memref<8x32x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>>
    %41 = rock.transform %40 by #transform_map23 : memref<8x32x1x2x8xvector<8xf16>, #gpu.address_space<workgroup>> to memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
    %c0_3 = arith.constant 0 : index
    // CHECK: %[[multibuf_view2:.*]] = rock.reinterpret_multibuffer %[[multibuf1]] {{.*}} : memref<16384xi8, #gpu.address_space<workgroup>> to memref<2x512xvector<8xf16>, #gpu.address_space<workgroup>>
    %view_4 = memref.view %29[%c0_3][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<512xvector<8xf16>, #gpu.address_space<workgroup>>
    %c0_5 = arith.constant 0 : index
    %view_6 = memref.view %30[%c0_5][] : memref<8192xi8, #gpu.address_space<workgroup>> to memref<512xvector<8xf16>, #gpu.address_space<workgroup>>
    %42 = arith.divui %6, %c64 : index
    %43 = arith.divui %42, %c2 : index
    %44 = arith.remui %42, %c2 : index
    %c32 = arith.constant 32 : index
    %c32_7 = arith.constant 32 : index
    %45 = arith.muli %43, %c32 : index
    %46 = arith.muli %44, %c32_7 : index
    %47 = rock.alloc() : memref<8xvector<4xf16>, #gpu.address_space<private>>
    %48 = rock.alloc() : memref<8xvector<4xf16>, #gpu.address_space<private>>
    %49 = rock.alloc() : memref<1xvector<16xf32>, #gpu.address_space<private>>
    %cst = arith.constant dense<0.000000e+00> : vector<16xf32>
    rock.fill(%49, %cst) : memref<1xvector<16xf32>, #gpu.address_space<private>>, vector<16xf32>
    affine.for %arg3 = 0 to 16 {
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%2) [%arg3, %9, %16, %18, %6] -> %7 : memref<16x1x6x16x256x16xf16> -> memref<16xf16, #gpu.address_space<private>>
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%arg3, %9, %16, %18, %6] -> %8 : memref<16x1x6x16x256x16xf16> -> memref<16xf16, #gpu.address_space<private>>
      // CHECK: %[[mbIndex0:.*]] = affine.apply {{.*}}(%arg3)
      // CHECK: rock.threadwise_copy {{.*}} -> {{.*}}[%[[mbIndex0]]]
      // LOWERING: rock.transforming_for
      // LOWERING: strides [1, 8]
      // LOWERING: rock.in_bounds_store {{.*}} -> %[[multibuf0_view]][{{.*}}]
      rock.threadwise_copy %22 -> %24 : memref<8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>
      rock.threadwise_copy %26 -> %28 : memref<8x2xf16, #gpu.address_space<private>> -> memref<8x2xf16, #gpu.address_space<private>>
      // CHECK: %[[mbIndex1:.*]] = affine.apply {{.*}}(%arg3)
      // CHECK: %[[subview0:.*]] = memref.subview {{.*}}[%[[mbIndex1]], 0]
      // CHECK: %[[mbIndex2:.*]] = affine.apply {{.*}}(%arg3)
      // CHECK: rock.threadwise_write_all {{.*}} %[[subview0]] -> [](%[[t7]]) [%[[mbIndex2]], %6] by  set : memref<16xf16, strided<[1], offset: ?>, #gpu.address_space<private>> -> memref<2x256x16xvector<8xf16>, #gpu.address_space<workgroup>>
      rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %19 -> [](%37) [%6] by  set : memref<16xf16, #gpu.address_space<private>> -> memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
      rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %20 -> [](%41) [%6] by  set : memref<16xf16, #gpu.address_space<private>> -> memref<256x16xvector<8xf16>, #gpu.address_space<workgroup>>
      rock.lds_barrier
      // CHECK: %[[multibuf_idx:.*]] = affine.apply {{.*}}(%arg3)
      // CHECK: %[[subview1:.*]] = memref.subview %[[multibuf_view2]][%[[multibuf_idx]], 0] [1, 512] [1, 1]
      rock.blockwise_gemm_accel %49 += %47 from %view_4 * %48 from %view_6 features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a", blockSize = 256 : i32, inMPerThread = 2 : i32, inNPerThread = 2 : i32, params = #xldops_gemm_params, rotateMWithK} : memref<1xvector<16xf32>, #gpu.address_space<private>> += memref<8xvector<4xf16>, #gpu.address_space<private>> from memref<512xvector<8xf16>, #gpu.address_space<workgroup>> * memref<8xvector<4xf16>, #gpu.address_space<private>> from memref<512xvector<8xf16>, #gpu.address_space<workgroup>>
      rock.lds_barrier
    }
    return
  }
}
