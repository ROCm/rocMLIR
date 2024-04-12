// RUN: rocmlir-driver --rock-linalg-align --rock-pipeline %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4 * 2, d5 + d6 * 2)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, d2 floordiv 28, d1 mod 3, d2 mod 28)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 4 + d5) * 8 + d7, d3 * 16 + d4 + d6)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 4, d4 mod 4, 0, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 128 + d1, d2, d3, d4)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 4 + d5) * 8 + d7, d2 * 16 + d4 + d6)>
#map12 = affine_map<(d0, d1) -> (d1)>
#map13 = affine_map<(d0, d1) -> (d0 + d1)>
#transform_map = #rock.transform_map<#map by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{1} ["exp2"] at [2] -> [] at []>, <PassThrough ["dim0"] at [3] -> ["dim0"] at [0]>] bounds = [1, 1, 1, 3] -> [3]>
#transform_map1 = #rock.transform_map<#map1 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [128, 128, 3, 3] -> [1, 1, 1, 3]>
#transform_map2 = #rock.transform_map<#map3 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 56, 56] -> [1, 128, 56, 56]>
#transform_map3 = #rock.transform_map<#map4 by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [1, 1, 128, 58, 58] -> [1, 1, 128, 56, 56]>
#transform_map4 = #rock.transform_map<#map5 by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 2} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, <Embed{1, 2} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [1, 1, 128, 3, 28, 3, 28] -> [1, 1, 128, 58, 58]>
#transform_map5 = #rock.transform_map<#map6 by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{128, 3, 3} ["gemmK"] at [1] -> ["ci", "y", "x"] at [2, 3, 5]>, <Merge{1, 28, 28} ["gemmN"] at [2] -> ["ni", "ho", "wo"] at [0, 4, 6]>] bounds = [1, 1152, 784] -> [1, 1, 128, 3, 28, 3, 28]>
#transform_map6 = #rock.transform_map<#map7 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{36, 4, 8} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{49, 16, 1} ["n_block", "n_thread", "n_iter"] at [3, 4, 6] -> ["n"] at [2]>, <AddDim{8} ["m_block"] at [2] -> [] at []>] bounds = [36, 1, 8, 49, 16, 4, 1, 8] -> [1, 1152, 784]>
#transform_map7 = #rock.transform_map<#map8 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{16, 4} ["tid"] at [4] -> ["n_thread", "k_thread"] at [4, 5]>, <Merge{1, 8} ["iter"] at [5] -> ["n_iter", "k_iter"] at [6, 7]>] bounds = [36, 1, 8, 49, 64, 8] -> [36, 1, 8, 49, 16, 4, 1, 8]>
#transform_map8 = #rock.transform_map<#map9 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 128} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 128, 128, 3, 3] -> [128, 128, 3, 3]>
#transform_map9 = #rock.transform_map<#map10 by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{128, 3, 3} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 1152, 128] -> [1, 128, 128, 3, 3]>
#transform_map10 = #rock.transform_map<#map11 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{36, 4, 8} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{8, 16, 1} ["m_block", "m_thread", "m_iter"] at [2, 4, 6] -> ["m"] at [2]>, <AddDim{49} ["n_block"] at [3] -> [] at []>] bounds = [36, 1, 8, 49, 16, 4, 1, 8] -> [1, 1152, 128]>
#transform_map11 = #rock.transform_map<#map8 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{16, 4} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{1, 8} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [36, 1, 8, 49, 64, 8] -> [36, 1, 8, 49, 16, 4, 1, 8]>
#transform_map12 = #rock.transform_map<#map12 by [<AddDim{1} ["i"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 2] -> [2]>
#transform_map13 = #rock.transform_map<#map12 by [<AddDim{1} ["j"] at [0] -> [] at []>, <PassThrough ["k"] at [1] -> ["k"] at [0]>] bounds = [1, 2] -> [2]>
#transform_map14 = #rock.transform_map<#map13 by [<Unmerge{1, 1} ["i", "j"] at [0, 1] -> ["offset"] at [0]>] bounds = [1, 1] -> [1]>
module {
  func.func @test(%arg0: memref<3xf16> {func.read_access}, %arg1: memref<1x128x56x56xf16> {func.read_access}, %arg2: memref<128x128x3x3xi8> {func.read_access}, %arg3: memref<1x128x28x28xf16> {func.write_access}) attributes {block_size = 64 : i32, grid_size = 392 : i32, kernel, original_func = @test} {
    %c1 = arith.constant 1 : index
    %c36 = arith.constant 36 : index
    %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x128x3x3xf16>
    %0 = rock.transform %arg0 by #transform_map : memref<3xf16> to memref<1x1x1x3xf16>
    %1 = rock.transform %0 by #transform_map1 : memref<1x1x1x3xf16> to memref<128x128x3x3xf16>
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %1 : memref<128x128x3x3xi8>, memref<128x128x3x3xf16>) outs(%alloc : memref<128x128x3x3xf16>) {
    ^bb0(%in: i8, %in_8: f16, %out: f16):
      %21 = arith.sitofp %in : i8 to f16
      %22 = arith.mulf %21, %in_8 : f16
      linalg.yield %22 : f16
    }
    %2 = rock.transform %arg1 by #transform_map2 : memref<1x128x56x56xf16> to memref<1x1x128x56x56xf16>
    %3 = rock.transform %2 by #transform_map3 : memref<1x1x128x56x56xf16> to memref<1x1x128x58x58xf16>
    %4 = rock.transform %3 by #transform_map4 : memref<1x1x128x58x58xf16> to memref<1x1x128x3x28x3x28xf16>
    %5 = rock.transform %4 by #transform_map5 : memref<1x1x128x3x28x3x28xf16> to memref<1x1152x784xf16>
    %6 = rock.transform %5 by #transform_map6 : memref<1x1152x784xf16> to memref<36x1x8x49x16x4x1x8xf16>
    %7 = rock.transform %6 by #transform_map7 : memref<36x1x8x49x16x4x1x8xf16> to memref<36x1x8x49x64x8xf16>
    %8 = rock.transform %alloc by #transform_map8 : memref<128x128x3x3xf16> to memref<1x128x128x3x3xf16>
    %9 = rock.transform %8 by #transform_map9 : memref<1x128x128x3x3xf16> to memref<1x1152x128xf16>
    %10 = rock.transform %9 by #transform_map10 : memref<1x1152x128xf16> to memref<36x1x8x49x16x4x1x8xf16>
    %11 = rock.transform %10 by #transform_map11 : memref<36x1x8x49x16x4x1x8xf16> to memref<36x1x8x49x64x8xf16>
    %12 = rock.workgroup_id : index
    %13 = rock.workitem_id : index
    %14 = arith.divui %12, %c1 : index
    %alloc_0 = memref.alloc() : memref<8xf16, #gpu.address_space<private>>
    %15 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %alloc_1 = memref.alloc() : memref<8xf16, #gpu.address_space<private>>
    %16 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %alloc_2 = memref.alloc() : memref<8xf16, #gpu.address_space<private>>
    %17 = rock.alloc() : memref<16xi8, #gpu.address_space<private>>
    %alloc_3 = memref.alloc() : memref<8xf16, #gpu.address_space<private>>
    %alloc_4 = memref.alloc() : memref<2xvector<4xf16>, #gpu.address_space<private>>
    %alloc_5 = memref.alloc() : memref<2xvector<4xf16>, #gpu.address_space<private>>
    %alloc_6 = memref.alloc() : memref<1xvector<4xf32>, #gpu.address_space<private>>
    %18 = rock.transform %alloc_4 by #transform_map12 : memref<2xvector<4xf16>, #gpu.address_space<private>> to memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %19 = rock.transform %alloc_5 by #transform_map13 : memref<2xvector<4xf16>, #gpu.address_space<private>> to memref<1x2xvector<4xf16>, #gpu.address_space<private>>
    %20 = rock.transform %alloc_6 by #transform_map14 : memref<1xvector<4xf32>, #gpu.address_space<private>> to memref<1x1xvector<4xf32>, #gpu.address_space<private>>
    %alloc_7 = memref.alloc() : memref<64x8xvector<8xf16>, #gpu.address_space<workgroup>>
    affine.for %arg4 = 0 to 1 {
      memref.store %cst, %alloc_6[%arg4] : memref<1xvector<4xf32>, #gpu.address_space<private>>
    }
    scf.for %arg4 = %c0 to %c36 step %c1 {
      rock.stage {
        rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%11) [%arg4, %14, %12, %13, %13] -> %alloc_0 : memref<36x1x8x49x64x8xf16> -> memref<8xf16, #gpu.address_space<private>>
        rock.yield
      } {name = "GlobalRead"}
      rock.stage {
        rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %alloc_2 -> [](%alloc_7) [%13] by  set : memref<8xf16, #gpu.address_space<private>> -> memref<64x8xvector<8xf16>, #gpu.address_space<workgroup>>
        rock.yield
      } {name = "LDSWrite"}
      rock.stage {
        %21 = rock.workitem_id : index
        rock.threadwise_accel_gemm %20 += %18 * %19 at[%21, %21, %21] features =  mfma|dot|atomic_add {arch = "gfx90a", params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 16, kpack = 8, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>} : memref<1x1xvector<4xf32>, #gpu.address_space<private>> += memref<1x2xvector<4xf16>, #gpu.address_space<private>> * memref<1x2xvector<4xf16>, #gpu.address_space<private>>
        rock.yield
      } {name = "MMA"}
    } {pipeline = #rock.pipeline<2>}
    return
  }
}


// To test the input argument loads in the prolog and software-pipelined kernel would use the same buffer
// CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%{{.*}}) [{{.*}}] -> %[[BUFFER_0:.+]] : memref<36x1x8x49x64x8xi8> -> memref<8xi8, #gpu.address_space<private>>
// CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%{{.+}}) [{{.*}}] -> %[[BUFFER_1:.+]] : memref<36x1x8x49x64x8xf16> -> memref<8xf16, #gpu.address_space<private>>
// CHECK: linalg.generic {{{.*}}} ins(%[[BUFFER_0]], %[[BUFFER_1]]
// CHECK: scf.for
// CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%{{.*}}) [{{.*}}] -> %[[BUFFER_0:.+]] : memref<36x1x8x49x64x8xi8> -> memref<8xi8, #gpu.address_space<private>>
// CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%{{.+}}) [{{.*}}] -> %[[BUFFER_1:.+]] : memref<36x1x8x49x64x8xf16> -> memref<8xf16, #gpu.address_space<private>>
// CHECK: linalg.generic {{{.*}}} ins(%[[BUFFER_0]], %[[BUFFER_1]]
