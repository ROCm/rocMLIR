// RUN: rocmlir-opt -rock-linalg-align < %s | FileCheck %s
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 64, (d3 mod 64) floordiv 32, (d3 mod 32) floordiv 16, d3 mod 16, d4 floordiv 32, (d4 mod 32) floordiv 8, d4 mod 8)>
#map19 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, (((d1 * 4 + d7) * 2 + d3) * 8 + d9) * 2 + d5, ((d2 * 4 + d8) * 2 + d4) * 16 + d6)>
#map20 = affine_map<() -> ()>
#map21 = affine_map<(d0) -> (0, 0, d0)>
#map22 = affine_map<(d0, d1, d2) -> (d2)>
#map23 = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#map24 = affine_map<(d0, d1, d2) -> ()>
#map25 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map26 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#transform_map22 = #rock.transform_map<#map18 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 2, 2, 16} ["tid"] at [3] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [3, 4, 5, 6]>, <Merge{4, 4, 8} ["item"] at [4] -> ["rep_i", "rep_j", "item_i"] at [7, 8, 9]>] bounds = [32, 3, 24, 128, 128] -> [32, 3, 24, 2, 2, 2, 16, 4, 4, 8]>
#transform_map23 = #rock.transform_map<#map19 by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{3, 4, 2, 8, 2} ["m_block", "rep_i", "wave_m", "item_i", "m_tid"] at [1, 7, 3, 9, 5] -> ["gemmM"] at [1]>, <Unmerge{24, 4, 2, 16} ["n_block", "rep_j", "wave_n", "n_tid"] at [2, 8, 4, 6] -> ["gemmN"] at [2]>] bounds = [32, 3, 24, 2, 2, 2, 16, 4, 4, 8] -> [32, 384, 3072]>
#transform_map24 = #rock.transform_map<#map21 by [<Merge{1, 1, 3072} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [3072] -> [1, 1, 3072]>
#transform_map25 = #rock.transform_map<#map22 by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <PassThrough ["dim0"] at [2] -> ["dim0"] at [0]>] bounds = [1, 1, 3072] -> [3072]>
#transform_map26 = #rock.transform_map<#map23 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [32, 384, 3072] -> [1, 1, 3072]>
#transform_map27 = #rock.transform_map<#map24 by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>, <AddDim{1} ["exp2"] at [2] -> [] at []>] bounds = [1, 1, 1] -> []>
#transform_map28 = #rock.transform_map<#map25 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [32, 384, 3072] -> [1, 1, 1]>
// A cut down version of the input from
// https://github.com/ROCm/rocMLIR-internal/issues/1098
// right before it headed down to linalg.generic. The actuall gemm part has been
// removed for test simplicity.
module {
  func.func @mlir_simplified_issue_(%arg0: memref<1x1x3072xf32>, %arg1: memref<32x384x768xi8>, %arg2: memref<32x768x3072xi8>, %arg3: memref<32x384x3072xi8>) attributes {arch = "gfx1100", block_size = 128 : i32, grid_size = 2304 : i32, kernel = "mixr", num_cu = 48 : i64, wave_size = 32 : i32} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 34.907238 : f32
    %cst_4 = arith.constant 1.270000e+02 : f32
    %cst_5 = arith.constant -1.280000e+02 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<32x384x3072xi32>
    // CHECK: [[gemmTile:%.+]] = rock.alloc() : memref<128xi32
    %47 = rock.alloc() : memref<128xi32, #gpu.address_space<private>>
    rock.threadwise_write_all features =  dot|atomic_add|atomic_fmax_f32|wmma {forceUnroll, useIndexDiffs} %47 -> [#transform_map22, #transform_map23](%alloc) [%c0, %c0, %c0, %c0] by  set : memref<128xi32, #gpu.address_space<private>> -> memref<32x384x3072xi32>
    // CHECK: [[cstTile:%.+]] = rock.alloc() : memref<128xf32, #gpu.address_space<private>>
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: outs([[cstTile]]
    %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<f32>
    linalg.generic {indexing_maps = [#map20], iterator_types = []} outs(%alloc_18 : memref<f32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    }
    %48 = rock.transform %arg0 by #transform_map24 : memref<1x1x3072xf32> to memref<3072xf32>
    %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<32x384x3072xi8>
    %49 = rock.transform %48 by #transform_map25 : memref<3072xf32> to memref<1x1x3072xf32>
    %50 = rock.transform %49 by #transform_map26 : memref<1x1x3072xf32> to memref<32x384x3072xf32>
    %51 = rock.transform %alloc_18 by #transform_map27 : memref<f32> to memref<1x1x1xf32>
    %52 = rock.transform %51 by #transform_map28 : memref<1x1x1xf32> to memref<32x384x3072xf32>
    // CHECK: [[globalTile:%.+]] = rock.alloc() : memref<128xf32
    // CHECK-NEXT: threadwise_read_into
    // CHECK-SAME -> [[globalTile]]
    // CHECK: linalg.generic
    // CHECK-SAME: ins([[gemmTile]], [[globalTile]], [[cstTile]]
    linalg.generic {indexing_maps = [#map26, #map26, #map26, #map26], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %50, %52 : memref<32x384x3072xi32>, memref<32x384x3072xf32>, memref<32x384x3072xf32>) outs(%alloc_19 : memref<32x384x3072xi8>) {
    ^bb0(%in: i32, %in_20: f32, %in_21: f32, %out: i8):
      %53 = arith.sitofp %in : i32 to f32
      %55 = arith.addf %53, %in_20 : f32
      %61 = arith.mulf %55, %in_21 : f32
      %62 = math.roundeven %61 : f32
      %63 = arith.minf %62, %cst_4 : f32
      %64 = arith.maxf %63, %cst_5 : f32
      %65 = arith.fptosi %64 : f32 to i8
      linalg.yield %65 : i8
    }
    memref.copy %alloc_19, %arg3 : memref<32x384x3072xi8> to memref<32x384x3072xi8>
    return
  }
}

