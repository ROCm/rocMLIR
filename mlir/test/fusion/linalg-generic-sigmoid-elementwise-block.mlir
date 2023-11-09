// RUN: rocmlir-opt -rock-linalg-align < %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map10 = affine_map<(d0, d1) -> (0, d1)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 32, (d3 mod 32) floordiv 16, (d3 mod 16) floordiv 4, d3 mod 4, d4 floordiv 8, (d4 mod 8) floordiv 4, (d4 mod 4) floordiv 2, d4 mod 2)>
#map19 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10) -> (d0, (((d1 * 2 + d7) * 2 + d3) * 4 + d5) * 2 + d8, (((d2 * 2 + d9) * 2 + d4) * 4 + d6) * 2 + d10)>
#map20 = affine_map<(d0, d1) -> (d0, d1)>
#map21 = affine_map<(d0) -> (0, d0)>
#map22 = affine_map<(d0, d1) -> (d1)>
#transform_map4 = #rock.transform_map<#map2 by [<Unmerge{1, 2} ["col0", "col1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 5] -> [2, 5]>
#transform_map8 = #rock.transform_map<#map5 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 30} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 27} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 32, 32] -> [1, 2, 5]>
#transform_map32 = #rock.transform_map<#map18 by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 2, 4, 4} ["tid"] at [3] -> ["m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"] at [3, 4, 5, 6]>, <Merge{2, 2, 2, 2} ["iter"] at [4] -> ["m_repeat", "m_thread", "n_repeat", "n_thread"] at [7, 8, 9, 10]>] bounds = [1, 1, 1, 64, 16] -> [1, 1, 1, 2, 2, 4, 4, 2, 2, 2, 2]>
#transform_map33 = #rock.transform_map<#map19 by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 2, 2, 4, 2} ["m_block", "m_repeat", "m_cuwaves", "m_cuwave", "m_thread"] at [1, 7, 3, 5, 8] -> ["gemmM"] at [1]>, <Unmerge{1, 2, 2, 4, 2} ["n_block", "n_repeat", "n_cuwaves", "n_cuwave", "n_thread"] at [2, 9, 4, 6, 10] -> ["gemmN"] at [2]>] bounds = [1, 1, 1, 2, 2, 4, 4, 2, 2, 2, 2] -> [1, 32, 32]>
#transform_map34 = #rock.transform_map<#map21 by [<Merge{1, 5} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>] bounds = [5] -> [1, 5]>
#transform_map35 = #rock.transform_map<#map22 by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <PassThrough ["dim0"] at [1] -> ["dim0"] at [0]>] bounds = [1, 5] -> [5]>
#transform_map36 = #rock.transform_map<#map10 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [2, 5] -> [1, 5]>
// Cut down version of a MIGraphX test that triggered a crash, see
// https://github.com/ROCmSoftwarePlatform/rocMLIR/issues/1188
// Arguments have been rearranged to make pattern matching easier.

module {
  // CHECK-LABEL: @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add
  // CHECK-SAME: ([[arg0:%[^:]+]]: memref<2x5xf32>, [[arg1:%[^:]+]]: memref<2x5xf32>
  func.func @mlir_transpose_slice_dot_add_add_tanh_add_sigmoid_sub_mul_mul_add(%arg0: memref<2x5xf32>, %arg1: memref<2x5xf32>, %arg2: memref<1x5xf32>, %arg3: memref<2x5xf32>,  %arg4: memref<2x5xf32>, %arg5: memref<2x5xf32>, %arg6: memref<2x5xf32>, %arg7: memref<15x5xf32>, %arg8: memref<2x5xf32>) attributes {mhal.arch = "gfx1100", rock.block_size = 64 : i32, rock.grid_size = 1 : i32, kernel = "mixr", num_cu = 48 : i64, rock.wave_size = 32 : i32} {
    %cst = arith.constant 1.000000e+00 : f32
    %alloc = memref.alloc() : memref<2x5xf32>
    %4 = rock.transform %alloc by #transform_map4 : memref<2x5xf32> to memref<1x2x5xf32>
    %8 = rock.transform %4 by #transform_map8 : memref<1x2x5xf32> to memref<1x32x32xf32>
    %c0 = arith.constant 0 : index
    // CHECK: [[gemmAlloc:%.+]] = rock.alloc() : memref<16xf32
    %11 = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
    rock.threadwise_write_all features =  dot|atomic_add|atomic_fmax_f32 {forceUnroll, useIndexDiffs} %11 -> [#transform_map32, #transform_map33](%8) [%c0, %c0, %c0, %c0] by  set : memref<16xf32, #gpu.address_space<private>> -> memref<1x32x32xf32>
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<2x5xf32>
    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<2x5xf32>

    // CHECK: [[arg0_alloc:%.+]] = rock.alloc() : memref<16xf32
    // CHECK-NEXT: [[arg0_group:%.+]] = rock.transform [[arg0]]
    // CHECK-SAME: memref<2x5xf32> to memref<1x2x5xf32>
    // CHECK-NEXT: [[arg0_padded:%.+]] = rock.transform [[arg0_group]]
    // CHECK-SAME: memref<1x2x5xf32> to memref<1x32x32xf32>
    // CHECK-NEXT: rock.threadwise_read_into
    // CHECK-SAME: [[arg0_padded]]
    // CHECK-SAME: [[arg0_alloc]]
    // CHECK-NEXT: [[arg1_alloc:%.+]] = rock.alloc() : memref<16xf32
    // CHECK-NEXT: [[arg1_group:%.+]] = rock.transform [[arg1]]
    // CHECK-SAME: memref<2x5xf32> to memref<1x2x5xf32>
    // CHECK-NEXT: [[arg1_padded:%.+]] = rock.transform [[arg1_group]]
    // CHECK-SAME: memref<1x2x5xf32> to memref<1x32x32xf32>
    // CHECK-NEXT: rock.threadwise_read_into
    // CHECK-SAME: [[arg1_padded]]
    // CHECK-SAME: [[arg1_alloc]]
    // CHECK-NEXT: [[cst_alloc0:%.+]] = rock.alloc() : memref<16xf32
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: ins([[arg0_alloc]], [[arg1_alloc]]
    // CHECK-SAME: outs([[cst_alloc0]]

    // Since the original allocation is read twice by the second generic, we
    // have two copies of said generic floating around, as is needed for general
    // correctness.
    // CHECK: rock.threadwise_read_into
    // CHECK: rock.threadwise_read_into
    // CHECK-NEXT: [[cst_alloc1:%.+]] = rock.alloc
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: outs([[cst_alloc1]]
    linalg.generic {indexing_maps = [#map20, #map20, #map20], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x5xf32>, memref<2x5xf32>) outs(%alloc_13 : memref<2x5xf32>) {
    ^bb0(%in: f32, %in_14: f32, %out: f32):
      %54 = arith.addf %in, %in_14 : f32
      %55 = arith.negf %54 : f32
      %56 = math.exp %55 : f32
      %57 = arith.addf %56, %cst : f32
      %58 = arith.divf %cst, %57 : f32
      linalg.yield %58 : f32
    }
    %51 = rock.transform %arg2 by #transform_map34 : memref<1x5xf32> to memref<5xf32>
    %52 = rock.transform %51 by #transform_map35 : memref<5xf32> to memref<1x5xf32>
    %53 = rock.transform %52 by #transform_map36 : memref<1x5xf32> to memref<2x5xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: [[cst_alloc1]]
    // CHECK-SAME: [[cst_alloc0]]
    linalg.generic {indexing_maps = [#map20, #map20, #map20, #map20, #map20, #map20, #map20, #map20], iterator_types = ["parallel", "parallel"]} ins(%arg4, %alloc_13, %arg3, %alloc, %53, %alloc_13, %arg5 : memref<2x5xf32>, memref<2x5xf32>, memref<2x5xf32>, memref<2x5xf32>, memref<2x5xf32>, memref<2x5xf32>, memref<2x5xf32>) outs(%alloc_12 : memref<2x5xf32>) {
    ^bb0(%in: f32, %in_14: f32, %in_15: f32, %in_16: f32, %in_17: f32, %in_18: f32, %in_19: f32, %out: f32):
      %54 = arith.addf %in_16, %in_17 : f32
      %55 = arith.addf %in_15, %54 : f32
      %56 = math.tanh %55 : f32
      %57 = arith.subf %in, %in_14 : f32
      %58 = arith.mulf %57, %56 : f32
      %59 = arith.mulf %in_18, %in_19 : f32
      %60 = arith.addf %58, %59 : f32
      linalg.yield %60 : f32
    }
    memref.copy %alloc_12, %arg8 : memref<2x5xf32> to memref<2x5xf32>
    return
  }
}

