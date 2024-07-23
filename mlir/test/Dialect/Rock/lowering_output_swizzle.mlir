// RUN: rocmlir-opt --rock-gemm-output-swizzle %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: func.func @rock_gemm_output_swizzle
func.func @rock_gemm_output_swizzle(%matrix_c: memref<1x1280x2048xf16>) attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  %registers = rock.alloc() : memref<32xf16, #priv>
  %blockid = rock.workgroup_id : index
  %threadid = rock.workitem_id : index

  %c22 = arith.constant 22 : index
  %c320 = arith.constant 320 : index
  %c20 = arith.constant 20 : index
  %c352 = arith.constant 352 : index
  %16 = arith.divui %blockid, %c320 : index
  %17 = arith.remui %blockid, %c320 : index
  %18 = arith.divui %17, %c352 : index
  %19 = arith.muli %18, %c22 : index
  %20 = arith.subi %c20, %19 : index
  %21 = arith.minui %20, %c22 : index
  %22 = arith.remui %17, %21 : index
  %23 = arith.addi %19, %22 : index
  %24 = arith.remui %17, %c352 : index
  %25 = arith.divui %24, %21 : index

  // CHECK: rock.alloc() : memref<24576xi8, #gpu.address_space<workgroup>>
  // CHECK: memref.view
  // CHECK: memref.view
  %28 = rock.alloc() : memref<8192xi8, #gpu.address_space<workgroup>>
  %29 = rock.alloc() : memref<16384xi8, #gpu.address_space<workgroup>>

  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_read_into
  // CHECK: rock.threadwise_write_all
  rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %registers -> [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 64, (d3 mod 64) floordiv 32, d3 mod 32, d4 floordiv 16, 0, (d4 mod 16) floordiv 4, d4 mod 4)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{4, 2, 32} ["tid"] at [3] -> ["wave", "m_tid", "n_tid"] at [3, 4, 5]>, <Merge{2, 1, 4, 4} ["item"] at [4] -> ["i", "j", "vec_group", "vec_item"] at [6, 7, 8, 9]>] bounds = [1, 20, 16, 256, 32] -> [1, 20, 16, 4, 2, 32, 2, 1, 4, 4]>, #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3 floordiv 2, d3 mod 2, d4, d5, 0, d6, 0, 0, d8, d9)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 2} ["wave"] at [3] -> ["wave_m", "wave_n"] at [3, 4]>, <PassThrough ["m_tid", "n_tid"] at [4, 5] -> ["m_tid", "n_tid"] at [5, 6]>, <Merge{1, 2} ["i"] at [6] -> ["m_i", "n_i"] at [7, 8]>, <Merge{1, 1} ["j"] at [7] -> ["blk_row", "blk_col"] at [9, 10]>, <PassThrough ["vec_group", "vec_item"] at [8, 9] -> ["vec_group", "vec_item"] at [11, 12]>] bounds = [1, 20, 16, 4, 2, 32, 2, 1, 4, 4] -> [1, 20, 16, 2, 2, 2, 32, 1, 2, 1, 1, 4, 4]>, #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (d0, ((((d1 + d7) * 2 + d3 + d9) * 4 + d11) * 2 + d5) * 4 + d12, ((d2 * 2 + d8) * 2 + d4 + d10) * 32 + d6)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{20, 1, 2, 1, 4, 2, 4} ["m_block", "m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [1, 7, 3, 9, 11, 5, 12] -> ["gemmM"] at [1]>, <Unmerge{16, 2, 2, 1, 32} ["n_block", "n_i", "wave_n", "blk_col", "n_tid"] at [2, 8, 4, 10, 6] -> ["gemmN"] at [2]>] bounds = [1, 20, 16, 2, 2, 2, 32, 1, 2, 1, 1, 4, 4] -> [1, 1280, 2048]>, #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1280, 2048] -> [1, 1280, 2048]>, #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1280, 2048] -> [1, 1280, 2048]>](%matrix_c) [%16, %23, %25, %threadid] by  set : memref<32xf16, #gpu.address_space<private>> -> memref<1x1280x2048xf16>
  return
}