// RUN: rocmlir-opt --canonicalize --rock-gemm-output-swizzle %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: func.func @rock_gemm_output_swizzle
func.func @rock_gemm_output_swizzle(%matrix_c: memref<1x1280x2048xf16>) attributes{arch = "", block_size = 256 : i32, grid_size = 320 : i32, kernel} {
  %registers = rock.alloc() : memref<32xf16, #priv>
  %registers2 = rock.alloc() : memref<32xf16, #priv>
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

  // CHECK: %[[ALLOC:.*]] = rock.alloc() : memref<32768xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET:.*]] = arith.constant 0 : index
  // CHECK: memref.view %[[ALLOC]][%[[OFFSET]]][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  // CHECK: %[[OFFSET2:.*]] = arith.constant 16384 : index
  // CHECK: memref.view %[[ALLOC]][%[[OFFSET2]]][] : memref<32768xi8, #gpu.address_space<workgroup>> to memref<16384xi8, #gpu.address_space<workgroup>>
  %28 = rock.alloc() : memref<16384xi8, #wg>
  %29 = rock.alloc() : memref<16384xi8, #wg>
  
  %c0 = arith.constant 0 : index
  %view_29 = memref.view %29[%c0][] : memref<16384xi8, #wg> to memref<8192xf16, #wg>
  %view_29_2 = rock.transform %view_29 by <affine_map<(d0, d1, d2) -> ((d1 * 256 + d0) * 8 + d2)> by [<Unmerge{4, 256, 8} ["iter", "tid", "numElements"] at [1, 0, 2] -> ["flattenBlock"] at [0]>] bounds = [256, 4, 8] -> [8192]> : memref<8192xf16, #gpu.address_space<workgroup>> to memref<256x4x8xf16, #gpu.address_space<workgroup>>
  %view_29_3 = rock.transform %view_29_2 by <affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{4, 8} ["iter"] at [1] -> ["iter", "numElements"] at [1, 2]>] bounds = [256, 32] -> [256, 4, 8]> : memref<256x4x8xf16, #gpu.address_space<workgroup>> to memref<256x32xf16, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%view_29_3) [%threadid] -> %registers : memref<256x32xf16, #wg> -> memref<32xf16, #priv>
  
  %view_28 = memref.view %28[%c0][] : memref<16384xi8, #wg> to memref<8192xf16, #wg>
  %view_28_2 = rock.transform %view_28 by <affine_map<(d0, d1, d2) -> ((d1 * 256 + d0) * 8 + d2)> by [<Unmerge{4, 256, 8} ["iter", "tid", "numElements"] at [1, 0, 2] -> ["flattenBlock"] at [0]>] bounds = [256, 4, 8] -> [8192]> : memref<8192xf16, #gpu.address_space<workgroup>> to memref<256x4x8xf16, #gpu.address_space<workgroup>>
  %view_28_3 = rock.transform %view_28_2 by <affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)> by [<PassThrough ["tid"] at [0] -> ["tid"] at [0]>, <Merge{4, 8} ["iter"] at [1] -> ["iter", "numElements"] at [1, 2]>] bounds = [256, 32] -> [256, 4, 8]> : memref<256x4x8xf16, #gpu.address_space<workgroup>> to memref<256x32xf16, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%view_28_3) [%threadid] -> %registers2 : memref<256x32xf16, #wg> -> memref<32xf16, #priv>
  
  // add registers
  %load = rock.in_bounds_load %registers[%c0] : memref<32xf16, #priv>, index -> vector<32xf16>
  %load2 = rock.in_bounds_load %registers2[%c0] : memref<32xf16, #priv>, index -> vector<32xf16>
  %add = arith.addf %load, %load2 : vector<32xf16>
  rock.in_bounds_store %add -> %registers[%c0] : vector<32xf16> -> memref<32xf16, #priv>, index

  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_read_into
  // CHECK: rock.threadwise_write_all
  rock.threadwise_write_all features =  mfma|dot|atomic_add {forceUnroll, useIndexDiffs} %registers -> [#rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 floordiv 64, (d3 mod 64) floordiv 32, d3 mod 32, d4 floordiv 16, 0, (d4 mod 16) floordiv 4, d4 mod 4)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{4, 2, 32} ["tid"] at [3] -> ["wave", "m_tid", "n_tid"] at [3, 4, 5]>, <Merge{2, 1, 4, 4} ["item"] at [4] -> ["i", "j", "vec_group", "vec_item"] at [6, 7, 8, 9]>] bounds = [1, 20, 16, 256, 32] -> [1, 20, 16, 4, 2, 32, 2, 1, 4, 4]>, #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, d3 floordiv 2, d3 mod 2, d4, d5, 0, d6, 0, 0, d8, d9)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{2, 2} ["wave"] at [3] -> ["wave_m", "wave_n"] at [3, 4]>, <PassThrough ["m_tid", "n_tid"] at [4, 5] -> ["m_tid", "n_tid"] at [5, 6]>, <Merge{1, 2} ["i"] at [6] -> ["m_i", "n_i"] at [7, 8]>, <Merge{1, 1} ["j"] at [7] -> ["blk_row", "blk_col"] at [9, 10]>, <PassThrough ["vec_group", "vec_item"] at [8, 9] -> ["vec_group", "vec_item"] at [11, 12]>] bounds = [1, 20, 16, 4, 2, 32, 2, 1, 4, 4] -> [1, 20, 16, 2, 2, 2, 32, 1, 2, 1, 1, 4, 4]>, #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (d0, ((((d1 + d7) * 2 + d3 + d9) * 4 + d11) * 2 + d5) * 4 + d12, ((d2 * 2 + d8) * 2 + d4 + d10) * 32 + d6)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{20, 1, 2, 1, 4, 2, 4} ["m_block", "m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [1, 7, 3, 9, 11, 5, 12] -> ["gemmM"] at [1]>, <Unmerge{16, 2, 2, 1, 32} ["n_block", "n_i", "wave_n", "blk_col", "n_tid"] at [2, 8, 4, 10, 6] -> ["gemmN"] at [2]>] bounds = [1, 20, 16, 2, 2, 2, 32, 1, 2, 1, 1, 4, 4] -> [1, 1280, 2048]>, #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1280, 2048] -> [1, 1280, 2048]>, #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 1280, 2048] -> [1, 1280, 2048]>](%matrix_c) [%16, %23, %25, %threadid] by  set : memref<32xf16, #gpu.address_space<private>> -> memref<1x1280x2048xf16>
  return
}
