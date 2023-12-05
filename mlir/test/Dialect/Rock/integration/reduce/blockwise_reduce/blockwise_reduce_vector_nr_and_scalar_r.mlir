
// RUN: rocmlir-gen -ph -print-results -rand none - < %s | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK-COUNT-24576: 32

#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, d3 floordiv 32, d3 mod 32, 0, 0, d4 floordiv 4, d4 mod 4)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{1, 2, 32} ["tid"] at [3] -> ["wave", "m_tid", "n_tid"] at [3, 4, 5]>, <Merge{1, 1, 4, 4} ["item"] at [4] -> ["i", "j", "vec_group", "vec_item"] at [6, 7, 8, 9]>] bounds = [1, 2, 12, 64, 16] -> [1, 2, 12, 1, 2, 32, 1, 1, 4, 4]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> (d0, d1, d2, 0, 0, d4, d5, 0, 0, 0, 0, d8, d9)> by [<PassThrough ["g_block", "m_block", "n_block"] at [0, 1, 2] -> ["g_block", "m_block", "n_block"] at [0, 1, 2]>, <Merge{1, 1} ["wave"] at [3] -> ["wave_m", "wave_n"] at [3, 4]>, <PassThrough ["m_tid", "n_tid"] at [4, 5] -> ["m_tid", "n_tid"] at [5, 6]>, <Merge{1, 1} ["i"] at [6] -> ["m_i", "n_i"] at [7, 8]>, <Merge{1, 1} ["j"] at [7] -> ["blk_row", "blk_col"] at [9, 10]>, <PassThrough ["vec_group", "vec_item"] at [8, 9] -> ["vec_group", "vec_item"] at [11, 12]>] bounds = [1, 2, 12, 1, 2, 32, 1, 1, 4, 4] -> [1, 2, 12, 1, 1, 2, 32, 1, 1, 1, 1, 4, 4]>
#transform_map3 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12) -> (d0, (((d1 + d7 + d3 + d9) * 4 + d11) * 2 + d5) * 4 + d12, (d2 + d8 + d4 + d10) * 32 + d6)> by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{1, 1, 1, 1, 4, 2, 4} ["m_block", "m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [1, 7, 3, 9, 11, 5, 12] -> ["gemmM"] at [1]>, <Unmerge{1, 1, 1, 1, 32} ["n_block", "n_i", "wave_n", "blk_col", "n_tid"] at [2, 8, 4, 10, 6] -> ["gemmN"] at [2]>] bounds = [1, 2, 12, 1, 1, 2, 32, 1, 1, 1, 1, 4, 4] -> [1, 32, 32]>

#map12 = affine_map<(d0, d1) -> (0, d0 floordiv 32, d0 mod 32, 0, 0, d1 floordiv 4, d1 mod 4)>
#map13 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (0, 0, d1, d2, 0, 0, 0, 0, d5, d6)>
#map14 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9) -> ((((d4 + d0 + d6) * 4 + d8) * 2 + d2) * 4 + d9, (d5 + d1 + d7) * 32 + d3)>
#map12_tid = affine_map<(d0) -> (0, d0 floordiv 32, d0 mod 32)>
#map13_tid = affine_map<(d0, d1, d2) -> (0, 0, d1, d2)>
#map14_tid = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map12_iter = affine_map<(d0) -> (0, 0, d0 floordiv 4, d0 mod 4)>
#map13_iter = affine_map<(d0, d1, d2, d3) -> (0, 0, 0, 0, d2, d3)>
#map14_iter = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4 * 4 + d5, 0)>

#transform_map21 = #rock.transform_map<#map12 by [<Merge{1, 2, 32} ["tid"] at [0] -> ["wave", "m_tid", "n_tid"] at [0, 1, 2]>, <Merge{1, 1, 4, 4} ["item"] at [1] -> ["i", "j", "vec_group", "vec_item"] at [3, 4, 5, 6]>] bounds = [64, 16] -> [1, 2, 32, 1, 1, 4, 4]>
#transform_map22 = #rock.transform_map<#map13 by [<Merge{1, 1} ["wave"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["m_tid", "n_tid"] at [1, 2] -> ["m_tid", "n_tid"] at [2, 3]>, <Merge{1, 1} ["i"] at [3] -> ["m_i", "n_i"] at [4, 5]>, <Merge{1, 1} ["j"] at [4] -> ["blk_row", "blk_col"] at [6, 7]>, <PassThrough ["vec_group", "vec_item"] at [5, 6] -> ["vec_group", "vec_item"] at [8, 9]>] bounds = [1, 2, 32, 1, 1, 4, 4] -> [1, 1, 2, 32, 1, 1, 1, 1, 4, 4]>
#transform_map23 = #rock.transform_map<#map14 by [<Unmerge{1, 1, 1, 4, 2, 4} ["m_i", "wave_m", "blk_row", "vec_group", "m_tid", "vec_item"] at [4, 0, 6, 8, 2, 9] -> ["gemmM"] at [0]>, <Unmerge{1, 1, 1, 32} ["n_i", "wave_n", "blk_col", "n_tid"] at [5, 1, 7, 3] -> ["gemmN"] at [1]>] bounds = [1, 1, 2, 32, 1, 1, 1, 1, 4, 4] -> [32, 32]>

#transform_map21_tid = #rock.transform_map<#map12_tid by [<Merge{1, 2, 32} ["tid"] at [0] -> ["wave", "m_tid", "n_tid"] at [0, 1, 2]>] bounds = [64] -> [1, 2, 32]>
#transform_map22_tid = #rock.transform_map<#map13_tid by [<Merge{1, 1} ["wave"] at [0] -> ["wave_m", "wave_n"] at [0, 1]>, <PassThrough ["m_tid", "n_tid"] at [1, 2] -> ["m_tid", "n_tid"] at [2, 3]>] bounds = [1, 2, 32] -> [1, 1, 2, 32]>
#transform_map23_tid = #rock.transform_map<#map14_tid by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmM"] at [0]>, <Unmerge{1, 32} ["wave_n","n_tid"] at [1, 3] -> ["gemmN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>

#transform_map21_iter = #rock.transform_map<#map12_iter by [<Merge{1, 1, 4, 4} ["item"] at [0] -> ["i", "j", "vec_group", "vec_item"] at [0, 1, 2, 3]>] bounds = [16] -> [1, 1, 4, 4]>
#transform_map22_iter = #rock.transform_map<#map13_iter by [<Merge{1, 1} ["i"] at [0] -> ["m_i", "n_i"] at [0, 1]>, <Merge{1, 1} ["j"] at [1] -> ["blk_row", "blk_col"] at [2, 3]>, <PassThrough ["vec_group", "vec_item"] at [2, 3] -> ["vec_group", "vec_item"] at [4, 5]>] bounds = [1, 1, 4, 4] -> [1, 1, 1, 1, 4, 4]>
#transform_map23_iter = #rock.transform_map<#map14_iter by [<Unmerge{1, 1, 4, 4} ["m_i", "blk_row", "vec_group", "vec_item"] at [0, 2, 4, 5] -> ["gemmM"] at [0]>, <Unmerge{1, 1} ["n_i", "blk_col"] at [1, 3] -> ["gemmN"] at [1]>] bounds = [1, 1, 1, 1, 4, 4] -> [16, 1]>

func.func @rock_blockwise_vector_nr_and_scalar_r(%input : memref<1x64x384xf32>,  %output : memref<1x64x384xf32>) attributes{arch = "", block_size = 64 : i32, grid_size = 24 : i32, kernel} {
  %input_reg = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<16xf32, #gpu.address_space<private>>
  %ws_lds = rock.alloc() : memref<1024xf32, #gpu.address_space<workgroup>>

  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %bid = rock.workgroup_id : index
  %m_block_id = arith.remui %bid, %c2 : index
  %n_block_id = arith.divui %bid, %c2 : index

  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map1, #transform_map2, #transform_map3](%input)[%c0, %m_block_id, %n_block_id, %tid] -> %input_reg : memref<1x64x384xf32> ->  memref<16xf32, #gpu.address_space<private>>

  rock.blockwise_broadcast_reduce  sum [#transform_map21, #transform_map22, #transform_map23] %input_reg into %output_reg using [#transform_map21_tid, #transform_map22_tid, #transform_map23_tid][#transform_map21_iter, #transform_map22_iter, #transform_map23_iter] %ws_lds {axis = 0 : index, blockSize = 64 : i32} : memref<16xf32, #gpu.address_space<private>> using memref<1024xf32, #gpu.address_space<workgroup>> into memref<16xf32, #gpu.address_space<private>>

  rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %output_reg -> [#transform_map1, #transform_map2, #transform_map3](%output)[%c0, %m_block_id, %n_block_id, %tid] by set : memref<16xf32, #gpu.address_space<private>> -> memref<1x64x384xf32>
  return
}
