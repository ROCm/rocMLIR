// RUN: rocmlir-gen -ph -print-results -rand none - < %s | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK-COUNT-160: 20

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 160} ["nr"] at [0] -> ["x", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["y"] at [1]>] bounds = [160, 20] -> [1, 20, 160]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 20 + d1, d2)> by [<Unmerge{8, 20} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [8, 20, 20] -> [160, 20]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [8, 20, 20] -> [8, 20, 20]>

#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 160} ["nr"] at [0] -> ["x", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["y"] at [1]>] bounds = [160, 1] -> [1, 1, 160]>
#transform_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 20 + d2, d1)> by [<Unmerge{8, 20} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [8, 1, 20] -> [160, 1]>

#transform_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [20, 20] -> [20, 20]>

func.func @rock_blockwise_reducesum_nr_threads_gt_blocksize(%input : memref<1x20x160xf32>,  %output : memref<1x1x160xf32>) attributes{arch = "", block_size = 20 : i32, grid_size = 8 : i32, kernel} {
  %input_reg = rock.alloc() : memref<20xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<20xf32, #gpu.address_space<private>>
  %ws_lds = rock.alloc() : memref<400xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map2, #transform_map1, #transform_map0](%input)[%bid, %tid] -> %input_reg : memref<1x20x160xf32> ->  memref<20xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce sum [#transform_map5]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 20 : i32} : memref<20xf32, #gpu.address_space<private>> using memref<400xf32, #gpu.address_space<workgroup>> into memref<20xf32, #gpu.address_space<private>>
  rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %output_reg -> [#transform_map4, #transform_map3](%output)[%bid, %tid] by set : memref<20xf32, #gpu.address_space<private>> -> memref<1x1x160xf32>
  return
}
