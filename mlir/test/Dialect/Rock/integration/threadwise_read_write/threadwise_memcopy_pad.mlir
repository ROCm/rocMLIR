// RUN: cat %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -gO 2 -arch %arch -c  | rocmlir-opt --canonicalize| mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK-COUNT-493: 2

#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1 - 2, d2 - 1)> by [<Pad{2, 1, 1, 2} ["y", "z"] at [1, 2] -> ["y", "z"] at [1, 2]>, <PassThrough ["x"] at [0] -> ["x"] at [0]>] bounds = [1, 20, 32] -> [1, 17, 29]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["x", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["y"] at [1]>] bounds = [32, 20] -> [1, 20, 32]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d1, d2)> by [<Unmerge{8, 4} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [8, 4, 20] -> [32, 20]>
#transform_map3 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [8, 20, 4] -> [8, 4, 20]>

#transform_map4 = #rock.transform_map<affine_map<(d0, d1) -> (d0, d1 - 2)> by [<Pad{2, 1} ["iter"] at [1] -> ["iter"] at [1]>, <PassThrough ["tid"] at [0] -> ["tid"] at [0]>] bounds = [4, 20] -> [4, 17]>
#transform_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [20, 4] -> [4, 20]>

func.func @rock_threadwise_memcopy_pad(%input : memref<1x17x29xf32>,  %output : memref<1x17x29xf32>) attributes{arch = "", block_size = 20 : i32, grid_size = 8 : i32, kernel} {
  %input_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %ws_lds = rock.alloc() : memref<4x17xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map3, #transform_map2, #transform_map1, #transform_map0](%input)[%bid, %tid] -> %input_reg : memref<1x17x29xf32> ->  memref<4xf32, #gpu.address_space<private>>

  // This is to do a compute see if read_into s and write_all s are functioning
  %c0 = arith.constant 0 : index
  %load = rock.in_bounds_load %input_reg[%c0] : memref<4xf32, #gpu.address_space<private>>, index -> vector<4xf32>
  %add = arith.addf %load, %load : vector<4xf32>
  rock.in_bounds_store %add -> %input_reg[%c0] : vector<4xf32> -> memref<4xf32, #gpu.address_space<private>>, index

  rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %input_reg -> [#transform_map5, #transform_map4](%ws_lds)[%tid] by set : memref<4xf32, #gpu.address_space<private>> -> memref<4x17xf32, #gpu.address_space<workgroup>>
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map5, #transform_map4](%ws_lds)[%tid] -> %output_reg : memref<4x17xf32, #gpu.address_space<workgroup>> ->  memref<4xf32, #gpu.address_space<private>>
  rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %output_reg -> [#transform_map3, #transform_map2, #transform_map1, #transform_map0](%output)[%bid, %tid] by set : memref<4xf32, #gpu.address_space<private>> -> memref<1x17x29xf32>
  return
}
