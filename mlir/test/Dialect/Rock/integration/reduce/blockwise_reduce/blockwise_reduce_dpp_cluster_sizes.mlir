// RUN: rocmlir-gen -ph -print-results -rand none -fut test_dpp_cluster4 - < %s | sed s/##TOKEN_ARCH##/%arch/g | rocmlir-driver -arch %arch -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_C4
// RUN: rocmlir-gen -ph -print-results -rand none -fut test_dpp_cluster8 - < %s | sed s/##TOKEN_ARCH##/%arch/g | rocmlir-driver -arch %arch -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_C8
// RUN: rocmlir-gen -ph -print-results -rand none -fut test_dpp_cluster16 - < %s | sed s/##TOKEN_ARCH##/%arch/g | rocmlir-driver -arch %arch -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_C16
// RUN: rocmlir-gen -ph -print-results -rand fixed -fut test_dpp_cluster32 - < %s | sed s/##TOKEN_ARCH##/%arch/g | rocmlir-driver -arch %arch -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_C32
// RUN: rocmlir-gen -ph -print-results -rand fixed -fut test_dpp_cluster64 - < %s | sed s/##TOKEN_ARCH##/%arch/g | rocmlir-driver -arch %arch -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_C64
// CHECK_C4-COUNT-32: 4
// CHECK_C8-COUNT-32: 8
// CHECK_C16-COUNT-32: 16
// CHECK_C32: 0.75
// CHECK_C64: 0.75

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 4] -> [1, 4, 32]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d1, d2)> by [<Unmerge{8, 4} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [8, 4, 4] -> [32, 4]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [8, 16, 4] -> [8, 4, 16]>
#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 1] -> [1, 1, 32]>
#transform_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d2, d1)> by [<Unmerge{8, 4} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [8, 1, 4] -> [32, 1]>
#transform_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [16, 4] -> [4, 16]>
#transform_map5_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{4, 4} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [16] -> [4, 4]>
#transform_map5_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{4, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [4] -> [4, 1]>

func.func @test_dpp_cluster4(%input : memref<1x4x32xf32>, %output : memref<1x1x32xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 16 : i32, grid_size = 8 : i32, kernel} {
  %input_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %ws_lds_bytes = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %ws_lds = memref.view %ws_lds_bytes[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map2, #transform_map1, #transform_map0](%input)[%bid, %tid] -> %input_reg : memref<1x4x32xf32> -> memref<4xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce sum [#transform_map5][#transform_map5_tid][#transform_map5_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 16 : i32} : memref<4xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<4xf32, #gpu.address_space<private>>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %output_reg -> [#transform_map4, #transform_map3](%output)[%bid, %tid] by set : memref<4xf32, #gpu.address_space<private>> -> memref<1x1x32xf32>
  return
}

#c8_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 8] -> [1, 8, 32]>
#c8_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d1, d2)> by [<Unmerge{8, 4} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [8, 4, 8] -> [32, 8]>
#c8_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [8, 32, 4] -> [8, 4, 32]>
#c8_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 1] -> [1, 1, 32]>
#c8_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d2, d1)> by [<Unmerge{8, 4} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [8, 1, 4] -> [32, 1]>
#c8_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [32, 4] -> [4, 32]>
#c8_map5_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{4, 8} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [32] -> [4, 8]>
#c8_map5_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{4, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [4] -> [4, 1]>

func.func @test_dpp_cluster8(%input : memref<1x8x32xf32>, %output : memref<1x1x32xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 32 : i32, grid_size = 8 : i32, kernel} {
  %input_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  %ws_lds_bytes = rock.alloc() : memref<512xi8, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %ws_lds = memref.view %ws_lds_bytes[%c0][] : memref<512xi8, #gpu.address_space<workgroup>> to memref<128xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#c8_map2, #c8_map1, #c8_map0](%input)[%bid, %tid] -> %input_reg : memref<1x8x32xf32> -> memref<4xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce sum [#c8_map5][#c8_map5_tid][#c8_map5_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 32 : i32} : memref<4xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<4xf32, #gpu.address_space<private>>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %output_reg -> [#c8_map4, #c8_map3](%output)[%bid, %tid] by set : memref<4xf32, #gpu.address_space<private>> -> memref<1x1x32xf32>
  return
}

#c16_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 16] -> [1, 16, 32]>
#c16_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{16, 2} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [16, 2, 16] -> [32, 16]>
#c16_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [16, 32, 2] -> [16, 2, 32]>
#c16_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 1] -> [1, 1, 32]>
#c16_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 2 + d2, d1)> by [<Unmerge{16, 2} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [16, 1, 2] -> [32, 1]>
#c16_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [32, 2] -> [2, 32]>
#c16_map5_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{2, 16} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [32] -> [2, 16]>
#c16_map5_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{2, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [2] -> [2, 1]>

func.func @test_dpp_cluster16(%input : memref<1x16x32xf32>, %output : memref<1x1x32xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 32 : i32, grid_size = 16 : i32, kernel} {
  %input_reg = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
  %ws_lds_bytes = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %ws_lds = memref.view %ws_lds_bytes[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#c16_map2, #c16_map1, #c16_map0](%input)[%bid, %tid] -> %input_reg : memref<1x16x32xf32> -> memref<2xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce sum [#c16_map5][#c16_map5_tid][#c16_map5_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 32 : i32} : memref<2xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<2xf32, #gpu.address_space<private>>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %output_reg -> [#c16_map4, #c16_map3](%output)[%bid, %tid] by set : memref<2xf32, #gpu.address_space<private>> -> memref<1x1x32xf32>
  return
}

#c32_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 1} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [1, 32] -> [1, 32, 1]>
#c32_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 1 + d1, d2)> by [<Unmerge{1, 1} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [1, 1, 32] -> [1, 32]>
#c32_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [1, 32, 1] -> [1, 1, 32]>
#c32_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 1} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [1, 1] -> [1, 1, 1]>
#c32_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 1 + d2, d1)> by [<Unmerge{1, 1} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [1, 1, 1] -> [1, 1]>
#c32_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [32, 1] -> [1, 32]>
#c32_map5_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 32} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [32] -> [1, 32]>
#c32_map5_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{1, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [1] -> [1, 1]>

func.func @test_dpp_cluster32(%input : memref<1x32x1xf32>, %output : memref<1x1x1xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 32 : i32, grid_size = 1 : i32, kernel} {
  %input_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
  %ws_lds_bytes = rock.alloc() : memref<128xi8, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %ws_lds = memref.view %ws_lds_bytes[%c0][] : memref<128xi8, #gpu.address_space<workgroup>> to memref<32xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#c32_map2, #c32_map1, #c32_map0](%input)[%bid, %tid] -> %input_reg : memref<1x32x1xf32> -> memref<1xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce max [#c32_map5][#c32_map5_tid][#c32_map5_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 32 : i32} : memref<1xf32, #gpu.address_space<private>> using memref<32xf32, #gpu.address_space<workgroup>> into memref<1xf32, #gpu.address_space<private>>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %output_reg -> [#c32_map4, #c32_map3](%output)[%bid, %tid] by set : memref<1xf32, #gpu.address_space<private>> -> memref<1x1x1xf32>
  return
}

// cluster_size=64: on RDNA (waveSize=32) this falls back to tree reduction
#c64_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 1} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [1, 64] -> [1, 64, 1]>
#c64_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 1 + d1, d2)> by [<Unmerge{1, 1} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [1, 1, 64] -> [1, 64]>
#c64_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [1, 64, 1] -> [1, 1, 64]>
#c64_map3 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 1} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [1, 1] -> [1, 1, 1]>
#c64_map4 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 1 + d2, d1)> by [<Unmerge{1, 1} ["bid", "iter"] at [0, 2] -> ["nr"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [1]>] bounds = [1, 1, 1] -> [1, 1]>
#c64_map5 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [64, 1] -> [1, 64]>
#c64_map5_tid = #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 64} ["tid"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [64] -> [1, 64]>
#c64_map5_iter = #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<Merge{1, 1} ["iter"] at [0] -> ["nr_per_bid", "r"] at [0, 1]>] bounds = [1] -> [1, 1]>

func.func @test_dpp_cluster64(%input : memref<1x64x1xf32>, %output : memref<1x1x1xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 64 : i32, grid_size = 1 : i32, kernel} {
  %input_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
  %output_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
  %ws_lds_bytes = rock.alloc() : memref<256xi8, #gpu.address_space<workgroup>>
  %c0 = arith.constant 0 : index
  %ws_lds = memref.view %ws_lds_bytes[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
  %bid = rock.workgroup_id : index
  %tid = rock.workitem_id : index
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#c64_map2, #c64_map1, #c64_map0](%input)[%bid, %tid] -> %input_reg : memref<1x64x1xf32> -> memref<1xf32, #gpu.address_space<private>>
  rock.blockwise_broadcast_reduce max [#c64_map5][#c64_map5_tid][#c64_map5_iter]%input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<1xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<1xf32, #gpu.address_space<private>>
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %output_reg -> [#c64_map4, #c64_map3](%output)[%bid, %tid] by set : memref<1xf32, #gpu.address_space<private>> -> memref<1x1x1xf32>
  return
}
