// RUN: cat %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK-COUNT-300: 3

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0) -> (d0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 2} ["tid", "iter"] at [0, 1] -> ["dim0"] at [0]>] bounds = [256, 2] -> [512]>
#transform_map1 = #rock.transform_map<#map1 by [<Pad{0, 212} ["dim0"] at [0] -> ["dim0"] at [0]>] bounds = [512] -> [300]>

#map4 = affine_map<(d0, d1, d2) -> (d0, d1 * 2 + d2)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{256, 2} ["tid", "iter"] at [1, 2] -> ["dim0"] at [1]>, <PassThrough ["bid"] at [0] -> ["bid"] at [0]>] bounds = [1, 256, 2] -> [1, 512]>
#transform_map5 = #rock.transform_map<#map5 by [<Pad{0, 212} ["tid"] at [1] -> ["tid"] at [1]>, <PassThrough ["bid"] at [0] -> ["bid"] at [0]>] bounds = [1, 512] -> [1, 300]>

func.func @rock_blockwise_fill_scalar_case1(%output : memref<1x300xf32>) attributes{arch = "", block_size = 256 : i32, grid_size = 1 : i32, kernel} {
    %output_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    %ldsbuf = rock.alloc() : memref<300xf32, #gpu.address_space<workgroup>>
    %c1 = arith.constant 3.0 : f32
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<300xf32, #gpu.address_space<workgroup>>, f32
    rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map, #transform_map1](%ldsbuf) -> %output_reg : memref<300xf32, #gpu.address_space<workgroup>> ->  memref<1xf32, #gpu.address_space<private>>
    rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %output_reg -> [#transform_map4, #transform_map5](%output) by set : memref<1xf32, #gpu.address_space<private>> -> memref<1x300xf32>
    return
}
