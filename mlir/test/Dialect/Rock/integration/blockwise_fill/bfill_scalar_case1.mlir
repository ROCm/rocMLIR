// RUN: cat %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK-COUNT-256: 1

#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0) -> (d0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 1} ["tid", "iter"] at [0, 1] -> ["dim0"] at [0]>] bounds = [256, 1] -> [256]>
#transform_map1 = #rock.transform_map<#map1 by [<Pad{0, 0} ["dim0"] at [0] -> ["dim0"] at [0]>] bounds = [256] -> [256]>

func.func @rock_blockwise_fill_scalar_case1(%output : memref<1x256x1xf32>) attributes{arch = "", block_size = 256 : i32, grid_size = 1 : i32, kernel} {
    %output_reg = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    %ldsbuf = rock.alloc() : memref<256xf32, #gpu.address_space<workgroup>>
    %c1 = arith.constant 1.0 : f32
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<256xf32, #gpu.address_space<workgroup>>, f32
    rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map, #transform_map1](%ldsbuf) -> %output_reg : memref<256xf32, #gpu.address_space<workgroup>> ->  memref<1xf32, #gpu.address_space<private>>
    rock.threadwise_write_all features = none {forceUnroll, useIndexDiffs} %output_reg -> [](%output) by set : memref<1xf32, #gpu.address_space<private>> -> memref<1x256x1xf32>
    return
}
