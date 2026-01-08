// RUN: rocmlir-gen -fut mlir_remove_casts --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_remove_casts_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]
module {
func.func @mlir_remove_casts(%arg0: !migraphx.shaped<1x75352x5x128xf16, 48225280x640x128x1>, %arg1: !migraphx.shaped<1x75352x5x128xf16, 48225280x640x128x1>, %arg2: !migraphx.shaped<1x75352x5x128xf16, 48225280x640x128x1>) -> !migraphx.shaped<1x5x75352x128xf16, 48225280x9645056x128x1> attributes {kernel = "mixr"} {
    %0 = migraphx.literal(dense<8.837890e-02> : tensor<1xf16>) : <1xf16, 0>
    %1 = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x75352x5x128xf16, 48225280x640x128x1> -> <1x5x75352x128xf16, 48225280x128x640x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 2, 3, 1]} : <1x75352x5x128xf16, 48225280x640x128x1> -> <1x5x128x75352xf16, 48225280x128x1x640>
    %3 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x75352x5x128xf16, 48225280x640x128x1> -> <1x5x75352x128xf16, 48225280x128x640x1>
    %4 = migraphx.dot %1, %2 : <1x5x75352x128xf16, 48225280x128x640x1>, <1x5x128x75352xf16, 48225280x128x1x640> -> <1x5x75352x75352xf16, 28389619520x5677923904x75352x1>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 5, 75352, 75352]} : <1xf16, 0> -> <1x5x75352x75352xf16, 0x0x0x0>
    %6 = migraphx.convert %4 {target_type = 2 : i64} : <1x5x75352x75352xf16, 28389619520x5677923904x75352x1> to <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %7 = migraphx.convert %5 {target_type = 2 : i64} : <1x5x75352x75352xf16, 0x0x0x0> to <1x5x75352x75352xf32, 0x0x0x0>
    %8 = migraphx.mul %6, %7 : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>, <1x5x75352x75352xf32, 0x0x0x0> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %9 = migraphx.reshape %8 {dims = [1, 5, 75352, 75352]} : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %10 = migraphx.reduce_max %9 {axes = [3]} : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> -> <1x5x75352x1xf32, 376760x75352x1x1>
    %11 = migraphx.reshape %10 {dims = [1, 5, 75352, 1]} : <1x5x75352x1xf32, 376760x75352x1x1> -> <1x5x75352x1xf32, 376760x75352x1x1>
    %12 = migraphx.multibroadcast %11 {out_dyn_dims = [], out_lens = [1, 5, 75352, 75352]} : <1x5x75352x1xf32, 376760x75352x1x1> -> <1x5x75352x75352xf32, 376760x75352x1x0>
    %13 = migraphx.sub %8, %12 : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>, <1x5x75352x75352xf32, 376760x75352x1x0> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %14 = migraphx.exp %13 : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %15 = migraphx.reshape %14 {dims = [1, 5, 75352, 75352]} : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %16 = migraphx.reduce_sum %15 {axes = [3]} : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> -> <1x5x75352x1xf32, 376760x75352x1x1>
    %17 = migraphx.reshape %16 {dims = [1, 5, 75352, 1]} : <1x5x75352x1xf32, 376760x75352x1x1> -> <1x5x75352x1xf32, 376760x75352x1x1>
    %18 = migraphx.multibroadcast %17 {out_dyn_dims = [], out_lens = [1, 5, 75352, 75352]} : <1x5x75352x1xf32, 376760x75352x1x1> -> <1x5x75352x75352xf32, 376760x75352x1x0>
    %19 = migraphx.div %14, %18 : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>, <1x5x75352x75352xf32, 376760x75352x1x0> -> <1x5x75352x75352xf32, 28389619520x5677923904x75352x1>
    %20 = migraphx.convert %19 {target_type = 1 : i64} : <1x5x75352x75352xf32, 28389619520x5677923904x75352x1> to <1x5x75352x75352xf16, 28389619520x5677923904x75352x1>
    %21 = migraphx.dot %20, %3 : <1x5x75352x75352xf16, 28389619520x5677923904x75352x1>, <1x5x75352x128xf16, 48225280x128x640x1> -> <1x5x75352x128xf16, 48225280x9645056x128x1>
    return %21 : !migraphx.shaped<1x5x75352x128xf16, 48225280x9645056x128x1>
  }
}