// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -fut mlir_remove_casts --arch %arch --clone-harness - | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_remove_casts_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver --arch %arch --kernel-pipeline=migraphx,highlevel - | rocmlir-driver -arch %arch -c -mlir-print-ir-after=rock-remove-redundant-casts 2>&1 | FileCheck %s --check-prefix=NO-FPEXT

// CHECK: [1 1 1]
// NO-FPEXT-NOT: llvm.fpext
module {
func.func @mlir_remove_casts(%arg0: !migraphx.shaped<1x64x5x128xf16, 40960x640x128x1>, %arg1: !migraphx.shaped<1x64x5x128xf16, 40960x640x128x1>, %arg2: !migraphx.shaped<1x64x5x128xf16, 40960x640x128x1>) -> !migraphx.shaped<1x5x64x128xf16, 40960x8192x128x1> attributes {arch="gfx950", kernel = "mixr"} {
    %0 = migraphx.literal(dense<8.837890e-02> : tensor<1xf16>) : <1xf16, 0>
    %1 = migraphx.transpose %arg0 {permutation = [0, 2, 1, 3]} : <1x64x5x128xf16, 40960x640x128x1> -> <1x5x64x128xf16, 40960x128x640x1>
    %2 = migraphx.transpose %arg1 {permutation = [0, 2, 3, 1]} : <1x64x5x128xf16, 40960x640x128x1> -> <1x5x128x64xf16, 40960x128x1x640>
    %3 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <1x64x5x128xf16, 40960x640x128x1> -> <1x5x64x128xf16, 40960x128x640x1>
    %4 = migraphx.dot %1, %2 : <1x5x64x128xf16, 40960x128x640x1>, <1x5x128x64xf16, 40960x128x1x640> -> <1x5x64x64xf16, 20480x4096x64x1>
    %5 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 5, 64, 64]} : <1xf16, 0> -> <1x5x64x64xf16, 0x0x0x0>
    %6 = migraphx.convert %4 {target_type = 2 : i64} : <1x5x64x64xf16, 20480x4096x64x1> to <1x5x64x64xf32, 20480x4096x64x1>
    %7 = migraphx.convert %5 {target_type = 2 : i64} : <1x5x64x64xf16, 0x0x0x0> to <1x5x64x64xf32, 0x0x0x0>
    %8 = migraphx.mul %6, %7 : <1x5x64x64xf32, 20480x4096x64x1>, <1x5x64x64xf32, 0x0x0x0> -> <1x5x64x64xf32, 20480x4096x64x1>
    %9 = migraphx.reshape %8 {dims = [1, 5, 64, 64]} : <1x5x64x64xf32, 20480x4096x64x1> -> <1x5x64x64xf32, 20480x4096x64x1>
    %10 = migraphx.reduce_max %9 {axes = [3]} : <1x5x64x64xf32, 20480x4096x64x1> -> <1x5x64x1xf32, 320x64x1x1>
    %11 = migraphx.reshape %10 {dims = [1, 5, 64, 1]} : <1x5x64x1xf32, 320x64x1x1> -> <1x5x64x1xf32, 320x64x1x1>
    %12 = migraphx.multibroadcast %11 {out_dyn_dims = [], out_lens = [1, 5, 64, 64]} : <1x5x64x1xf32, 320x64x1x1> -> <1x5x64x64xf32, 320x64x1x0>
    %13 = migraphx.sub %8, %12 : <1x5x64x64xf32, 20480x4096x64x1>, <1x5x64x64xf32, 320x64x1x0> -> <1x5x64x64xf32, 20480x4096x64x1>
    %14 = migraphx.exp %13 : <1x5x64x64xf32, 20480x4096x64x1> -> <1x5x64x64xf32, 20480x4096x64x1>
    %15 = migraphx.reshape %14 {dims = [1, 5, 64, 64]} : <1x5x64x64xf32, 20480x4096x64x1> -> <1x5x64x64xf32, 20480x4096x64x1>
    %16 = migraphx.reduce_sum %15 {axes = [3]} : <1x5x64x64xf32, 20480x4096x64x1> -> <1x5x64x1xf32, 320x64x1x1>
    %17 = migraphx.reshape %16 {dims = [1, 5, 64, 1]} : <1x5x64x1xf32, 320x64x1x1> -> <1x5x64x1xf32, 320x64x1x1>
    %18 = migraphx.multibroadcast %17 {out_dyn_dims = [], out_lens = [1, 5, 64, 64]} : <1x5x64x1xf32, 320x64x1x1> -> <1x5x64x64xf32, 320x64x1x0>
    %19 = migraphx.div %14, %18 : <1x5x64x64xf32, 20480x4096x64x1>, <1x5x64x64xf32, 320x64x1x0> -> <1x5x64x64xf32, 20480x4096x64x1>
    %20 = migraphx.convert %19 {target_type = 1 : i64} : <1x5x64x64xf32, 20480x4096x64x1> to <1x5x64x64xf16, 20480x4096x64x1>
    %21 = migraphx.dot %20, %3 : <1x5x64x64xf16, 20480x4096x64x1>, <1x5x64x128xf16, 40960x128x640x1> -> <1x5x64x128xf16, 40960x8192x128x1>
    return %21 : !migraphx.shaped<1x5x64x128xf16, 40960x8192x128x1>
  }
}

