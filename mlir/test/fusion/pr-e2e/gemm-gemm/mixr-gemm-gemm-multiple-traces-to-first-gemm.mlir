// RUN: rocmlir-gen -fut mlir_gemm_gemm --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_gemm_gemm_wrapper  --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
  func.func @mlir_gemm_gemm(%arg0: !migraphx.shaped<124x976xf32, 976x1>, %arg1: !migraphx.shaped<976x664xf32, 664x1>, %arg2: !migraphx.shaped<664xf32, 1>, %arg3: !migraphx.shaped<664xf32, 1>, %arg4: !migraphx.shaped<664xf32, 1>, %arg5: !migraphx.shaped<664xf32, 1>, %arg6: !migraphx.shaped<664x88xf32, 88x1>) -> !migraphx.shaped<124x88xf32, 88x1>  {
    %0 = migraphx.literal(dense<1.000000e+00> : tensor<1xf32>) : <1xf32, 0>
    %1 = migraphx.dot %arg0, %arg1 : <124x976xf32, 976x1>, <976x664xf32, 664x1> -> <124x664xf32, 664x1>
    %2 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [124, 664]} : <664xf32, 1> -> <124x664xf32, 0x1>
    %3 = migraphx.multibroadcast %arg3 {out_dyn_dims = [], out_lens = [124, 664]} : <664xf32, 1> -> <124x664xf32, 0x1>
    %4 = migraphx.multibroadcast %arg4 {out_dyn_dims = [], out_lens = [124, 664]} : <664xf32, 1> -> <124x664xf32, 0x1>
    %5 = migraphx.multibroadcast %arg5 {out_dyn_dims = [], out_lens = [124, 664]} : <664xf32, 1> -> <124x664xf32, 0x1>
    %6 = migraphx.add %2, %1 : <124x664xf32, 0x1>, <124x664xf32, 664x1> -> <124x664xf32, 664x1>
    %7 = migraphx.mul %6, %3 : <124x664xf32, 664x1>, <124x664xf32, 0x1> -> <124x664xf32, 664x1>
    %8 = migraphx.add %7, %4 : <124x664xf32, 664x1>, <124x664xf32, 0x1> -> <124x664xf32, 664x1>
    %9 = migraphx.sigmoid %8 : <124x664xf32, 664x1> -> <124x664xf32, 664x1>
    %10 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [124, 664]} : <1xf32, 0> -> <124x664xf32, 0x0>
    %11 = migraphx.sub %10, %9 : <124x664xf32, 0x0>, <124x664xf32, 664x1> -> <124x664xf32, 664x1>
    %12 = migraphx.mul %11, %5 : <124x664xf32, 664x1>, <124x664xf32, 0x1> -> <124x664xf32, 664x1>
    %13 = migraphx.add %9, %12 : <124x664xf32, 664x1>, <124x664xf32, 664x1> -> <124x664xf32, 664x1>
    %14 = migraphx.mul %13, %6 : <124x664xf32, 664x1>, <124x664xf32, 664x1> -> <124x664xf32, 664x1>
    %15 = migraphx.dot %14, %arg6 : <124x664xf32, 664x1>, <664x88xf32, 88x1> -> <124x88xf32, 88x1>
    return %15 : !migraphx.shaped<124x88xf32, 88x1>
  }
}
