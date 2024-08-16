// RUN: rocmlir-gen -fut mlir_attention_where --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_where_wrapper -RMS_threshold 0.01 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
  func.func @mlir_attention_where(%arg0: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg1: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg2: !migraphx.shaped<1x12x256x256xi8, 786432x65536x256x1>, %arg3: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>, %arg4: !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>) -> !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>  // attributes {arch = "gfx942", kernel = "mixr"} 
  {
    %0 = migraphx.literal(dense<1.250000e-01> : tensor<1xf16>) : <1xf16, 0>
    %1 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x1x256>
    %2 = migraphx.dot %arg0, %1 : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x1x256> -> <1x12x256x256xf16, 786432x65536x256x1>
    %3 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 12, 256, 256]} : <1xf16, 0> -> <1x12x256x256xf16, 0x0x0x0>
    %4 = migraphx.mul %2, %3 : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 0x0x0x0> -> <1x12x256x256xf16, 786432x65536x256x1>
    %5 = migraphx.where %arg2, %4, %arg3 : <1x12x256x256xi8, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x256x1>
    %6 = migraphx.softmax %5 {axis = 3 : i64} : <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x256x1>
    %7 = migraphx.dot %6, %arg4 : <1x12x256x256xf16, 786432x65536x256x1>, <1x12x256x256xf16, 786432x65536x256x1> -> <1x12x256x256xf16, 786432x65536x256x1>
    return %7 : !migraphx.shaped<1x12x256x256xf16, 786432x65536x256x1>
  }
