// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]
module {
func.func @mlir_attention(%arg0: !migraphx.shaped<1x77x768xf16, 59136x768x1>, %arg1: !migraphx.shaped<1x77x768xf16, 59136x768x1>, %arg2: !migraphx.shaped<1x1x77x77xf16, 5929x5929x77x1>, %arg3: !migraphx.shaped<1x77x768xf16, 59136x768x1>) -> (!migraphx.shaped<12x77x64xf16, 4928x64x1>) {
    %0 = migraphx.reshape %arg0  {dims = [1, 77, 12, 64]} : <1x77x768xf16, 59136x768x1> -> <1x77x12x64xf16, 59136x768x64x1>
    %1 = migraphx.transpose %0  {permutation = [2, 0, 1, 3]} : <1x77x12x64xf16, 59136x768x64x1> -> <12x1x77x64xf16, 64x59136x768x1>
    %2 = migraphx.reshape %1  {dims = [12, 77, 64]} : <12x1x77x64xf16, 64x59136x768x1> -> <12x77x64xf16, 64x768x1>
    %3 = migraphx.reshape %arg1  {dims = [1, 77, 12, 64]} : <1x77x768xf16, 59136x768x1> -> <1x77x12x64xf16, 59136x768x64x1>
    %4 = migraphx.transpose %3  {permutation = [2, 3, 0, 1]} : <1x77x12x64xf16, 59136x768x64x1> -> <12x64x1x77xf16, 64x1x59136x768>
    %5 = migraphx.reshape %4  {dims = [12, 64, 77]} : <12x64x1x77xf16, 64x1x59136x768> -> <12x64x77xf16, 64x1x768>
    %6 = migraphx.multibroadcast %arg2  {out_lens = [1, 12, 77, 77]} : <1x1x77x77xf16, 5929x5929x77x1> -> <1x12x77x77xf16, 5929x0x77x1>
    %7 = migraphx.reshape %arg3  {dims = [1, 77, 12, 64]} : <1x77x768xf16, 59136x768x1> -> <1x77x12x64xf16, 59136x768x64x1>
    %8 = migraphx.transpose %7  {permutation = [2, 0, 1, 3]} : <1x77x12x64xf16, 59136x768x64x1>-><12x1x77x64xf16, 64x59136x768x1>
    %9 = migraphx.reshape %8  {dims = [12, 77, 64]} : <12x1x77x64xf16, 64x59136x768x1>-><12x77x64xf16, 64x768x1>
    %10 = migraphx.dot %2, %5 : <12x77x64xf16, 64x768x1>, <12x64x77xf16, 64x1x768> -> <12x77x77xf16, 5929x77x1>
    %11 = migraphx.reshape %10  {dims = [1, 12, 77, 77]} : <12x77x77xf16, 5929x77x1> -> <1x12x77x77xf16, 71148x5929x77x1>
    %12 = migraphx.add %11, %6 : <1x12x77x77xf16, 71148x5929x77x1>, <1x12x77x77xf16, 5929x0x77x1> -> <1x12x77x77xf16, 71148x5929x77x1>
    %13 = migraphx.reshape %12  {dims = [12, 77, 77]} : <1x12x77x77xf16, 71148x5929x77x1> -> <12x77x77xf16, 5929x77x1>
    %14 = migraphx.convert %13 {target_type = 2 : i64} : <12x77x77xf16, 5929x77x1> to <12x77x77xf32, 5929x77x1>
    %15 = migraphx.reshape %14  {dims = [12, 77, 77]} : <12x77x77xf32, 5929x77x1> -> <12x77x77xf32, 5929x77x1>
    %16 = migraphx.reduce_max %15 {axes = [2]} : <12x77x77xf32, 5929x77x1> -> <12x77x1xf32, 77x1x1>
    %17 = migraphx.reshape %16  {dims = [12, 77, 1]} : <12x77x1xf32, 77x1x1> -> <12x77x1xf32, 77x1x1>
    %18 = migraphx.multibroadcast %17 {out_lens = [12, 77, 77]} : <12x77x1xf32, 77x1x1> -> <12x77x77xf32, 77x1x0>
    %19 = migraphx.sub %14, %18 : <12x77x77xf32, 5929x77x1>, <12x77x77xf32, 77x1x0> -> <12x77x77xf32, 5929x77x1>
    %20 = migraphx.exp %19 : <12x77x77xf32, 5929x77x1> -> <12x77x77xf32, 5929x77x1>
    %21 = migraphx.reshape %20  {dims = [12, 77, 77]} : <12x77x77xf32, 5929x77x1> -> <12x77x77xf32, 5929x77x1>
    %22 = migraphx.reduce_sum %21  {axes = [2]} : <12x77x77xf32, 5929x77x1> -> <12x77x1xf32, 77x1x1>
    %23 = migraphx.reshape %22  {dims = [12, 77, 1]} : <12x77x1xf32, 77x1x1> -> <12x77x1xf32, 77x1x1>
    %24 = migraphx.multibroadcast %23  {out_lens = [12, 77, 77]} : <12x77x1xf32, 77x1x1> -> <12x77x77xf32, 77x1x0>
    %25 = migraphx.div %20, %24 : <12x77x77xf32, 5929x77x1>, <12x77x77xf32, 77x1x0> -> <12x77x77xf32, 5929x77x1>
    %26 = migraphx.convert %25 {target_type = 1 : i64} : <12x77x77xf32, 5929x77x1> to <12x77x77xf16, 5929x77x1>
    %27 = migraphx.dot %26, %9 : <12x77x77xf16, 5929x77x1>, <12x77x64xf16, 64x768x1> -> <12x77x64xf16, 4928x64x1>
    return %27 : !migraphx.shaped<12x77x64xf16, 4928x64x1>
  }
}
