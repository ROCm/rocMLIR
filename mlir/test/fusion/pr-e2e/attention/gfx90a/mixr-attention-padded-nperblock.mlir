// RUN: rocmlir-gen -fut mlir_attention --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_attention_wrapper -relDiff_threshold 0.0001 --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                                    %arg1: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                                    %arg2: !migraphx.shaped<1xf32, 1>,
                                    %arg3: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                                    %arg4: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                                    %arg5: !migraphx.shaped<1x256x256xf32, 65536x256x1>,
                                    %arg6: !migraphx.shaped<1x256x256xf32, 65536x256x1>)
                                    -> !migraphx.shaped<1x256x256xf32, 65536x256x1>
                                    attributes {rock.arch = "", rock.enable_splitk_for_tuning, rock.kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1xf32, 1> -> <1x256x256xf32, 0x0x0>
    %1 = migraphx.add %arg0, %arg1 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %2 = migraphx.mul %1, %0 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 0x0x0> -> <1x256x256xf32, 65536x256x1>
    %3 = migraphx.tanh %2 : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %4 = migraphx.add %arg3, %arg4 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %5 = migraphx.tanh %4 : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %6 = migraphx.add %arg5, %arg6 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %7 = migraphx.tanh %6 : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %8 = migraphx.transpose %5 {permutation = [0, 2, 1]} : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x1x256>
    %9 = migraphx.transpose %7 {permutation = [0, 2, 1]} : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x1x256>
    %10 = migraphx.dot %3, %8 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x1x256> -> <1x256x256xf32, 65536x256x1>
    %11 = migraphx.reshape %10 {dims = [1, 256, 256]} : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %12 = migraphx.reduce_max %11 {axes = [2]} : <1x256x256xf32, 65536x256x1> -> <1x256x1xf32, 256x1x1>
    %13 = migraphx.reshape %12 {dims = [1, 256, 1]} : <1x256x1xf32, 256x1x1> -> <1x256x1xf32, 256x1x1>
    %14 = migraphx.multibroadcast %13 {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1x256x1xf32, 256x1x1> -> <1x256x256xf32, 256x1x0>
    %15 = migraphx.sub %10, %14 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 256x1x0> -> <1x256x256xf32, 65536x256x1>
    %16 = migraphx.exp %15 : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %17 = migraphx.reshape %16 {dims = [1, 256, 256]} : <1x256x256xf32, 65536x256x1> -> <1x256x256xf32, 65536x256x1>
    %18 = migraphx.reduce_sum %17 {axes = [2]} : <1x256x256xf32, 65536x256x1> -> <1x256x1xf32, 256x1x1>
    %19 = migraphx.reshape %18 {dims = [1, 256, 1]} : <1x256x1xf32, 256x1x1> -> <1x256x1xf32, 256x1x1>
    %20 = migraphx.multibroadcast %19 {out_dyn_dims = [], out_lens = [1, 256, 256]} : <1x256x1xf32, 256x1x1> -> <1x256x256xf32, 256x1x0>
    %21 = migraphx.div %16, %20 : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 256x1x0> -> <1x256x256xf32, 65536x256x1>
    %22 = migraphx.dot %21, %9 {perf_config = "attn:v3:64,64,16,8,16,16,16,4,1,2,2,0,1"} : <1x256x256xf32, 65536x256x1>, <1x256x256xf32, 65536x1x256> -> <1x256x256xf32, 65536x256x1>
    return %22 : !migraphx.shaped<1x256x256xf32, 65536x256x1>
  }
}
