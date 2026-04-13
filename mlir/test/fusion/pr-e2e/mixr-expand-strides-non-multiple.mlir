// RUN: rocmlir-gen -fut mlir_dot_sigmoid --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel -targets %arch | rocmlir-gen -ph -rand 1 -rand_type float -fut mlir_dot_sigmoid_wrapper --verifier clone -print-verify-results=always - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s 

// Non-contiguous strides from a concat-like layout (4x5x24 and 4x7x24 in a
// 4x12x24 logical tensor). Ensures dot+sigmoid + expand-strides placement;
// with correct GPU staging, GPU matches CPU on the full output.
// CHECK: relDiff = 0 : 1152/1152 (100.000000%)

module {
  func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x5x16xf16, 80x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x5x24xf16, 288x24x1> attributes {rock.arch = "gfx1201", rock.kernel = "mixr", rock.num_cu = 32 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <4x5x16xf16, 80x16x1>, <4x16x24xf16, 384x24x1> -> <4x5x24xf16, 120x24x1>
    %1 = migraphx.sigmoid %0 : <4x5x24xf16, 120x24x1> -> <4x5x24xf16, 288x24x1>
    return %1 : !migraphx.shaped<4x5x24xf16, 288x24x1>
  }
}
