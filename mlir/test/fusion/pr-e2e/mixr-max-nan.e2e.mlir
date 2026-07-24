// RUN: rocmlir-gen -fut dot_max_nan --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_max_nan_wrapper - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen -fut dot_max_nan --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx-linalg,highlevel -host-pipeline=migraphx-linalg,highlevel | rocmlir-gen -ph -print-results -rand none -fut dot_max_nan_wrapper - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// Max propagates NaNs from either operand even though its scalar lowering uses
// `nsz`. There is deliberately no signed-zero result requirement.
// CHECK-COUNT-15: nan

module {
  func.func @dot_max_nan(
      %arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>,
      %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>)
      -> !migraphx.shaped<1x5x3xf32, 15x3x1>
      attributes {rock.kernel, rock.arch = ""} {
    %dot = migraphx.dot %arg0, %arg1
        : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1>
        -> <1x5x3xf32, 15x3x1>
    %zero = migraphx.sub %dot, %dot
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %one_literal = migraphx.literal(
        dense<1.0> : tensor<1x5x3xf32>)
        : <1x5x3xf32, 15x3x1>
    %one = migraphx.add %zero, %one_literal
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %lhs_literal = migraphx.literal(
        dense<[[[0x7FC00000, 1.0, 0x7FC00000],
                 [1.0, 0x7FC00000, 1.0],
                 [0x7FC00000, 1.0, 0x7FC00000],
                 [1.0, 0x7FC00000, 1.0],
                 [0x7FC00000, 1.0, 0x7FC00000]]]> : tensor<1x5x3xf32>)
        : <1x5x3xf32, 15x3x1>
    %rhs_literal = migraphx.literal(
        dense<[[[1.0, 0x7FC00000, 1.0],
                 [0x7FC00000, 1.0, 0x7FC00000],
                 [1.0, 0x7FC00000, 1.0],
                 [0x7FC00000, 1.0, 0x7FC00000],
                 [1.0, 0x7FC00000, 1.0]]]> : tensor<1x5x3xf32>)
        : <1x5x3xf32, 15x3x1>
    %lhs = migraphx.mul %lhs_literal, %one
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %rhs = migraphx.mul %rhs_literal, %one
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    %result = migraphx.max %lhs, %rhs
        : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1>
        -> <1x5x3xf32, 15x3x1>
    return %result : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
