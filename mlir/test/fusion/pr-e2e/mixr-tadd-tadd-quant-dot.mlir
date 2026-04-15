// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx-linalg,highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module {
  // CHECK: [28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28,  28]
  func.func @mlir_add(%arg0: !migraphx.shaped<3x2x7x2xf8E4M3FN, 28x14x2x1>,
                      %arg1: !migraphx.shaped<3x2x5x7xf8E4M3FN, 70x35x7x1>) -> !migraphx.shaped<3x2x2x5xf32, 20x10x5x1> attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = migraphx.transpose %arg0 {permutation = [0, 1, 3, 2]} : <3x2x7x2xf8E4M3FN, 28x14x2x1> -> <3x2x2x7xf8E4M3FN, 28x14x1x2>
    %1 = migraphx.add %0, %0 : <3x2x2x7xf8E4M3FN, 28x14x1x2>, <3x2x2x7xf8E4M3FN, 28x14x1x2> -> <3x2x2x7xf8E4M3FN, 28x14x1x2>
    %2 = migraphx.transpose %arg1 {permutation = [0, 1, 3, 2]} : <3x2x5x7xf8E4M3FN, 70x35x7x1> -> <3x2x7x5xf8E4M3FN, 70x35x1x7>
    %3 = migraphx.add %2, %2 : <3x2x7x5xf8E4M3FN, 70x35x1x7>, <3x2x7x5xf8E4M3FN, 70x35x1x7> -> <3x2x7x5xf8E4M3FN, 70x35x1x7>
    %4 = migraphx.quant_dot %1, %3 {perf_config="v3:64,32,4,32,16,16,1,1,2,1,1"} : <3x2x2x7xf8E4M3FN, 28x14x1x2>, <3x2x7x5xf8E4M3FN, 70x35x1x7> -> <3x2x2x5xf32, 20x10x5x1>
    return %4 : !migraphx.shaped<3x2x2x5xf32, 20x10x5x1>
  }
}

