// RUN: rocmlir-gen -fut matmul_const_nonsquare --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx,highlevel -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -print-results -rand 1 -rand_type float -fut matmul_const_nonsquare_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal,runner -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

// This kernel performs matrix multiplication C = A * B
// where B is a constant matrix filled with 0s, 1s, and -1s.
// A is a 3x4 input matrix (batched with batch size 1)
// B is a 4x5 constant matrix with values {-1, 0, 1}
// C is the 3x5 output matrix

module {
  func.func @matmul_const_nonsquare(%arg0: !migraphx.shaped<1x3x4xf32, 12x4x1>) -> !migraphx.shaped<1x3x5xf32, 15x5x1> attributes {kernel, arch = ""} {
    // Constant matrix B (4x5) filled with 0s, 1s, and -1s
    // Layout: [batch=1, K=4, N=5]
    // The constant values form a pattern useful for feature selection/combination:
    // [[ 1,  0,  0, -1,  1],
    //  [ 0,  1, -1,  0,  0],
    //  [-1,  0,  1,  1,  0],
    //  [ 0, -1,  0,  0,  1]]
    %const_b = migraphx.literal(dense<[[
      [ 1.0,  0.0,  0.0, -1.0,  1.0],
      [ 0.0,  1.0, -1.0,  0.0,  0.0],
      [-1.0,  0.0,  1.0,  1.0,  0.0],
      [ 0.0, -1.0,  0.0,  0.0,  1.0]
    ]]> : tensor<1x4x5xf32>) : <1x4x5xf32, 20x5x1>

    // Perform matrix multiplication: C = A * B
    // A: [1, 3, 4] * B: [1, 4, 5] -> C: [1, 3, 5]
    %result = migraphx.dot %arg0, %const_b : <1x3x4xf32, 12x4x1>, <1x4x5xf32, 20x5x1> -> <1x3x5xf32, 15x5x1>

    return %result : !migraphx.shaped<1x3x5xf32, 15x5x1>
  }
}

