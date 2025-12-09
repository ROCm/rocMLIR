// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 256 -n 128 -k 64 -g 1 -t f32 --transB True \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   --runner-path mlir-runner \
// RUN:   --libs "%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext" \
// RUN:   -arch %arch | FileCheck %s

// Verify hipblaslt GEMM f32 256x128x64 with transB produces correct results
// CHECK: PASSED

