// REQUIRES: rock-any-e2e
// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 256 -n 256 -k 128 -g 1 -t f16 \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   --runner-path mlir-runner \
// RUN:   --libs "%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext" \
// RUN:   -arch %arch --tolerance 0.05 | FileCheck %s

// Verify hipblaslt GEMM f16 256x256x128 produces correct results
// CHECK: PASSED

