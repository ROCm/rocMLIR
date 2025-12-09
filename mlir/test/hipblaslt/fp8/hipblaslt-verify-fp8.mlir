// RUN: %python %mlir_src_root/utils/performance/hipblaslt-benchmark-driver/verify_hipblaslt.py -m 256 -n 256 -k 128 -g 1 -t fp8 \
// RUN:   --hipblaslt-path hipblaslt-benchmark-driver \
// RUN:   --rocmlir-gen-path rocmlir-gen \
// RUN:   --rocmlir-driver-path rocmlir-driver \
// RUN:   --runner-path mlir-runner \
// RUN:   --libs "%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext" \
// RUN:   -arch %arch --tolerance 0.1 | FileCheck %s

// Verify hipblaslt GEMM fp8 256x256x128 produces correct results
// CHECK: PASSED

