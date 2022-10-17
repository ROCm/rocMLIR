// RUN: rocmlir-gen --arch %arch -p -t f16 %pv %rocmlir_gen_flags | rocmlir-driver -c | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BF16

// CHECK_BF16: [1 1 1]
