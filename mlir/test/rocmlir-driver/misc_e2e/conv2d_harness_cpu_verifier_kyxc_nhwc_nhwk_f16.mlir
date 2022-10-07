// RUN: rocmlir-gen --arch %arch -p -pv -rand 1 -t f16 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk %s | rocmlir-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E


// E2E: [1 1 1]
