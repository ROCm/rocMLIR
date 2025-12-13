// RUN: rocmlir-gen --arch %arch --operation attention -t f16 --causal --prefix_offset=3 --current_seq_len=4 -seq_len_q 8 -seq_len_k 16 -head_dim_qk 32 -head_dim_v 32 -rand 1 -rand_type float -pv | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK: [1 1 1]

