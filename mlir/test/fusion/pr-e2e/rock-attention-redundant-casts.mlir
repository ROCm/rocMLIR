// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -g 16 -seq_len_q 1500 -seq_len_k 4096 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 1 -num_heads_kv 1 -transK=true -pv | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void | FileCheck %s
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -g 16 -seq_len_q 1500 -seq_len_k 4096 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 1 -num_heads_kv 1 -transK=true | rocmlir-driver -arch %arch -c -mlir-print-ir-after=rock-remove-redundant-casts 2>&1 | FileCheck %s --check-prefix=NO-FPEXT

// CHECK: [1 1 1]
// NO-FPEXT-NOT: llvm.fpext
