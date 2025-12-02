// RUN: rocmlir-gen -rand 1 -return_lse -split_kv 4 -current_seq_len=17,1,32 -g 3 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 1 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias --transQ=false --transK=false --transV=false --transO=false  --operation attention -t f16 --arch %arch -pv \
// RUN: | rocmlir-driver -c \
// RUN: | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext -entry-point-result=void \
// RUN: | FileCheck %s

// CHECK: [1 1 1]
