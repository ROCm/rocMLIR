// Verify permlane-swap cross-lane reduction for f16/i8 attention on gfx950.
// Two partialR variants:
//   partialR=4 (mfma 16x16): v_permlane16_swap + v_permlane32_swap
//   partialR=2 (mfma 32x32): v_permlane32_swap only
// Each case checks IR intrinsics and correctness [1 1 1].
// Problem: g=1 q=77 k=77 d=64.

// --- f16 partialR=4: permlane16_swap + permlane32_swap ---
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,2,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_F16_4WAY_IR
// CHECK_F16_4WAY_IR-DAG: rocdl.permlane16.swap
// CHECK_F16_4WAY_IR-DAG: rocdl.permlane32.swap

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,2,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16_4WAY
// CHECK_F16_4WAY: [1 1 1]

// --- f16 partialR=2: permlane32_swap only ---
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,4,32,32,32,4,1,2,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_F16_2WAY_IR
// CHECK_F16_2WAY_IR: rocdl.permlane32.swap
// CHECK_F16_2WAY_IR-NOT: rocdl.permlane16.swap

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,4,32,32,32,4,1,2,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16_2WAY
// CHECK_F16_2WAY: [1 1 1]

// --- i8 partialR=4: permlane16_swap + permlane32_swap ---
// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,8,16,16,16,4,1,2,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_I8_4WAY_IR
// CHECK_I8_4WAY_IR-DAG: rocdl.permlane16.swap
// CHECK_I8_4WAY_IR-DAG: rocdl.permlane32.swap

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,8,16,16,16,4,1,2,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_I8_4WAY
// CHECK_I8_4WAY: [1 1 1]

// --- i8 partialR=2: permlane32_swap only ---
// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,8,32,32,32,4,1,2,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_I8_2WAY_IR
// CHECK_I8_2WAY_IR: rocdl.permlane32.swap
// CHECK_I8_2WAY_IR-NOT: rocdl.permlane16.swap

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,8,32,32,32,4,1,2,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_I8_2WAY
// CHECK_I8_2WAY: [1 1 1]
