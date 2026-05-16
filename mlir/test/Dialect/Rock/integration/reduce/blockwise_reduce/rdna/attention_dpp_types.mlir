// Verify DPP-based blockwise reduction for f16/i8 attention on RDNA (gfx11xx/gfx12xx).
// Each case checks: (1) gpu.subgroup_reduce with expected cluster size, (2) correctness [1 1 1].
// Problem: g=12 q=77 k=77 d=64

// --- f16 cluster=4 ---
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 12 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,4,16,16,16,4,1,1,2,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_F16_IR
// CHECK_F16_IR: gpu.subgroup_reduce {{.*}} cluster(size = 4)

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 12 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:32,32,32,4,16,16,16,4,1,1,2,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16
// CHECK_F16: [1 1 1]

// --- i8 cluster=8 ---
// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 12 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:128,128,16,64,32,16,16,4,1,1,2,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_I8_IR
// CHECK_I8_IR: gpu.subgroup_reduce {{.*}} cluster(size = 8)

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 12 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:128,128,16,64,32,16,16,4,1,1,2,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_I8
// CHECK_I8: [1 1 1]
