// Verify ds_swizzle wave32 cross-lane reduction for f16/i8 attention on
// RDNA3 (gfx11xx). Uses ds_swizzle_b32 XOR=16 to swap lanes 0-15 <-> 16-31.
// Each case checks: (1) amdgpu.swizzle_bitmode in IR, (2) correctness [1 1 1].
// Problem: g=1 q=77 k=77 d=64.

// --- f16 ds_swizzle wave32 ---
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_F16_IR
// CHECK_F16_IR: amdgpu.swizzle_bitmode

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16
// CHECK_F16: [1 1 1]

// --- i8 ds_swizzle wave32 ---
// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_I8_IR
// CHECK_I8_IR: amdgpu.swizzle_bitmode

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_I8
// CHECK_I8: [1 1 1]
