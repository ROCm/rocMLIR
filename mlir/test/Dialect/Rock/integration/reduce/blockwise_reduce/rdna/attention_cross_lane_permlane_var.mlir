// Verify permlanex16_var cross-lane reduction for f16/i8 attention on
// RDNA4 (gfx12xx). Uses v_permlanex16_var_b32 with cross=true.
// Each case checks: (1) amdgpu.permlane_var in IR, (2) correctness [1 1 1].
// Problem: g=1 q=77 k=77 d=64.

// --- f16 permlanex16_var ---
// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_F16_IR
// CHECK_F16_IR: amdgpu.permlane_var {{.*}} cross(true)

// RUN: rocmlir-gen --arch %arch --operation attention -t f16 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_F16
// CHECK_F16: [1 1 1]

// --- i8 permlanex16_var ---
// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" | rocmlir-driver --kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_I8_IR
// CHECK_I8_IR: amdgpu.permlane_var {{.*}} cross(true)

// RUN: rocmlir-gen --arch %arch --operation attention -t i8 -num_heads_q 1 -seq_len_q 77 -seq_len_k 77 -head_dim_qk 64 -head_dim_v 64 --perf_config "attn:v3:16,16,16,4,16,16,16,4,1,1,2,0,1" -rand 1 -rand_type int -pv -relDiff_threshold 0.02 -RMS_threshold 0.015 | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_I8
// CHECK_I8: [1 1 1]
