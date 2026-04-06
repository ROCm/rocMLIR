// E2E correctness tests for Winograd-eligible convolution shapes.
// These use rocmlir-gen with -pv (populate & verify) to check that
// the GEMM pipeline produces correct results on shapes where
// Winograd could be applicable.
//
// REQUIRES: rocm-runner

// ============================================================================
// FP32 configs
// ============================================================================

// --- ResNet-50: 1x64x56x56, K=64, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F32_RESNET

// E2E_F32_RESNET: [1 1 1]

// --- ResNet-50: 1x128x28x28, K=128, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 128 --in_h 28 --in_w 28 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F32_RN128

// E2E_F32_RN128: [1 1 1]

// --- Small spatial: 1x64x14x14, K=64, 3x3, pad=0 ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 14 --in_w 14 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F32_PAD0

// E2E_F32_PAD0: [1 1 1]

// --- Small spatial: 1x512x7x7, K=512, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 512 --in_h 7 --in_w 7 \
// RUN:   --out_channels 512 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F32_SMALL

// E2E_F32_SMALL: [1 1 1]

// --- Valid conv (pad=0): 1x64x56x56, K=64, 3x3 ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F32_VALID

// E2E_F32_VALID: [1 1 1]

// ============================================================================
// FP16 configs
// ============================================================================

// --- fp16 1x64x56x56, K=64, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F16_RESNET

// E2E_F16_RESNET: [1 1 1]

// --- fp16 1x256x14x14, K=256, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F16_MED

// E2E_F16_MED: [1 1 1]

// --- fp16 diffusion model: 2x320x64x64, K=320, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 2 --in_channels 320 --in_h 64 --in_w 64 \
// RUN:   --out_channels 320 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F16_DIFFUSION

// E2E_F16_DIFFUSION: [1 1 1]

// --- fp16 large batch: 64x64x56x56, K=64, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 64 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F16_BATCH

// E2E_F16_BATCH: [1 1 1]

// --- fp16 large spatial: 1x128x100x100, K=128, 3x3, pad=1 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 128 --in_h 100 --in_w 100 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_F16_LARGE

// E2E_F16_LARGE: [1 1 1]

// ============================================================================
// REGRESSION: non-Winograd configs must still produce correct results
// ============================================================================

// --- 1x1 conv (not Winograd-eligible) must still work ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 1 --fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_REGRESS_1x1

// E2E_REGRESS_1x1: [1 1 1]

// --- Strided 3x3 (not Winograd-eligible on Rage) must still work ---
// RUN: rocmlir-gen --operation conv -t f32 --arch %arch %rocmlir_gen_flags \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 %pv \
// RUN:   | rocmlir-driver -kernel-pipeline full -host-pipeline runner \
// RUN:   | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext \
// RUN:     --entry-point-result=void \
// RUN:   | FileCheck %s --check-prefix=E2E_REGRESS_STRIDE

// E2E_REGRESS_STRIDE: [1 1 1]
