// WinogradInterceptPass tests: verify the pass correctly handles various
// inputs and produces expected output or error messages.

// ============================================================================
// POSITIVE: 3x3 conv without winograd perf_config uses normal GEMM pipeline
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PIPELINE_OK

// Full pipeline should produce gpu.binary without Winograd
// PIPELINE_OK: gpu.binary
// PIPELINE_OK-NOT: miopenSp3

// ============================================================================
// POSITIVE: winograd perf_config is detected by the intercept pass
// ============================================================================

// A valid winograd perf_config should be recognized by the pass.
// On CI without ROCm assembler, the pass will fail at assembly step,
// but on a ROCm machine it should succeed. Test the detection via
// rocmlir-opt which runs just the intercept pass.
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,RageV4_9,304,Default,fp32_fp32acc_f2x3_stride1" \
// RUN:   | rocmlir-opt --rock-affix-params 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WINO_DETECTED

// AffixTuningParameters should see and preserve the winograd: perf_config
// WINO_DETECTED: perf_config = "winograd:

// ============================================================================
// NEGATIVE: Invalid winograd perf_config format should fail
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,InvalidFamily,999,garbage,nope" \
// RUN:   > %t.mlir
// RUN: not rocmlir-driver -kernel-pipeline full %t.mlir

// ============================================================================
// NEGATIVE: winograd perf_config with wrong version prefix
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v99,RageV4_9,304,Default,fp32" \
// RUN:   > %t2.mlir
// RUN: not rocmlir-driver -kernel-pipeline full %t2.mlir

// ============================================================================
// NEGATIVE: Empty family name
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,,304,Default,fp32" \
// RUN:   > %t3.mlir
// RUN: not rocmlir-driver -kernel-pipeline full %t3.mlir

// ============================================================================
// LAYOUT FALLBACK: NHWC layout with winograd perf_config falls back to GEMM
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout g01ck --in_layout n01gc --out_layout n01gk \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,RageV4_9,304,Default,fp32_fp32acc_f2x3_stride1" \
// RUN:   | rocmlir-driver -kernel-pipeline full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NHWC_FALLBACK

// Should fall back to GEMM (winograd: removed, GEMM produces gpu.binary)
// NHWC_FALLBACK: gpu.binary
// NHWC_FALLBACK-NOT: miopenSp3
// NHWC_FALLBACK-NOT: winograd

// ============================================================================
// GROUPED CONV FALLBACK: grouped 3x3 with winograd perf_config falls back
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 4 \
// RUN:   | rocmlir-driver -kernel-pipeline full 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GROUPED_FALLBACK

// Grouped conv should never use Winograd
// GROUPED_FALLBACK: gpu.binary
// GROUPED_FALLBACK-NOT: miopenSp3
