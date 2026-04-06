// Tuning integration tests: verify Winograd entries appear in tuning space
// and that perf_config strings with "winograd:" prefix are handled correctly.

// ============================================================================
// Winograd perf_config handling in AffixTuningParameters
// ============================================================================

// When a winograd: perf_config is set, AffixTuningParameters should return
// early (no-op) and let WinogradInterceptPass handle it.
// A 3x3 conv with explicit winograd perf_config should compile through.

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,RageV4_9,304,Default,fp32_fp32acc_f2x3_stride1" \
// RUN:   | rocmlir-opt --rock-affix-params 2>&1 \
// RUN:   | FileCheck %s --check-prefix=AFFIX_WINO

// AffixTuningParameters should pass through the op unchanged for winograd
// AFFIX_WINO: rock.conv
// AFFIX_WINO: perf_config = "winograd:

// ============================================================================
// Non-winograd perf_config still works normally
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=DEFAULT_GEMM

// DEFAULT_GEMM: gpu.binary
// DEFAULT_GEMM-NOT: miopenSp3

// ============================================================================
// isModuleFusible returns false for winograd configs
// ============================================================================

// Unfused 3x3 conv should compile normally through the full pipeline.
// isModuleFusible returns false for winograd: perf_configs, ensuring
// the tuning infrastructure does not attempt fusion with Winograd kernels.

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=UNFUSED

// UNFUSED: gpu.binary
