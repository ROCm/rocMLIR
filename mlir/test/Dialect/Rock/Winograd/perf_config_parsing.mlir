// Tests for Winograd perf_config string handling through the pipeline.
// Verifies that winograd: and non-winograd perf_configs are dispatched correctly.

// ============================================================================
// Winograd perf_config passes through AffixTuningParameters unchanged
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   -perf_config "winograd:v1,RageV4_9,304,Default,fp32_fp32acc_f2x3_stride1" \
// RUN:   | rocmlir-opt --rock-affix-params 2>&1 \
// RUN:   | FileCheck %s --check-prefix=WINO_PASSTHROUGH

// AffixTuningParameters should leave winograd: perf_config intact
// WINO_PASSTHROUGH: rock.conv
// WINO_PASSTHROUGH: perf_config = "winograd:

// ============================================================================
// Non-winograd perf_config (GEMM v4) is handled normally
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-opt --rock-affix-params 2>&1 \
// RUN:   | FileCheck %s --check-prefix=GEMM_NORMAL

// Without explicit perf_config, AffixTuningParameters assigns GEMM params
// GEMM_NORMAL: rock.conv
// GEMM_NORMAL-NOT: winograd
