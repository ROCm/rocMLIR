// Negative tests: configurations that must NOT use Winograd.
// These verify the solver correctly rejects ineligible configs
// and falls through to the GEMM pipeline.

// ============================================================================
// FILTER SIZE REJECTION: Winograd requires 3x3 filters
// ============================================================================

// --- 1x1 convolution: must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 1 --fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=FILTER_1x1

// FILTER_1x1: gpu.binary
// FILTER_1x1-NOT: winograd
// FILTER_1x1-NOT: miopenSp3

// --- 5x5 convolution: must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 28 --in_w 28 \
// RUN:   --out_channels 64 --fil_h 5 --fil_w 5 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 2 --padding_w 2 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=FILTER_5x5

// FILTER_5x5: gpu.binary
// FILTER_5x5-NOT: winograd
// FILTER_5x5-NOT: miopenSp3

// --- 7x7 convolution: must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 3 --in_h 224 --in_w 224 \
// RUN:   --out_channels 64 --fil_h 7 --fil_w 7 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 \
// RUN:   --padding_h 3 --padding_w 3 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=FILTER_7x7

// FILTER_7x7: gpu.binary
// FILTER_7x7-NOT: winograd
// FILTER_7x7-NOT: miopenSp3

// ============================================================================
// DATA TYPE REJECTION: Winograd only supports fp16, fp32, bf16
// ============================================================================

// --- int8 convolution: must use GEMM ---
// RUN: rocmlir-gen --operation conv -t i8 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=DTYPE_INT8

// DTYPE_INT8: gpu.binary
// DTYPE_INT8-NOT: winograd
// DTYPE_INT8-NOT: miopenSp3

// ============================================================================
// STRIDE REJECTION: Rage/Fury families require stride=1
// ============================================================================

// --- stride=2 convolution: must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=STRIDE_2

// STRIDE_2: gpu.binary
// STRIDE_2-NOT: winograd
// STRIDE_2-NOT: miopenSp3

// ============================================================================
// GROUP CONVOLUTION REJECTION: Winograd requires groupCount=1
// ============================================================================

// --- grouped conv (g=4): must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 4 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=GROUPED

// GROUPED: gpu.binary
// GROUPED-NOT: winograd
// GROUPED-NOT: miopenSp3

// ============================================================================
// LAYOUT REJECTION: Winograd only supports NCHW layout
// ============================================================================

// --- NHWC layout: must fall back to GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout g01ck --in_layout n01gc --out_layout n01gk \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=NHWC_LAYOUT

// NHWC_LAYOUT: gpu.binary
// NHWC_LAYOUT-NOT: winograd
// NHWC_LAYOUT-NOT: miopenSp3

// ============================================================================
// SPATIAL SIZE REJECTION: Too small for Winograd overhead
// ============================================================================

// --- Very small spatial (3x3 input): must use GEMM ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 3 --in_w 3 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=TINY_SPATIAL

// TINY_SPATIAL: gpu.binary
// TINY_SPATIAL-NOT: winograd
// TINY_SPATIAL-NOT: miopenSp3

// ============================================================================
// REGRESSION: existing paths must remain unchanged
// ============================================================================

// --- 1x1 conv on gfx908: standard GEMM path ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx908 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 1024 --fil_h 1 --fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=REGRESS_1x1

// REGRESS_1x1: gpu.binary
// REGRESS_1x1-NOT: winograd
// REGRESS_1x1-NOT: miopenSp3

// --- 3x3 conv on gfx1100 (V30 family applicable) ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx1100 \
// RUN:   --num_cu 48 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=GFX1100

// GFX1100: gpu.binary
// GFX1100-NOT: miopenSp3
