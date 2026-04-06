// Regression test: ensure existing convolution paths work unchanged
// when Winograd code is linked in.
//
// These tests use existing rocmlir-gen conv configs that should NOT
// trigger the Winograd path (1x1 filters, int8, grouped, etc.)

// ============================================================================
// 1x1 conv should always use GEMM path
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx908 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 1024 --fil_h 1 --fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=GEMM_1x1

// GEMM_1x1: gpu.binary
// GEMM_1x1-NOT: winograd
// GEMM_1x1-NOT: miopenSp3

// ============================================================================
// 3x3 conv WITHOUT winograd perf_config should still use GEMM
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx908 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=GEMM_3x3

// GEMM_3x3: gpu.binary
// GEMM_3x3-NOT: winograd
// GEMM_3x3-NOT: miopenSp3

// ============================================================================
// fp16 1x1 conv on gfx942 should use GEMM
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f16 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 512 --in_h 7 --in_w 7 \
// RUN:   --out_channels 2048 --fil_h 1 --fil_w 1 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=F16_1x1

// F16_1x1: gpu.binary
// F16_1x1-NOT: winograd
// F16_1x1-NOT: miopenSp3

// ============================================================================
// Strided 3x3 conv should use GEMM
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=STRIDE2_3x3

// STRIDE2_3x3: gpu.binary
// STRIDE2_3x3-NOT: miopenSp3

// ============================================================================
// Grouped 3x3 conv should use GEMM
// ============================================================================

// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 256 --in_h 14 --in_w 14 \
// RUN:   --out_channels 256 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 4 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=GROUPED_3x3

// GROUPED_3x3: gpu.binary
// GROUPED_3x3-NOT: miopenSp3

// ============================================================================
// int8 3x3 conv should use GEMM (Winograd doesn't support int8)
// ============================================================================

// RUN: rocmlir-gen --operation conv -t i8 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full | FileCheck %s --check-prefix=INT8_3x3

// INT8_3x3: gpu.binary
// INT8_3x3-NOT: miopenSp3
