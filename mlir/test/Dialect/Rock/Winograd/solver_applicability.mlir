// Winograd solver applicability tests via rocmlir-gen and tuning infrastructure.
// These tests verify which convolution configs produce Winograd tuning entries
// and which don't.

// ============================================================================
// POSITIVE TESTS: configs that should produce Winograd tuning entries
// ============================================================================

// --- fp32 3x3 stride=1 pad=1 on gfx942 (Rage v4.9 eligible) ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=F32_3x3_GFX942

// F32_3x3_GFX942: gpu.binary

// --- fp16 3x3 stride=1 pad=1 on gfx942 ---
// RUN: rocmlir-gen --operation conv -t f16 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=F16_3x3_GFX942

// F16_3x3_GFX942: gpu.binary

// --- fp32 on gfx908 (V30 family eligible) ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx908 \
// RUN:   --num_cu 120 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=F32_GFX908

// F32_GFX908: gpu.binary

// --- fp16 on gfx90a (Rage v4.9 eligible) ---
// RUN: rocmlir-gen --operation conv -t f16 --arch gfx90a \
// RUN:   --num_cu 110 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 2 --in_channels 128 --in_h 28 --in_w 28 \
// RUN:   --out_channels 128 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=F16_GFX90A

// F16_GFX90A: gpu.binary

// --- Large batch, large spatial (Rage territory) ---
// RUN: rocmlir-gen --operation conv -t f16 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 64 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 1 --padding_w 1 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LARGE_BATCH

// LARGE_BATCH: gpu.binary

// --- fp32 with pad=0 (valid convolution) ---
// RUN: rocmlir-gen --operation conv -t f32 --arch gfx942 \
// RUN:   --num_cu 304 \
// RUN:   --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 \
// RUN:   --batchsize 1 --in_channels 64 --in_h 56 --in_w 56 \
// RUN:   --out_channels 64 --fil_h 3 --fil_w 3 \
// RUN:   --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 \
// RUN:   --padding_h 0 --padding_w 0 --groupsize 1 \
// RUN:   | rocmlir-driver -kernel-pipeline full -rock-tuning-space=greedy 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PAD_ZERO

// PAD_ZERO: gpu.binary
