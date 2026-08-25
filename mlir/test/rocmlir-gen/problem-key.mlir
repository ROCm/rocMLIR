// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_1
// CHECK_1: -t f32 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false -supportsSplitK true
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_2
// CHECK_2: -t f16 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_3
// CHECK_3: -t i8 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_4
// CHECK_4: -t i8 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 4 -num_heads_kv 4 -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_5
// CHECK_5: -t i8 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 4 -num_heads_kv 2 -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_6
// CHECK_6: -t i8 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 4 -num_heads_kv 2 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16,16,17,1,30,40,38,12 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_7
// CHECK_7: -t i8 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 4 -num_heads_kv 2 -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -causal -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_8
// CHECK_8: -t f16 -transQ false -transK false -transV false -transO false -causal true -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -return_lse -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_9
// CHECK_9: -t f16 -transQ false -transK false -transV false -transO false -causal false -return_lse true -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -return_lse -split_kv 8 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_10
// CHECK_10: -t f16 -transQ false -transK false -transV false -transO false -causal false -return_lse true -split_kv 8 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false

// Attention scale/bias/transposed-bias fusion is part of the problem key. These
// flags must round-trip through --emit-tuning-key (kept in sync with
// AttentionConfiguration in perfRunner.py).
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 --with-attn-scale --with-attn-bias | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_SCALE_BIAS
// CHECK_SCALE_BIAS: -t f32 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale true -with-attn-bias true -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 --with-attn-scale | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_SCALE_ONLY
// CHECK_SCALE_ONLY: -head_dim_v 32 -with-attn-scale true -with-attn-bias false -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 --with-attn-bias | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_BIAS_ONLY
// CHECK_BIAS_ONLY: -head_dim_v 32 -with-attn-scale false -with-attn-bias true -transBias false
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 --with-attn-bias --transBias | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_TRANS_BIAS
// CHECK_TRANS_BIAS: -head_dim_v 32 -with-attn-scale false -with-attn-bias true -transBias true
// Quantized (i8) dequantization inputs must not be mistaken for attn scale/bias.
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 --with-attn-scale --with-attn-bias | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_I8_SCALE_BIAS
// CHECK_I8_SCALE_BIAS: -t i8 {{.*}} -head_dim_v 32 -with-attn-scale true -with-attn-bias true -transBias false
// The same i8 kernel without scale/bias must classify both as false, proving the
// dequantization mul/add are not counted as attn scale/bias.
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_I8_NO_SCALE_BIAS
// CHECK_I8_NO_SCALE_BIAS: -t i8 {{.*}} -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false

// sliding_window_size is only emitted when set. current_seq_len remains runtime
// data and defaults to seq_len_k - 1 when this key is reconstructed for tuning.
// RUN: rocmlir-gen --arch gfx942 --operation attention -sliding_window_size 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_SW
// CHECK_SW: -t f16 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -sliding_window_size 8 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias false -transBias false

// Sliding-window and transposed-bias fields are independent and have stable
// relative positions in the attention tuning key.
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16 -sliding_window_size 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 1 --with-attn-bias --transBias | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_SW_TRANSBIAS
// CHECK_SW_TRANSBIAS: -split_kv 1 -sliding_window_size 8 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -with-attn-scale false -with-attn-bias true -transBias true

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 256 --perf_config=v3:32,256,2,32,32,4,1,1,2,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_DEPTHWISE_CONV
// CHECK_DEPTHWISE_CONV: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 256 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 256

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --perf_config=v3:32,256,2,32,32,4,1,1,2,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_GROUP_CONV
// CHECK_GROUP_CONV: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 256 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 128

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 512 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --perf_config=v3:32,256,2,32,32,4,1,1,2,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_GROUP_CONV2
// CHECK_GROUP_CONV2: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 512 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 128

// Checking numCU

// RUN: rocmlir-gen --arch gfx942 --num_cu 304 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_NUMCU
// CHECK_NUMCU: 304

// RUN: rocmlir-gen --arch gfx950 --operation gemm --scaledGemm --transB --transScaleB -g 12 -m 256 -n 256 -k 64 -t f4E2M1FN -out_dtype f32 | rocmlir-gen --emit-tuning-key - | FileCheck %s --check-prefixes=CHECK_SCALED_GEMM
// CHECK_SCALED_GEMM: -t f4E2M1FN -out_datatype f32 -transA false -transB true -scaledGemm -scale_a_dtype f8E8M0FNU -scale_b_dtype f8E8M0FNU -transScaleA false -transScaleB true -g 12 -m 256 -n 256 -k 64

// Checking numCU and numChiplets

// RUN: rocmlir-gen --arch gfx942 --num_cu 80 --num_chiplets 4 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_NUMCHIPLETS
// CHECK_NUMCHIPLETS: 80 4
