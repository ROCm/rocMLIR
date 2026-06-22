// Full-lowering sanity coverage for accelerated (MFMA) data types and
// operations that the existing sanity tests do not exercise end to end:
// sanity_xdlops.mlir only covers gfx908 GEMM (f16/bf16/i8), fp8_ops.mlir only
// lowers GEMM to gpu,rocdl, and conv backward / attention have no full gpu
// pipeline coverage. Each config is lowered through the Rock kernel pipeline
// (-kernel-pipeline=gpu, which runs every Rock transform) and then all the way
// to a GPU binary; --verify-passes plus a final rocmlir-opt parse asserts that
// every step produces valid IR. No GPU execution is required.

// ---- GEMM: fp8 / bf8 on MFMA ----
// RUN: rocmlir-gen --arch gfx942 --operation gemm -g 1 -m 128 -k 128 -n 128 -t fp8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation gemm -g 1 -m 128 -k 128 -n 128 -t fp8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation gemm -g 1 -m 128 -k 128 -n 128 -t bf8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation gemm -g 1 -m 128 -k 128 -n 128 -t bf8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt

// ---- Convolution: forward fp8 ----
// RUN: rocmlir-gen --arch gfx942 --operation conv -t fp8 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv -t fp8 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt

// ---- Convolution: backward-data (f32, f16) ----
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -t f32 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -t f32 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt

// ---- Convolution: backward-weight (f16) ----
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_weight -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_weight -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt

// ---- Attention (f16, f32) ----
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f32 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -t f32 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt

// ---- GEMM fp8 on gfx950 (distinct accel lowering from gfx942) ----
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 128 -k 128 -n 128 -t fp8 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 1 -m 128 -k 128 -n 128 -t fp8 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx950 | rocmlir-opt

// ---- WMMA path: GEMM and convolution on gfx1100 ----
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -g 1 -m 128 -k 128 -n 128 -t f16 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -g 1 -m 128 -k 128 -n 128 -t f16 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx1100 | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx1100 --operation conv -t f16 -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx1100 | rocmlir-opt

// ---- Convolution with NCHW/KCYX layout (exercises layout normalization) ----
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f32 -fil_layout kcyx -in_layout nchw -out_layout nkhw -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu --verify-passes | rocmlir-opt
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f32 -fil_layout kcyx -in_layout nchw -out_layout nkhw -batchsize 4 -in_channels 32 -out_channels 32 -in_h 14 -in_w 14 -fil_h 3 -fil_w 3 | rocmlir-driver -kernel-pipeline=gpu,binary --verify-passes --arch=gfx942 | rocmlir-opt
