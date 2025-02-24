// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f32 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_1
// CHECK_1: -t f32 -transQ false -transK false -transV false -transO false -g 1 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_2
// CHECK_2: -t f16 -transQ false -transK false -transV false -transO false -g 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_3
// CHECK_3: -t i8 -transQ false -transK false -transV false -transO false -g 8 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_4
// CHECK_4: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_5
// CHECK_5: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_6
// CHECK_6: -t i8 -transQ false -transK false -transV false -transO false -g 4 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32
// RUN: rocmlir-gen --arch gfx942 --operation attention -current_seq_len=16,16,17,1,30,40,38,12 -num_heads_q 4 -num_heads_kv 2 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t i8 -g 8 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_7
// CHECK_7: -t i8 -transQ false -transK false -transV false -transO false -g 32 -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 256 --kernel-repeats 1 --perf_config=v2:32,256,2,32,32,4,1,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_DEPTHWISE_CONV
// CHECK_DEPTHWISE_CONV: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 256 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 256

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 256 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --kernel-repeats 1 --perf_config=v2:32,256,2,32,32,4,1,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_GROUP_CONV
// CHECK_GROUP_CONV: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 256 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 128

// RUN: rocmlir-gen --arch gfx942 --operation conv -t f16 --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 64 --in_channels 256 --in_h 20 --in_w 20 --out_channels 512 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 3 --padding_w 3 --groupsize 128 --kernel-repeats 1 --perf_config=v2:32,256,2,32,32,4,1,1,1 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_GROUP_CONV2
// CHECK_GROUP_CONV2: convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 64 -c 256 -H 20 -W 20 -k 512 -y 7 -x 7 -p 3 -q 3 -u 1 -v 1 -l 1 -j 1 -g 128

// Checking numCU

// RUN: rocmlir-gen --arch gfx942 --num_cu 304 --operation attention -seq_len_q 256 -seq_len_k 512 -head_dim_qk 64 -head_dim_v 32 -t f16 -g 4 | rocmlir-gen --emit-tuning-key - | FileCheck %s  --check-prefixes=CHECK_NUMCU
// CHECK_NUMCU: 304
