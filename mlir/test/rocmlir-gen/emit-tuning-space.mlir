// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-NAVI
// CHECK-NAVI: v3:64,32,32,4,2,4,1,1,2

// RUN: rocmlir-gen --arch gfx90a --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=104 --emit-tuning-space=full | FileCheck %s --check-prefixes=CHECK-MI
// CHECK-MI: v4:64,64,8,32,32,16,4,4,1,2,0,0,1,1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-DOUBLEBUFFER
// CHECK-EXHAUSTIVE-DOUBLEBUFFER: v4:64,64,8,16,16,16,4,4,2,2,0,0,1,1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-DIRECTTOLDS-SINGLE
// CHECK-EXHAUSTIVE-DIRECTTOLDS-SINGLE: v4:64,64,8,16,16,16,4,4,3,2,0,0,1,1

// RUN: rocmlir-gen --arch gfx950 --operation=gemm -t f32 -g 1 -m 64 -k 128 -n 64 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-DIRECTTOLDS-DOUBLE
// CHECK-EXHAUSTIVE-DIRECTTOLDS-DOUBLE: v4:64,64,8,16,16,16,4,4,4,2,0,0,1,1

// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN-DIRECTTOLDS-SINGLE
// CHECK-EXHAUSTIVE-ATTN-DIRECTTOLDS-SINGLE: attn:v3:32,64,32,16,32,32,16,8,1,3,2,0,1

// RUN: rocmlir-gen --arch gfx950 --operation=attention -t f32 -g 1 -head_dim_qk 32 -head_dim_v 32 -num_heads_q 128 -num_heads_kv 128 -seq_len_q 1024 -seq_len_k 1024 --num_cu=256 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-EXHAUSTIVE-ATTN-DIRECTTOLDS-DOUBLE
// CHECK-EXHAUSTIVE-ATTN-DIRECTTOLDS-DOUBLE: attn:v3:32,64,32,16,32,32,16,8,1,4,2,0,1

// RUN: rocmlir-gen -p --arch gfx1100 --operation=attention -t f16 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-SCHEDULING-ATTENTION
// CHECK-SCHEDULING-ATTENTION: attn:v3:128,128,256,64,32,32,16,16,1,2,2,0,1

// RUN: rocmlir-gen -p --arch gfx1100 --operation=gemm -t f16 --emit-tuning-space=exhaustive | FileCheck %s --check-prefixes=CHECK-SCHEDULING-GEMM
// CHECK-SCHEDULING-GEMM: v4:256,256,8,64,128,16,16,1,2,2,0,0,1,1
