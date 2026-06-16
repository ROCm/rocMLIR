// --emit-features prints the learned-model feature vector(s) as CSV using the
// same C++ feature extraction as smart tuning, so it is the parity source of
// truth for the offline trainer. The header row is the canonical feature names
// (prefixed by the perf_config column), followed by one row per config.

// Default mode enumerates the exhaustive applicable space: header + >=1 row.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f16 -m 1024 -k 768 -n 512 --emit-features | FileCheck %s --check-prefix=GEMM
// GEMM: perf_config,trans_a,trans_b,g,m,n,k,log_m,log_n,log_k
// GEMM: v{{[0-9]+}}:{{[0-9].*}}

// Single-config mode: pipe one config from the tuning space into stdin
// (-perf_config=-) and get exactly that row back.
// RUN: rocmlir-gen --arch gfx942 --operation=gemm -t f16 -m 1024 -k 768 -n 512 --emit-tuning-space=quick | head -n 1 | rocmlir-gen --arch gfx942 --operation=gemm -t f16 -m 1024 -k 768 -n 512 -perf_config=- --emit-features | FileCheck %s --check-prefix=STDIN
// STDIN: perf_config,trans_a,trans_b
// STDIN-COUNT-1: v{{[0-9]+}}:
// STDIN-NOT: v{{[0-9]+}}:

// Conv features.
// RUN: rocmlir-gen --arch gfx942 --operation conv -t f32 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=1 -in_channels=64 -out_channels=64 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --emit-features | FileCheck %s --check-prefix=CONV
// CONV: perf_config,is_fwd,is_bwd,is_wrw,n,c,h,w,k,y,x
// CONV: v{{[0-9]+}}:

// Attention features (configs are emitted as "attn:vN:...").
// RUN: rocmlir-gen --arch gfx942 --operation=attention -t f16 -g 1 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 8 -num_heads_kv 8 -seq_len_q 1024 -seq_len_k 1024 --emit-features | FileCheck %s --check-prefix=ATTN
// ATTN: perf_config,
// ATTN: attn:v{{[0-9]+}}:
