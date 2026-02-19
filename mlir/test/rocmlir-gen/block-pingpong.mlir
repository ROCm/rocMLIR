// Test --use-block-pingpong flag sets the rock.use_block_pingpong attribute

// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f16 -g 1 -m 512 -n 512 -k 256 --use-block-pingpong | FileCheck %s --check-prefix=GEMM_PINGPONG
// GEMM_PINGPONG: func.func @rock_gemm
// GEMM_PINGPONG-SAME: rock.use_block_pingpong

// RUN: rocmlir-gen --arch gfx90a --operation gemm -t f16 -g 1 -m 512 -n 512 -k 256 | FileCheck %s --check-prefix=GEMM_NO_PINGPONG
// GEMM_NO_PINGPONG: func.func @rock_gemm
// GEMM_NO_PINGPONG-NOT: rock.use_block_pingpong

// RUN: rocmlir-gen --arch gfx90a --operation attention -t f16 --seq_len_q 64 --seq_len_k 64 --head_dim_qk 128 --head_dim_v 128 --num_heads_q 8 --num_heads_kv 8 --use-block-pingpong | FileCheck %s --check-prefix=ATTN_PINGPONG
// ATTN_PINGPONG: func.func @rock_attention
// ATTN_PINGPONG-SAME: rock.use_block_pingpong
