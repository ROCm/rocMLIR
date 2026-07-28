// Negative tests for rocmlir-gen command-line option validation.

// --kernel-repeats only valid with -ph or -pv.
// RUN: not rocmlir-gen -operation gemm -t f32 -out_datatype f32 --arch %arch -g 1 -m 1024 -k 1024 -n 1024 -transA=False -transB=False --kernel-repeats=100 2>&1 | FileCheck %s --check-prefix=ERR_KERNEL_REPEATS
// ERR_KERNEL_REPEATS: --kernel-repeats is only supported with host harness (-ph) or CPU validation (-pv).

// --verifier=cpp is not implemented for any of the in-tree operations.
// RUN: not rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 64 -k 64 -n 64 -ph --verifier=cpp 2>&1 | FileCheck %s --check-prefix=ERR_CPP_GEMM
// ERR_CPP_GEMM: External gemm validator is not available

// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -ph --verifier=cpp 2>&1 | FileCheck %s --check-prefix=ERR_CPP_ATTN
// ERR_CPP_ATTN: External attention validator is not available

// --verifier=clone is only valid when wrapping an existing input module.
// RUN: not rocmlir-gen --arch %arch --operation gemm -t f32 -g 1 -m 64 -k 64 -n 64 --verifier=clone 2>&1 | FileCheck %s --check-prefix=ERR_CLONE_NO_INPUT
// ERR_CLONE_NO_INPUT: Clone validation is not compatible with kernel generation.

// --arch is mandatory; the driver must reject runs that omit it.
// RUN: not rocmlir-gen --operation gemm -t f32 -g 1 -m 64 -k 64 -n 64 2>&1 | FileCheck %s --check-prefix=ERR_NO_ARCH
// ERR_NO_ARCH: --arch is not set

// GEMM requires -g, -m, -k, -n. The detector walks {groupsize, m, k, n} in
// order; groupsize defaults to 1 so it passes the <=0 check, leaving `m` as
// the first missing arg the diagnostic reports.
// RUN: not rocmlir-gen --arch %arch --operation gemm -t f32 2>&1 | FileCheck %s --check-prefix=ERR_GEMM_MISSING
// ERR_GEMM_MISSING: Value for: m not specified

// Mixed input/filter dtypes require an explicit output dtype.
// RUN: not rocmlir-gen --arch %arch -p -fil_dtype f16 -in_dtype f32 2>&1 | FileCheck %s --check-prefix=ERR_MIXED_DTYPE
// ERR_MIXED_DTYPE: Missing output type for mixed input types

// Flash-decoding (split_kv > 1) needs return_lse; otherwise the driver bails out.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -split_kv 2 2>&1 | FileCheck %s --check-prefix=ERR_SPLITKV
// ERR_SPLITKV: If split-kv > 1 (flash decoding), we need to return LSE

// Transposed bias layout is only meaningful when an attention bias is present.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -transBias 2>&1 | FileCheck %s --check-prefix=ERR_TRANS_BIAS_WITHOUT_BIAS
// ERR_TRANS_BIAS_WITHOUT_BIAS: --transBias requires --with-attn-bias

// Sliding-window masking is relative to the KV-cache position.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -sliding_window_size=16 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_WINDOW
// ERR_SLIDING_WINDOW: sliding_window_size requires current_seq_len to be set

// A negative value is invalid; zero is the disabled value.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 -sliding_window_size=-16 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_WINDOW_NEG
// ERR_SLIDING_WINDOW_NEG: sliding_window_size must be non-negative

// The window cannot exceed the compile-time maximum key sequence length.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -current_seq_len=32 -sliding_window_size=128 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_WINDOW_TOO_LARGE
// ERR_SLIDING_WINDOW_TOO_LARGE: sliding_window_size must not exceed seq_len_k

// The window is materialized in i32 attributes and constants.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -current_seq_len=32 -sliding_window_size=2147483648 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_WINDOW_I32
// ERR_SLIDING_WINDOW_I32: sliding_window_size must fit in a 32-bit integer

// Attention, gemm+gemm, and conv+gemm pipelines require -t (dataTypeAlias).
// RUN: not rocmlir-gen --arch %arch --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 2>&1 | FileCheck %s --check-prefix=ERR_NO_DTYPE
// ERR_NO_DTYPE: Type of the attention/gemm+gemm/conv+gemm operation is not specified

// Passing both `padding_h` and `padding_h_l/r` with mismatched values logs a
// warning. rocmlir-gen does not abort so we just match the diagnostic on stderr.
// RUN: rocmlir-gen --arch %arch -p -padding_h 2 -padding_h_l 1 2>&1 | FileCheck %s --check-prefix=WARN_PADDING_H
// WARN_PADDING_H: you can't use both padding_h and (padding_h_l,padding_h_r).
// RUN: rocmlir-gen --arch %arch -p -padding_w 2 -padding_w_r 1 2>&1 | FileCheck %s --check-prefix=WARN_PADDING_W
// WARN_PADDING_W: you can't use both padding_w and (padding_w_l,padding_w_r).
