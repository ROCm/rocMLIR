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

// Zero is not a look-back distance and must not act as a disable sentinel.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=32 -sliding_window_look_back=0 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_LOOK_BACK_ZERO
// ERR_SLIDING_LOOK_BACK_ZERO: sliding_window_look_back must be -1 or a positive integer

// Values below the public -1 sentinel are invalid.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=32 -sliding_window_look_back=-2 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_LOOK_BACK_NEG
// ERR_SLIDING_LOOK_BACK_NEG: sliding_window_look_back must be -1 or a positive integer

// A look-back larger than seq_len_k - 1 is rejected by the Rock verifier; catch
// it in the driver too so the error is reported up front rather than after
// lowering.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=32 -sliding_window_look_back=64 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_LOOK_BACK_TOO_LARGE
// ERR_SLIDING_LOOK_BACK_TOO_LARGE: sliding_window_look_back must not exceed seq_len_k - 1

// The look-back is materialized in i32 attributes and constants.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=32 -sliding_window_look_back=2147483648 2>&1 | FileCheck %s --check-prefix=ERR_SLIDING_LOOK_BACK_I32
// ERR_SLIDING_LOOK_BACK_I32: sliding_window_look_back must fit in a 32-bit integer

// P is an inclusive index, so negative values and P == seq_len_k are invalid.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=-1 2>&1 | FileCheck %s --check-prefix=ERR_LAST_VALID_NEG
// ERR_LAST_VALID_NEG: last_valid_kv_index values must satisfy 0 <= P < seq_len_k
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=64 2>&1 | FileCheck %s --check-prefix=ERR_LAST_VALID_TOO_LARGE
// ERR_LAST_VALID_TOO_LARGE: last_valid_kv_index values must satisfy 0 <= P < seq_len_k

// Each attention group requires exactly one last-valid K/V index.
// RUN: not rocmlir-gen --arch %arch --operation attention -t f16 -g 2 -seq_len_q 1 -seq_len_k 64 -head_dim_qk 32 -head_dim_v 32 -last_valid_kv_index=32 2>&1 | FileCheck %s --check-prefix=ERR_LAST_VALID_COUNT
// ERR_LAST_VALID_COUNT: last_valid_kv_index must contain one value per group (expected 2, got 1)

// Attention, gemm+gemm, and conv+gemm pipelines require -t (dataTypeAlias).
// RUN: not rocmlir-gen --arch %arch --operation attention -seq_len_q 256 -seq_len_k 256 -head_dim_qk 32 -head_dim_v 32 2>&1 | FileCheck %s --check-prefix=ERR_NO_DTYPE
// ERR_NO_DTYPE: Type of the attention/gemm+gemm/conv+gemm operation is not specified

// Passing both `padding_h` and `padding_h_l/r` with mismatched values logs a
// warning. rocmlir-gen does not abort so we just match the diagnostic on stderr.
// RUN: rocmlir-gen --arch %arch -p -padding_h 2 -padding_h_l 1 2>&1 | FileCheck %s --check-prefix=WARN_PADDING_H
// WARN_PADDING_H: you can't use both padding_h and (padding_h_l,padding_h_r).
// RUN: rocmlir-gen --arch %arch -p -padding_w 2 -padding_w_r 1 2>&1 | FileCheck %s --check-prefix=WARN_PADDING_W
// WARN_PADDING_W: you can't use both padding_w and (padding_w_l,padding_w_r).
