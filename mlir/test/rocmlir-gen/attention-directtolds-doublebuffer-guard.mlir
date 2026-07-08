// Attention lowering must reject the DirectToLDSDoubleBuffer schedule (v4) for
// the second GEMM (P*V -> O) when its K tile is padded (seq_len_k not a multiple
// of gemm1KPerBlock) and head_dim_v spans more than one M block
// (head_dim_v > gemm1MPerBlock). That combination miscompiles the O tensor by
// reading uninitialized LDS in the double-buffered direct-to-LDS V load, so it
// must be reported as inapplicable instead of silently returning wrong results.

// Rejected: schedule v4, padded seq_len_k (20 not a multiple of gemm1KPerBlock=32),
// and head_dim_v 48 > gemm1MPerBlock 32 (=> more than one gemm1 M block).
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- -operation attention -t f16 -seq_len_q 64 -seq_len_k 20 -head_dim_qk 32 -head_dim_v 48 --with-attn-scale --perf_config=attn:v3:32,32,32,8,32,32,16,4,1,4,2,2,1 | not rocmlir-driver --kernel-pipeline=applicability - 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: DirectToLDSDoubleBuffer schedule is unsupported for attention

// Accepted: same shape with the single-stage direct-to-LDS schedule (v3) still
// lowers, confirming the guard is specific to the double-buffered variant.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- -operation attention -t f16 -seq_len_q 64 -seq_len_k 20 -head_dim_qk 32 -head_dim_v 48 --with-attn-scale --perf_config=attn:v3:32,32,32,8,32,32,16,4,1,3,2,2,1 | rocmlir-driver --kernel-pipeline=applicability - | FileCheck %s --check-prefix=OK
// OK: func.func @rock_attention
