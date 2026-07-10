// The direct-to-LDS gather (amdgpu.gather_to_lds) drops out-of-bounds writes
// instead of writing 0, so a padded tile would be left uninitialized in LDS and
// poison the second attention GEMM (NaN*0 = NaN). For direct-to-LDS loads that
// can go out of bounds (non-trivial padding), the LDS tile must be
// cooperatively zeroed (rock.blockwise_fill) before the gather. Aligned loads
// need no zeroing and must not pay for it.

// Padded: seq_len_k (20) is not a multiple of the K-per-block, so the V load
// can go out of bounds and the LDS tile must be zeroed first.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- -operation attention -t f16 -seq_len_q 64 -seq_len_k 20 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --perf_config=attn:v3:32,32,32,8,32,32,16,4,1,4,2,2,1 | rocmlir-driver --kernel-pipeline=applicability - --mlir-print-ir-after=rock-blockwise-load-tile-to-threadwise 2>&1 | FileCheck %s --check-prefix=PADDED
// PADDED: rock.blockwise_fill

// Aligned: all dims are multiples of the tile, no out-of-bounds is possible, so
// no LDS zeroing is emitted.
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- -operation attention -t f16 -seq_len_q 64 -seq_len_k 64 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --perf_config=attn:v3:32,32,32,8,32,32,16,4,1,4,2,2,1 | rocmlir-driver --kernel-pipeline=applicability - --mlir-print-ir-after=rock-blockwise-load-tile-to-threadwise 2>&1 | FileCheck %s --check-prefix=ALIGNED
// ALIGNED-NOT: rock.blockwise_fill
