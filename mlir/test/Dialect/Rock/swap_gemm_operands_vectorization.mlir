// This test verifies that SwapGemmOperands improves store vectorization.
// After the swap, the C matrix is written through a transposed view, which
// makes the store contiguous along the fast-changing dimension, enabling
// wider vectorized stores.
//
// We check the rock.global_store `length` attribute which indicates how many
// elements are stored per global store operation.

// ============================================================
// f16: 256x256x256 - from GemmVariants test suite
// ============================================================

// With swap: length should be 4 (4 f16 = 8 bytes = dwordx2)
// RUN: rocmlir-gen --operation gemm -m 256 -n 256 -k 256 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-swap-gemm-operands \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=SWAP-F16

// Without swap: length should be 1 (scalar store)
// RUN: rocmlir-gen --operation gemm -m 256 -n 256 -k 256 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=NOSWAP-F16

// SWAP-F16: rock.global_store
// SWAP-F16-SAME: length = 4
// NOSWAP-F16: rock.global_store
// NOSWAP-F16-SAME: length = 1

// ============================================================
// bf16: 128x256x512
// ============================================================

// With swap: length should be 4 (4 bf16 = 8 bytes)
// RUN: rocmlir-gen --operation gemm -m 128 -n 256 -k 512 -g 1 --arch gfx950 -t bf16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-swap-gemm-operands \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=SWAP-BF16

// Without swap: length should be 1
// RUN: rocmlir-gen --operation gemm -m 128 -n 256 -k 512 -g 1 --arch gfx950 -t bf16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=NOSWAP-BF16

// SWAP-BF16: rock.global_store
// SWAP-BF16-SAME: length = 4
// NOSWAP-BF16: rock.global_store
// NOSWAP-BF16-SAME: length = 1

// ============================================================
// f16: asymmetric 128x256x512 - common ML dimension
// ============================================================

// With swap: length should be 4
// RUN: rocmlir-gen --operation gemm -m 128 -n 256 -k 512 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-swap-gemm-operands \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=SWAP-F16-ASYM

// Without swap: length should be 1
// RUN: rocmlir-gen --operation gemm -m 128 -n 256 -k 512 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=NOSWAP-F16-ASYM

// SWAP-F16-ASYM: rock.global_store
// SWAP-F16-ASYM-SAME: length = 4
// NOSWAP-F16-ASYM: rock.global_store
// NOSWAP-F16-ASYM-SAME: length = 1

// ============================================================
// f16: large asymmetric 256x512x1024 - stress test
// ============================================================

// With swap: length should be 4
// RUN: rocmlir-gen --operation gemm -m 256 -n 512 -k 1024 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-swap-gemm-operands \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=SWAP-F16-LARGE

// Without swap: length should be 1
// RUN: rocmlir-gen --operation gemm -m 256 -n 512 -k 1024 -g 1 --arch gfx950 -t f16 \
// RUN:   | rocmlir-opt --rock-affix-params --rock-conv-to-gemm \
// RUN:     --rock-gemm-to-gridwise --rock-regularize \
// RUN:     --rock-gridwise-gemm-to-blockwise \
// RUN:     --rock-blockwise-load-tile-to-threadwise \
// RUN:     --rock-linalg-align --rock-blockwise-gemm-to-threadwise \
// RUN:     --rock-pipeline --canonicalize --convert-linalg-to-affine-loops \
// RUN:     --rock-vectorize-fusions --rock-add-async-wait \
// RUN:     --rock-annotate-liveness --rock-reuse-lds \
// RUN:     --rock-threadwise-gemm-lowering \
// RUN:   | FileCheck %s --check-prefix=NOSWAP-F16-LARGE

// SWAP-F16-LARGE: rock.global_store
// SWAP-F16-LARGE-SAME: length = 4
// NOSWAP-F16-LARGE: rock.global_store
// NOSWAP-F16-LARGE-SAME: length = 1
