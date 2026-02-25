//===- LdsTransposeLoad.h - MLIR helper for rock.lds_transpose_load -------===//
//
// Copyright 2025 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines helper functions for MLIR code generation related to
// rock.lds_transpose_load operations. It provides utilities for computing
// matrix accelerator tile offsets, generating indices, and emitting calls
// to the LDS transpose load operation in an accelerator-friendly layout.
//
// It is intended to simplify the IR generation logic and ensure
// consistent handling of f16/bf16/fp8/bf8 matrix accelerator tile loads from
// LDS memory.
//
// Supported element types:
// - f16, bf16: uses ds_read_tr16_b64 (returns 4 elements per thread)
// - f8E4M3FN, f8E5M2 (OCP FP8): uses ds_read_tr8_b64 (returns 8 elements)
//
// Supported MFMA geometries:
// - Standard: (16,16), (16,32), (32,8), (32,16) - single-rate or double-rate
// - Scaled FP8: (16,128) - quad-rate (4 ds_read_tr8 calls per K tile)
// - Scaled FP8: (32,64) - quad-rate (4 ds_read_tr8 calls per K tile)
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H
#define MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H

#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::rock::hwtranspose {

// Operand selector (A or B matrix)
enum class OperandKind { A, B };

// Build LDS transpose config attribute from already-computed MFMA params.
// Used in BlockwiseLoadTileToThreadwise when decision was made upstream.
// Requires mfmaDDim > 0 and mfmaKDim > 0 (asserted).
// Valid combinations: (16,16), (16,32), (16,128), (32,8), (32,16), (32,64)
LDSTransposeConfigAttr buildTransposeAttrFromParams(
    PatternRewriter &rewriter, int64_t mfmaDDim, int64_t mfmaKDim,
    int64_t mPerBlock, int64_t nPerBlock, int64_t kPerBlock, int64_t mPerWave,
    int64_t nPerWave, bool doubleBuffering, bool isOperandA);

// Emits the actual hardware transpose load sequence.
// Reads configuration directly from the op's LDSTransposeConfigAttr.
LogicalResult emitThreadwiseHWTranspose(PatternRewriter &b,
                                        ThreadwiseReadIntoOp op,
                                        int64_t blockSize, int64_t waveSize);

// Result of LDS transpose decision making for both operands
struct LDSTransposeDecision {
  bool enableA{false}; // Enable for operand A
  bool enableB{false}; // Enable for operand B
  int64_t mfmaDDim{0}; // MFMA D dimension (M or N, 16 or 32)
  int64_t mfmaKDim{0}; // MFMA K dimension (8, 16, 32, 64, or 128)
};

// Decides whether to enable LDS transpose for operands A and B
// based on architecture, MFMA geometry, kpack constraints, and layout config.
// Parameters:
//   - bLoadsFromLDS: Whether operand B actually loads from LDS.
//     If false (e.g., Q matrix prefetched to registers), B will be disabled
//     for LDS transpose regardless of other constraints.
LDSTransposeDecision decideLDSTransposeForOperands(
    const rock::accel::AccelEmitter *accelEmitter, StringRef arch,
    Type elementTypeA, Type elementTypeB, bool directToLDS,
    const LDSLayoutConfigDim &ldsLayoutConfigA,
    const LDSLayoutConfigDim &ldsLayoutConfigB, int64_t mPerBlock,
    int64_t nPerBlock, int64_t kPerBlock, int64_t mPerWave, int64_t nPerWave,
    int64_t kpack, bool doubleBuffering, bool bLoadsFromLDS = true);

} // namespace mlir::rock::hwtranspose

#endif // MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H
