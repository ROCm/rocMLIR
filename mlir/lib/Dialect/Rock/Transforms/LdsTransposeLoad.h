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
// panel offsets, generating indices, and emitting calls to the LDS
// transpose load operation in a MFMA-friendly layout.
//
// It is intended to simplify the IR generation logic and ensure
// consistent handling of f16/bf16 panel loads from LDS memory.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H
#define MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::rock::hwtranspose {

// Operand selector (A or B matrix)
enum class OperandKind { A, B };

// Simplified layout kinds
enum class LayoutKind { None, L16x32, L32x16, L16x16, L32x8 };

// Shape of a single MFMA instruction this load cooperates with.
struct MfmaInstrShape {
  int64_t mnMfma;
  int64_t kMfma;
};

// Structure to hold the outcome of the hardware transpose analysis.
struct Decision {
  bool usable{false};
  LayoutKind layout{LayoutKind::None};
  OperandKind operand{OperandKind::A};
  int64_t mPerBlock{1};
  int64_t nPerBlock{1};
  int64_t kPerBlock{1};
  int64_t mPerWave{1};
  int64_t nPerWave{1};
  bool doubleBuffering{false};
};

// Determines whether hardware-assisted LDS transpose optimization can be
// applied for the given GEMM configuration. Checks architecture support,
// data types, MFMA instruction shape, and buffer layout compatibility.
// Returns a Decision struct indicating applicability and selected layout.
Decision makeDecision(StringRef arch, Type elemTypeA, Type elemTypeB,
                      bool DirectToLds, const MfmaInstrShape &shape,
                      OperandKind operand, int64_t mPerBlock, int64_t nPerBlock,
                      int64_t kPerBlock, int64_t mPerWave, int64_t nPerWave,
                      bool doubleBuffering);

// Select a layout kind based on the MFMA instruction shape.
LayoutKind selectLayout(int64_t nonKDim, int64_t instrK);

// Attach attributes to the ThreadwiseReadIntoOp based on the decision.
DictionaryAttr buildTransposeAttr(const Decision &dec, bool isOperandA,
                                  PatternRewriter &rewriter);

// Lowering-time description.
// Set to true once all required attributes are present so lowering may proceed.
struct LoweringInfo {
  bool usable{false};
  LayoutKind layout{LayoutKind::None};
  OperandKind operand{OperandKind::A};
  Type elemType{};
  int64_t mPerBlock{1};
  int64_t nPerBlock{1};
  int64_t kPerBlock{1};
  int64_t mPerWave{1};
  int64_t nPerWave{1};
  bool doubleBuffering{false};
};

// Derives lowering information from the attributes of a ThreadwiseReadIntoOp.
LoweringInfo deriveLoweringInfo(ThreadwiseReadIntoOp op, PatternRewriter &b);

// Emits the actual hardware transpose load sequence.
LogicalResult emitThreadwiseHWTranspose(ThreadwiseReadIntoOp op,
                                        const LoweringInfo &info,
                                        PatternRewriter &b, int64_t blockSize,
                                        int64_t waveSize);

// Utility to get the string name of a layout.
StringRef layoutName(LayoutKind kind);

} // namespace mlir::rock::hwtranspose

#endif // MLIR_LIB_DIALECT_ROCK_TRANSFORMS_LDS_TRANSPOSE_LOAD_H
