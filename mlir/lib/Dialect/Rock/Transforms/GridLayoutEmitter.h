//===- GridLayoutEmitter.h - MLIR helper that contains the layout logic -===//
//
// Copyright 2020 The MLIR Authors.
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
// This class tries to abstract away the code-generation details needed to
// generated calls to matrix multiply accelerator intrinsics (wmma, mfma).
//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H
#define MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"

namespace mlir {
namespace rock {
namespace layout {

/// Struct containing the {g,m,n} block coordinates of a block
/// with a given bid. I.e., block bid will compute C[g_block, m_block, n_block]
/// output
struct GridCoordinates {
  Value g_block;
  Value m_block;
  Value n_block;
};

/// Struct containing information that guide the layout heuristic selection
struct GridLayoutInfo {
  int64_t gBlocks;
  int64_t mBlocks;
  int64_t nBlocks;
  int64_t numCU;
  Type inputType;
  Type outputType;
};

/// This function emits the right triplet of <group,block_m,block_n> identifers,
/// given a flat blockId. This has been adapted from:
/// https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py
///
GridCoordinates makeGroupedGridLayout(PatternRewriter &b, Location loc,
                                      Value bid, GridLayoutInfo info);

GridCoordinates makeGroupedGridLayoutXCCMiddle(PatternRewriter &b, Location loc,
                                      Value bid, GridLayoutInfo info);

GridCoordinates makeGroupedGridLayoutXCCSlowest(PatternRewriter &b, Location loc,
                                      Value bid, GridLayoutInfo info);

GridCoordinates makeGxMxNGridLayout(PatternRewriter &b, Location loc, Value bid,
                                    GridLayoutInfo info);

} // namespace layout
} // namespace rock
} // namespace mlir

#endif // MLIR_LIB_DIALECT_ROCK_TRANSFORMS_MLIR_LAYOUT_EMITTER_H
