//===- GridLayoutEmitter.cpp - MLIR helper that contains the layout logic -===//
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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "llvm/Support/Debug.h"

#include "GridLayoutEmitter.h"

#define DEBUG_TYPE "rock-grid-layout-emitter"

using namespace mlir;
using namespace mlir::rock;
using namespace mlir::arith;
using namespace mlir::rock::layout;

GridCoordinates rock::layout::makeGroupedGridLayout(PatternRewriter &b,
                                                    Location loc, Value bid,
                                                    GridLayoutInfo info) {

  // Heurisitc to compute groupSize
  // This also covers the cases where the output width is larger
  // than the input width
  int64_t bitWidthIn = info.inputType.getIntOrFloatBitWidth();
  int64_t bitWidthOut = info.outputType.getIntOrFloatBitWidth();
  int64_t groupSize =
      std::ceil(std::sqrt(info.numCU)) * (bitWidthOut / bitWidthIn);

  Value mBlocksPerGroup = b.createOrFold<ConstantIndexOp>(loc, groupSize);
  Value blocksPerGroup =
      b.createOrFold<ConstantIndexOp>(loc, groupSize * info.nBlocks);
  Value mBlocksValue = b.createOrFold<ConstantIndexOp>(loc, info.mBlocks);

  // Compute g_block first and the bid in the actual group g_block
  Value gridSize =
      b.createOrFold<ConstantIndexOp>(loc, info.mBlocks * info.nBlocks);
  Value g_block = b.create<DivUIOp>(loc, bid, gridSize);
  bid = b.create<RemUIOp>(loc, bid, gridSize);

  // Group together the workgroups in g_block
  Value groupId = b.create<DivUIOp>(loc, bid, blocksPerGroup);
  Value firstBidM = b.create<MulIOp>(loc, groupId, mBlocksPerGroup);
  Value thisMBlocksPerGroup = b.create<MinUIOp>(
      loc, b.create<SubIOp>(loc, mBlocksValue, firstBidM), mBlocksPerGroup);
  Value m_block = b.create<AddIOp>(
      loc, firstBidM, b.create<RemUIOp>(loc, bid, thisMBlocksPerGroup));
  Value n_block = b.create<DivUIOp>(
      loc, b.create<RemUIOp>(loc, bid, blocksPerGroup), thisMBlocksPerGroup);
  return {g_block, m_block, n_block};
}
