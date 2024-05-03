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

static Value rearrangeWorkgroupsForXCC(Location loc, PatternRewriter &b,
                                       Value bid, int64_t gridSize,
                                       int64_t numChipletsPerGroup) {
  Value numChipletsVal =
      b.createOrFold<ConstantIndexOp>(loc, numChipletsPerGroup);
  int64_t wgsPerChiplet = (gridSize) / numChipletsPerGroup;
  Value wgsPerChipletVal = b.createOrFold<ConstantIndexOp>(loc, wgsPerChiplet);
  Value logicalChipletId = b.create<RemUIOp>(loc, bid, numChipletsVal);
  Value wgIdPerLogicalChiplet = b.create<DivUIOp>(loc, bid, numChipletsVal);
  Value rearrangedBid = b.create<AddIOp>(
      loc, wgIdPerLogicalChiplet,
      b.create<MulIOp>(loc, logicalChipletId, wgsPerChipletVal));
  int64_t lastNumChipletMultiple =
      (gridSize - 1) - (gridSize % numChipletsPerGroup);
  Value lastNumChipletMultipleVal =
      b.createOrFold<ConstantIndexOp>(loc, lastNumChipletMultiple);
  Value isBidLargerThanlastNumChipletMultiple = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, bid, lastNumChipletMultipleVal);
  bid = b.create<arith::SelectOp>(loc, isBidLargerThanlastNumChipletMultiple,
                                  bid, rearrangedBid);
  return bid;
}

GridCoordinates rock::layout::makeGroupedGridLayout(PatternRewriter &b,
                                                    Location loc, Value bid,
                                                    GridLayoutInfo info,
                                                    StringRef arch) {
  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  int64_t numChiplets = rock::lookupArchInfo(arch).maxNumXCC;
  if (numChiplets > 1) {
    // It was emphircally found that two chiplets as a group
    // computing a spatial mxn tile has better locality throughout.
    int64_t numChipletsPerGroup = std::ceil(numChiplets / 2);
    int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChipletsPerGroup);
  }

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
  Value mnBlocks =
      b.createOrFold<ConstantIndexOp>(loc, info.mBlocks * info.nBlocks);
  Value g_block = b.create<DivUIOp>(loc, bid, mnBlocks);
  bid = b.create<RemUIOp>(loc, bid, mnBlocks);

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

GridCoordinates rock::layout::makeGxMxNGridLayout(PatternRewriter &b,
                                                  Location loc, Value bid,
                                                  GridLayoutInfo info) {
  Value g1MxNBlockCountVal =
      b.createOrFold<ConstantIndexOp>(loc, info.mBlocks * info.nBlocks);
  Value g1NBlockCountVal = b.createOrFold<ConstantIndexOp>(loc, info.nBlocks);
  Value gBlockIdx = b.create<arith::DivUIOp>(loc, bid, g1MxNBlockCountVal);
  Value nonGBlockIdx = b.create<arith::RemUIOp>(loc, bid, g1MxNBlockCountVal);
  Value mBlockIdx =
      b.create<arith::DivUIOp>(loc, nonGBlockIdx, g1NBlockCountVal);
  Value nBlockIdx =
      b.create<arith::RemUIOp>(loc, nonGBlockIdx, g1NBlockCountVal);

  return {gBlockIdx, mBlockIdx, nBlockIdx};
}

GridCoordinates rock::layout::makeGxNGridLayout(PatternRewriter &b,
                                                Location loc, Value bid,
                                                Value mIter, int64_t nBlocks,
                                                int64_t gridSize,
                                                StringRef arch) {
  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  int64_t numChiplets = rock::lookupArchInfo(arch).maxNumXCC;
  if (numChiplets > 1) {
    // It was emphircally found that two chiplets as a group
    // computing a spatial mxn tile has better locality throughout.
    int64_t numChipletsPerGroup = std::ceil(numChiplets / 2);
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChipletsPerGroup);
  }

  Value g1NBlockCountVal = b.createOrFold<ConstantIndexOp>(loc, nBlocks);
  Value gBlockIdx = b.create<arith::DivUIOp>(loc, bid, g1NBlockCountVal);
  Value nBlockIdx = b.create<arith::RemUIOp>(loc, bid, g1NBlockCountVal);
  return {gBlockIdx, mIter, nBlockIdx};
}
