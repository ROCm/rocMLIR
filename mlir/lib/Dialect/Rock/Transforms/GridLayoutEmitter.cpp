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
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
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
  Value logicalChipletId = RemUIOp::create(b, loc, bid, numChipletsVal);
  Value wgIdPerLogicalChiplet = DivUIOp::create(b, loc, bid, numChipletsVal);
  Value rearrangedBid = AddIOp::create(
      b, loc, wgIdPerLogicalChiplet,
      MulIOp::create(b, loc, logicalChipletId, wgsPerChipletVal));
  int64_t lastNumChipletMultiple =
      (gridSize - 1) - (gridSize % numChipletsPerGroup);
  Value lastNumChipletMultipleVal =
      b.createOrFold<ConstantIndexOp>(loc, lastNumChipletMultiple);
  Value isBidLargerThanlastNumChipletMultiple = arith::CmpIOp::create(
      b, loc, arith::CmpIPredicate::sgt, bid, lastNumChipletMultipleVal);
  bid = arith::SelectOp::create(b, loc, isBidLargerThanlastNumChipletMultiple,
                                bid, rearrangedBid);
  return bid;
}

static int64_t getNumChiplets(StringRef arch, int64_t numCU) {
  int64_t numChiplets = rock::lookupArchInfo(arch).maxNumXCC;
  // TODO: hack until we find a better way to determine number of chiplets
  if (arch.contains("gfx942") && numCU == 80) {
    numChiplets = 4;
  }
  if (arch.contains("gfx942") && numCU == 20) {
    numChiplets = 1;
  }
  return numChiplets;
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
  int64_t numChiplets = getNumChiplets(arch, info.numCU);
  if (numChiplets > 1) {
    // It was empirically found that two chiplets as a group
    // computing a spatial mxn tile has better locality throughout.
    int64_t numChipletsPerGroup = std::ceil(numChiplets / 2);
    int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChipletsPerGroup);
  }

  // Heuristic to compute groupSize
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
  Value g_block = DivUIOp::create(b, loc, bid, mnBlocks);
  bid = RemUIOp::create(b, loc, bid, mnBlocks);

  // Group together the workgroups in g_block
  Value groupId = DivUIOp::create(b, loc, bid, blocksPerGroup);
  Value firstBidM = MulIOp::create(b, loc, groupId, mBlocksPerGroup);
  Value thisMBlocksPerGroup = MinUIOp::create(
      b, loc, SubIOp::create(b, loc, mBlocksValue, firstBidM), mBlocksPerGroup);
  Value m_block = AddIOp::create(
      b, loc, firstBidM, RemUIOp::create(b, loc, bid, thisMBlocksPerGroup));
  Value n_block =
      DivUIOp::create(b, loc, RemUIOp::create(b, loc, bid, blocksPerGroup),
                      thisMBlocksPerGroup);
  // no need to get splitKFactor here
  return {g_block, m_block, n_block};
}

AttnGridCoordinates
rock::layout::makeGxNGridLayout(PatternRewriter &b, Location loc, Value bid,
                                Value mIter, int64_t nBlocks, int64_t gridSize,
                                StringRef arch, int64_t numCU, Value splitKV) {
  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  int64_t numChiplets = getNumChiplets(arch, numCU);
  if (numChiplets > 1) {
    // It was empirically found that two chiplets as a group
    // computing a spatial mxn tile has better locality throughout.
    int64_t numChipletsPerGroup = std::ceil(numChiplets / 2);
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChipletsPerGroup);
  }
  Value g1NBlockCountVal = b.createOrFold<ConstantIndexOp>(loc, nBlocks);

  Value gBlockIdx, nBlockIdx, splitKVIdx;
  if (splitKV) {
    Value noGSize = arith::MulIOp::create(b, loc, splitKV, g1NBlockCountVal);
    gBlockIdx = arith::DivUIOp::create(b, loc, bid, noGSize);
    nBlockIdx = arith::RemUIOp::create(b, loc, bid, g1NBlockCountVal);
    Value outerIdx = arith::DivUIOp::create(b, loc, bid, g1NBlockCountVal);
    splitKVIdx = arith::RemUIOp::create(b, loc, outerIdx, splitKV);
  } else {
    gBlockIdx = arith::DivUIOp::create(b, loc, bid, g1NBlockCountVal);
    nBlockIdx = arith::RemUIOp::create(b, loc, bid, g1NBlockCountVal);
    splitKVIdx = nullptr;
  }
  // braces for init of the base class: GridCoordinates
  return {{gBlockIdx, mIter, nBlockIdx}, splitKVIdx};
}
