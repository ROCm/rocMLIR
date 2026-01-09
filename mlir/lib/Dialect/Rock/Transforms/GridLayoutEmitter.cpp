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

// based on
// https://github.com/HazyResearch/HipKittens/blob/7f6986b502396aa865c0c80625121daf7caa756d/include/common/util.cuh#L78
static Value rearrangeWorkgroupsForXCC(Location loc, PatternRewriter &b,
                                       Value bid, int64_t gridSize,
                                       int64_t numChiplets, int64_t chunkSize) {
  Value numChipletsVal = b.createOrFold<ConstantIndexOp>(loc, numChiplets);
  Value chunkSizeVal = b.createOrFold<ConstantIndexOp>(loc, chunkSize);

  // Current XCD
  Value xcd = RemUIOp::create(b, loc, bid, numChipletsVal);

  // Largest full (numChiplets*chunkSize)-aligned block
  int64_t block = numChiplets * chunkSize;
  int64_t limit = (gridSize / block) * block;
  Value blockVal = b.createOrFold<ConstantIndexOp>(loc, block);
  Value limitVal = b.createOrFold<ConstantIndexOp>(loc, limit);

  // Local BID (within round-robin assignment)
  Value localBid = DivUIOp::create(b, loc, bid, numChipletsVal);
  Value chunkIdx = DivUIOp::create(b, loc, localBid, chunkSizeVal);
  Value posInChunk = RemUIOp::create(b, loc, localBid, chunkSizeVal);

  // New BID
  // newBid = chunkIdx * block + xcd * chunkSize + posInChunk;
  Value newBid = AddIOp::create(
      b, loc,
      AddIOp::create(b, loc, MulIOp::create(b, loc, chunkIdx, blockVal),
                     MulIOp::create(b, loc, xcd, chunkSizeVal)),
      posInChunk);

  // If bid beyond the last full block, leave unchanged
  // if (bid > limit) return bid;
  Value isBidLargerThanLastFullBlock =
      arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sgt, bid, limitVal);
  bid = arith::SelectOp::create(b, loc, isBidLargerThanLastFullBlock, bid,
                                newBid);

  return bid;
}

static int64_t getNumChiplets(StringRef arch, int64_t numCU) {
  int64_t numChiplets = rock::lookupArchInfo(arch).maxNumXCC;
  // TODO: hack until we find a better way to determine number of chiplets
  if (arch.contains("gfx942") && numCU == 80) {
    numChiplets = 4;
  }
  return numChiplets;
}

GridCoordinates rock::layout::makeGroupedGridLayout(PatternRewriter &b,
                                                    Location loc, Value bid,
                                                    GridLayoutInfo info,
                                                    StringRef arch) {
  // Heuristic to compute groupSize
  // This also covers the cases where the output width is larger
  // than the input width
  int64_t numChiplets = getNumChiplets(arch, info.numCU);
  int64_t bitWidthIn = info.inputType.getIntOrFloatBitWidth();
  int64_t bitWidthOut = info.outputType.getIntOrFloatBitWidth();
  int64_t groupSize = std::ceil(std::sqrt(info.numCU / numChiplets)) *
                      (bitWidthOut / bitWidthIn);
  // use gridGroupSize if it's not zero
  if (info.gridGroupSize != 0) {
    groupSize = info.gridGroupSize;
    LLVM_DEBUG(llvm::dbgs() << "Setting groupSize by using tuning params to "
                            << groupSize << "\n");
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "Using heuristic to set groupSize to " << groupSize << "\n");
  }

  // Currently the firmware will launch workgroups
  // in a round-robin fashion to each chiplet. However
  // we would want a group (>=1) of chiplets to perform
  // a spatially local tile.
  // Therefore, adjust bid to make every consecutive #groups of chiplets
  // be slowest changing in the grid.
  if (numChiplets > 1) {
    int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
    int64_t chunkSize = std::min(groupSize * groupSize,
                                 std::max(int64_t{1}, gridSize / numChiplets));
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChiplets,
                                    chunkSize);
  }

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
    int64_t chunkSize = std::max(int64_t{1}, gridSize / numChiplets);
    bid = rearrangeWorkgroupsForXCC(loc, b, bid, gridSize, numChiplets,
                                    chunkSize);
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
