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
#include "mlir/Dialect/Rock/utility/math.h"

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


GridCoordinates rock::layout::makeGroupedGridLayoutXCCMiddle(PatternRewriter &b,
                                                             Location loc, Value bid,
                                                             GridLayoutInfo info) {

  // Heurisitc to compute groupSize
  // This also covers the cases where the output width is larger
  // than the input width
  int64_t bitWidthIn = info.inputType.getIntOrFloatBitWidth();
  int64_t bitWidthOut = info.outputType.getIntOrFloatBitWidth();
  int64_t numChiplets = 8;
  assert(info.numCU / numChiplets == 38);
  int64_t groupSize =
      std::ceil(std::sqrt(info.numCU)) * (bitWidthOut / bitWidthIn);

  Value mBlocksPerGroup = b.createOrFold<ConstantIndexOp>(loc, groupSize);
  Value blocksPerGroup =
      b.createOrFold<ConstantIndexOp>(loc, groupSize * info.nBlocks);
  Value mBlocksValue = b.createOrFold<ConstantIndexOp>(loc, info.mBlocks);

  // Re-order workgroup-id to make chiplets slowest moving dimension
  int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
  if(gridSize >= info.numCU){
    Value gridSizeVal =
        b.createOrFold<ConstantIndexOp>(loc, gridSize);
    // We use the GCD because if the gridsize is not divisble by number
    // of chiplets then the following bid bit re-ordering might make it
    // larger than the gridsize. In this scenarios we will try to cater
    // to some chiplets by using the gcd.
    // int64_t logicalNumChiplets = math_util::gcd(gridSize, numChiplets);
    Value numChipletsVal = b.createOrFold<ConstantIndexOp>(loc, numChiplets);
    int64_t cusPerChiplet = info.numCU / numChiplets;
    Value cusPerChipletVal = b.createOrFold<ConstantIndexOp>(loc, cusPerChiplet);

    Value numCUVal = b.createOrFold<ConstantIndexOp>(loc, info.numCU);
    Value GpuIters = b.create<DivUIOp>(loc, bid, numCUVal);
    Value bidPerGrid = b.create<RemUIOp>(loc, bid, numCUVal);
    Value logicalChipletId = b.create<RemUIOp>(loc, bidPerGrid, numChipletsVal);
    Value wgIdPerLogicalChiplet = b.create<DivUIOp>(loc, bidPerGrid, numChipletsVal);

    // construct new bid
    Value bid_new = b.create<AddIOp>(
        loc, 
        wgIdPerLogicalChiplet, 
        b.create<MulIOp>(
            loc, 
            logicalChipletId, 
            cusPerChipletVal
        )
    );
    bid_new = b.create<AddIOp>(
        loc,
        bid_new,
        b.create<MulIOp>(
            loc, 
            numCUVal, 
            GpuIters
        )
    );

    int64_t lastNumCuMultiple = (gridSize - 1) - (gridSize % info.numCU);
    Value lastNumCuMultipleVal = b.createOrFold<ConstantIndexOp>(loc, lastNumCuMultiple);
    Value isBidLargerThanlastNumCuMultiple = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             bid, lastNumCuMultipleVal);
    bid = b.create<arith::SelectOp>(loc, isBidLargerThanlastNumCuMultiple, bid, bid_new);
  }

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

GridCoordinates rock::layout::makeGroupedGridLayoutXCCSlowest(PatternRewriter &b,
                                                    Location loc, Value bid,
                                                    GridLayoutInfo info) {
  // Heurisitc to compute groupSize
  // This also covers the cases where the output width is larger
  // than the input width
  int64_t bitWidthIn = info.inputType.getIntOrFloatBitWidth();
  int64_t bitWidthOut = info.outputType.getIntOrFloatBitWidth();
  int64_t numChiplets = 8;
  assert(info.numCU / numChiplets == 38);
  int64_t groupSize =
      std::ceil(std::sqrt(info.numCU)) * (bitWidthOut / bitWidthIn);

  Value mBlocksPerGroup = b.createOrFold<ConstantIndexOp>(loc, groupSize);
  Value blocksPerGroup =
      b.createOrFold<ConstantIndexOp>(loc, groupSize * info.nBlocks);
  Value mBlocksValue = b.createOrFold<ConstantIndexOp>(loc, info.mBlocks);

  // Re-order workgroup-id to make chiplets slowest moving dimension
  int64_t gridSize = info.gBlocks * info.mBlocks * info.nBlocks;
  if(gridSize >= numChiplets){
    Value gridSizeVal =
        b.createOrFold<ConstantIndexOp>(loc, gridSize);
    // We use the GCD because if the gridsize is not divisble by number
    // of chiplets then the following bid bit re-ordering might make it
    // larger than the gridsize. In this scenarios we will try to cater
    // to some chiplets by using the gcd.
    // int64_t logicalNumChiplets = math_util::gcd(gridSize, numChiplets);
    Value numChipletsVal = b.createOrFold<ConstantIndexOp>(loc, numChiplets);
    int64_t wgsPerChiplet = (gridSize) / numChiplets;
    Value wgsPerChipletVal = b.createOrFold<ConstantIndexOp>(loc, wgsPerChiplet);

    // Value numCUVal = b.createOrFold<ConstantIndexOp>(loc, info.numCU);
    // Value GpuIters = b.create<DivUIOp>(loc, bid, numCUVal);
    // Value bidPerGrid = b.create<RemUIOp>(loc, bid, numCUVal);
    Value logicalChipletId = b.create<RemUIOp>(loc, bid, numChipletsVal);
    Value wgIdPerLogicalChiplet = b.create<DivUIOp>(loc, bid, numChipletsVal);

    // construct new bid
    Value bid_new = b.create<AddIOp>(
        loc, 
        wgIdPerLogicalChiplet, 
        b.create<MulIOp>(
            loc, 
            logicalChipletId, 
            wgsPerChipletVal
        )
    );

    int64_t lastNumChipletMultiple = (gridSize - 1) - (gridSize % numChiplets);
    Value lastNumChipletMultipleVal = b.createOrFold<ConstantIndexOp>(loc, lastNumChipletMultiple);
    Value isBidLargerThanlastNumChipletMultiple = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             bid, lastNumChipletMultipleVal);
    bid = b.create<arith::SelectOp>(loc, isBidLargerThanlastNumChipletMultiple, bid, bid_new);
  }

  Value g1MxNBlockCountVal =
      b.createOrFold<ConstantIndexOp>(loc, info.mBlocks * info.nBlocks);
  Value gBlockIdx = b.create<arith::DivUIOp>(loc, bid, g1MxNBlockCountVal);
  Value nonGBlockIdx = b.create<arith::RemUIOp>(loc, bid, g1MxNBlockCountVal);
  Value g1NBlockCountVal = b.createOrFold<ConstantIndexOp>(loc, info.nBlocks);
  Value g1MBlockCountVal = b.createOrFold<ConstantIndexOp>(loc, info.mBlocks);
  Value nBlockIdx, mBlockIdx;
  if(info.nBlocks > info.mBlocks){
    nBlockIdx =
        b.create<arith::DivUIOp>(loc, nonGBlockIdx, g1MBlockCountVal);
    mBlockIdx =
        b.create<arith::RemUIOp>(loc, nonGBlockIdx, g1MBlockCountVal);
  }
  else{
    mBlockIdx =
        b.create<arith::DivUIOp>(loc, nonGBlockIdx, g1NBlockCountVal);
    nBlockIdx =
        b.create<arith::RemUIOp>(loc, nonGBlockIdx, g1NBlockCountVal);
  }

  return {gBlockIdx, mBlockIdx, nBlockIdx};

//   // Compute g_block first and the bid in the actual group g_block
//   Value mnBlocks =
//       b.createOrFold<ConstantIndexOp>(loc, info.mBlocks * info.nBlocks);
//   Value g_block = b.create<DivUIOp>(loc, bid, mnBlocks);
//   bid = b.create<RemUIOp>(loc, bid, mnBlocks);

//   // Group together the workgroups in g_block
//   Value groupId = b.create<DivUIOp>(loc, bid, blocksPerGroup);
//   Value firstBidM = b.create<MulIOp>(loc, groupId, mBlocksPerGroup);
//   Value thisMBlocksPerGroup = b.create<MinUIOp>(
//       loc, b.create<SubIOp>(loc, mBlocksValue, firstBidM), mBlocksPerGroup);
//   Value m_block = b.create<AddIOp>(
//       loc, firstBidM, b.create<RemUIOp>(loc, bid, thisMBlocksPerGroup));
//   Value n_block = b.create<DivUIOp>(
//       loc, b.create<RemUIOp>(loc, bid, blocksPerGroup), thisMBlocksPerGroup);
//   return {g_block, m_block, n_block};
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
