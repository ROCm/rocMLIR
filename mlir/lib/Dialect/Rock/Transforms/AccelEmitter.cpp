//===- AccelEmitter.cpp - MLIR helper to emit acceleration intrinsics
//---------------===//
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

#include "AccelEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using namespace mlir::rock::accel;

// ************************
// Generic helper functions
// ************************

void AccelEmitter::validateAcceleratorProperties() {
  // Extract relevant tuning parameters
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  int64_t kBase = accelEmitterParams.kBase;

  if (kPack > 1 && (kPack < kBase || kPack % kBase != 0)) {
    llvm_unreachable(
        "Tuning parameter selection guarantees kPack is multiple of k_base,"
        "this should never happen");
  }
}

AccelEmitter::AccelEmitter(StringRef arch,
                           RockAccelTuningParamAttrInterface tuningParams,
                           AccelEmitterParams accelEmitterParams)
    : tuningParams(tuningParams), accelEmitterParams(accelEmitterParams),
      waveSize(rock::lookupArchInfo(arch).waveSize) {
  validateAcceleratorProperties();
}

Value AccelEmitter::computeOutputConversion(PatternRewriter &b, Location loc,
                                            int64_t matrixM, int64_t matrixN,
                                            int64_t blockSize, int64_t gridSize,
                                            Value regVectorOrig, Value regDest,
                                            bool forceUnroll) {

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  int64_t nResultVectors = accelEmitterParams.nResultVectors;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  Type destType = regDest.getType().dyn_cast<MemRefType>().getElementType();

  int64_t accVectorLen = accVectorType.getNumElements();
  int64_t numElements = accVectorLen * (mRepeats * nRepeats * nResultVectors);
  auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

  BottomUpTMBuilder toRegCScalar(b, {"scalar"}, {numElements}, loc);
  toRegCScalar.embed({"vector"}, {0}, {mRepeats * nRepeats * nResultVectors},
                     "scalar", {accVectorLen});
  TransformMapAttr toRegCScalarAttr = toRegCScalar.get();

  auto convertLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{{zeroConstantOp}, {zeroConstantOp}},
      ArrayRef<Attribute>{b.getArrayAttr({}), b.getArrayAttr(toRegCScalarAttr)},
      /*bounds=*/ArrayRef<int64_t>{mRepeats * nRepeats * nResultVectors},
      /*strides=*/std::nullopt, forceUnroll, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(convertLoop.getBody());
    Value loaded =
        b.create<memref::LoadOp>(loc, accVectorType, regVectorOrig,
                                 convertLoop.getLowerCoords(/*domain*/ 0));
    Value cast = loaded;
    if (destType != accVectorType.getElementType()) {
      VectorType destVectorType = accVectorType.clone(destType);
      cast = createTypeConversionOp(b, loc, loaded, destVectorType);
    }
    b.create<InBoundsStoreOp>(loc, cast, regDest,
                              convertLoop.getLowerCoords(/*domain*/ 1));
  }
  return regDest;
}

// **************************
// Mfma accelerator interface
// **************************

MfmaEmitter::MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(mfmaGroup, tuningParams)},
      mfmaGroup{mfmaGroup} {}

AccelEmitterParams MfmaEmitter::initAccelEmitterParams(
    MfmaInsnGroup mfmaGroup, RockAccelTuningParamAttrInterface tuningParams) {
  AccelEmitterParams params;
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();

  // Extract relevant tuning parameters
  int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();
  int64_t K = kpackPerBlock * kPack;

  // Accelerator parameters
  params.kBase = mfmaAttr.k_base;
  params.kBasePerThread =
      (mfmaAttr.isKReduction ? K / mfmaAttr.inputSpansPerMfmaIn : K) /
      params.kBase;
  params.mRepeats = mfmaGroup.getMRepeats(mPerWave);
  params.nRepeats = mfmaGroup.getNRepeats(nPerWave);
  params.nResultVectors = mfmaGroup.getImms().size();
  params.mPerAccel = mPerWave / params.mRepeats;
  params.nPerAccel = nPerWave / params.nRepeats;
  params.kpackPerThread =
      (mfmaAttr.isKReduction ? kpackPerBlock / mfmaAttr.inputSpansPerMfmaIn
                             : kpackPerBlock);

  // Accelerator data types
  params.argTypeA = mfmaGroup.getArgTypeA();
  params.argTypeB = mfmaGroup.getArgTypeB();
  params.accVectorType = mfmaGroup.getRetType();

  return params;
}

void MfmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset) {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mfmaNonKDim = mfmaAttr.mfmaNonKDim;
  auto imms = mfmaGroup.getImms();
  int64_t nResultVectors = imms.size();
  VectorType vectorType = mfmaGroup.getRetType();
  for (int64_t i = 0; i < nResultVectors; ++i) {
    Value offset = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    offset = b.create<AddIOp>(loc, offset, regCOffset);

    auto vectorC = b.create<memref::LoadOp>(loc, vectorType, bufferC, offset);
    auto mfma = b.create<amdgpu::MFMAOp>(
        loc, vectorType, mfmaNonKDim, mfmaNonKDim, mfmaAttr.k,
        mfmaAttr.blocksMfma, argA, argB, vectorC, /*cbsz=*/imms[i].cbsz,
        /*abid=*/imms[i].abid,
        /*blgp=*/imms[i].blgp, /*reducePrecision=*/false, /*negateA=*/false,
        /*negateB=*/false, /*negateC=*/false);
    auto vectorD = mfma.getDestD();

    b.create<memref::StoreOp>(loc, vectorD, bufferC, offset);
  }
}

static int64_t calculateGridSize(ArrayRef<int64_t> bidGridLengths) {
  int64_t gridSizeVal = 1;
  for (int64_t gLen : bidGridLengths) {
    gridSizeVal *= gLen;
  }
  return gridSizeVal;
}

static TopDownTMBuilder
createTopSplitTMBuilder(PatternRewriter &b, Location loc, int64_t numElements,
                        std::optional<ArrayRef<int64_t>> bidGridLengths,
                        std::optional<int64_t> blockSize) {
  if (bidGridLengths.has_value()) {
    int64_t gridSizeVal = calculateGridSize(bidGridLengths.value());
    return TopDownTMBuilder(b, {"bid", "tid", "item"},
                            {gridSizeVal, blockSize.value(), numElements}, loc);
  }
  if (blockSize.has_value()) {
    return TopDownTMBuilder(b, {"tid", "item"},
                            {blockSize.value(), numElements}, loc);
  }
  return TopDownTMBuilder(b, {"item"}, {numElements}, loc);
}

ArrayAttr MfmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
    std::optional<int64_t> blockSize,
    std::optional<ArrayRef<int64_t>> bidGridLengths) {

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  int64_t nResultVectors = accelEmitterParams.nResultVectors;
  VectorType accVectorType = accelEmitterParams.accVectorType;
  int64_t mPerAccel = accelEmitterParams.mPerAccel;
  int64_t nPerAccel = accelEmitterParams.nPerAccel;

  auto mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mPerRepeat = mPerWave / mRepeats;
  int64_t nPerRepeat = nPerWave / nRepeats;
  int64_t nWaves = nPerBlock / nPerWave;
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t rowGroupSize = mfmaAttr.rowGroupSize;
  int64_t rowGroupsPerBlock = mfmaAttr.rowGroupsPerBlock;
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t m = mfmaAttr.mfmaNonKDim;

  // Note n has the 4x4 => 4x64 behavior that necessitated
  // inputSpansPerMfmaIn
  int64_t n = mfmaAttr.inputSpanLen;
  int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
  int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
  int64_t blocksPerRepeat = (mPerRepeat * nPerRepeat) / (m * n);

  int64_t retNumElements = accVectorType.getNumElements();
  int64_t numElements = retNumElements * mRepeats * nRepeats * nResultVectors;
  TopDownTMBuilder splitMemoryCoords =
      createTopSplitTMBuilder(b, loc, numElements, bidGridLengths, blockSize);
  {
    unsigned lowIdx = 0;
    if (bidGridLengths.has_value()) {
      splitMemoryCoords.merge({"g", "m", "n"}, {lowIdx, lowIdx + 1, lowIdx + 2},
                              {"bid"}, bidGridLengths.value());
      lowIdx += 3;
    }
    if (blockSize.has_value()) {
      int64_t wavesInKernelBlock = blockSize.value() / waveSize;
      splitMemoryCoords.merge(
          {"wave", "m_tid", "n_tid"}, {lowIdx, lowIdx + 1, lowIdx + 2}, "tid",
          {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
      lowIdx += 3;
    }
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"},
        {lowIdx, lowIdx + 1, lowIdx + 2, lowIdx + 3}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
  }
  TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

  // "blkMajor" and "blkMinor" are placeholder names because we don't know
  // if they'll be column or row until we check for broadcast-ness.
  auto toRowsAndCols =
      TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
  llvm::StringMap<uint32_t> rowsAndColsIdxs;
  if (blockSize.has_value()) {
    rowsAndColsIdxs = expandNamesInPlace(splitMemoryCoords,
                                         {{"wave", {"wave_m", "wave_n"}},
                                          {"i", {"m_i", "n_i"}},
                                          {"j", {"blkMajor", "blkMinor"}}});
  } else {
    rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords,
        {{"i", {"m_i", "n_i"}}, {"j", {"blkMajor", "blkMinor"}}});
  }
  TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
  if (bidGridLengths.has_value()) {
    rowsAndColsWrap.passThrough({"g", "m", "n"});
  }
  if (blockSize.has_value()) {
    int64_t wavesInKernelBlock = blockSize.value() / waveSize;
    rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                          {wavesInKernelBlock / nWaves, nWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
  }
  rowsAndColsWrap.merge({"m_i", "n_i"}, "i",
                        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});

  // Here we use the full builder API since we want index and name control
  bool isABroadcast = (nPerRepeat >= mPerRepeat);
  SmallVector<StringRef, 2> rowsFirst = {"blk_row", "blk_col"};
  SmallVector<StringRef, 2> colsFirst = {"blk_col", "blk_row"};
  toRowsAndCols.merge(
      isABroadcast ? rowsFirst : colsFirst,
      {rowsAndColsIdxs["blkMajor"], rowsAndColsIdxs["blkMinor"]}, "j",
      {splitMemoryCoords.endSize("j") / blocksInOutRegs, blocksInOutRegs});
  toRowsAndCols.passThrough(
      {"vec_group", "vec_item"},
      {rowsAndColsIdxs["vec_group"], rowsAndColsIdxs["vec_item"]},
      {"vec_group", "vec_item"});

  TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();

  auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
  if (bidGridLengths.has_value()) {
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});
  }

  // Note that `wave_m` and `wave_n` are strided by mPerAccel/nPerAccel, i.e.,
  // all the waves will compute next to each other and then they will move to
  // the next subtile in the workgroup
  {
    SmallVector<StringRef, 7> dimNamesM{/*0=*/"m",
                                        /*1=*/"m_i",
                                        /*2=*/"wave_m",
                                        /*3=*/"blk_row",
                                        /*4=*/"vec_group",
                                        /*5=*/"m_tid",
                                        /*6=*/"vec_item"};
    SmallVector<int64_t, 7> orderedDimStridesM{/*0=*/mPerBlock,
                                               /*1=*/mPerAccel * mWaves,
                                               /*2=*/mPerAccel,
                                               /*3=*/m,
                                               /*4=*/inputSpansPerMfmaIn *
                                                   rowGroupSize,
                                               /*5=*/rowGroupSize,
                                               /*6=*/1};
    SmallVector<int64_t, 7> dimSizes;
    convertDimStridestoSizes(orderedDimStridesM, mLen, dimSizes);
    if (bidGridLengths.has_value()) {
      toMatrixC.unmerge("gemmM", 1, dimNamesM, dimSizes);
    } else if (blockSize.has_value()) {
      toMatrixC.unmerge("gemmM", 1, ArrayRef<StringRef>{dimNamesM}.slice(1),
                        ArrayRef<int64_t>{dimSizes}.slice(1));
    } else {
      toMatrixC.unmerge(
          "gemmM", 1, {dimNamesM[1], dimNamesM[3], dimNamesM[4], dimNamesM[6]},
          {dimSizes[1], dimSizes[3], dimSizes[4], dimSizes[6]});
    }
  }
  {
    SmallVector<StringRef, 5> dimNamesN{/*0=*/"n",
                                        /*1=*/"n_i",
                                        /*2=*/"wave_n",
                                        /*3=*/"blk_col",
                                        /*4=*/"n_tid"};
    SmallVector<int64_t, 5> orderedDimStridesN{/*0=*/nPerBlock,
                                               /*1=*/nPerAccel * nWaves,
                                               /*2=*/nPerAccel,
                                               /*3=*/n,
                                               /*4=*/1};
    SmallVector<int64_t, 7> dimSizes;
    convertDimStridestoSizes(orderedDimStridesN, nLen, dimSizes);
    if (bidGridLengths.has_value()) {
      toMatrixC.unmerge("gemmN", 2, dimNamesN, dimSizes);
    } else if (blockSize.has_value()) {
      toMatrixC.unmerge("gemmN", 2, ArrayRef<StringRef>{dimNamesN}.slice(1),
                        ArrayRef<int64_t>{dimSizes}.slice(1));
    } else {
      toMatrixC.unmerge("gemmN", 2, {dimNamesN[1], dimNamesN[3]},
                        {dimSizes[1], dimSizes[3]});
    }
  }
  TransformMapAttr toMatrixCAttr = toMatrixC.get();
  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

Value MfmaEmitter::computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                                          OpBuilder &dBuilder, Value d_i,
                                          OpBuilder &builder, Value dPerBlock,
                                          Location loc, Value sourceBase,
                                          Value dWaves, Value laneId) {

  // Extract relevant emitter parameters
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  bool isKReduction = mfmaAttr.isKReduction;
  Value inputSpanLenConstantOp =
      builder.create<ConstantIndexOp>(loc, inputSpanLen);

  Value sourceOffset = sourceBase;
  if (!isKReduction) {
    // Compute source offset as
    // sourceOffset = k_i * MN + laneId + waveOffset * d_i;
    sourceOffset = builder.create<AddIOp>(loc, sourceOffset, laneId);

    Value waveOffset =
        builder.create<MulIOp>(loc, dWaves, inputSpanLenConstantOp);

    sourceOffset = dBuilder.create<AddIOp>(
        loc, sourceOffset, dBuilder.create<MulIOp>(loc, waveOffset, d_i));
    sourceOffset = kBuilder.create<AddIOp>(
        loc, sourceOffset, kBuilder.create<MulIOp>(loc, dPerBlock, k_i));
  } else {
    Value inputSpanLenConstantOp =
        builder.create<ConstantIndexOp>(loc, inputSpanLen);
    Value kpackPerThreadConstantOp =
        builder.create<ConstantIndexOp>(loc, accelEmitterParams.kpackPerThread);

    Value blk_id = builder.create<DivUIOp>(loc, laneId, inputSpanLenConstantOp);
    Value blk_td = builder.create<RemUIOp>(loc, laneId, inputSpanLenConstantOp);
    Value waveOffset =
        builder.create<MulIOp>(loc, dWaves, inputSpanLenConstantOp);

    // rowOffset = (k_i + kpackPerBlock * blk_id)
    Value rowOffset = kBuilder.create<AddIOp>(
        loc, kBuilder.create<MulIOp>(loc, blk_id, kpackPerThreadConstantOp),
        k_i);

    // Compute source offset as
    // sourceOffset =  rowOffset * MN + blk_td + d_i * waveOffset;
    sourceOffset = builder.create<AddIOp>(loc, sourceOffset, blk_td);
    sourceOffset = dBuilder.create<AddIOp>(
        loc, sourceOffset, dBuilder.create<MulIOp>(loc, waveOffset, d_i));

    sourceOffset = kBuilder.create<AddIOp>(
        loc, sourceOffset, kBuilder.create<MulIOp>(loc, rowOffset, dPerBlock));
  }
  return sourceOffset;
}

// **************************
// Wmma accelerator interface
// **************************

WmmaEmitter::WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(wmmaInsn, tuningParams)},
      wmmaInsn(wmmaInsn) {}

AccelEmitterParams WmmaEmitter::initAccelEmitterParams(
    WmmaInsn wmmaInsn, RockAccelTuningParamAttrInterface tuningParams) {
  AccelEmitterParams params;

  // Extract relevant tuning parameters
  int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
  int64_t kPack = tuningParams.getKpack();

  params.mRepeats = wmmaInsn.mRepeats;
  params.nRepeats = wmmaInsn.nRepeats;
  params.nResultVectors = 1;
  params.kpackPerThread = kpackPerBlock;
  params.kBase = wmmaInsn.inputLen;
  params.mPerAccel = wmmaInsn.inputLen;
  params.nPerAccel = wmmaInsn.inputLen;
  params.kBasePerThread = kpackPerBlock * kPack / params.kBase;

  params.argTypeA = wmmaInsn.argTypeA;
  params.argTypeB = wmmaInsn.argTypeB;
  params.accVectorType = wmmaInsn.retType;

  return params;
}

Value WmmaEmitter::computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                                          OpBuilder &dBuilder, Value d_i,
                                          OpBuilder &builder, Value dPerBlock,
                                          Location loc, Value sourceBase,
                                          Value dWaves, Value laneId) {

  Value mPerAccel =
      builder.create<ConstantIndexOp>(loc, accelEmitterParams.mPerAccel);
  Value waveOffset = builder.create<MulIOp>(loc, dWaves, mPerAccel);

  // Compute source offset as
  // sourceOffset = k_i * MN + (laneId % wmmaInputLen) + waveOffset * mn_i;
  Value inputLen = builder.create<ConstantIndexOp>(loc, wmmaInsn.inputLen);
  Value sourceOffset = builder.create<AddIOp>(
      loc, sourceBase, builder.create<RemUIOp>(loc, laneId, inputLen));
  sourceOffset = dBuilder.create<AddIOp>(
      loc, sourceOffset, dBuilder.create<MulIOp>(loc, waveOffset, d_i));
  sourceOffset = kBuilder.create<AddIOp>(
      loc, sourceOffset, kBuilder.create<MulIOp>(loc, dPerBlock, k_i));
  return sourceOffset;
}

void WmmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset) {
  VectorType vectorType = wmmaInsn.retType;
  auto vectorC = b.create<memref::LoadOp>(loc, vectorType, bufferC, regCOffset);

  auto mfma = b.create<amdgpu::WMMAOp>(loc, vectorType, argA, argB, vectorC,
                                       /*subwordOffset=*/0, /*unsignedA=*/false,
                                       /*unsignedB=*/false, /*clamp=*/true);
  auto vectorD = mfma.getDestD();

  b.create<memref::StoreOp>(loc, vectorD, bufferC, regCOffset);
}

ArrayAttr WmmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t mLen, int64_t nLen,
    std::optional<int64_t> blockSize,
    std::optional<ArrayRef<int64_t>> bidGridLengths) {

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  int64_t nWaves = nPerBlock / nPerWave;
  int64_t mWaves = mPerBlock / mPerWave;

  // High level code for this loop
  // source: https://gpuopen.com/learn/wmma_on_rdna3/
  //
  // laneIdx = tid % waveSize;
  // for (int item = 0; item < 8; item++){
  //    m_tid = laneIdx / 16;
  //    col = laneIdx % 16;
  //    row = (2*item + m_tid);
  //    c[16 * row+col] = regC[item];
  // }
  //
  //

  int64_t retNumElements = accVectorType.getNumElements();
  TopDownTMBuilder splitMemoryCoords = createTopSplitTMBuilder(
      b, loc, mRepeats * nRepeats * retNumElements, bidGridLengths, blockSize);
  {
    unsigned lowIdx = 0;
    if (bidGridLengths.has_value()) {
      splitMemoryCoords.merge({"g", "m", "n"}, {lowIdx, lowIdx + 1, lowIdx + 2},
                              {"bid"}, bidGridLengths.value());
      lowIdx += 3;
    }
    if (blockSize.has_value()) {
      int64_t wavesInKernelBlock = blockSize.value() / waveSize;
      splitMemoryCoords.merge(
          {"wave_m", "wave_n", "m_tid", "n_tid"},
          {lowIdx, lowIdx + 1, lowIdx + 2, lowIdx + 3}, "tid",
          {wavesInKernelBlock / nWaves, nWaves, waveSize / wmmaInsn.inputLen,
           wmmaInsn.inputLen});
      lowIdx += 4;
    }
    splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"},
                            {lowIdx, lowIdx + 1, lowIdx + 2}, "item",
                            {mRepeats, nRepeats, retNumElements});
  }
  TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

  auto toMatrixC =
      TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);

  if (bidGridLengths.has_value()) {
    toMatrixC.passThrough({"gemmG"}, {0}, {"g"});
  }

  // m_tid is liimited to 0 or 1 (or 0,1,2,3 for wave64). Basically every
  // workitem is computing 8 values, which represent a sub-column of a 16x16
  // tile. For workitems 0 to 15 m_tid is 0 and they write to 0,2,4,...,14.  For
  // workitems 16 to 31 m_tid is 1 and they write at positions 1,3,5,..,15 .
  // This is outlined in https://gpuopen.com/learn/wmma_on_rdna3/
  //
  // Note that `wave_m` and `wave_n` are strided by the inputLen, i.e., all the
  // waves will compute next to each other and then they will move to the next
  // subtile in the workgroup
  {
    SmallVector<StringRef, 5> dimNamesM{/*0=*/"m",
                                        /*1=*/"rep_i",
                                        /*2=*/"wave_m",
                                        /*3=*/"item_i",
                                        /*4=*/"m_tid"};
    SmallVector<int64_t, 5> orderedDimStridesM{/*0=*/mPerBlock,
                                               /*1=*/mWaves * wmmaInsn.inputLen,
                                               /*2=*/wmmaInsn.inputLen,
                                               /*3=*/wmmaInsn.outputStride,
                                               /*4=*/1};
    SmallVector<int64_t, 7> dimSizes;
    convertDimStridestoSizes(orderedDimStridesM, mLen, dimSizes);
    if (bidGridLengths.has_value()) {
      toMatrixC.unmerge("gemmM", 1, dimNamesM, dimSizes);
    } else if (blockSize.has_value()) {
      toMatrixC.unmerge("gemmM", 1, ArrayRef<StringRef>{dimNamesM}.slice(1),
                        ArrayRef<int64_t>{dimSizes}.slice(1));
    } else {
      toMatrixC.unmerge("gemmM", 1, {dimNamesM[1], dimNamesM[3]},
                        {dimSizes[1], dimSizes[3]});
    }
  }
  {
    SmallVector<StringRef, 5> dimNamesN{/*0=*/"n",
                                        /*1=*/"rep_j",
                                        /*2=*/"wave_n",
                                        /*3=*/"n_tid"};
    SmallVector<int64_t, 5> orderedDimStridesN{/*0=*/nPerBlock,
                                               /*1=*/nWaves * wmmaInsn.inputLen,
                                               /*2=*/wmmaInsn.inputLen,
                                               /*3=*/1};
    SmallVector<int64_t, 7> dimSizes;
    convertDimStridestoSizes(orderedDimStridesN, nLen, dimSizes);
    if (bidGridLengths.has_value()) {
      toMatrixC.unmerge("gemmN", 2, dimNamesN, dimSizes);
    } else if (blockSize.has_value()) {
      toMatrixC.unmerge("gemmN", 2, ArrayRef<StringRef>{dimNamesN}.slice(1),
                        ArrayRef<int64_t>{dimSizes}.slice(1));
    } else {
      toMatrixC.unmerge("gemmN", 2, {dimNamesN[1]}, {dimSizes[1]});
    }
  }
  TransformMapAttr toMatrixCAttr = toMatrixC.get();
  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

std::unique_ptr<AccelEmitter>
AccelEmitter::select(GemmFeatures features, Type dataTypeA, Type dataTypeB,
                     StringRef arch,
                     RockAccelTuningParamAttrInterface tuningParams) {
  bool isMfma = rock::bitEnumContainsAll(features, GemmFeatures::mfma);
  bool isWmma = rock::bitEnumContainsAll(features, GemmFeatures::wmma);
  if (isMfma) {
    auto maybeMfmaInsnGroup = MfmaInsnGroup::select(dataTypeA, dataTypeB, arch,
                                                    tuningParams.getMPerWave(),
                                                    tuningParams.getNPerWave());
    if (failed(maybeMfmaInsnGroup)) {
      return nullptr;
    }
    return std::make_unique<MfmaEmitter>(*maybeMfmaInsnGroup, arch,
                                         tuningParams);
  } else if (isWmma) {
    int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto maybeWmmaInsnGroup = WmmaInsn::select(dataTypeA, dataTypeB, waveSize,
                                               tuningParams.getMPerWave(),
                                               tuningParams.getNPerWave());
    if (failed(maybeWmmaInsnGroup)) {
      return nullptr;
    }
    return std::make_unique<WmmaEmitter>(*maybeWmmaInsnGroup, arch,
                                         tuningParams);
  } else {
    return nullptr;
  }
}
