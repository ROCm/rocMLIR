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

#include "mlir/Dialect/Rock/IR/AccelEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/WmmaInsnGroup.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;
using namespace mlir::rock::accel;

// ************************
// Generic helper functions
// ************************

AccelEmitter::AccelEmitter(StringRef arch,
                           RockAccelTuningParamAttrInterface tuningParams,
                           AccelEmitterParams accelEmitterParams,
                           AccelEmitterKind kind)
    : tuningParams(tuningParams), accelEmitterParams(accelEmitterParams),
      waveSize(rock::lookupArchInfo(arch).waveSize), kind(kind) {
  if (failed(validateAcceleratorProperties()))
    llvm_unreachable("Accelerator parameters validation failed");
}

void AccelEmitter::computeOutputConversion(PatternRewriter &b, Location loc,
                                           Value regVectorOrig, Value regDest,
                                           bool forceUnroll) {

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  int64_t nResultVectors = accelEmitterParams.nResultVectors;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  Type destType = dyn_cast<MemRefType>(regDest.getType()).getElementType();

  int64_t accVectorLen = accVectorType.getNumElements();
  int64_t numElements = accVectorLen * (mRepeats * nRepeats * nResultVectors);
  auto zeroConstantOp = ConstantIndexOp::create(b, loc, 0);

  BottomUpTMBuilder toRegCScalar(b, {"scalar"}, {numElements}, loc);
  toRegCScalar.embed({"vector"}, {0}, {mRepeats * nRepeats * nResultVectors},
                     "scalar", {accVectorLen});
  TransformMapAttr toRegCScalarAttr = toRegCScalar.get();

  auto convertLoop = TransformingForOp::create(
      b, loc, ArrayRef<ValueRange>{{zeroConstantOp}, {zeroConstantOp}},
      ArrayRef<Attribute>{b.getArrayAttr({}), b.getArrayAttr(toRegCScalarAttr)},
      /*bounds=*/ArrayRef<int64_t>{mRepeats * nRepeats * nResultVectors},
      /*strides=*/std::nullopt, forceUnroll, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(convertLoop.getBody());
    Value loaded =
        memref::LoadOp::create(b, loc, accVectorType, regVectorOrig,
                               convertLoop.getLowerCoords(/*domain*/ 0));
    Value cast = loaded;
    if (destType != accVectorType.getElementType()) {
      VectorType destVectorType = accVectorType.clone(destType);
      cast = createTypeConversionOp(b, loc, loaded, destVectorType);
    }
    InBoundsStoreOp::create(b, loc, cast, regDest,
                            convertLoop.getLowerCoords(/*domain*/ 1));
  }
}

Value AccelEmitter::generateThreadwiseViewBufferA(PatternRewriter &b,
                                                  Location loc,
                                                  Value rawBufferA) {
  TopDownTMBuilder bufferAikTransform(
      b, {"i", "k"}, {1, accelEmitterParams.kBasePerThread}, loc);
  bufferAikTransform.ignore("i");
  bufferAikTransform.passThrough({"k"}, 0, {"k"});
  auto viewA = rock::transform(
      b, rawBufferA,
      b.getArrayAttr(SmallVector<Attribute>{bufferAikTransform.get()}));
  return viewA;
}

Value AccelEmitter::generateThreadwiseViewBufferB(PatternRewriter &b,
                                                  Location loc,
                                                  Value rawBufferB) {
  TopDownTMBuilder bufferBjkTransform(
      b, {"j", "k"}, {1, accelEmitterParams.kBasePerThread}, loc);
  bufferBjkTransform.ignore("j");
  bufferBjkTransform.passThrough({"k"}, 0, {"k"});
  auto viewB = rock::transform(
      b, rawBufferB,
      b.getArrayAttr(SmallVector<Attribute>{bufferBjkTransform.get()}));
  return viewB;
}

Value AccelEmitter::generateThreadwiseViewBufferC(PatternRewriter &b,
                                                  Location loc,
                                                  Value rawBufferC) {
  TopDownTMBuilder bufferCijTransform(
      b, {"i", "j"}, {accelEmitterParams.mRepeats, accelEmitterParams.nRepeats},
      loc);
  bufferCijTransform.unmerge(
      "offset", 0, {"i", "j"},
      {accelEmitterParams.mRepeats, accelEmitterParams.nRepeats});
  auto viewC = rock::transform(
      b, rawBufferC,
      b.getArrayAttr(SmallVector<Attribute>{bufferCijTransform.get()}));
  return viewC;
}

// **************************
// Mfma accelerator interface
// **************************

MfmaEmitter::MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(mfmaGroup, tuningParams, arch),
                   AccelEmitterKind::AEK_MFMAEmitter},
      mfmaGroup{mfmaGroup} {}

AccelEmitterParams MfmaEmitter::initAccelEmitterParams(
    MfmaInsnGroup mfmaGroup, RockAccelTuningParamAttrInterface tuningParams,
    StringRef arch) {
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
                                     ValueRange regCOffset, Value scaleA,
                                     Value scaleB) {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mfmaDDim = mfmaAttr.mfmaDDim;
  auto imms = mfmaGroup.getImms();
  int64_t nResultVectors = imms.size();
  Value nResultVectorsConst = ConstantIndexOp::create(b, loc, nResultVectors);
  VectorType vectorType = mfmaGroup.getRetType();
  auto outputOffset = llvm::to_vector(regCOffset);
  bool isScaled = scaleA && scaleB;

  for (int64_t i = 0; i < nResultVectors; ++i) {
    Value offset = b.createOrFold<arith::ConstantIndexOp>(loc, i);
    offset = AddIOp::create(
        b, loc, offset,
        MulIOp::create(b, loc, outputOffset.back(), nResultVectorsConst));
    outputOffset.back() = offset;
    auto vectorC =
        memref::LoadOp::create(b, loc, vectorType, bufferC, outputOffset);

    Value vectorD;
    if (isScaled) {
      auto mfma = amdgpu::ScaledMFMAOp::create(
          b, loc, vectorType, mfmaDDim, mfmaDDim, mfmaAttr.k, argA, argB,
          vectorC, scaleA, scaleB, /*scalesIdxA=*/0, /*scalesIdxB=*/0);
      vectorD = mfma.getDestD();
    } else {
      auto mfma = amdgpu::MFMAOp::create(
          b, loc, vectorType, mfmaDDim, mfmaDDim, mfmaAttr.k,
          mfmaAttr.blocksMfma, argA, argB, vectorC, /*cbsz=*/imms[i].cbsz,
          /*abid=*/imms[i].abid, /*blgp=*/imms[i].blgp,
          /*reducePrecision=*/false, /*negateA=*/false, /*negateB=*/false,
          /*negateC=*/false);
      vectorD = mfma.getDestD();
    }

    memref::StoreOp::create(b, loc, vectorD, bufferC, outputOffset);
  }
}

static void
makeViewsForRowsAndCols(TopDownTMBuilder &viewBuilder, int64_t mPerRepeat,
                        int64_t nPerRepeat,
                        const llvm::StringMap<uint32_t> &rowsAndColsIdxs,
                        int64_t endSizeJ, int64_t blocksInOutRegs) {
  // Here we use the full builder API since we want index and name control
  bool isABroadcast = (nPerRepeat >= mPerRepeat);
  SmallVector<StringRef, 2> rowsFirst = {"blk_row", "blk_col"};
  SmallVector<StringRef, 2> colsFirst = {"blk_col", "blk_row"};
  viewBuilder.merge(
      isABroadcast ? rowsFirst : colsFirst,
      {rowsAndColsIdxs.lookup("blkMajor"), rowsAndColsIdxs.lookup("blkMinor")},
      "j", {endSizeJ / blocksInOutRegs, blocksInOutRegs});
  viewBuilder.passThrough(
      {"vec_group", "vec_item"},
      {rowsAndColsIdxs.lookup("vec_group"), rowsAndColsIdxs.lookup("vec_item")},
      {"vec_group", "vec_item"});
}

struct Dim {
  StringRef name;
  int64_t size;
};

static std::tuple<SmallVector<StringRef>, SmallVector<int64_t>>
getDimNamesAndSize(ArrayRef<Dim> dims) {
  SmallVector<StringRef> names;
  SmallVector<int64_t> sizes;
  for (const Dim &d : dims) {
    names.push_back(d.name);
    sizes.push_back(d.size);
  }
  return {names, sizes};
}

llvm::FailureOr<RegsAsMatrixSubTiles> MfmaEmitter::computeOutputTransforms(
    OpBuilder &b, Location loc, int64_t mLen, int64_t nLen, int64_t blockSize,
    ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
    int64_t inNPerThread, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN) {

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
  int64_t m = mfmaAttr.mfmaDDim;

  // Note n has the 4x4 => 4x64 behavior that necessitated
  // inputSpansPerMfmaIn
  int64_t n = mfmaAttr.inputSpanLen;
  int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
  int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
  int64_t blocksPerRepeat = (mPerRepeat * nPerRepeat) / (m * n);

  int64_t retNumElements = accVectorType.getNumElements();
  int64_t numElements = retNumElements * mRepeats * nRepeats * nResultVectors;
  int64_t wavesInKernelBlock = blockSize / waveSize;

  // Note that `wave_m` and `wave_n` are strided by mPerAccel/nPerAccel, i.e.,
  // all the waves will compute next to each other and then they will move to
  // the next subtile in the workgroup

  // M sub dims
  Dim mBlock{"m_block", mLen / mPerBlock};
  Dim mi{"m_i", mPerWave / mPerAccel};
  Dim waveM{"wave_m", mWaves};
  Dim blkRow{"blk_row", mPerAccel / m};
  Dim vecGroup{"vec_group", m / (inputSpansPerMfmaIn * rowGroupSize)};
  Dim mTid{"m_tid", inputSpansPerMfmaIn};
  Dim vecItem{"vec_item", rowGroupSize};

  SmallVector<StringRef> dimNamesM;
  SmallVector<int64_t, 7> dimSizesM;
  std::tie(dimNamesM, dimSizesM) =
      getDimNamesAndSize({mi, waveM, blkRow, vecGroup, mTid, vecItem});

  // N sub dims
  Dim nBlock{"n_block", nLen / nPerBlock};
  Dim ni{"n_i", nPerWave / nPerAccel};
  Dim waveN{"wave_n", nWaves};
  Dim blkCol{"blk_col", (nPerAccel / n)};
  Dim nTid{"n_tid", n};
  SmallVector<StringRef> dimNamesN;
  SmallVector<int64_t, 7> dimSizesN;
  std::tie(dimNamesN, dimSizesN) =
      getDimNamesAndSize({ni, waveN, blkCol, nTid});

  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"g_block", "m_block", "n_block", "tid", "item"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
         numElements},
        loc);
    splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
    splitMemoryCoords.merge(
        {"wave", "m_tid", "n_tid"}, {3, 4, 5}, "tid",
        {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
    splitMemoryCoords.merge(
        {"i", "j", "vec_group", "vec_item"}, {6, 7, 8, 9}, "item",
        {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
         blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();
    auto toRowsAndCols =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    // "blkMajor" and "blkMinor" are placeholder names because we don't know
    // if they'll be column or row until we check for broadcast-ness.
    llvm::StringMap<uint32_t> rowsAndColsIdxs = expandNamesInPlace(
        splitMemoryCoords, {{"wave", {"wave_m", "wave_n"}},
                            {"i", {"m_i", "n_i"}},
                            {"j", {"blkMajor", "blkMinor"}}});
    TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
    rowsAndColsWrap.passThrough({"g_block", "m_block", "n_block"});
    rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                          {wavesInKernelBlock / nWaves, nWaves});
    rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
    rowsAndColsWrap.merge(
        {"m_i", "n_i"}, "i",
        {splitMemoryCoords.endSize("i") / nRepeats, nRepeats});
    makeViewsForRowsAndCols(toRowsAndCols, mPerRepeat, nPerRepeat,
                            rowsAndColsIdxs, splitMemoryCoords.endSize("j"),
                            blocksInOutRegs);
    TransformMapAttr toRowsAndColsAttr = toRowsAndCols.get();
    auto toMatrixC = TopDownTMBuilder::below(toRowsAndCols, toRowsAndColsAttr);
    toMatrixC.passThrough({"g_block", mBlock.name, nBlock.name});
    toMatrixC.unmerge("gemmBlockM", 3, dimNamesM, dimSizesM);
    toMatrixC.unmerge("gemmBlockN", 4, dimNamesN, dimSizesN);

    // Before returning the output view, if necessary, swap back the
    // threadid/iter dimensions on both the M/N axis.
    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr,
                                          toRowsAndColsAttr};
    FailureOr<TopDownTMBuilder> swapRes = mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /*isBlockwise=*/false, transformAttrs);
    if (failed(swapRes))
      return failure();

    ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as blockwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block"};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    ret.blockSubTile = maybeBlockSubTile.value();
  }

  {
    // Create views for tid slice of blockwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block", "item"};
    FailureOr<ArrayAttr> maybeBlockSubTileTidSlice =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTileTidSlice)) {
      return failure();
    }
    ret.blockSubTileTidSlice = maybeBlockSubTileTidSlice.value();
  }

  {
    // Create views as threadwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block", "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    ret.threadSubTile = maybeThreadSubTile.value();
  }

  return ret;
}

Value MfmaEmitter::wrapLDSBufferForLoad(
    OpBuilder &b, Location loc, Value buffer,
    const BlockwiseMatrixParamsAttr &matrixParams, int64_t blockSize,
    StringRef dName, bool useLdsTransposeLoad) const {

  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";

  // Extract relevant tuning parameters
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPack = tuningParams.getKpack();
  bool rotateDWithK = matrixParams.getRotateDWithK();
  bool ldsLayoutDxK = matrixParams.getLDSLayoutDxK();
  int64_t dPerThread = matrixParams.getInDPerThread();

  // Extract relevant emitter parameters
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  int64_t kBase = accelEmitterParams.kBase;
  bool isKReduction = mfmaAttr.isKReduction;
  int64_t kIter = kpackPerThread;
  int64_t kVec = 1;
  // Note that when directToLDS is disabled, we are loading vector<kpackxdtype>
  // from LDS, so we load kpackPerThread. When directToLDS is enabled, we
  // load vector<1xdtype>, so each thread will load kpackPerThread * kPack.
  if (matrixParams.getDirectToLDS()) {
    // kVec is kPack for directToLDS because as explained above, the
    // non-directToLDS case, has a dtype=vector<kpackxdtype>. So, we need to
    // handle both cases.
    kVec = kPack;
    kPerBlock *= kPack;
    assert(!rotateDWithK && "rotateDWithK must not be enabled for directToLds");
  }

  // Extract relevant derived parameters
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);

  SmallVector<Attribute> transformAttrs;
  if (!isKReduction) {
    TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                              {blockSize, dRepeats, kIter * kVec});
    splitTid.merge({"wave_id", "lane_id"}, {0, 1}, "tid",
                   {blockSize / waveSize, waveSize});
    splitTid.merge({"k_iter", "k_vec"}, {3, 4}, "k_iter", {kIter, kVec});

    splitTid.passThrough({"d_iter"}, {2}, {"d_iter"});
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);

    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id",
                      {mWaves, nWaves});
    splitWaveId.passThrough({"lane_id", "d_iter", "k_iter", "k_vec"},
                            {2, 3, 4, 5},
                            {"lane_id", "d_iter", "k_iter", "k_vec"});
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);

    TopDownTMBuilder toLDSRowCol =
        TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);

    // d = d_i*dWaves*dPerAccel + wave_d*dPerAccel + lane_id
    toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "lane_id"},
                        {dRepeats, dWaves, dPerAccel});

    // k = k_i
    toLDSRowCol.unmerge("k", 1, {"k_iter", "k_vec"}, {kIter, kVec});
    toLDSRowCol.ignore(otherWaveDim);

    TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();

    transformAttrs.push_back(toLDSRowColAttr);

    int64_t stride = (kPack == 1 ? dPerThread : 1);
    auto offset =
        rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride, "d",
                 dPerBlock, 0, "k", kPerBlock, {}, {"k"}, transformAttrs);

    if (ldsLayoutDxK)
      offset.unmerge("source_offset", 0, {"d", "k"}, {dPerBlock, kPerBlock});
    else
      offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

    TransformMapAttr offsetAttr = offset.get();
    transformAttrs.push_back(offsetAttr);

  } else {
    TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                              {blockSize, dRepeats, kIter * kVec});
    splitTid.merge(
        {"wave_id", "blk_id", "blk_td"}, {0, 1, 2}, "tid",
        {blockSize / waveSize, waveSize / inputSpanLen, inputSpanLen});
    splitTid.merge({"k_iter", "k_vec"}, {4, 5}, "k_iter", {kIter, kVec});

    splitTid.passThrough({"d_iter"}, {3}, {"d_iter"});
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);

    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id",
                      {mWaves, nWaves});
    splitWaveId.passThrough({"blk_id", "blk_td", "d_iter", "k_iter", "k_vec"},
                            {2, 3, 4, 5, 6},
                            {"blk_id", "blk_td", "d_iter", "k_iter", "k_vec"});
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);

    TopDownTMBuilder toLDSRowCol(b, {}, {}, loc);

    // Use LDS transpose compatible K formula when this operand uses LDS
    // transpose load. Handles both kVec >= kBase and kVec < kBase cases.
    if (useLdsTransposeLoad) {
      // K access pattern must match the transpose load's pattern.
      // For double-rate MFMA, properly distribute K across threads
      int64_t instrK = mfmaAttr.k;
      int64_t numBlksInK = instrK / kBase;
      int64_t numBlksInD = (waveSize / inputSpanLen) / numBlksInK;

      // Split blk_id into blk_d (for D dimension) and blk_k (for K dimension)
      TopDownTMBuilder splitBlkId =
          TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
      splitBlkId.passThrough({"wave_m", "wave_n"}, {0, 1},
                             {"wave_m", "wave_n"});
      splitBlkId.merge({"blk_d", "blk_k"}, {2, 3}, "blk_id",
                       {numBlksInD, numBlksInK});
      splitBlkId.passThrough({"blk_td", "d_iter", "k_iter", "k_vec"},
                             {4, 5, 6, 7},
                             {"blk_td", "d_iter", "k_iter", "k_vec"});
      TransformMapAttr splitBlkIdAttr = splitBlkId.get();
      transformAttrs.push_back(splitBlkIdAttr);

      if (kVec >= kBase) {
        // Case 1: kVec >= kBase - split k_vec into k_mfma and k_base
        int64_t numMfmaPerKVec = kVec / kBase;

        TopDownTMBuilder splitKVec =
            TopDownTMBuilder::below(splitBlkId, splitBlkIdAttr);
        splitKVec.passThrough({"wave_m", "wave_n"}, {0, 1},
                              {"wave_m", "wave_n"});
        splitKVec.passThrough({"blk_d", "blk_k", "blk_td", "d_iter", "k_iter"},
                              {2, 3, 4, 5, 6},
                              {"blk_d", "blk_k", "blk_td", "d_iter", "k_iter"});
        splitKVec.merge({"k_mfma", "k_base"}, {7, 8}, "k_vec",
                        {numMfmaPerKVec, kBase});
        TransformMapAttr splitKVecAttr = splitKVec.get();
        transformAttrs.push_back(splitKVecAttr);

        toLDSRowCol = TopDownTMBuilder::below(splitKVec, splitKVecAttr);

        // d = d_iter * dWaves * numBlksInD * inputSpanLen + wave_d * numBlksInD
        // * inputSpanLen + blk_d * inputSpanLen + blk_td
        toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "blk_d", "blk_td"},
                            {dRepeats, dWaves, numBlksInD, inputSpanLen});

        // k = k_iter * (numMfmaPerKVec * instrK) + k_mfma * instrK + blk_k *
        // kBase + k_base
        toLDSRowCol.unmerge("k", 1, {"k_iter", "k_mfma", "blk_k", "k_base"},
                            {kIter, numMfmaPerKVec, numBlksInK, kBase});
      } else {
        // Case 2: kVec < kBase - split k_iter to accumulate multiple kVec
        // loads into one kBase worth of data (e.g., kVec=4, kBase=8)
        int64_t numKVecPerMfma = kBase / kVec;
        int64_t kOuter = kIter / numKVecPerMfma;

        TopDownTMBuilder splitKIter =
            TopDownTMBuilder::below(splitBlkId, splitBlkIdAttr);
        splitKIter.passThrough({"wave_m", "wave_n"}, {0, 1},
                               {"wave_m", "wave_n"});
        splitKIter.passThrough({"blk_d", "blk_k", "blk_td", "d_iter"},
                               {2, 3, 4, 5},
                               {"blk_d", "blk_k", "blk_td", "d_iter"});
        splitKIter.merge({"k_outer", "k_inner"}, {6, 7}, "k_iter",
                         {kOuter, numKVecPerMfma});
        splitKIter.passThrough({"k_vec"}, {8}, {"k_vec"});
        TransformMapAttr splitKIterAttr = splitKIter.get();
        transformAttrs.push_back(splitKIterAttr);

        toLDSRowCol = TopDownTMBuilder::below(splitKIter, splitKIterAttr);

        // d formula same as kVec >= kBase case
        toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "blk_d", "blk_td"},
                            {dRepeats, dWaves, numBlksInD, inputSpanLen});

        // k = k_outer * instrK + blk_k * kBase + k_inner * kVec + k_vec
        // This accumulates numKVecPerMfma loads of kVec elements into kBase
        toLDSRowCol.unmerge("k", 1, {"k_outer", "blk_k", "k_inner", "k_vec"},
                            {kOuter, numBlksInK, numKVecPerMfma, kVec});
      }
    } else {
      // Standard formula for regular load scenarios
      toLDSRowCol = TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);

      // d = blk_td + d_i * waveOffset
      toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "blk_td"},
                          {dRepeats, dWaves, inputSpanLen});
      if (matrixParams.getSplitKAcrossThreadsFirst()) {
        // k = blk_id + (waveSize / inputSpanLen) * k_i
        toLDSRowCol.unmerge("k", 1, {"k_iter", "blk_id", "k_vec"},
                            {kIter, waveSize / inputSpanLen, kVec});
      } else {
        // k = k_i + kpackPerBlock * blk_id
        toLDSRowCol.unmerge("k", 1, {"blk_id", "k_iter", "k_vec"},
                            {waveSize / inputSpanLen, kIter, kVec});
      }
    }

    toLDSRowCol.ignore(otherWaveDim);

    TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();
    transformAttrs.push_back(toLDSRowColAttr);

    int64_t stride = (kPack == 1 ? dPerThread : 1);
    auto offset =
        rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride, "d",
                 dPerBlock, 0, "k", kPerBlock, {}, {"k"}, transformAttrs);

    if (ldsLayoutDxK)
      offset.unmerge("source_offset", 0, {"d", "k"}, {dPerBlock, kPerBlock});
    else
      offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

    TransformMapAttr offsetAttr = offset.get();
    transformAttrs.push_back(offsetAttr);
  }

  ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
  return transform(b, buffer, ldsRead);
}

bool MfmaEmitter::isKReduction() const {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  return mfmaAttr.isKReduction;
}

int64_t MfmaEmitter::getRowGroupSize() const {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  return mfmaAttr.rowGroupSize;
}

int64_t MfmaEmitter::getKDim() const {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  return mfmaAttr.k;
}

int64_t MfmaEmitter::getDDim(StringRef dName) const {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  return mfmaAttr.mfmaDDim;
}

llvm::FailureOr<RegsAsMatrixSubTiles>
MfmaEmitter::createAccelGemmOperandTransforms(
    OpBuilder &b, Location loc, int64_t kIters,
    ArrayRef<int64_t> bidGridLengths, int64_t blockSize,
    int64_t dInCopyPerThread, StringRef dName, bool isKContiguousDim,
    bool rotateDWithK, bool doSplitKAcrossThreadsFirst,
    bool otherOperandUsesLdsTranspose) const {
  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  int64_t thisDimNumBlocks =
      dName == "m" ? bidGridLengths[1] : bidGridLengths[2];

  // Extract relevant tuning parameters
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPackPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  int64_t kBase = accelEmitterParams.kBase;
  bool isKReduction = mfmaAttr.isKReduction;
  int64_t instrK = mfmaAttr.k;

  // Extract relevant derived parameters
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);

  RegsAsMatrixSubTiles ret;
  // compute grid sub tile transforms
  {
    SmallVector<Attribute> transformAttrs;
    // First coordinate transform
    TopDownTMBuilder splitIter(
        b, {"k_loop", "g_block", "m_block", "n_block", "tid", "iter"},
        {kIters, bidGridLengths[0], bidGridLengths[1], bidGridLengths[2],
         blockSize, dRepeats * kpackPerThread * kPack},
        loc);
    {
      splitIter.passThrough({"k_loop", "g_block", "m_block", "n_block", "tid"});
      if (isKContiguousDim) {
        splitIter.merge({"drepeat", "kpack_iter", "kpack"}, {5, 6, 7}, "iter",
                        {dRepeats, kpackPerThread, kPack});
      } else {
        splitIter.merge({"kpack_iter", "drepeat", "kpack"}, {5, 6, 7}, "iter",
                        {kpackPerThread, dRepeats, kPack});
      }
    }
    TransformMapAttr splitIterAttr = splitIter.get();
    transformAttrs.push_back(splitIterAttr);
    // Second coordinate transform
    TopDownTMBuilder splitTid =
        TopDownTMBuilder::below(splitIter, splitIterAttr);
    {
      unsigned int dims = 0;
      splitTid.passThrough({"k_loop", "g_block"});
      splitTid.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitTid.passThrough({"kpack"}, {3}, {"kpack"});
      if (isKReduction) {
        splitTid.merge(
            {"wave_id", "blk_id", "blk_td"}, {4, 5, 6}, "tid",
            {blockSize / waveSize, waveSize / inputSpanLen, inputSpanLen});
        dims = 7;
      } else {
        splitTid.merge({"wave_id", "lane_id"}, {4, 5}, "tid",
                       {blockSize / waveSize, waveSize});
        dims = 6;
      }
      splitTid.passThrough({"d_iter", "k_iter"}, {dims, dims + 1},
                           {"drepeat", "kpack_iter"});
    }
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);
    // Third coordinate transform
    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    {
      splitWaveId.passThrough({"k_loop", "g_block"});
      splitWaveId.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitWaveId.passThrough({"kpack"}, {3}, {"kpack"});
      splitWaveId.merge({"wave_m", "wave_n"}, {4, 5}, "wave_id",
                        {mWaves, nWaves});
      if (isKReduction) {
        splitWaveId.passThrough({"blk_id", "blk_td", "d_iter", "k_iter"},
                                {6, 7, 8, 9},
                                {"blk_id", "blk_td", "d_iter", "k_iter"});
      } else {
        splitWaveId.passThrough({"lane_id", "d_iter", "k_iter"}, {6, 7, 8},
                                {"lane_id", "d_iter", "k_iter"});
      }
    }
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);
    // Fourth coordinate transform
    // Check if we need LDS transpose compatible K formula.
    // When prefetch is used: kPack >= kBase allows LDS transpose load,
    // kPack < kBase disables it (falls back to regular load).
    bool useLdsTransposeCompatibleK =
        otherOperandUsesLdsTranspose && isKReduction;
    int64_t numBlksInK = instrK / kBase;
    int64_t numBlksInD = (waveSize / inputSpanLen) / numBlksInK;

    TransformMapAttr toLDSRowColAttr;
    if (useLdsTransposeCompatibleK) {
      // LDS transpose compatible path: split blk_id into blk_d and blk_k
      // Also split kpack into k_mfma and k_base to match LDS transpose pattern
      int64_t numMfmaPerKPack = kPack / kBase;

      // First, add a transform to split blk_id
      TopDownTMBuilder splitBlkId =
          TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
      splitBlkId.passThrough({"k_loop", "g_block"});
      splitBlkId.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitBlkId.passThrough({"kpack"}, {3}, {"kpack"});
      splitBlkId.passThrough({"wave_m", "wave_n"}, {4, 5},
                             {"wave_m", "wave_n"});
      splitBlkId.merge({"blk_d", "blk_k"}, {6, 7}, "blk_id",
                       {numBlksInD, numBlksInK});
      splitBlkId.passThrough({"blk_td", "d_iter", "k_iter"}, {8, 9, 10},
                             {"blk_td", "d_iter", "k_iter"});
      TransformMapAttr splitBlkIdAttr = splitBlkId.get();
      transformAttrs.push_back(splitBlkIdAttr);

      // Split kpack into k_mfma and k_base (similar to wrapLDSBufferForLoad)
      TopDownTMBuilder splitKpack =
          TopDownTMBuilder::below(splitBlkId, splitBlkIdAttr);
      splitKpack.passThrough({"k_loop", "g_block"});
      splitKpack.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitKpack.merge({"k_mfma", "k_base"}, {3, 4}, "kpack",
                       {numMfmaPerKPack, kBase});
      splitKpack.passThrough({"wave_m", "wave_n"}, {5, 6},
                             {"wave_m", "wave_n"});
      splitKpack.passThrough({"blk_d", "blk_k", "blk_td", "d_iter", "k_iter"},
                             {7, 8, 9, 10, 11},
                             {"blk_d", "blk_k", "blk_td", "d_iter", "k_iter"});
      TransformMapAttr splitKpackAttr = splitKpack.get();
      transformAttrs.push_back(splitKpackAttr);

      // Then create the coordinate transform
      TopDownTMBuilder toLDSRowCol =
          TopDownTMBuilder::below(splitKpack, splitKpackAttr);
      toLDSRowCol.passThrough({"k_loop", "g_block"});
      toLDSRowCol.passThrough({thisBlockDim}, {2}, {thisBlockDim});

      // d = d_iter * dWaves * numBlksInD * inputSpanLen + wave_d * numBlksInD *
      // inputSpanLen + blk_d * inputSpanLen + blk_td
      toLDSRowCol.unmerge("d", 3, {"d_iter", thisWaveDim, "blk_d", "blk_td"},
                          {dRepeats, dWaves, numBlksInD, inputSpanLen});

      // k = k_iter * (numMfmaPerKPack * instrK) + k_mfma * instrK + blk_k *
      // kBase + k_base This matches the formula in wrapLDSBufferForLoad
      toLDSRowCol.unmerge("k", 4, {"k_iter", "k_mfma", "blk_k", "k_base"},
                          {kpackPerThread, numMfmaPerKPack, numBlksInK, kBase});

      toLDSRowCol.ignore(otherWaveDim);
      toLDSRowColAttr = toLDSRowCol.get();
      transformAttrs.push_back(toLDSRowColAttr);

      // Fifth coordinate transform for LDS transpose compatible path
      // Note: rotateDWithK should not be used with LDS transpose compatible
      {
        TopDownTMBuilder offset =
            TopDownTMBuilder::below(toLDSRowCol, toLDSRowColAttr);
        offset.passThrough({"G"}, {0}, {"g_block"});
        offset.unmerge({"K"}, 1, {"k_loop", "k"},
                       {kIters, kPackPerBlock * kPack});
        offset.unmerge("D", 2, {thisBlockDim, "d"},
                       {thisDimNumBlocks, dPerBlock});
        TransformMapAttr offsetAttr = offset.get();
        transformAttrs.push_back(offsetAttr);
      }
      ret.gridSubTile = b.getArrayAttr(transformAttrs);
    } else {
      // Regular path
      TopDownTMBuilder toLDSRowCol =
          TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
      toLDSRowCol.passThrough({"k_loop", "g_block"});
      toLDSRowCol.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      toLDSRowCol.passThrough({"kpack"}, {3}, {"kpack"});
      if (isKReduction) {
        // d = blk_td + d_i * waveOffset
        toLDSRowCol.unmerge("d", 4, {"d_iter", thisWaveDim, "blk_td"},
                            {dRepeats, dWaves, inputSpanLen});
        if (doSplitKAcrossThreadsFirst) {
          // k = blk_id + (waveSize / inputSpanLen) * k_i
          toLDSRowCol.unmerge("k", 5, {"k_iter", "blk_id"},
                              {kpackPerThread, waveSize / inputSpanLen});
        } else {
          // k = k_i + kpackPerBlock * blk_id
          toLDSRowCol.unmerge("k", 5, {"blk_id", "k_iter"},
                              {waveSize / inputSpanLen, kpackPerThread});
        }
      } else {
        // d = d_i*dWaves*dPerAccel + wave_d*dPerAccel + lane_id
        toLDSRowCol.unmerge("d", 4, {"d_iter", thisWaveDim, "lane_id"},
                            {dRepeats, dWaves, dPerAccel});
        // k = k_i
        toLDSRowCol.passThrough({"k"}, 5, {"k_iter"});
      }
      toLDSRowCol.ignore(otherWaveDim);
      toLDSRowColAttr = toLDSRowCol.get();
      transformAttrs.push_back(toLDSRowColAttr);

      // Fifth coordinate transform
      {
        int64_t stride = (kPack == 1 ? dInCopyPerThread : 1);
        auto offset = rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr,
                               stride, "d", dPerBlock, 3, "k", kPackPerBlock,
                               {"k_loop", "g_block", thisBlockDim, "kpack"},
                               {"k"}, transformAttrs);
        offset.passThrough({"G"}, {0}, {"g_block"});
        offset.unmerge({"K"}, 1, {"k_loop", "k", "kpack"},
                       {kIters, kPackPerBlock, kPack});
        offset.unmerge("D", 2, {thisBlockDim, "d"},
                       {thisDimNumBlocks, dPerBlock});
        TransformMapAttr offsetAttr = offset.get();
        transformAttrs.push_back(offsetAttr);
      }
      ret.gridSubTile = b.getArrayAttr(transformAttrs);
    }
  }
  // compute block sub tile transforms
  {
    StringSet<> dimensionsToRemove{"k_loop", "g_block", "m_block", "n_block"};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    ret.blockSubTile = maybeBlockSubTile.value();
  }
  // compute thread sub tile transforms
  {
    StringSet<> dimensionsToRemove{"k_loop", "g_block", "m_block", "n_block",
                                   "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    ret.threadSubTile = maybeThreadSubTile.value();
  }
  return ret;
}

LogicalResult MfmaEmitter::validateAcceleratorProperties() {
  // Extract relevant tuning parameters
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  int64_t kBase = accelEmitterParams.kBase;

  if (kPack > 1 && (kPack < kBase || kPack % kBase != 0))
    return failure();

  return success();
}

// **************************
// Wmma accelerator interface
// **************************

WmmaEmitter::WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
                         RockAccelTuningParamAttrInterface tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(wmmaInsn, tuningParams, arch),
                   AccelEmitterKind::AEK_WMMAEmitter},
      wmmaInsn(wmmaInsn), isGfx11(arch.contains("gfx11")) {}

AccelEmitterParams WmmaEmitter::initAccelEmitterParams(
    WmmaInsn wmmaInsn, RockAccelTuningParamAttrInterface tuningParams,
    StringRef arch) {
  AccelEmitterParams params;

  // Extract relevant tuning parameters
  int64_t kpackPerBlock = tuningParams.getKpackPerBlock();
  int64_t kPack = tuningParams.getKpack();
  int64_t inputVectorLen = wmmaInsn.argTypeA.getNumElements();
  params.kBase = inputVectorLen;
  params.mRepeats = wmmaInsn.mRepeats;
  params.nRepeats = wmmaInsn.nRepeats;
  params.nResultVectors = 1;

  params.kpackPerThread = kpackPerBlock;
  params.mPerAccel = wmmaInsn.mPerAccel;
  params.nPerAccel = wmmaInsn.nPerAccel;
  // Pre-gfx12 each thread in the wave is loading an entire groups
  // of Ks to reduce. So, if there are 32 threads in a wave and
  // and we want to do a(16x16) * b(16x16), 16 threads are loading a vector
  // of 16 Ks and the other 16 threads are replicating those values.
  int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
  // isGfx11 flag is set after call to this function. Therefore can not use
  // isGfx11 flag yet from inside this function.
  if (!arch.contains("gfx11")) {
    // Post-gfx12 each thread is loading a partial set of values
    // to reduce. For instance, with the previous example, each
    // thread is loading a vector of 8 Ks. The first 16 threads are
    // loading k=[0:8] the second 16 threads are loading k=[8:16] threads
    assert(wmmaInsn.mPerAccel == wmmaInsn.nPerAccel &&
           "Currently only supported for equal mPerAccel and nPerAccel");
    int64_t numReductions = waveSize / wmmaInsn.mPerAccel;
    params.kpackPerThread /= numReductions;
  }
  params.kBasePerThread = (params.kpackPerThread * kPack) / params.kBase;
  params.argTypeA = wmmaInsn.argTypeA;
  params.argTypeB = wmmaInsn.argTypeB;
  params.accVectorType = wmmaInsn.retType;

  return params;
}

int64_t WmmaEmitter::getKDim() const {
  // K dimension is encoded in the input vector length
  return wmmaInsn.kDim;
}

int64_t WmmaEmitter::getDDim(StringRef dName) const {
  if (dName.empty()) {
    llvm_unreachable("dName must be specified for WMMA");
  }

  return dName == "m" ? wmmaInsn.mPerAccel : wmmaInsn.nPerAccel;
}

Value WmmaEmitter::wrapLDSBufferForLoad(
    OpBuilder &b, Location loc, Value buffer,
    const BlockwiseMatrixParamsAttr &matrixParams, int64_t blockSize,
    StringRef dName, bool useLdsTransposeLoad) const {
  // Note: WMMA does not support LDS transpose load, so the parameter is unused.
  (void)useLdsTransposeLoad;

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();
  // TODO: gfx10 supports directToLDS. Implement it.
  assert(!matrixParams.getDirectToLDS() &&
         "direct to LDS not supported for WMMA");
  assert(!matrixParams.getLDSLayoutDxK() &&
         "WMMA only supports LDS layout KxD for now");

  // Extract relevant emitter parameters
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);

  // Extract relevant derived parameters
  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;

  SmallVector<Attribute> transformAttrs;

  // Compute source offset as
  // sourceOffset = k_i * MN + (laneId % wmmaInputLen) + waveOffset * mn_i;
  TopDownTMBuilder splitTid(b, {"tid", "d_iter", "k_iter"},
                            {blockSize, dRepeats, kpackPerThread});
  splitTid.merge({"wave_id", "lane_id"}, {0, 1}, "tid",
                 {blockSize / waveSize, waveSize});

  splitTid.passThrough({"d_iter", "k_iter"}, {2, 3}, {"d_iter", "k_iter"});
  TransformMapAttr splitTidAttr = splitTid.get();
  transformAttrs.push_back(splitTidAttr);

  TopDownTMBuilder splitWaveId =
      TopDownTMBuilder::below(splitTid, splitTidAttr);
  splitWaveId.merge({"wave_m", "wave_n"}, {0, 1}, "wave_id", {mWaves, nWaves});
  splitWaveId.passThrough({"lane_id", "d_iter", "k_iter"}, {2, 3, 4},
                          {"lane_id", "d_iter", "k_iter"});
  TransformMapAttr splitWaveIdAttr = splitWaveId.get();
  transformAttrs.push_back(splitWaveIdAttr);

  TopDownTMBuilder replicateLanes =
      TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
  replicateLanes.passThrough({"wave_m", "wave_n", "d_iter", "k_iter"},
                             {0, 1, 4, 5},
                             {"wave_m", "wave_n", "d_iter", "k_iter"});

  replicateLanes.merge({"block_id", "block_td"}, {2, 3}, "lane_id",
                       {waveSize / dPerAccel, dPerAccel});
  TransformMapAttr replicateLanesAttr = replicateLanes.get();
  transformAttrs.push_back(replicateLanesAttr);

  TopDownTMBuilder toLDSRowCol =
      TopDownTMBuilder::below(replicateLanes, replicateLanesAttr);
  if (isGfx11) {
    toLDSRowCol.passThrough({"k"}, {1}, {"k_iter"});
    toLDSRowCol.ignore("block_id");
  } else {
    toLDSRowCol.unmerge({"k"}, 1, {"block_id", "k_iter"},
                        {wmmaInsn.outputStride, kpackPerThread});
  }
  toLDSRowCol.unmerge("d", 0, {"d_iter", thisWaveDim, "block_td"},
                      {dRepeats, dWaves, dPerAccel});
  toLDSRowCol.ignore(otherWaveDim);

  TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();
  transformAttrs.push_back(toLDSRowColAttr);

  int64_t stride = (kPack == 1 ? matrixParams.getInDPerThread() : 1);
  auto offset = rotateIf(matrixParams.getRotateDWithK(), toLDSRowCol,
                         toLDSRowColAttr, stride, "d", dPerBlock, 0, "k",
                         kPerBlock, {}, {"k"}, transformAttrs);

  offset.unmerge("source_offset", 0, {"k", "d"}, {kPerBlock, dPerBlock});

  TransformMapAttr offsetAttr = offset.get();
  transformAttrs.push_back(offsetAttr);

  ArrayAttr ldsRead = b.getArrayAttr(transformAttrs);
  return transform(b, buffer, ldsRead);
}

llvm::FailureOr<RegsAsMatrixSubTiles>
WmmaEmitter::createAccelGemmOperandTransforms(
    OpBuilder &b, Location loc, int64_t kIters,
    ArrayRef<int64_t> bidGridLengths, int64_t blockSize,
    int64_t dInCopyPerThread, StringRef dName, bool isKContiguousDim,
    bool rotateDWithK, bool doSplitKAcrossThreadsFirst,
    bool otherOperandUsesLdsTranspose) const {
  // Note: WMMA does not support LDS transpose load, so the parameter is unused
  (void)otherOperandUsesLdsTranspose;
  StringRef thisWaveDim = dName == "m" ? "wave_m" : "wave_n";
  StringRef otherWaveDim = dName == "m" ? "wave_n" : "wave_m";
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  int64_t thisDimNumBlocks =
      dName == "m" ? bidGridLengths[1] : bidGridLengths[2];

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t kPackPerBlock = tuningParams.getKpackPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();

  // Extract relevant emitter parameters
  int64_t kpackPerThread = accelEmitterParams.kpackPerThread;
  int64_t dRepeats = (dName == "m" ? accelEmitterParams.mRepeats
                                   : accelEmitterParams.nRepeats);
  int64_t dPerAccel = (dName == "m" ? accelEmitterParams.mPerAccel
                                    : accelEmitterParams.nPerAccel);

  // Extract relevant derived parameters
  int64_t dWaves = (dName == "m" ? mPerBlock / mPerWave : nPerBlock / nPerWave);
  int64_t dPerBlock = (dName == "m" ? mPerBlock : nPerBlock);
  int64_t mWaves = mPerBlock / mPerWave;
  int64_t nWaves = nPerBlock / nPerWave;

  RegsAsMatrixSubTiles ret;
  // compute grid sub tile transforms
  {
    SmallVector<Attribute> transformAttrs;
    // First coordinate transform
    TopDownTMBuilder splitIter(
        b, {"k_loop", "g_block", "m_block", "n_block", "tid", "iter"},
        {kIters, bidGridLengths[0], bidGridLengths[1], bidGridLengths[2],
         blockSize, dRepeats * kpackPerThread * kPack},
        loc);
    {
      splitIter.passThrough({"k_loop", "g_block", "m_block", "n_block", "tid"});
      if (isKContiguousDim) {
        splitIter.merge({"drepeat", "kpack_iter", "kpack"}, {5, 6, 7}, "iter",
                        {dRepeats, kpackPerThread, kPack});
      } else {
        splitIter.merge({"kpack_iter", "drepeat", "kpack"}, {5, 6, 7}, "iter",
                        {kpackPerThread, dRepeats, kPack});
      }
    }
    TransformMapAttr splitIterAttr = splitIter.get();
    transformAttrs.push_back(splitIterAttr);
    // Second coordinate transform
    TopDownTMBuilder splitTid =
        TopDownTMBuilder::below(splitIter, splitIterAttr);
    {
      splitTid.passThrough({"k_loop", "g_block"});
      splitTid.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitTid.passThrough({"kpack"}, {3}, {"kpack"});
      splitTid.merge({"wave_id", "lane_id"}, {4, 5}, "tid",
                     {blockSize / waveSize, waveSize});
      splitTid.passThrough({"d_iter", "k_iter"}, {6, 7},
                           {"drepeat", "kpack_iter"});
    }
    TransformMapAttr splitTidAttr = splitTid.get();
    transformAttrs.push_back(splitTidAttr);
    // Second coordinate transform
    TopDownTMBuilder splitWaveId =
        TopDownTMBuilder::below(splitTid, splitTidAttr);
    {
      splitWaveId.passThrough({"k_loop", "g_block"});
      splitWaveId.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      splitWaveId.passThrough({"kpack"}, {3}, {"kpack"});
      splitWaveId.merge({"wave_m", "wave_n"}, {4, 5}, "wave_id",
                        {mWaves, nWaves});
      splitWaveId.passThrough({"lane_id", "d_iter", "k_iter"}, {6, 7, 8},
                              {"lane_id", "d_iter", "k_iter"});
    }
    TransformMapAttr splitWaveIdAttr = splitWaveId.get();
    transformAttrs.push_back(splitWaveIdAttr);
    // Third coordinate transform
    TopDownTMBuilder replicateLanes =
        TopDownTMBuilder::below(splitWaveId, splitWaveIdAttr);
    {
      replicateLanes.passThrough({"k_loop", "g_block"});
      replicateLanes.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      replicateLanes.passThrough({"kpack"}, {3}, {"kpack"});
      replicateLanes.passThrough({"wave_m", "wave_n", "d_iter", "k_iter"},
                                 {4, 5, 8, 9},
                                 {"wave_m", "wave_n", "d_iter", "k_iter"});

      replicateLanes.merge({"block_id", "block_td"}, {6, 7}, "lane_id",
                           {waveSize / dPerAccel, dPerAccel});
    }
    TransformMapAttr replicateLanesAttr = replicateLanes.get();
    transformAttrs.push_back(replicateLanesAttr);
    // Fourth coordinate transform
    TopDownTMBuilder toLDSRowCol =
        TopDownTMBuilder::below(replicateLanes, replicateLanesAttr);
    {
      toLDSRowCol.passThrough({"k_loop", "g_block"});
      toLDSRowCol.passThrough({thisBlockDim}, {2}, {thisBlockDim});
      toLDSRowCol.passThrough({"kpack"}, {3}, {"kpack"});
      if (isGfx11) {
        toLDSRowCol.passThrough({"k"}, {5}, {"k_iter"});
        toLDSRowCol.ignore("block_id");
      } else {
        toLDSRowCol.unmerge({"k"}, 5, {"block_id", "k_iter"},
                            {wmmaInsn.outputStride, kpackPerThread});
      }
      toLDSRowCol.ignore(otherWaveDim);
      toLDSRowCol.unmerge("d", 4, {"d_iter", thisWaveDim, "block_td"},
                          {dRepeats, dWaves, dPerAccel});
    }
    TransformMapAttr toLDSRowColAttr = toLDSRowCol.get();
    transformAttrs.push_back(toLDSRowColAttr);
    // Fifth coordinate transform
    {
      int64_t stride = (kPack == 1 ? dInCopyPerThread : 1);
      auto offset = rotateIf(rotateDWithK, toLDSRowCol, toLDSRowColAttr, stride,
                             "d", dPerBlock, 3, "k", kPackPerBlock,
                             {"k_loop", "g_block", thisBlockDim, "kpack"},
                             {"k"}, transformAttrs);
      offset.passThrough({"G"}, {0}, {"g_block"});
      offset.unmerge({"K"}, 1, {"k_loop", "k", "kpack"},
                     {kIters, kPackPerBlock, kPack});
      offset.unmerge("D", 2, {thisBlockDim, "d"},
                     {thisDimNumBlocks, dPerBlock});
      TransformMapAttr offsetAttr = offset.get();
      transformAttrs.push_back(offsetAttr);
    }
    ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }
  // compute block sub tile transforms
  {
    StringSet<> dimensionsToRemove{"k_loop", "g_block", "m_block", "n_block"};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    ret.blockSubTile = maybeBlockSubTile.value();
  }
  // compute thread sub tile transforms
  {
    StringSet<> dimensionsToRemove{"k_loop", "g_block", "m_block", "n_block",
                                   "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    ret.threadSubTile = maybeThreadSubTile.value();
  }
  return ret;
}

void WmmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     ValueRange regCOffset, Value scaleA,
                                     Value scaleB) {
  bool isScaled = scaleA && scaleB;
  if (isScaled) {
    llvm::report_fatal_error(
        "Scaled WMMA not implemented yet for WMMA Emitter");
  }

  VectorType vectorType = wmmaInsn.retType;
  auto vectorC =
      memref::LoadOp::create(b, loc, vectorType, bufferC, regCOffset);

  // WMMAOp requires explicit m, n, k dimensions as IntegerAttrs
  auto mAttr = b.getI32IntegerAttr(wmmaInsn.mPerAccel);
  auto nAttr = b.getI32IntegerAttr(wmmaInsn.nPerAccel);
  auto kAttr = b.getI32IntegerAttr(wmmaInsn.kDim);
  auto subwordOffsetAttr = b.getI32IntegerAttr(0);

  // Clamp flag is only valid for integer output types
  Type elementType = vectorType.getElementType();
  UnitAttr clampAttr =
      isa<IntegerType>(elementType) ? b.getUnitAttr() : nullptr;

  auto mfma = amdgpu::WMMAOp::create(b, loc, vectorType, mAttr, nAttr, kAttr,
                                     argA, argB, vectorC, subwordOffsetAttr,
                                     /*unsignedA=*/nullptr,
                                     /*unsignedB=*/nullptr, clampAttr);
  auto vectorD = mfma.getDestD();

  memref::StoreOp::create(b, loc, vectorD, bufferC, regCOffset);
}

llvm::FailureOr<RegsAsMatrixSubTiles> WmmaEmitter::computeOutputTransforms(
    OpBuilder &b, Location loc, int64_t mLen, int64_t nLen, int64_t blockSize,
    ArrayRef<int64_t> bidGridLengths, int64_t inMPerThread,
    int64_t inNPerThread, bool doSwapThreadIterSubDimsForM,
    bool doSwapThreadIterSubDimsForN) {

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t mPerWave = tuningParams.getMPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  int64_t nWaves = nPerBlock / nPerWave;
  int64_t mWaves = mPerBlock / mPerWave;
  SmallVector<Attribute> transformAttrs;

  int64_t retNumElements = accVectorType.getNumElements();

  SmallVector<StringRef, 5> dimNamesM{/*0=*/"m_block",
                                      /*1=*/"rep_i",
                                      /*2=*/"wave_m"};
  if (isGfx11) {
    dimNamesM.push_back(/*3=*/"item_i");
    dimNamesM.push_back(/*4=*/"m_tid");
  } else {
    dimNamesM.push_back(/*3=*/"m_tid");
    dimNamesM.push_back(/*4=*/"item_i");
  }
  SmallVector<int64_t, 7> orderedDimStridesM{/*0=*/mPerBlock,
                                             /*1=*/mWaves * wmmaInsn.mPerAccel,
                                             /*2=*/wmmaInsn.mPerAccel};
  if (isGfx11) {
    orderedDimStridesM.push_back(/*3=*/wmmaInsn.outputStride);
  } else {
    orderedDimStridesM.push_back(/*3=*/accVectorType.getNumElements());
  }
  orderedDimStridesM.push_back(/*4=*/1);

  SmallVector<int64_t, 7> dimSizesM;
  convertDimStridestoSizes(orderedDimStridesM, mLen, dimSizesM);

  SmallVector<StringRef, 5> dimNamesN{/*0=*/"n_block",
                                      /*1=*/"rep_j",
                                      /*2=*/"wave_n",
                                      /*3=*/"n_tid"};
  SmallVector<int64_t, 5> orderedDimStridesN{/*0=*/nPerBlock,
                                             /*1=*/nWaves * wmmaInsn.nPerAccel,
                                             /*2=*/wmmaInsn.nPerAccel,
                                             /*3=*/1};
  SmallVector<int64_t, 7> dimSizesN;
  convertDimStridestoSizes(orderedDimStridesN, nLen, dimSizesN);

  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder splitMemoryCoords(
        b, {"g_block", "m_block", "n_block", "tid", "item"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], blockSize,
         mRepeats * nRepeats * retNumElements},
        loc);
    splitMemoryCoords.passThrough({"g_block", "m_block", "n_block"});
    assert(wmmaInsn.mPerAccel == wmmaInsn.nPerAccel &&
           "Currently only supported for equal mPerAccel and nPerAccel");
    splitMemoryCoords.merge(
        {"wave_m", "wave_n", "m_tid", "n_tid"}, {3, 4, 5, 6}, "tid",
        {mWaves, nWaves, waveSize / wmmaInsn.mPerAccel, wmmaInsn.mPerAccel});
    splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"}, {7, 8, 9}, "item",
                            {mRepeats, nRepeats, retNumElements});
    TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

    auto toMatrixC =
        TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
    toMatrixC.passThrough({"g_block", dimNamesM[0], dimNamesN[0]});
    toMatrixC.unmerge("gemmBlockM", 3, ArrayRef<StringRef>{dimNamesM}.slice(1),
                      ArrayRef<int64_t>{dimSizesM}.slice(1));
    toMatrixC.unmerge("gemmBlockN", 4, ArrayRef<StringRef>{dimNamesN}.slice(1),
                      ArrayRef<int64_t>{dimSizesN}.slice(1));

    SmallVector<Attribute> transformAttrs{splitMemoryCoordsAttr};
    FailureOr<TopDownTMBuilder> swapRes = mlir::rock::swapThreadIdAndIteration(
        toMatrixC, /*mBlocks=*/bidGridLengths[1], /*nBlocks=*/bidGridLengths[2],
        inMPerThread, inNPerThread, mPerBlock, nPerBlock,
        doSwapThreadIterSubDimsForM, doSwapThreadIterSubDimsForN,
        /**isBlockwise=*/false, transformAttrs);
    if (failed(swapRes))
      return failure();

    ret.gridSubTile = b.getArrayAttr(transformAttrs);
  }

  {
    // Create views as blockwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block"};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    ret.blockSubTile = maybeBlockSubTile.value();
  }

  {
    // Create views for tid slice of blockwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block", "item"};
    FailureOr<ArrayAttr> maybeBlockSubTileTidSlice =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTileTidSlice)) {
      return failure();
    }
    ret.blockSubTileTidSlice = maybeBlockSubTileTidSlice.value();
  }

  {
    // Create views as threadwise sub-tile of C
    StringSet<> dimensionsToRemove{"g_block", "m_block", "n_block", "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, ret.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    ret.threadSubTile = maybeThreadSubTile.value();
  }

  return ret;
}

std::unique_ptr<AccelEmitter>
AccelEmitter::select(GemmFeatures features, Type dataTypeA, Type dataTypeB,
                     StringRef arch,
                     RockAccelTuningParamAttrInterface tuningParams) {
  bool isMfma = rock::bitEnumContainsAll(features, GemmFeatures::mfma);
  bool isWmma = rock::bitEnumContainsAll(features, GemmFeatures::wmma);
  if (isMfma) {
    auto maybeMfmaInsnGroup = MfmaInsnGroup::select(
        dataTypeA, dataTypeB, arch, tuningParams.getMnPerXdl(),
        tuningParams.getKpack(), tuningParams.getKpackPerBlock(),
        tuningParams.getScheduleVersion());
    if (failed(maybeMfmaInsnGroup)) {
      return nullptr;
    }
    return std::make_unique<MfmaEmitter>(*maybeMfmaInsnGroup, arch,
                                         tuningParams);
  } else if (isWmma) {
    AccelGemmParamsAttr wmmaParams = cast<AccelGemmParamsAttr>(tuningParams);
    int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto maybeWmmaInsnGroup =
        WmmaInsn::select(dataTypeA, dataTypeB, waveSize, arch,
                         wmmaParams.getMPerWave(), wmmaParams.getNPerWave(),
                         wmmaParams.getKpack(), wmmaParams.getKpackPerBlock());
    if (failed(maybeWmmaInsnGroup)) {
      return nullptr;
    }
    return std::make_unique<WmmaEmitter>(*maybeWmmaInsnGroup, arch,
                                         tuningParams);
  } else {
    return nullptr;
  }
}
