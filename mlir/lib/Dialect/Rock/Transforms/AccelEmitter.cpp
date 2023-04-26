#include "AccelEmitter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

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

AccelEmitter::AccelEmitter(StringRef arch, XdlopsGemmParamsAttr tuningParams,
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
                         XdlopsGemmParamsAttr tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(mfmaGroup, tuningParams)},
      mfmaGroup{mfmaGroup} {

}

AccelEmitterParams
MfmaEmitter::initAccelEmitterParams(MfmaInsnGroup mfmaGroup,
                                    XdlopsGemmParamsAttr tuningParams) {
  AccelEmitterParams params;
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();

  // Extract relevant tuning parameters
  int64_t kPerBlock = tuningParams.getKPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();
  int64_t kPack = tuningParams.getKpack();
  int64_t K = kPerBlock * kPack;

  // Accelerator parameters
  params.kBase = mfmaAttr.k_base;
  params.kBasePerThread =
      (mfmaAttr.isKReduction ? K / mfmaAttr.inputSpansPerMfmaIn : K) / params.kBase;
  params.mRepeats = mfmaGroup.getMRepeats(mPerWave);
  params.nRepeats = mfmaGroup.getNRepeats(nPerWave);
  params.nResultVectors = mfmaGroup.getImms().size();
  params.mPerAccel = mfmaGroup.getLenPerMfmaGroup(mPerWave);
  params.nPerAccel = mfmaGroup.getLenPerMfmaGroup(nPerWave);
  params.kPerThread =
      (mfmaAttr.isKReduction ? kPerBlock / mfmaAttr.inputSpansPerMfmaIn : kPerBlock);

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

ArrayAttr MfmaEmitter::computeOutputTransforms(PatternRewriter &b, Location loc,
                                               int64_t matrixM, int64_t matrixN,
                                               int64_t blockSize,
                                               int64_t gridSize, Value regC) {

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

  auto mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mPerRepeat = mPerWave / mRepeats;
  int64_t nPerRepeat = nPerWave / nRepeats;
  int64_t nBlocks = matrixN / nPerBlock;
  int64_t mBlocks = matrixM / mPerBlock;
  int64_t gStride = mBlocks * nBlocks;
  int64_t nWaves = nPerBlock / nPerWave;
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
  int64_t wavesInKernelBlock = blockSize / waveSize;

  int64_t retNumElements = accVectorType.getNumElements();
  int64_t numElements = retNumElements * mRepeats * nRepeats * nResultVectors;
  TopDownTMBuilder splitMemoryCoords(b, {"bid", "tid", "item"},
                                     {gridSize, blockSize, numElements}, loc);
  splitMemoryCoords.merge({"g", "m", "n"}, {0, 1, 2}, {"bid"},
                          {gridSize / gStride, gStride / nBlocks, nBlocks});
  splitMemoryCoords.merge(
      {"wave", "m_tid", "n_tid"}, {3, 4, 5}, "tid",
      {wavesInKernelBlock, waveSize / inputSpanLen, inputSpanLen});
  splitMemoryCoords.merge(
      {"i", "j", "vec_group", "vec_item"}, {6, 7, 8, 9}, "item",
      {numElements / (blocksPerRepeat * rowGroupsPerBlock * rowGroupSize),
       blocksPerRepeat, rowGroupsPerBlock, rowGroupSize});
  TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

  // "blkMajor" and "blkMinor" are placeholder names because we don't know
  // if they'll be column or row until we check for broadcast-ness.
  auto toRowsAndCols =
      TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);
  llvm::StringMap<uint32_t> rowsAndColsIdxs =
      expandNamesInPlace(splitMemoryCoords, {{"wave", {"wave_m", "wave_n"}},
                                             {"i", {"m_i", "n_i"}},
                                             {"j", {"blkMajor", "blkMinor"}}});
  TopDownTMBottomDimsWrapper rowsAndColsWrap(toRowsAndCols, rowsAndColsIdxs);
  rowsAndColsWrap.passThrough({"g", "m", "n"});
  rowsAndColsWrap.merge({"wave_m", "wave_n"}, "wave",
                        {wavesInKernelBlock / nWaves, nWaves});
  rowsAndColsWrap.passThrough({"m_tid", "n_tid"});
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
  toMatrixC.passThrough({"gemmG"}, {0}, {"g"});

  toMatrixC.embed(
      "gemmM", 1, matrixM,
      {"m", "wave_m", "m_tid", "m_i", "blk_row", "vec_group", "vec_item"},
      {mPerBlock, mPerWave, rowGroupSize, mPerRepeat, m,
       inputSpansPerMfmaIn * rowGroupSize, 1});
  toMatrixC.embed("gemmN", 2, matrixN,
                  {"n", "wave_n", "n_i", "blk_col", "n_tid"},
                  {nPerBlock, nPerWave, nPerRepeat, n, 1});
  TransformMapAttr toMatrixCAttr = toMatrixC.get();

  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

Value MfmaEmitter::computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                                          OpBuilder &dBuilder, Value d_i,
                                          OpBuilder &builder, Value dPerBlock,
                                          Location loc, Value sourceBase,
                                          Value laneId) {

  // Extract relevant emitter parameters
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t nPerAccel = accelEmitterParams.nPerAccel;
  int64_t inputSpanLen = mfmaAttr.inputSpanLen;
  int64_t inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
  bool isKReduction = mfmaAttr.isKReduction;

  Value sourceOffset = sourceBase;
  if (!isKReduction) {
    // srcOffset = k_i * MN + laneId + mPerMfmaGroup * mn_i;
    Value mnPerMfmaGroup = builder.create<ConstantIndexOp>(loc, nPerAccel);
    sourceOffset = builder.create<AddIOp>(loc, sourceOffset, laneId);
    sourceOffset = dBuilder.create<AddIOp>(
        loc, sourceOffset, dBuilder.create<MulIOp>(loc, mnPerMfmaGroup, d_i));
    sourceOffset = kBuilder.create<AddIOp>(
        loc, sourceOffset, kBuilder.create<MulIOp>(loc, dPerBlock, k_i));
  } else {
    // srcOffset = (k_i * input_span_per_mfma + blk_id) * MN + blk_td + mn_i
    // * input_span_length;
    Value inputSpanLenConstantOp =
        builder.create<ConstantIndexOp>(loc, inputSpanLen);
    Value inputSpansPerMfmaInConstantOp =
        builder.create<ConstantIndexOp>(loc, inputSpansPerMfmaIn);
    Value blk_id = builder.create<DivUIOp>(loc, laneId, inputSpanLenConstantOp);
    Value blk_td = builder.create<RemUIOp>(loc, laneId, inputSpanLenConstantOp);

    sourceOffset = builder.create<AddIOp>(loc, sourceOffset, blk_td);
    sourceOffset = dBuilder.create<AddIOp>(
        loc, sourceOffset,
        dBuilder.create<MulIOp>(loc, inputSpanLenConstantOp, d_i));
    sourceOffset = kBuilder.create<AddIOp>(
        loc, sourceOffset,
        kBuilder.create<MulIOp>(
            loc,
            kBuilder.create<AddIOp>(
                loc,
                kBuilder.create<MulIOp>(loc, k_i,
                                        inputSpansPerMfmaInConstantOp),
                blk_id),
            dPerBlock));
  }
  return sourceOffset;
}

// **************************
// Wmma accelerator interface
// **************************

WmmaEmitter::WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
                         XdlopsGemmParamsAttr tuningParams)
    : AccelEmitter{arch, tuningParams,
                   initAccelEmitterParams(wmmaInsn, tuningParams)},
      wmmaInsn(wmmaInsn) {}

AccelEmitterParams
WmmaEmitter::initAccelEmitterParams(WmmaInsn wmmaInsn,
                                    XdlopsGemmParamsAttr tuningParams) {
  AccelEmitterParams params;

  // Extract relevant tuning parameters
  int64_t kPerBlock = tuningParams.getKPerBlock();
  int64_t kPack = tuningParams.getKpack();

  params.mRepeats = wmmaInsn.mRepeats;
  params.nRepeats = wmmaInsn.nRepeats;
  params.nResultVectors = 1;
  params.kPerThread = kPerBlock;
  params.kBase = wmmaInsn.inputLen;
  params.mPerAccel = wmmaInsn.inputLen;
  params.nPerAccel = wmmaInsn.inputLen;
  params.kBasePerThread = kPerBlock * kPack / params.kBase;

  params.argTypeA = wmmaInsn.argTypeA;
  params.argTypeB = wmmaInsn.argTypeB;
  params.accVectorType = wmmaInsn.retType;

  return params;
}

Value WmmaEmitter::computeLdsSourceOffset(OpBuilder &kBuilder, Value k_i,
                                          OpBuilder &dBuilder, Value d_i,
                                          OpBuilder &builder, Value dPerBlock,
                                          Location loc, Value sourceBase,
                                          Value laneId) {

  // srcOffset = k_i * MN + (laneId % wmmaInputLen) + wmmaInputLen * mn_i;
  Value inputLen = builder.create<ConstantIndexOp>(loc, wmmaInsn.inputLen);
  Value sourceOffset = builder.create<AddIOp>(
      loc, sourceBase, builder.create<RemUIOp>(loc, laneId, inputLen));
  sourceOffset = dBuilder.create<AddIOp>(
      loc, sourceOffset, dBuilder.create<MulIOp>(loc, inputLen, d_i));
  sourceOffset = kBuilder.create<AddIOp>(
      loc, sourceOffset, kBuilder.create<MulIOp>(loc, dPerBlock, k_i));
  return sourceOffset;
}

void WmmaEmitter::emitThreadwiseLoop(OpBuilder &b, Location loc, Value argA,
                                     Value argB, Value bufferC,
                                     Value regCOffset) {
  VectorType vectorType = wmmaInsn.retType;
  auto vectorC = b.create<memref::LoadOp>(loc, vectorType, bufferC, regCOffset);

  auto mfma = b.create<amdgpu::WMMAOp>(loc, vectorType, argA, argB, vectorC);
  auto vectorD = mfma.getDestD();

  b.create<memref::StoreOp>(loc, vectorD, bufferC, regCOffset);
}

ArrayAttr WmmaEmitter::computeOutputTransforms(PatternRewriter &b, Location loc,
                                               int64_t matrixM, int64_t matrixN,
                                               int64_t blockSize,
                                               int64_t gridSize, Value regC) {

  // Extract relevant tuning parameters
  int64_t mPerBlock = tuningParams.getMPerBlock();
  int64_t nPerBlock = tuningParams.getNPerBlock();
  int64_t mPerWave = tuningParams.getMPerWave();
  int64_t nPerWave = tuningParams.getNPerWave();

  // Extract relevant emitter parameters
  int64_t mRepeats = accelEmitterParams.mRepeats;
  int64_t nRepeats = accelEmitterParams.nRepeats;
  VectorType accVectorType = accelEmitterParams.accVectorType;

  int64_t nBlocks = matrixN / nPerBlock;
  int64_t mBlocks = matrixM / mPerBlock;
  int64_t gStride = mBlocks * nBlocks;
  int64_t nWaves = nPerBlock / nPerWave;

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
  TopDownTMBuilder splitMemoryCoords(
      b, {"bid", "tid", "item"},
      {gridSize, blockSize, mRepeats * nRepeats * retNumElements}, loc);
  splitMemoryCoords.merge({"g", "m", "n"}, {0, 1, 2}, {"bid"},
                          {gridSize / gStride, gStride / nBlocks, nBlocks});

  int64_t wavesInKernelBlock = blockSize / waveSize;
  splitMemoryCoords.merge({"wave_m", "wave_n", "m_tid", "n_tid"}, {3, 4, 5, 6},
                          "tid",
                          {wavesInKernelBlock / nWaves, nWaves,
                           waveSize / wmmaInsn.inputLen, wmmaInsn.inputLen});

  splitMemoryCoords.merge({"rep_i", "rep_j", "item_i"}, {7, 8, 9}, "item",
                          {mRepeats, nRepeats, retNumElements});
  TransformMapAttr splitMemoryCoordsAttr = splitMemoryCoords.get();

  auto toMatrixC =
      TopDownTMBuilder::below(splitMemoryCoords, splitMemoryCoordsAttr);

  toMatrixC.passThrough({"gemmG"}, {0}, {"g"});

  // m_tid is liimited to 0 or 1 (or 0,1,2,3 for wave64). Basically every
  // workitem is computing 8 values, which represent a sub-column of a 16x16
  // tile. For workitems 0 to 15 m_tid is 0 and they write to 0,2,4,...,14.  For
  // workitems 16 to 31 m_tid is 1 and they write at positions 1,3,5,..,15 .
  // This is outlined in https://gpuopen.com/learn/wmma_on_rdna3/
  toMatrixC.embed(
      "gemmM", 1, matrixM, {"m", "wave_m", "m_tid", "rep_i", "item_i"},
      {mPerBlock, mPerWave, 1, wmmaInsn.inputLen, wmmaInsn.outputStride});

  toMatrixC.embed("gemmN", 2, matrixN, {"n", "wave_n", "n_tid", "rep_j"},
                  {nPerBlock, nPerWave, 1, wmmaInsn.inputLen});

  TransformMapAttr toMatrixCAttr = toMatrixC.get();

  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

std::unique_ptr<AccelEmitter>
AccelEmitter::select(GemmFeatures features, Type dataTypeA, Type dataTypeB,
                     StringRef arch, XdlopsGemmParamsAttr tuningParams) {
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
    auto maybeWmmaInsnGroup =
        WmmaInsn::select(dataTypeA, dataTypeB, tuningParams.getMPerWave(),
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
