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
  if (kPack > 1 && (kPack < kBase || kPack % kBase != 0)) {
    llvm_unreachable(
        "Tuning parameter selection guarantees kPack is multiple of k_base,"
        "this should never happen");
  }
}

AccelEmitter::AccelEmitter(StringRef arch, XdlopsGemmParamsAttr tuningParams) {
  mPerBlock = tuningParams.getMPerBlock();
  nPerBlock = tuningParams.getNPerBlock();
  kPerBlock = tuningParams.getKPerBlock();
  mPerWave = tuningParams.getMPerWave();
  nPerWave = tuningParams.getNPerWave();
  kPack = tuningParams.getKpack();
  waveSize = rock::lookupArchInfo(arch).waveSize;
}

// **************************
// Mfma accelerator interface
// **************************

MfmaEmitter::MfmaEmitter(MfmaInsnGroup mfmaGroup, StringRef arch,
                         XdlopsGemmParamsAttr tuningParams)
    : AccelEmitter{arch, tuningParams}, mfmaGroup{mfmaGroup} {
  MfmaInsnAttr mfmaAttr = mfmaGroup.getInsnAttr();

  // Specific mfma parameters
  int64_t K = kPerBlock * kPack;
  int64_t blocksInOutRegs = mfmaAttr.blocksInOutRegs;
  isKReduction = (blocksInOutRegs == 1) && (inputSpansPerMfmaIn > 1);
  inputSpansPerMfmaIn = mfmaAttr.inputSpansPerMfmaIn;
  inputSpanLen = mfmaAttr.inputSpanLen;

  // Accelerator parameters
  mRepeats = mfmaGroup.getMRepeats(mPerWave);
  nRepeats = mfmaGroup.getNRepeats(nPerWave);
  nResultVectors = mfmaGroup.getImms().size();
  mPerAccel = mfmaGroup.getLenPerMfmaGroup(mPerWave);
  nPerAccel = mfmaGroup.getLenPerMfmaGroup(nPerWave);
  kPerThread = (isKReduction ? kPerBlock / inputSpansPerMfmaIn : kPerBlock);
  numOutputVectorElements = mfmaGroup.getRetType().getNumElements() *
                            nResultVectors * mRepeats * nRepeats;
  kBase = mfmaAttr.k_base;
  kBasePerThread = (isKReduction ? K / inputSpansPerMfmaIn : K) / kBase;
  inputBufferSize = K / ((isKReduction ? inputSpansPerMfmaIn : 1) * kBase);

  // Accelerator data types
  argTypeA = mfmaGroup.getArgTypeA();
  argTypeB = mfmaGroup.getArgTypeB();
  accVectorType = mfmaGroup.getRetType();

  validateAcceleratorProperties();
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

ArrayAttr MfmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t M, int64_t N, int64_t blockSize,
    int64_t gridSize, Value regCAllocOp, Value convertedC) {

  auto mfmaAttr = mfmaGroup.getInsnAttr();
  int64_t mPerRepeat = mPerWave / mRepeats;
  int64_t nPerRepeat = nPerWave / nRepeats;
  int64_t nBlocks = N / nPerBlock;
  int64_t mBlocks = M / mPerBlock;
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
      "gemmM", 1, M,
      {"m", "wave_m", "m_tid", "m_i", "blk_row", "vec_group", "vec_item"},
      {mPerBlock, mPerWave, rowGroupSize, mPerRepeat, m,
       inputSpansPerMfmaIn * rowGroupSize, 1});
  toMatrixC.embed("gemmN", 2, N, {"n", "wave_n", "n_i", "blk_col", "n_tid"},
                  {nPerBlock, nPerWave, nPerRepeat, n, 1});
  TransformMapAttr toMatrixCAttr = toMatrixC.get();

  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toRowsAndColsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

Value MfmaEmitter::computeOutputConversion(PatternRewriter &b, Location loc,
                                           int64_t M, int64_t N,
                                           int64_t blockSize, int64_t gridSize,
                                           Value regVectorOrig, Value regDest,
                                           bool forceUnroll) {

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

Value MfmaEmitter::computeLdsSourceOffset(OpBuilder &kb, Value k_i,
                                          OpBuilder &mnb, Value mn_i,
                                          OpBuilder &b, Value MN, Location loc,
                                          Value sourceOffset, Value laneId) {

  if (!isKReduction) {
    // srcOffset = k_i * MN + laneId + mPerMfmaGroup * mn_i;
    Value mnPerMfmaGroup = b.create<ConstantIndexOp>(loc, nPerAccel);
    sourceOffset = b.create<AddIOp>(loc, sourceOffset, laneId);
    sourceOffset = mnb.create<AddIOp>(
        loc, sourceOffset, mnb.create<MulIOp>(loc, mnPerMfmaGroup, mn_i));
    sourceOffset =
        kb.create<AddIOp>(loc, sourceOffset, kb.create<MulIOp>(loc, MN, k_i));
  } else {
    // srcOffset = (k_i * input_span_per_mfma + blk_id) * MN + blk_td + mn_i
    // * input_span_length;
    Value inputSpanLenConstantOp = b.create<ConstantIndexOp>(loc, inputSpanLen);
    Value inputSpansPerMfmaInConstantOp =
        b.create<ConstantIndexOp>(loc, inputSpansPerMfmaIn);
    Value blk_id = b.create<DivUIOp>(loc, laneId, inputSpanLenConstantOp);
    Value blk_td = b.create<RemUIOp>(loc, laneId, inputSpanLenConstantOp);

    sourceOffset = b.create<AddIOp>(loc, sourceOffset, blk_td);
    sourceOffset = mnb.create<AddIOp>(
        loc, sourceOffset,
        mnb.create<MulIOp>(loc, inputSpanLenConstantOp, mn_i));
    sourceOffset = kb.create<AddIOp>(
        loc, sourceOffset,
        kb.create<MulIOp>(
            loc,
            kb.create<AddIOp>(
                loc, kb.create<MulIOp>(loc, k_i, inputSpansPerMfmaInConstantOp),
                blk_id),
            MN));
  }
  return sourceOffset;
}

// **************************
// Wmma accelerator interface
// **************************

WmmaEmitter::WmmaEmitter(WmmaInsn wmmaInsn, StringRef arch,
                         XdlopsGemmParamsAttr tuningParams)
    : AccelEmitter{arch, tuningParams}, wmmaInsn(wmmaInsn) {
  mRepeats = wmmaInsn.mRepeats;
  nRepeats = wmmaInsn.nRepeats;
  nResultVectors = 1;
  kPerThread = kPerBlock;
  kBase = wmmaInsn.inputLen;
  mPerAccel = wmmaInsn.inputLen;
  nPerAccel = wmmaInsn.inputLen;
  kBasePerThread = kPerBlock * kPack / kBase;
  inputBufferSize = (kPerBlock * kPack) / kBase;

  argTypeA = wmmaInsn.argTypeA;
  argTypeB = wmmaInsn.argTypeB;
  accVectorType = wmmaInsn.retType;

  numOutputVectorElements =
      (accVectorType.getNumElements()) * nResultVectors * mRepeats * nRepeats;
  reducedVectorType = accVectorType.cloneWith(accVectorType.getNumElements(),
                                              accVectorType.getElementType());
  validateAcceleratorProperties();
}

Value WmmaEmitter::computeLdsSourceOffset(OpBuilder &kb, Value k_i,
                                          OpBuilder &mnb, Value mn_i,
                                          OpBuilder &b, Value MN, Location loc,
                                          Value sourceOffset, Value laneId) {

  // srcOffset = k_i * MN + (laneId % wmmaInputLen) + wmmaInputLen * mn_i;
  Value inputLen = b.create<ConstantIndexOp>(loc, wmmaInsn.inputLen);
  sourceOffset = b.create<AddIOp>(loc, sourceOffset,
                                  b.create<RemUIOp>(loc, laneId, inputLen));
  sourceOffset = mnb.create<AddIOp>(loc, sourceOffset,
                                    mnb.create<MulIOp>(loc, inputLen, mn_i));
  sourceOffset =
      kb.create<AddIOp>(loc, sourceOffset, kb.create<MulIOp>(loc, MN, k_i));
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

ArrayAttr WmmaEmitter::computeOutputTransforms(
    PatternRewriter &b, Location loc, int64_t M, int64_t N, int64_t blockSize,
    int64_t gridSize, Value regCAllocOp, Value convertedC) {

  int64_t nBlocks = N / nPerBlock;
  int64_t mBlocks = M / mPerBlock;
  int64_t gStride = mBlocks * nBlocks;
  int64_t nWaves = nPerBlock / nPerWave;

  // High level code for this loop
  // source: https://gpuopen.com/learn/wmma_on_rdna3/
  //
  // for (int ele = 0; ele < 8; ele++){
  //    col = laneIdx % 16;
  //    row = (2*ele + laneIdx / 16);
  //    c[16 * row+col] = reg[ele];
  // }
  //
  int64_t retNumElements = reducedVectorType.getNumElements();
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

  toMatrixC.embed("gemmM", 1, M, {"m", "wave_m", "m_tid", "rep_i", "item_i"},
                  {mPerBlock, mPerWave, 1, wmmaInsn.inputLen, 2});

  toMatrixC.embed("gemmN", 2, N, {"n", "wave_n", "n_tid", "rep_j"},
                  {nPerBlock, nPerWave, 1, wmmaInsn.inputLen});

  TransformMapAttr toMatrixCAttr = toMatrixC.get();

  ArrayAttr idToMatrixCMaps =
      b.getArrayAttr({splitMemoryCoordsAttr, toMatrixCAttr});
  return idToMatrixCMaps;
}

Value WmmaEmitter::computeOutputConversion(PatternRewriter &b, Location loc,
                                           int64_t M, int64_t N,
                                           int64_t blockSize, int64_t gridSize,
                                           Value regCAllocOp, Value convertedC,
                                           bool forceUnroll) {

  auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

  Value registerC = regCAllocOp;
  Type destType = convertedC.getType().dyn_cast<MemRefType>().getElementType();

  BottomUpTMBuilder toRegCScalar(b, {"scalar"}, {numOutputVectorElements}, loc);
  toRegCScalar.embed({"vector"}, {0}, {mRepeats * nRepeats}, "scalar",
                     {reducedVectorType.getNumElements()});
  TransformMapAttr toRegCScalarAttr = toRegCScalar.get();

  auto convertLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{{zeroConstantOp}, {zeroConstantOp}},
      ArrayRef<Attribute>{b.getArrayAttr({}), b.getArrayAttr(toRegCScalarAttr)},
      /*bounds=*/ArrayRef<int64_t>{mRepeats * nRepeats},
      /*strides=*/std::nullopt, forceUnroll, /*useIndexDiffs=*/true);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(convertLoop.getBody());
    Value loaded =
        b.create<memref::LoadOp>(loc, accVectorType, registerC,
                                 convertLoop.getLowerCoords(/*domain*/ 0));
    Value cast = loaded;
    if (destType != accVectorType.getElementType()) {
      VectorType destVectorType = accVectorType.clone(destType);
      cast = createTypeConversionOp(b, loc, loaded, destVectorType);
    }
    b.create<InBoundsStoreOp>(loc, cast, convertedC,
                              convertLoop.getLowerCoords(/*domain*/ 1));
  }
  return convertedC;
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
        WmmaInsn::select(dataTypeA, dataTypeB, arch, tuningParams.getMPerWave(),
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
