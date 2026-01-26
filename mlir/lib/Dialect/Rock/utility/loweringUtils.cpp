//===- loweringUtils.cpp - Rock utility functions -----------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/utility/math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>
using namespace mlir;
using namespace mlir::rock;

#define DEBUG_TYPE "rock-lowering-utils"

bool mlir::rock::isWrWAtomicKernel(GemmFeatures features, Type dataType,
                                   bool requiredPadding) {
  return isAccel(features) &&
         bitEnumContainsAll(features, GemmFeatures::atomic_add) &&
         (dataType.isF32() || dataType.isF16()) && !requiredPadding;
}

bool mlir::rock::is4GBMemoryType(ShapedType type) {
  if (!type.hasStaticShape())
    return true;
  int64_t elemBytes;
  if (auto shapedElemTy = dyn_cast<ShapedType>(type.getElementType()))
    elemBytes = (shapedElemTy.getNumElements() *
                 shapedElemTy.getElementTypeBitWidth()) /
                8;
  else
    elemBytes = type.getElementTypeBitWidth() / 8;

  return (type.getNumElements() * elemBytes) >
         (int64_t)std::numeric_limits<uint32_t>::max();
}

LogicalResult mlir::rock::calculateKBlockNum(const int64_t batchSize,
                                             const GemmSize &gemmSize,
                                             int64_t MPerBlock,
                                             int64_t NPerBlock,
                                             int64_t KPerBlock, int64_t KPack,
                                             int64_t num_cu, int64_t &nKBlock) {
  const int64_t gemmM = gemmSize.m;
  const int64_t gemmN = gemmSize.n;
  const int64_t gemmK = gemmSize.k;

  int64_t gemmKBlock = 1;

  assert(gemmM > 0 && gemmN > 0 && gemmK > 0);
  assert(MPerBlock > 0 && NPerBlock > 0 && KPerBlock > 0 && KPack > 0 &&
         batchSize > 0);

  if ((gemmM % MPerBlock != 0) || (gemmN % NPerBlock != 0) ||
      (gemmK % (KPerBlock * KPack) != 0))
    return failure();

  const int64_t gridSize =
      gemmSize.g * (gemmM / MPerBlock) * (gemmN / NPerBlock);
  const int64_t maxGridSize = 20 * num_cu;

  gemmKBlock = std::max(maxGridSize / gridSize, static_cast<int64_t>(1));
  gemmKBlock = std::min(gemmKBlock, batchSize);

  for (; gemmKBlock > 1; --gemmKBlock) {
    if (batchSize % gemmKBlock != 0)
      continue;

    if (gemmK % (gemmKBlock * KPerBlock * KPack) != 0)
      continue;

    break;
  }
  // not more than n
  gemmKBlock = std::min(batchSize, gemmKBlock);
  // not less than 1
  gemmKBlock = std::max((int64_t)1, gemmKBlock);

  nKBlock = gemmKBlock;
  return success();
}

bool mlir::rock::isEveryElementWrittenBwdData(ArrayRef<int64_t> strideDims,
                                              ArrayRef<int64_t> dilationDims,
                                              ArrayRef<int64_t> filterDims) {
  bool result = true;
  for (const auto &[stride, dilation, filterSize] :
       zip(strideDims, dilationDims, filterDims)) {
    if (!(dilation == 1 && stride <= filterSize))
      result = false;
  }
  return result;
}

SmallVector<int64_t>
mlir::rock::backwardDataKernelIds(ArrayRef<int64_t> strideDims,
                                  ArrayRef<int64_t> dilationDims,
                                  ArrayRef<int64_t> filterDims, bool usesV4R1) {
  assert(strideDims.size() == dilationDims.size());
  SmallVector<int64_t, 5> gcdStrideDilations;
  for (const auto &[stride, dilation] : zip(strideDims, dilationDims))
    gcdStrideDilations.push_back(math_util::gcd(stride, dilation));

  SmallVector<int64_t, 5> filTilda;
  for (const auto &[stride, gcdSD] : zip(strideDims, gcdStrideDilations))
    filTilda.push_back(stride / gcdSD);

  // Populate the kernel IDs according to the current backward data convolution
  // algorithm implementation.
  llvm::SmallVector<int64_t> kernelIds;
  int64_t subproduct = 1;
  int64_t product;
  for (size_t i = 1; i < filterDims.size(); i++)
    subproduct *= filTilda[i];
  product = subproduct * filTilda[0];
  for (int64_t kernelId = 0; kernelId < product; ++kernelId) {
    // gemmK size is different for each GEMM
    SmallVector<int64_t, 3> iTilda;
    SmallVector<int64_t, 3> iDotSlice;
    int64_t divisor = 1;
    iTilda.resize(filterDims.size());
    switch (filterDims.size()) {
    default:
      llvm_unreachable("Only 2-D and 3-D have been implemented.");
      break;
    case 3:
      divisor = filTilda[2];
      iTilda[2] = kernelId % divisor;
      [[fallthrough]];
    case 2:
      iTilda[1] = (kernelId % subproduct) / divisor;
      iTilda[0] = kernelId / subproduct;
    }
    for (size_t i = 0; i < filterDims.size(); i++)
      iDotSlice.push_back(math_util::integer_divide_ceil(
          filterDims[i] - iTilda[i], filTilda[i]));

    // gemmK must > 0, otherwise not need to run
    int64_t gemmKproduct = 1;
    for (int64_t ds : iDotSlice)
      gemmKproduct *= ds;
    if (gemmKproduct > 0) {
      kernelIds.push_back(kernelId);
    }
  }

  return kernelIds;
}

// TODO(kdrewnia): Could rank-0 vectors clear some of this up?
Type mlir::rock::vectorTypeOrSelf(Type elementType, int64_t len) {
  if (len == 1)
    return elementType;
  return VectorType::get({len}, elementType);
}

FailureOr<RegsAsMatrixSubTiles> mlir::rock::getLoadRegsAsTileViews(
    OpBuilder &b, Location loc, Value globalBuffer, StringRef dName,
    ArrayRef<int64_t> bidGridLengths, int64_t kPerBlock, int64_t dPerBlock) {
  SmallVector<StringRef, 3> bidGridOrder = {"g_block", "m_block", "n_block"};
  if (dName != "m" && dName != "n") {
    return emitError(loc, "expected dName to be m or n but got " + dName);
  }
  StringRef thisBlockDim = dName == "m" ? "m_block" : "n_block";
  StringRef otherBlockDim = dName == "m" ? "n_block" : "m_block";
  // Matrix A has shape [g, m, k] (isKFirst=false), Matrix B has shape [g, k, n]
  // (isKFirst=true)
  bool isKFirst = dName != "m";

  ShapedType matrixType = cast<ShapedType>(globalBuffer.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
  // For matrix B (isKFirst=true): k at index 1, d at index 2
  // For matrix A (isKFirst=false): k at index 2, d at index 1
  int64_t kGlobal = isKFirst ? matrixShape[1] : matrixShape[2];
  int64_t dGlobal = isKFirst ? matrixShape[2] : matrixShape[1];

  int64_t kIters = kGlobal / kPerBlock;

  std::string dIterName = llvm::formatv("{0}_iter", dName);

  std::string firstDim = dIterName;
  int firstDimLen = dPerBlock;
  std::string secondDim = "k_iter";
  int secondDimLen = kPerBlock;
  if (isKFirst) {
    std::swap(firstDim, secondDim);
    std::swap(firstDimLen, secondDimLen);
  }

  RegsAsMatrixSubTiles gpuViews;
  {
    TopDownTMBuilder toGlobalIdx(b,
                                 {"k_loop", bidGridOrder[0], bidGridOrder[1],
                                  bidGridOrder[2], firstDim, secondDim},
                                 {kIters, bidGridLengths[0], bidGridLengths[1],
                                  bidGridLengths[2], firstDimLen, secondDimLen},
                                 loc);

    toGlobalIdx.passThrough({"g"}, {0}, {"g_block"});
    // For matrix B (isKFirst): source is [g, k, n], k at index 1, n at index 2
    // For matrix A (!isKFirst): source is [g, m, k], m at index 1, k at index 2
    int kLowerIdx = isKFirst ? 1 : 2;
    int dLowerIdx = isKFirst ? 2 : 1;
    toGlobalIdx.unmerge("k", kLowerIdx, {"k_loop", "k_iter"},
                        {kIters, kPerBlock});
    toGlobalIdx.unmerge(dName, dLowerIdx, {thisBlockDim, dIterName},
                        {dGlobal / dPerBlock, dPerBlock});

    toGlobalIdx.ignore(otherBlockDim);
    TransformMapAttr toGlobalIdxAttr = toGlobalIdx.get();
    gpuViews.gridSubTile = b.getArrayAttr({toGlobalIdxAttr});
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2]};
    FailureOr<ArrayAttr> maybeBlockSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeBlockSubTile)) {
      return failure();
    }
    gpuViews.blockSubTile = maybeBlockSubTile.value();
  }
  {
    StringSet<> dimensionsToRemove{"k_loop", bidGridOrder[0], bidGridOrder[1],
                                   bidGridOrder[2], "tid"};
    FailureOr<ArrayAttr> maybeThreadSubTile =
        removeUpperDims(b, gpuViews.gridSubTile, dimensionsToRemove);

    if (failed(maybeThreadSubTile)) {
      return failure();
    }
    gpuViews.threadSubTile = maybeThreadSubTile.value();
  }
  return gpuViews;
}

Value mlir::rock::normalizeMatrix(Value matrix, OpBuilder &b, Location loc,
                                  bool doTranspose, StringRef firstDim,
                                  StringRef secondDim) {
  auto matrixType = cast<ShapedType>(matrix.getType());
  bool addGroup = matrixType.getShape().size() != 3;
  if (!addGroup && !doTranspose)
    return matrix;
  SmallVector<StringRef, 3> bottomNames;
  if (!addGroup)
    bottomNames.push_back("gemmG");
  if (doTranspose)
    bottomNames.append({secondDim, firstDim});
  else
    bottomNames.append({firstDim, secondDim});
  BottomUpTMBuilder normalizer(b, bottomNames, matrixType.getShape(), loc);

  if (addGroup)
    normalizer.addDim("gemmG", 0, 1);
  else
    normalizer.passThrough(normalizer.startName(0));

  normalizer.passThrough({firstDim, secondDim}, {1, 2}, {firstDim, secondDim});
  TransformMapAttr normalizeAttr = normalizer.get();
  return TransformOp::create(b, loc, matrix, normalizeAttr);
}

Value mlir::rock::padVector(Value vector, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad) {
  if (firstDimPad == 0)
    return vector;
  ArrayRef<int64_t> shape = cast<ShapedType>(vector.getType()).getShape();
  assert(shape.size() == 2);
  BottomUpTMBuilder padder(b, {"gemmG", firstDim}, shape, loc);
  padder.passThrough("gemmG");
  SmallString<8> paddedName;
  (firstDim + Twine("Pad")).toVector(paddedName);
  padder.pad(paddedName, firstDim, 0, firstDimPad);
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, vector, padAttr);
}

Value mlir::rock::padMatrix(Value matrix, OpBuilder &b, Location loc,
                            StringRef firstDim, int64_t firstDimPad,
                            StringRef secondDim, int64_t secondDimPad) {
  if (firstDimPad == 0 && secondDimPad == 0)
    return matrix;
  ArrayRef<int64_t> shape = cast<ShapedType>(matrix.getType()).getShape();
  BottomUpTMBuilder padder(b, {"gemmG", firstDim, secondDim}, shape, loc);
  padder.passThrough("gemmG");
  if (firstDimPad == 0) {
    padder.passThrough(firstDim);
  } else {
    SmallString<8> paddedName;
    (firstDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, firstDim, 0, firstDimPad);
  }
  if (secondDimPad == 0) {
    padder.passThrough(secondDim);
  } else {
    SmallString<8> paddedName;
    (secondDim + Twine("Pad")).toVector(paddedName);
    padder.pad(paddedName, secondDim, 0, secondDimPad);
  }
  TransformMapAttr padAttr = padder.get();
  return TransformOp::create(b, loc, matrix, padAttr);
}

template <typename AllocType>
static FailureOr<AllocType> findAlloc(Value value) {
  auto *curOp = value.getDefiningOp();
  auto maybeAllocOp = dyn_cast_or_null<AllocType>(curOp);
  while (!maybeAllocOp) {
    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp = dyn_cast_or_null<ViewLikeOpInterface>(curOp)) {
      curOp = viewOp.getViewSource().getDefiningOp();
    } else {
      return failure();
    }
    maybeAllocOp = dyn_cast_or_null<AllocType>(curOp);
  }
  if (!maybeAllocOp)
    return failure();

  return maybeAllocOp;
}

FailureOr<memref::AllocOp> mlir::rock::findMemrefAlloc(Value value) {
  return findAlloc<memref::AllocOp>(value);
}

FailureOr<BlockArgument> mlir::rock::findBlockArgument(Value value) {
  auto maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  while (!maybeBlockArg) {
    // Keep going until the operation that defines the value is a
    // view-like operation
    if (auto viewOp =
            dyn_cast_or_null<ViewLikeOpInterface>(value.getDefiningOp())) {
      value = viewOp.getViewSource();
    } else {
      return failure();
    }
    maybeBlockArg = dyn_cast_or_null<BlockArgument>(value);
  }

  return maybeBlockArg;
}

// Helper function to get attributes from parents
template <typename RetAttrType>
static FailureOr<RetAttrType> getAttrFromOpOrParents(
    Operation *op, StringRef opAttr,
    std::optional<StringRef> maybeDialectAttr = std::nullopt) {
  StringRef dialectAttr = maybeDialectAttr.value_or(opAttr);
  Operation *func = getParentFuncOp(op);
  RetAttrType attr;
  auto getAnyAttr = [&](ArrayRef<StringRef> attrNames, Operation *op) {
    for (StringRef attrName : attrNames) {
      if (!attr) {
        attr = op->getAttrOfType<RetAttrType>(attrName);
      } else {
        return;
      }
    }
  };

  // First check for the attribute on the op
  getAnyAttr({opAttr}, op);
  if (!attr) {
    // If that fails then try checking for the attribute on the func
    getAnyAttr({opAttr, dialectAttr}, func);
  }

  // If there is no desired attribute on the func, then check the nearest parent
  // with a symbol table (covers both ModuleOp and gpu::GPUModuleOp)
  if (!attr) {
    if (auto symbolTableOp = func->getParentWithTrait<OpTrait::SymbolTable>()) {
      getAnyAttr({opAttr, dialectAttr}, symbolTableOp);
      if (attr)
        return attr;
    }
  }

  if (!attr) {
    return failure();
  }
  return attr;
}

FailureOr<IntegerAttr> mlir::rock::getGridSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(op,
                                             rock::GridSizeAttr::getMnemonic());
}

FailureOr<IntegerAttr> mlir::rock::getBlockSize(Operation *op) {
  return getAttrFromOpOrParents<IntegerAttr>(
      op, rock::BlockSizeAttr::getMnemonic());
}

ReassociationIndices
mlir::rock::getReassociationForFlattening(ShapedType srcTp) {
  ReassociationIndices reassociation;
  for (int i = 0, e = srcTp.getRank(); i < e; i++)
    reassociation.push_back(i);
  return reassociation;
}

Value mlir::rock::getFlattenedMemref(OpBuilder &b, Value nonFlatMemRef) {
  Location loc = nonFlatMemRef.getLoc();
  MemRefType nonFlatMemRefType = cast<MemRefType>(nonFlatMemRef.getType());
  int64_t numElements = nonFlatMemRefType.getNumElements();
  if (nonFlatMemRefType.getRank() > 1) {
    Type nonFlatMemRefElType = nonFlatMemRefType.getElementType();
    auto flatMemRefType =
        MemRefType::get({numElements}, nonFlatMemRefElType, AffineMap{},
                        nonFlatMemRefType.getMemorySpace());
    auto reassociation = getReassociationForFlattening(nonFlatMemRefType);
    return memref::CollapseShapeOp::create(b, loc, flatMemRefType,
                                           nonFlatMemRef, reassociation);
  }
  return nonFlatMemRef;
}

TypedValue<MemRefType> mlir::rock::viewBufferAs(OpBuilder &b, Value buffer,
                                                Type elementType,
                                                ArrayRef<int64_t> dimensions) {
  Location loc = buffer.getLoc();
  Value zeroByteOffset = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  auto bufferType = cast<MemRefType>(buffer.getType());
  assert(bufferType.getRank() == 1 &&
         "Buffer type must be a 1D memref for viewBufferAs");
  assert(bufferType.getElementType() == b.getI8Type() &&
         "Buffer type must be a i8 memref for viewBufferAs");

  int64_t numBytes = bufferType.getShape()[0];
  int64_t numElements = std::accumulate(dimensions.begin(), dimensions.end(),
                                        int64_t{1}, std::multiplies<>());
  int64_t elementBitWidth =
      getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  int64_t vectorLength = isa<VectorType>(elementType)
                             ? cast<VectorType>(elementType).getNumElements()
                             : 1;
  int64_t totalBitWidthRequested = elementBitWidth * numElements * vectorLength;
  int64_t bufferBitWidth = numBytes * 8;
  assert(bufferBitWidth == totalBitWidthRequested &&
         "Can't evenly fit type into buffer");

  auto newBufferType = MemRefType::get(dimensions, elementType, nullptr,
                                       bufferType.getMemorySpace());
  auto view =
      memref::ViewOp::create(b, loc, newBufferType, buffer, zeroByteOffset,
                             /*dynamic dim sizes=*/ValueRange{});
  return TypedValue<MemRefType>(view.getResult());
}

TypedValue<MemRefType> mlir::rock::viewBufferAs(OpBuilder &b, Value buffer,
                                                Type elementType) {
  auto bufferType = cast<MemRefType>(buffer.getType());
  assert(bufferType.getRank() == 1 &&
         "Buffer type must be a 1D memref for viewBufferAs");
  assert(bufferType.getElementType() == b.getI8Type() &&
         "Buffer type must be a i8 memref for viewBufferAs");
  int64_t numBytes = bufferType.getShape()[0];
  int64_t bufferBitWidth = numBytes * 8;
  int64_t elementBitWidth =
      getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  int64_t vectorLength = isa<VectorType>(elementType)
                             ? cast<VectorType>(elementType).getNumElements()
                             : 1;
  assert(bufferBitWidth % (elementBitWidth * vectorLength) == 0 &&
         "Can't evenly fit type into buffer");
  int64_t length = bufferBitWidth / (elementBitWidth * vectorLength);
  return viewBufferAs(b, buffer, elementType, {length});
}

Value mlir::rock::gpuAlloc(OpBuilder &b, Location loc, int64_t bufferDim,
                           Type elementType,
                           gpu::AddressSpace memoryAddressSpace) {
  auto memoryAddressSpaceAttr =
      b.getAttr<gpu::AddressSpaceAttr>(memoryAddressSpace);

  // Note: we don't need to create views for register buffers, since those won't
  // have real memory accesses at the end of the day. This is important when
  // dealing with booleans and sub-byte types.
  if (memoryAddressSpace == gpu::AddressSpace::Private) {
    auto memType = MemRefType::get({bufferDim}, elementType, AffineMap{},
                                   memoryAddressSpaceAttr);
    return GpuAllocOp::create(b, loc, memType);
  }
  auto rawMemType =
      MemRefType::get({getPackedByteSize(bufferDim, elementType)},
                      b.getI8Type(), AffineMap{}, memoryAddressSpaceAttr);
  auto buffer = GpuAllocOp::create(b, loc, rawMemType);

  return viewBufferAs(b, buffer, elementType);
}

LogicalResult mlir::rock::checkLDSSize(StringAttr arch, int64_t ldsBytes) {
  // Check for arch limitations exceede
  const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
  return success(ldsBytes <= ldsSize);
}

static void traceAlloc(memref::AllocOp buffer,
                       const BufferDependencyAnalysis &deps,
                       SmallVector<BlockArgument> &args,
                       SmallVector<OpOperand *> &genericOpOperands) {
  IRRewriter rewriter(buffer.getContext());
  std::optional<llvm::SmallVector<OpOperand *>> readersOperands =
      deps.getReaders(buffer);
  if (!readersOperands.has_value())
    return;
  for (OpOperand *readerOperand : readersOperands.value()) {
    auto readOp = dyn_cast<MemoryEffectOpInterface>(readerOperand->getOwner());
    if (!readOp)
      continue;

    if (auto genericOp = dyn_cast<linalg::GenericOp>(readerOperand->getOwner()))
      genericOpOperands.push_back(readerOperand);

    SmallVector<MemoryEffects::EffectInstance> effects;
    readOp.getEffects(effects);
    for (const MemoryEffects::EffectInstance &effect : effects) {
      OpOperand *writerOperand = effect.getEffectValue<OpOperand *>();
      // Test against the write operand to guard against [MemRead, MemWrite]
      if (writerOperand && readerOperand != writerOperand &&
          isa<MemoryEffects::Write>(effect.getEffect())) {
        Value writerOperandValue = writerOperand->get();

        FailureOr<BlockArgument> maybeArg =
            findBlockArgument(writerOperandValue);
        if (succeeded(maybeArg))
          args.push_back(maybeArg.value());
        else if (memref::AllocOp writeBuffer =
                     writerOperandValue.getDefiningOp<memref::AllocOp>())
          traceAlloc(writeBuffer, deps, args, genericOpOperands);
      }
    }
  }
}

FailureOr<SmallVector<BlockArgument>>
mlir::rock::traceGemmOutputToArgs(Value matC, func::FuncOp func,
                                  const BufferDependencyAnalysis &deps) {
  if (func.getNumArguments() == 0)
    return failure();

  SmallVector<BlockArgument> args;
  auto funcArgs = func.getArguments();
  // check if matC is a kernel argument
  for (auto arg : funcArgs) {
    if (findBlockArgument(matC) == arg)
      args.push_back(arg);
  }
  assert(args.empty() || args.size() == 1);
  if (!args.empty())
    return args;

  // trace matC to its alloc
  FailureOr<memref::AllocOp> allocOp = findMemrefAlloc(matC);
  if (failed(allocOp))
    return failure();

  // trace gemm alloc to arg
  SmallVector<OpOperand *> genericOpOperands;
  traceAlloc(allocOp.value(), deps, args, genericOpOperands);
  for (auto arg : args) {
    bool containsArg =
        std::find(funcArgs.begin(), funcArgs.end(), arg) != funcArgs.end();
    assert(containsArg &&
           "Found BlockArgument does not belong to func.getArguments()");
  }
  if (!args.empty())
    return args;

  return failure();
}

FailureOr<SmallVector<OpOperand *>>
mlir::rock::traceGemmOutputToGenericOps(Value matC, func::FuncOp func,
                                        const BufferDependencyAnalysis &deps) {
  auto funcArgs = func.getArguments();
  // check if matC is a kernel argument
  for (auto arg : funcArgs) {
    // no possible linalg.generic output fusion if matC is a block arg
    if (findBlockArgument(matC) == arg)
      return {};
  }

  // trace matC to its alloc
  FailureOr<memref::AllocOp> allocOp = findMemrefAlloc(matC);
  if (failed(allocOp))
    return failure();

  // trace gemm alloc to arg, saving all genericOps
  SmallVector<OpOperand *> genericOpOperands;
  SmallVector<BlockArgument> args;
  traceAlloc(allocOp.value(), deps, args, genericOpOperands);

  return genericOpOperands;
}

llvm::FailureOr<RegsAsMatrixSubTiles>
mlir::rock::computeOutputTransforms(OpBuilder &b, Location loc,
                                    int64_t mPerBlock, int64_t nPerBlock,
                                    ArrayRef<int64_t> bidGridLengths) {
  RegsAsMatrixSubTiles ret;
  {
    // Create views as gridwise sub-tile of C
    TopDownTMBuilder toMatrixC(
        b, {"g_block", "m_block", "n_block", "m_iter", "n_iter"},
        {bidGridLengths[0], bidGridLengths[1], bidGridLengths[2], mPerBlock,
         nPerBlock},
        loc);

    toMatrixC.passThrough({"gemmG"}, {0}, {"g_block"});
    toMatrixC.unmerge("gemmM", 1, {"m_block", "m_iter"},
                      {bidGridLengths[1], mPerBlock});
    toMatrixC.unmerge("gemmN", 2, {"n_block", "n_iter"},
                      {bidGridLengths[2], nPerBlock});

    TransformMapAttr toMatrixCAttr = toMatrixC.get();
    // Before returning the output view, if necessary, swap back the
    // threadid/iter dimensions on both the M/N axis.
    SmallVector<Attribute> transformAttrs{toMatrixCAttr};

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
