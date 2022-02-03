//===- LowerMIOpenOps.cpp - MLIR MIOpen ops lowering passes ---------------===//
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
// These passes convert the MIOpen threadwise ops into constructs from the
// rest of MLIR so that they can be lowered to the GPU and LLVM dialects.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/MIOpen.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MIOpen/XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/utility/builderUtils.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"
#include <numeric>

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::miopen;

namespace {
struct LowerMIOpenOpsStep4Pass
    : public MIOpenOpsStep4PassBase<LowerMIOpenOpsStep4Pass> {
  void runOnOperation() override;
};

struct LowerMIOpenOpsStep5Pass
    : public MIOpenOpsStep5PassBase<LowerMIOpenOpsStep5Pass> {
  void runOnOperation() override;
};

// 2G ,INT MAX Value = 2147483647, use 2147483648 as offset and buffer
// store do nothing
constexpr int kTwoGB = 2147483647;

//===----------------------------------------------------------------------===//
// FIXME. XXX.
// Force the use of affine maps over index maps in the presence of padding on
// GEMM during threadwise load/store/copy when the gemm is padded due to bugs in
// the index diff map implementation (or incompletenesses in it?)
//===----------------------------------------------------------------------===//
bool overrideLoadStoreHack(const PaddingInfoAttr paddingInfo, bool original) {
  if (paddingInfo.getExtraM() > 0 || paddingInfo.getExtraK() > 0 ||
      paddingInfo.getExtraN() > 0) {
    return true;
  }
  return original;
}

//===----------------------------------------------------------------------===//
// Utility function to compute sliceLengths for threadwise_copy and
// threadwise_copy_v2 to determine the bounds of load/store loops.
//===----------------------------------------------------------------------===//
void computeSliceLengths(SmallVectorImpl<uint64_t> &sliceLengths,
                         const ArrayAttr bounds) {
  for (llvm::APInt v : bounds.getAsValueRange<IntegerAttr>()) {
    sliceLengths.push_back(v.getZExtValue());
  }
}

void computeSliceLengths(SmallVectorImpl<uint64_t> &sliceLengths,
                         const ArrayAttr sourceTransforms,
                         const ArrayAttr destTransforms, Type sourceType,
                         Type destType) {
  auto populateSliceLengthsWithTypeShape =
      [](SmallVectorImpl<uint64_t> &sliceLengths, Type type) {
        assert(type.isa<MemRefType>() || type.isa<VectorType>());
        if (type.isa<MemRefType>()) {
          // Use the shape of memref as initial slice lengths.
          for (auto dim : type.template cast<MemRefType>().getShape())
            sliceLengths.push_back(dim);
        } else if (type.isa<VectorType>()) {
          // Use the shape of vector as initial slice lengths.
          for (auto dim : type.template cast<VectorType>().getShape())
            sliceLengths.push_back(dim);
        }
      };

  // Order to decide the slice lengths:
  // - If both the source and destination have coordinate transforms,
  //    use the input shape of the source transform (this covers threadwise_copy
  //    to globals)
  // - shape of the dest in case only the source has affine transformations.
  // - shape of the source in case the source has no affine transfromations.
  if (sourceTransforms && sourceTransforms.size() > 0) {
    if (destTransforms && destTransforms.size() > 0) {
      auto firstSourceTransform = sourceTransforms[0].cast<TransformMapAttr>();
      ArrayRef<int64_t> bounds = firstSourceTransform.getUpperBounds();
      for (int64_t v : bounds) {
        assert(v >= 0 &&
               "Negative shapes not permitted, should've been caught earlier");
        sliceLengths.push_back(static_cast<uint64_t>(v));
      }
    } else {
      populateSliceLengthsWithTypeShape(sliceLengths, destType);
    }
  } else {
    populateSliceLengthsWithTypeShape(sliceLengths, sourceType);
  }
}

//===----------------------------------------------------------------------===//
// Utility function to emit load instructions with potentially OOB checks.
//===----------------------------------------------------------------------===//
Value emitLoadLogic(OpBuilder &b, Location loc, MemRefType sourceType,
                    Type loadedType, bool toEmitOOBLoadCheckLogic,
                    const ArrayRef<uint32_t> oobLoadCheckDims,
                    const Value source, const ArrayRef<Value> srcLowerIndices) {
  auto emitLoadInstruction = [&b, &loc](const ArrayRef<Value> &srcLowerIndices,
                                        MemRefType sourceType, Type loadedType,
                                        const Value &source) -> Value {
    Value loadedValue;
    if (loadedType.isa<VectorType>()) {
      // Issue vector load.
      if (sourceType.getMemorySpaceAsInt() == 0) {
        // Option 1: buffer load.
        // use buffer load if the source memref is on address space 0
        SmallVector<Value, 4> srcLowerIndicesI32;
        for (auto v : srcLowerIndices)
          srcLowerIndicesI32.push_back(
              b.create<IndexCastOp>(loc, v, b.getIntegerType(32)));
        loadedValue = b.create<gpu::MubufLoadOp>(loc, loadedType, source,
                                                 srcLowerIndicesI32);
      } else {
        // Option 2: scalar load + vector.insertelement
        VectorType loadedVectorType = loadedType.template cast<VectorType>();
        Type elementType = loadedVectorType.getElementType();
        int64_t vectorLength = loadedVectorType.getShape()[0];

        Value loadedVector =
            createZeroConstantFloatOp(b, loc, loadedVectorType);

        SmallVector<Value, 8> srcLowerIndicesUpdated;
        srcLowerIndicesUpdated.append(srcLowerIndices.begin(),
                                      srcLowerIndices.end());
        int64_t dim = sourceType.getRank() - 1;
        for (int64_t iter = 0; iter < vectorLength; ++iter) {
          auto iterIndex = b.create<ConstantIndexOp>(loc, iter);
          srcLowerIndicesUpdated[dim] =
              b.create<AddIOp>(loc, srcLowerIndices[dim], iterIndex);
          auto loadedElement = b.create<memref::LoadOp>(
              loc, elementType, source, srcLowerIndicesUpdated);

          loadedVector = b.create<vector::InsertElementOp>(
              loc, loadedVectorType, loadedElement, loadedVector, iterIndex);
        }
        loadedValue = loadedVector;
      }
    } else {
      // Issue scalar load.
      loadedValue =
          b.create<memref::LoadOp>(loc, loadedType, source, srcLowerIndices);
    }
    return loadedValue;
  };

  Value loadedValue;
  auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

  if (toEmitOOBLoadCheckLogic) {
    // Pre-populate srcLowerLoadOOBIndices. It will be modified inside
    // toEmitOOBCheckLogic basic block.
    SmallVector<Value, 8> srcLowerLoadOOBIndices;
    srcLowerLoadOOBIndices.append(srcLowerIndices.begin(),
                                  srcLowerIndices.end());

    // Emit a useful constant 0f for later use.
    Value zeroOp = createZeroConstantFloatOp(b, loc, loadedType);

    // Walkthrough all lower level indices where the dimension has
    // padding, check if the result lies within boundaries.

    // Logic in C++:
    // bool withinBounds = true;
    // for (auto dim : oobLoadCheckDims) {
    //   withBounds &=
    //     (srcLowerIndices[dim] >= 0 &&
    //      srcLowerIndices[dim] < sourceType.getShape()[dim]) {
    // }
    Value withinBoundsOp = b.create<ConstantIntOp>(loc, 1, b.getIntegerType(1));
    for (auto dim : oobLoadCheckDims) {
      Value coord = srcLowerIndices[dim];
      Value lowerBoundCheckOp =
          b.create<CmpIOp>(loc, CmpIPredicate::sge, coord, zeroConstantOp);
      Value upperBoundOp =
          b.create<ConstantIndexOp>(loc, sourceType.getShape()[dim]);
      Value upperBoundCheckOp =
          b.create<CmpIOp>(loc, CmpIPredicate::slt, coord, upperBoundOp);
      Value withinBoundInOneDimOp =
          b.create<AndIOp>(loc, lowerBoundCheckOp, upperBoundCheckOp);

      withinBoundsOp =
          b.create<AndIOp>(loc, withinBoundsOp, withinBoundInOneDimOp);

      // Prepare srcLowerLoadOOBIndices.
      srcLowerLoadOOBIndices[dim] = zeroConstantOp;
    }

    // Logic:
    // if (withinBounds) {
    //   // load address = lower indices from affine transform.
    // } else {
    //   // OOB. Prepare an address known NOT OOB.
    //   // For example, in NGCHW case:
    //   // load address = {N, G, C, 0, 0}
    //   // In NGHWC case:
    //   // load address = {N, G, 0, 0, C}
    // }
    // V = load(load address)
    // if (withinBounds) {
    //   return V
    // } else {
    //   return 0
    // }

    // Emit the first IfOp.
    auto firstIfWithinBoundsOp = b.create<scf::IfOp>(
        loc,
        TypeRange{b.getIndexType(), b.getIndexType(), b.getIndexType(),
                  b.getIndexType(), b.getIndexType()},
        withinBoundsOp, /*withElseRegion=*/true);

    // Then part.
    auto firstIfWithinBoundsThenBuilder =
        firstIfWithinBoundsOp.getThenBodyBuilder();
    firstIfWithinBoundsThenBuilder.create<scf::YieldOp>(
        loc,
        ValueRange{srcLowerIndices[0], srcLowerIndices[1], srcLowerIndices[2],
                   srcLowerIndices[3], srcLowerIndices[4]});

    // Else part.
    auto firstIfWithinBoundsElseBuilder =
        firstIfWithinBoundsOp.getElseBodyBuilder();
    firstIfWithinBoundsElseBuilder.create<scf::YieldOp>(
        loc, ValueRange{srcLowerLoadOOBIndices[0], srcLowerLoadOOBIndices[1],
                        srcLowerLoadOOBIndices[2], srcLowerLoadOOBIndices[3],
                        srcLowerLoadOOBIndices[4]});

    // Issue scalar load.
    SmallVector<Value, 8> srcLowerIndicesUpdated;
    for (unsigned iter = 0; iter < 5; ++iter)
      srcLowerIndicesUpdated.push_back(firstIfWithinBoundsOp.results()[iter]);
    loadedValue = emitLoadInstruction(srcLowerIndicesUpdated, sourceType,
                                      loadedType, source);

    // Emit the second IfOp.
    auto secondIfWithinBoundsOp =
        b.create<scf::IfOp>(loc, loadedType, withinBoundsOp, true);
    auto secondIfWithinBoundsThenBuilder =
        secondIfWithinBoundsOp.getThenBodyBuilder();
    secondIfWithinBoundsThenBuilder.create<scf::YieldOp>(loc, loadedValue);
    auto secondIfWithinBoundsElseBuilder =
        secondIfWithinBoundsOp.getElseBodyBuilder();
    secondIfWithinBoundsElseBuilder.create<scf::YieldOp>(loc, zeroOp);

    loadedValue = secondIfWithinBoundsOp.results()[0];

  } else {
    loadedValue =
        emitLoadInstruction(srcLowerIndices, sourceType, loadedType, source);
  }
  return loadedValue;
}

//===----------------------------------------------------------------------===//
// Utility function to emit store instructions with potentially OOB checks.
//===----------------------------------------------------------------------===//
void emitStoreLogic(
    BwdPaddingKernelInfo bwdPaddingInfo, OpBuilder &b, Location loc,
    MemRefType destType, Type typeToStore, bool toEmitOOBStoreCheckLogic,
    const ArrayRef<uint32_t> oobStoreCheckDims, const Value dest,
    const ArrayRef<Value> destLowerIndices, const Value value,
    InMemoryDataOperation memoryOp = InMemoryDataOperation::Set) {
  // Reduce noise of backward data padding info checks
  constexpr auto strideTwo = BwdPaddingKernelInfo::StrideTwo;
  constexpr auto isNCHW = BwdPaddingKernelInfo::isNCHW;
  constexpr auto xdlops = BwdPaddingKernelInfo::Xdlops;
  constexpr auto padM = BwdPaddingKernelInfo::PadM;
  constexpr auto padN = BwdPaddingKernelInfo::PadN;

  auto emitStoreInstruction = [&b,
                               &loc](const Value &value, MemRefType destType,
                                     Type typeToStore, const Value &dest,
                                     const ArrayRef<Value> &destLowerIndices,
                                     const Value &oob) {
    if (typeToStore.isa<VectorType>()) {
      // Issue vector store.
      if (destType.getMemorySpaceAsInt() == 0) {
        // use raw buffer store if the dest memref is on address space 0
        Value oobI32 = b.create<IndexCastOp>(loc, oob, b.getIntegerType(32));
        SmallVector<Value, 4> destLowerIndicesI32;
        for (auto v : destLowerIndices)
          destLowerIndicesI32.push_back(
              b.create<IndexCastOp>(loc, v, b.getIntegerType(32)));
        b.create<gpu::RawbufStoreOp>(loc, value, dest, oobI32,
                                     destLowerIndicesI32);
      } else {
        // Option 2: vector.extractelement + scalar store.
        assert(destType.getRank() == 1);
        assert(destLowerIndices.size() == 1);
        VectorType valueVectorType = typeToStore.template cast<VectorType>();
        Type elementType = destType.getElementType();
        int64_t vectorLength = valueVectorType.getShape()[0];
        SmallVector<Value, 8> destLowerIndicesUpdated;
        destLowerIndicesUpdated.append(destLowerIndices.begin(),
                                       destLowerIndices.end());
        for (int64_t iter = 0; iter < vectorLength; ++iter) {
          Value iterOp = b.create<ConstantIndexOp>(loc, iter);
          destLowerIndicesUpdated[0] =
              b.create<AddIOp>(loc, destLowerIndices[0], iterOp);
          auto element = b.create<vector::ExtractElementOp>(loc, elementType,
                                                            value, iterOp);
          b.create<memref::StoreOp>(loc, element, dest,
                                    destLowerIndicesUpdated);
        }
      }
    } else {
      // Issue scalar store.
      if (destType.getMemorySpaceAsInt() == 0) {
        // use raw buffer store if the dest memref is on address space 0
        SmallVector<Value, 4> destLowerIndicesI32;
        Value oobI32 = b.create<IndexCastOp>(loc, oob, b.getIntegerType(32));
        for (auto v : destLowerIndices)
          destLowerIndicesI32.push_back(
              b.create<IndexCastOp>(loc, v, b.getIntegerType(32)));
        b.create<gpu::RawbufStoreOp>(loc, value, dest, oobI32,
                                     destLowerIndicesI32);
      } else {
        b.create<memref::StoreOp>(loc, value, dest, destLowerIndices);
      }
    }
  };

  auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
  auto oobAddrOp = b.create<ConstantIndexOp>(loc, kTwoGB);

  if (toEmitOOBStoreCheckLogic) {
    SmallVector<Value, 8> destLowerStoreOOBIndices;
    destLowerStoreOOBIndices.append(destLowerIndices.begin(),
                                    destLowerIndices.end());

    // Logic in C++:
    // bool withinBounds = true;
    // for (auto dim : oobStoreCheckDims) {
    //   withBounds &=
    //     (destLowerIndices[dim] >= 0 &&
    //      destLowerIndices[dim] < destType.getShape()[dim]) {
    // }

    Value withinStoreBoundsOp =
        b.create<ConstantIntOp>(loc, 1, b.getIntegerType(1));
    for (auto dim : oobStoreCheckDims) {
      Value coordStore = destLowerIndices[dim];
      Value lowerBoundCheckOp =
          b.create<CmpIOp>(loc, CmpIPredicate::sge, coordStore, zeroConstantOp);
      Value upperBoundOp =
          b.create<ConstantIndexOp>(loc, destType.getShape()[dim]);
      Value upperBoundCheckOp =
          b.create<CmpIOp>(loc, CmpIPredicate::slt, coordStore, upperBoundOp);
      Value withinBoundInOneDimOp =
          b.create<AndIOp>(loc, lowerBoundCheckOp, upperBoundCheckOp);

      withinStoreBoundsOp =
          b.create<AndIOp>(loc, withinStoreBoundsOp, withinBoundInOneDimOp);

      destLowerStoreOOBIndices[dim] = zeroConstantOp;
    }

    auto ifWithinBoundsOp = b.create<scf::IfOp>(
        loc,
        TypeRange{b.getIndexType(), b.getIndexType(), b.getIndexType(),
                  b.getIndexType(), b.getIndexType(), b.getIndexType()},
        withinStoreBoundsOp, true);

    auto thenBuilder = ifWithinBoundsOp.getThenBodyBuilder();
    thenBuilder.create<scf::YieldOp>(
        loc, ValueRange{zeroConstantOp, destLowerIndices[0],
                        destLowerIndices[1], destLowerIndices[2],
                        destLowerIndices[3], destLowerIndices[4]});

    auto elseBuilder = ifWithinBoundsOp.getElseBodyBuilder();
    // here is workaround of backward data padding kernel to avoid compiler
    // issues.
    // Note that the xdlops kernel doesn't need a workaround unless there's
    // extra padding on the GEMM.
    // FIXME: Work out how these if statements can be consolidated
    if (bwdPaddingInfo == BwdPaddingKernelInfo::NA ||
        bwdPaddingInfo == (strideTwo | isNCHW | xdlops) ||
        bwdPaddingInfo == (strideTwo | xdlops)) {
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, destLowerStoreOOBIndices[0],
                     destLowerStoreOOBIndices[1], destLowerStoreOOBIndices[2],
                     destLowerStoreOOBIndices[3], destLowerStoreOOBIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | isNCHW)) {
      elseBuilder.create<scf::YieldOp>(
          loc, ValueRange{oobAddrOp, destLowerIndices[0], destLowerIndices[1],
                          destLowerIndices[2], zeroConstantOp, zeroConstantOp});
    } else if (bwdPaddingInfo == strideTwo) {
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, destLowerIndices[0], zeroConstantOp,
                     zeroConstantOp, destLowerIndices[3], destLowerIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | isNCHW | xdlops | padM)) {
      // c!=64 pad=0 nchw
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, destLowerIndices[0], destLowerIndices[1],
                     zeroConstantOp, destLowerIndices[3], destLowerIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | isNCHW | xdlops | padN)) {
      // n!=64 pad=0 nchw
      elseBuilder.create<scf::YieldOp>(
          loc, ValueRange{oobAddrOp, zeroConstantOp, destLowerIndices[1],
                          destLowerIndices[2], destLowerIndices[3],
                          destLowerIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | isNCHW | xdlops | padM | padN)) {
      // n!=64 c!=64 pad=0 nchw
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, zeroConstantOp, destLowerIndices[1],
                     zeroConstantOp, destLowerIndices[3], destLowerIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | isNCHW | xdlops | padN)) {
      // gemmn%64!=0 padh=padw=0 nhwc
      elseBuilder.create<scf::YieldOp>(
          loc, ValueRange{oobAddrOp, zeroConstantOp, destLowerIndices[1],
                          destLowerIndices[2], destLowerIndices[3],
                          destLowerIndices[4]});
    } else if (bwdPaddingInfo == (strideTwo | xdlops | padM)) {
      // gemmm%64!=0 padh=padw=0 nhwc
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, destLowerIndices[0], destLowerIndices[1],
                     destLowerIndices[2], destLowerIndices[3], zeroConstantOp});
    } else if (bwdPaddingInfo == (strideTwo | xdlops | padM | padN)) {
      // gemmm%64!=0 gemmN%64!=0 padh=padw=0 nhwc
      elseBuilder.create<scf::YieldOp>(
          loc,
          ValueRange{oobAddrOp, zeroConstantOp, destLowerIndices[1],
                     destLowerIndices[2], destLowerIndices[3], zeroConstantOp});
    } else {
      mlir::emitError(loc, "Unsupported backwards padding flags")
          << getBitsForBwdPaddingKernelInfo(bwdPaddingInfo);
    }

    // ifWithinBoundsOp results:
    // - 0 : oob address, 0 if inbound, 2GB if oob.
    // - 1~5 : 5D naive tensor address.
    SmallVector<Value, 8> destLowerIndicesUpdated;
    for (unsigned iter = 1; iter <= 5; ++iter)
      destLowerIndicesUpdated.push_back(ifWithinBoundsOp.getResults()[iter]);

    emitStoreInstruction(value, destType, typeToStore, dest,
                         destLowerIndicesUpdated,
                         /*oob=*/ifWithinBoundsOp.getResults()[0]);

  } else {
    if (memoryOp == InMemoryDataOperation::AtomicAdd) {
      SmallVector<Value, 8> destLowerStoreIndices;
      for (unsigned i = 0; i < destLowerIndices.size(); ++i) {
        auto dstIndex = b.create<IndexCastOp>(loc, destLowerIndices[i],
                                              b.getIntegerType(32));
        destLowerStoreIndices.push_back(dstIndex);
      }
      b.create<gpu::AtomicFAddOp>(loc, value, dest, destLowerStoreIndices);
    } else {
      emitStoreInstruction(value, destType, typeToStore, dest, destLowerIndices,
                           /*oob=*/zeroConstantOp);
    }
  }
}

//===----------------------------------------------------------------------===//
// Utility function to determine if we need to emit codes for OOB checks.
//===----------------------------------------------------------------------===//
bool obtainOOBCheckInfo(const Optional<AffineMap> composedTransform,
                        const ArrayAttr boundCheckAttr,
                        SmallVectorImpl<uint32_t> &oobCheckDims) {
  // Determine if we need to emit codes for out-of-bound check.
  bool ret = false;
  if (composedTransform && boundCheckAttr) {
    if (boundCheckAttr.size() == composedTransform->getNumResults()) {
      for (auto pair :
           llvm::enumerate(boundCheckAttr.getAsValueRange<BoolAttr>())) {
        if (pair.value()) {
          ret = true;
          oobCheckDims.push_back(pair.index());
        }
      }
    }
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Utility function to compute various information wrt threadwise_copy.
// - coordinate length : return value.
// - composed affine transform.
// - layered affine transform.
//===----------------------------------------------------------------------===//
uint32_t obtainGenericTensorTransformationInfo(
    const Type type, const ArrayAttr coordTransformsAttr,
    Optional<AffineMap> &composedTransform,
    Optional<ArrayAttr> boundsAttr = llvm::None) {
  // Infer info from a set of transformations.
  //
  // 1. If there are coordinate transforms defined, use the input rank of those
  // 2. Otherwise, use the rank of the affine map applied to the memref
  //  which is the identiy if unspecified
  // 3. A bounds attribute overrides this calculation
  assert(type.isa<MemRefType>() || type.isa<VectorType>());
  unsigned coordLength = 0;
  AffineMap typeAffineMap;
  if (type.isa<MemRefType>()) {
    MemRefType memrefType = type.cast<MemRefType>();
    coordLength = memrefType.getRank();
    if (!memrefType.getLayout().isIdentity())
      typeAffineMap = memrefType.getLayout().getAffineMap();
  } else if (type.isa<VectorType>()) {
    VectorType vectorType = type.cast<VectorType>();
    coordLength = vectorType.getShape().size();
    // Vector types doesn't have type-associated affine maps.
    // Keep typeAffineMaps uninitialized.
  }

  if (typeAffineMap) {
    coordLength = typeAffineMap.getNumInputs();
    composedTransform = typeAffineMap;
  }
  // Obtain metadata of coordinate transformations.
  if (coordTransformsAttr && coordTransformsAttr.size() > 0) {
    coordLength = coordTransformsAttr[0]
                      .cast<TransformMapAttr>()
                      .getMap()
                      .getValue()
                      .getNumInputs();
    SmallVector<AffineMap, 8> maps;
    for (auto attr : coordTransformsAttr.getAsRange<TransformMapAttr>()) {
      maps.push_back(attr.getMap().getValue());
    }
    composedTransform = composeTransforms(maps);
  }
  if (boundsAttr) {
    coordLength = boundsAttr->size();
  }
  // Return computed coordinate length.
  return coordLength;
}

//===----------------------------------------------------------------------===//
// Utility function to emit type conversion ops.
//===----------------------------------------------------------------------===//
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type sourceType, Type destType) {
  // Convert from sourceType to destType if necessary.
  Value result = source;
  Type sourceElemType = sourceType;
  Type destElemType = destType;
  if (auto sourceVec = sourceType.dyn_cast<VectorType>()) {
    if (auto destVec = destType.dyn_cast<VectorType>()) {
      assert(sourceVec.getNumElements() == destVec.getNumElements() &&
             "source and destinatioon have same length");
      sourceElemType = sourceVec.getElementType();
      destElemType = destVec.getElementType();
    } else {
      llvm_unreachable("Can't store vector sources to scalar destinations in "
                       "output writeback");
    }
  }
  if (sourceElemType != destElemType) {
    // Possible cases:
    // - fp16/bf16 -> fp32 : use fpext.
    // - fp32 -> fp16/bf16 : use fptrunc.
    // - fp16/fp32 -> bf16(i16) : use miopen.data_convert.
    // All these ops act elementwise on vectors
    if (sourceElemType.getIntOrFloatBitWidth() == 16 &&
        destElemType == b.getF32Type()) {
      result = b.create<arith::ExtFOp>(loc, source, destType);
    } else if (sourceElemType == b.getF32Type() &&
               destElemType.getIntOrFloatBitWidth() == 16) {
      result = b.create<arith::TruncFOp>(loc, source, destType);
    } else {
      llvm_unreachable("Only fp32, fp16, or bf16 targets for data conversion");
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Utility function to emit the logic to copy between naive tensors.
// This function is used within the lowering logic of threadwise_copy.
//===----------------------------------------------------------------------===//
void emitNaiveTensorCopyLogic(OpBuilder &b, Location loc, int64_t nSliceRow,
                              int64_t nSliceCol, int64_t dataPerAccess,
                              const OperandRange sourceCoord,
                              const OperandRange destCoord,
                              const Optional<AffineMap> composedSourceTransform,
                              const Optional<AffineMap> composedDestTransform,
                              Type sourceElementType, Type destElementType,
                              Value source, Value dest) {
  auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);

  SmallVector<Value, 8> srcUpperIndices;
  srcUpperIndices.append(sourceCoord.begin(), sourceCoord.end());
  SmallVector<Value, 8> destUpperIndices;
  destUpperIndices.append(destCoord.begin(), destCoord.end());

  // Emit fully-unrolled loops.
  for (unsigned ivo = 0; ivo < nSliceRow; ++ivo) {
    for (unsigned ivi = 0; ivi < nSliceCol; ivi += dataPerAccess) {
      // src_index = (0, ivo_i32, ivi_i32) + sourceCoord
      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> srcLowerIndices;
      if (composedSourceTransform)
        srcLowerIndices =
            expandAffineMap(b, loc, *composedSourceTransform, srcUpperIndices)
                .getValue();
      else
        srcLowerIndices = srcUpperIndices;

      // Load from source.
      // Issue scalar load.
      Value scalarValue = b.create<memref::LoadOp>(loc, sourceElementType,
                                                   source, srcLowerIndices);
      srcUpperIndices[sourceCoord.size() - 1] = b.create<AddIOp>(
          loc, srcUpperIndices[sourceCoord.size() - 1], oneConstantOp);
      // Convert from sourceElementType to destElementType if necessary.
      Value convertedScalarValue = createTypeConversionOp(
          b, loc, scalarValue, sourceElementType, destElementType);

      // dst_index = (0, ivo_i32, ivi_i32) + destCoord
      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> destLowerIndices;
      if (composedDestTransform)
        destLowerIndices =
            expandAffineMap(b, loc, *composedDestTransform, destUpperIndices)
                .getValue();
      else
        destLowerIndices = destUpperIndices;

      // Store to dest.
      // Issue scalar store.
      b.create<memref::StoreOp>(loc, convertedScalarValue, dest,
                                destLowerIndices);
      destUpperIndices[destCoord.size() - 1] = b.create<AddIOp>(
          loc, destUpperIndices[destCoord.size() - 1], oneConstantOp);

    } // ivi
    srcUpperIndices[1] =
        b.create<AddIOp>(loc, srcUpperIndices[1], oneConstantOp);
    destUpperIndices[1] =
        b.create<AddIOp>(loc, destUpperIndices[1], oneConstantOp);
  } // ivo
}

//===---------------------------------------------------------
// Determine if the operation provided is a constant, and return its value if it
// is
//===---------------------------------------------------------
Optional<int64_t> isConstantValue(Value v) {
  auto *op = v.getDefiningOp();
  while (auto cast = dyn_cast<IndexCastOp>(op)) {
    op = cast.in().getDefiningOp();
  }
  if (auto intOp = dyn_cast<ConstantIntOp>(op)) {
    return intOp.value();
  }
  if (auto indexOp = dyn_cast<ConstantIndexOp>(op)) {
    return indexOp.value();
  }
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// Utility function to compute index diff map.
//===----------------------------------------------------------------------===//
void computeIndexDiffMap(OpBuilder &b, Location loc,
                         const ArrayRef<Value> upperIndicesDiff,
                         const TransformMapAttr transformMap,
                         const ArrayRef<Value> lowerIndicesOriginal,
                         SmallVectorImpl<Value> &lowerIndicesDiff,
                         SmallVectorImpl<Value> &lowerIndicesUpdated) {
  Value zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
  // Obtain the shape of lower level memref.
  ArrayRef<int64_t> lowerLayerShape = transformMap.getLowerBounds();

  // Input:
  // - upper_diff
  // - upper_indices_original
  // - upper_layer_bounds
  // - lower_indices_original
  // - lower_layer_bounds
  // - F : a vector of functions mapping upper level dimensions to lower level
  // dimensions with attached metadata about how they're constructed
  //
  // Output:
  // - lower_diff : the computed diffs on the lower layer. such information
  //                would be passed to the next layer below as upper diff.
  // - lower_indices_updated : the updated lower layer indices. clients will
  //                           use the values to issue loads / stores.
  //
  // For each transform f specified in F:
  //   Let P be the upper dimensions used by f.
  //   Let Q be the lower dimensions used by f.
  //   Let T be upper_layer_bounds.
  //
  //   Switch f.type:
  //     Case Pad :
  //       |P| = |Q|
  //       For each i in P, and its counterpart j in Q
  //         lower_diff[j] = upper_diff[i]
  //         lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
  //
  //     Case PassThrough :
  //       |P| = |Q|
  //       For each i in P, and its counterpart j in Q
  //         lower_diff[j] = upper_diff[i]
  //         lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
  //
  //     Case Slice :
  //       |P| = |Q|
  //       For each i in P, and its counterpart j in Q
  //         lower_diff[j] = upper_diff[i]
  //         lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
  //
  //     Case Embed:
  //       |P| = k, currently k will be >= 2.
  //       |Q| shall be 1
  //       Let (p_{0}, ... , p_{k-1}) be elements in P, |P| = k
  //       Let (e_{0}, ... , e_{k-1}) be parameters of P
  //       Let j be the counterpart in q
  //       lower_diff[j] = sum_over_P(e_{i} * upper_diff[p_{i}])
  //       lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
  //
  //     Case UnMerge:
  //       |Q| shall be 1
  //       Let (p_{0}, ... , p_{k-1}) be elements in P, |P| = k
  //       Let (e_{0}, ... , e_{k-1}) be parameters of P
  //       Let (f_{0}, ... , f_{k-1})
  //         The value of f_{i} is defined as:
  //           f_{k-1} = 1
  //           f_{i} = mul_over_{domain: e_[i+1 .. k-1], iterator=l}(T_{l})
  //       Let j be the counterpart in q
  //         lower_diff[j] = sum_over_P(f_{i} * upper_diff[p_{i}])
  //         lower_indices_updated[j] = lower_indices_origina[j] + lower_diff[j]
  //
  //     Case Unfold:
  //       This transformation is currently only used on filter, when c/y/x
  //       dimensions are together.
  //       |P| shall be 1
  //       Let (q_{0}, ... , q_{k-1}) be elements in Q, |Q| = k
  //       Let (f_{0}, ... , f_{k-1}) be elements in F to compute from P to Q
  //       For each i in Q,
  //         lower_diff_tilda[i] = f_{i}(upper_diff)
  //       For each i in Q,
  //         lower_indices_modified[i] = lower_indices_original[i] +
  //           lower_diff_tilda[i]
  //       lower_diff = lower_diff_tilda
  //       lower_indices_updated = lower_indices_modified
  //
  //     Case Merge:
  //       |P| shall be 1
  //       Let (q_{0}, ... , q_{k-1}) be elements in Q, |Q| = k
  //       Let (f_{0}, ... , f_{k-1}) be elements in F to compute from P to Q
  //       For each i in Q,
  //         lower_diff_tilda[i] = f_{i}(upper_diff)
  //       For each i in Q,
  //         lower_indices_modified[i] = lower_indices_original[i] +
  //           lower_diff_tilda[i]
  //       For each i in Q, starting from i-1 down to 0 in descending order
  //         lower_indices_carrychecked[i] = carry/overflow check for
  //           lower_indices_modified[i]
  //       lower_diff = lower_indices_carrychecked - lower_indices_original
  //       lower_indices_updated = lower_indices_carrychecked
  //

  // llvm::errs() << "Transform metadata:\n";
  // llvm::errs() << transformMetadata << "\n";
  // llvm::errs() << "Upper indices diff size: "
  //              << upperIndicesDiff.size() << "\n";
  // llvm::errs() << "Lower indices original size: "
  //              << lowerIndicesOriginal.size() << "\n\n";

  // Look into layout attribute inside transform metadata.

  // lower level diff map
  // key : lower level dimension value.
  // value : lower level diff on that dimension.
  DenseMap<uint32_t, Value> lowerIndicesDiffMap;

  // lower level updated coordinate map
  // key : lower level dimension value.
  // value : lower level updated coordinate on that dimension.
  DenseMap<uint32_t, Value> lowerIndicesUpdatedMap;

  auto addToOriginal = [&b, loc](Value original, Value diff) -> Value {
    auto mbDiffConst = isConstantValue(diff);
    if (mbDiffConst.hasValue()) {
      int64_t diff = mbDiffConst.getValue();
      if (diff == 0) {
        return original;
      }
      auto mbOriginalConst = isConstantValue(original);
      if (mbOriginalConst.hasValue()) {
        return b.create<ConstantIndexOp>(loc,
                                         diff + mbOriginalConst.getValue());
      }
    }
    return b.create<AddIOp>(loc, original, diff);
  };

  // Iterate through all transformations specified in g.
  for (auto mapping : transformMap.getOps()) {
    // llvm::errs() << "f: " << f << "\n";

    // Obtain transformation information from f.
    TransformType transformation = mapping.getType();
    ArrayRef<uint32_t> p = mapping.getUpperDims();
    ArrayRef<uint32_t> q = mapping.getLowerDims();
    ArrayRef<int64_t> e = mapping.getParams();

    if (transformation == TransformType::Embed) {
      assert(e.size() == p.size());
      assert(q.size() == 1);
      Value lowerDiff = zeroConstantOp;
      for (unsigned iter = 0; iter < e.size(); ++iter) {
        int64_t coefficient = e[iter];
        uint32_t upperDim = p[iter];
        auto mbUpperDiff = isConstantValue(upperIndicesDiff[upperDim]);
        auto mbLowerDiff = isConstantValue(lowerDiff);
        if (mbUpperDiff.hasValue() && mbLowerDiff.hasValue()) {
          lowerDiff = b.create<ConstantIndexOp>(
              loc,
              mbLowerDiff.getValue() + coefficient * mbUpperDiff.getValue());
        } else {
          lowerDiff = b.create<AddIOp>(
              loc, lowerDiff,
              b.create<MulIOp>(loc, b.create<ConstantIndexOp>(loc, coefficient),
                               upperIndicesDiff[upperDim]));
        }
      }

      uint32_t lowerDim = q[0];
      lowerIndicesDiffMap[lowerDim] = lowerDiff;
      lowerIndicesUpdatedMap[lowerDim] =
          addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
    } else if (transformation == TransformType::Unmerge) {
      assert(e.size() == p.size());
      assert(q.size() == 1);
      uint32_t upperDim = p[0];
      Value lowerDiff = upperIndicesDiff[upperDim];
      for (unsigned iter = 1; iter < e.size(); ++iter) {
        int64_t coefficient = e[iter];
        uint32_t upperDim = p[iter];
        auto mbUpperDiff = isConstantValue(upperIndicesDiff[upperDim]);
        auto mbLowerDiff = isConstantValue(lowerDiff);
        if (mbUpperDiff.hasValue() && mbLowerDiff.hasValue()) {
          lowerDiff = b.create<ConstantIndexOp>(
              loc,
              mbUpperDiff.getValue() + coefficient * mbLowerDiff.getValue());
        } else {
          lowerDiff = b.create<AddIOp>(
              loc, upperIndicesDiff[upperDim],
              b.create<MulIOp>(loc, b.create<ConstantIndexOp>(loc, coefficient),
                               lowerDiff));
        }
      }
      uint32_t lowerDim = q[0];
      lowerIndicesDiffMap[lowerDim] = lowerDiff;
      lowerIndicesUpdatedMap[lowerDim] =
          addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
    } else if ((transformation == TransformType::PassThrough) ||
               (transformation == TransformType::Pad) ||
               (transformation == TransformType::Slice)) {
      assert(p.size() == q.size());
      for (unsigned iter = 0; iter < q.size(); ++iter) {
        uint32_t upperDim = p[iter];
        uint32_t lowerDim = q[iter];
        Value upperDiff = upperIndicesDiff[upperDim];
        Value lowerDiff = upperDiff;
        lowerIndicesDiffMap[lowerDim] = lowerDiff;
        lowerIndicesUpdatedMap[lowerDim] =
            addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiff);
      }
    } else if ((transformation == TransformType::Merge) ||
               (transformation == TransformType::Unfold)) {
      assert(p.size() == 1);
      uint32_t upperDim = p[0];

      // Obtain the affine map underlying the transform.
      AffineMap affineMap = transformMap.getMap().getAffineMap();

      SmallVector<Value, 8> lowerDiffModified;
      auto mbUpperDiffVal = isConstantValue(upperIndicesDiff[upperDim]);
      if (mbUpperDiffVal.hasValue()) {
        // In case upper level diff is a constant, use constantFold.
        int64_t upperDiff = mbUpperDiffVal.getValue();

        // Populate an upper diff vector with all indices 0, other than
        // upperDim dimension set as upperDiff.
        SmallVector<Attribute, 8> upperDiffModified;
        for (unsigned iter = 0; iter < upperIndicesDiff.size(); ++iter) {
          int64_t v = (iter == upperDim) ? upperDiff : 0;
          upperDiffModified.push_back(b.getI32IntegerAttr(v));
        }
        assert(upperDiffModified.size() == upperIndicesDiff.size());

        // Apply map to compute index lower diff, from index upper diff using
        // constantFold.
        SmallVector<Attribute, 8> lowerDiffModifiedAttr;
        (void)affineMap.constantFold(upperDiffModified, lowerDiffModifiedAttr);
        assert(lowerDiffModifiedAttr.size() == lowerIndicesOriginal.size());

        for (uint32_t iter = 0; iter < lowerDiffModifiedAttr.size(); ++iter) {
          lowerDiffModified.push_back(
              b.create<ConstantIndexOp>(loc, lowerDiffModifiedAttr[iter]
                                                 .template cast<IntegerAttr>()
                                                 .getInt()));
        }
        assert(lowerDiffModified.size() == lowerIndicesOriginal.size());
      } else {
        // In case upper level diff is not constant, use expandAffineMap.

        Value upperDiff = upperIndicesDiff[upperDim];

        // Populate an upper diff vector with all indices 0, other than
        // upperDim dimension set as upperDiff.
        SmallVector<Value, 8> upperDiffModified;
        for (uint32_t iter = 0; iter < upperIndicesDiff.size(); ++iter) {
          Value v = (iter == upperDim) ? upperDiff : zeroConstantOp;
          upperDiffModified.push_back(v);
        }
        assert(upperDiffModified.size() == upperIndicesDiff.size());

        // Apply map to compute index lower diff, from index upper diff using
        // expandAffineMap.
        lowerDiffModified =
            expandAffineMap(b, loc, affineMap, upperDiffModified).getValue();
        assert(lowerDiffModified.size() == lowerIndicesOriginal.size());
      }

      // Obtain lower diffs prior to carry check.
      SmallVector<Value, 8> lowerDiffs;
      for (unsigned iter = 0; iter < q.size(); ++iter) {
        uint32_t lowerDim = q[iter];
        Value lowerDiff = lowerDiffModified[lowerDim];
        lowerDiffs.push_back(lowerDiff);
      }
      assert(lowerDiffs.size() == q.size());

      // Compute updated lower indices by adding original lower indices with
      // lower diffs.
      SmallVector<Value, 8> lowerIndicesModified;
      for (uint32_t iter = 0; iter < q.size(); ++iter) {
        uint32_t lowerDim = q[iter];
        lowerIndicesModified.push_back(
            addToOriginal(lowerIndicesOriginal[lowerDim], lowerDiffs[iter]));
      }
      assert(lowerIndicesModified.size() == q.size());

      // Add carry check for Merge.
      // For Unfold it's not needed.
      if (transformation == TransformType::Merge) {
        // Carry checked lower indices.
        // FIXME: study how to properly lowerDiffsCarryChecked.
        DenseMap<uint32_t, Value> lowerDiffsCarryChecked;
        DenseMap<uint32_t, Value> lowerIndicesCarryChecked;
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          int64_t lowerDim = q[iter];
          lowerDiffsCarryChecked[lowerDim] = lowerDiffs[iter];
          lowerIndicesCarryChecked[lowerDim] = lowerIndicesModified[iter];
        }
        assert(lowerDiffsCarryChecked.size() == lowerIndicesModified.size());
        assert(lowerIndicesCarryChecked.size() == lowerIndicesModified.size());

        // We only implement carry logic. Borrow logic would never happen as
        // upper index diffs would always be positive in the current algorithm.
        Value overflowOp = zeroConstantOp;
        for (ssize_t iter = q.size() - 1; iter >= 0; --iter) {
          uint32_t lowerDim = q[iter];
          int64_t upperBound = e[iter];
          // If the overflow is statically 0, nothing gets added
          Value diff =
              addToOriginal(lowerDiffsCarryChecked[lowerDim], overflowOp);
          Value index =
              addToOriginal(lowerIndicesCarryChecked[lowerDim], overflowOp);

          // Don't generate overflow for the uppermost dimension,
          // as this can lead to oob loads
          if (iter == 0) {
            lowerDiffsCarryChecked[lowerDim] = diff;
            lowerIndicesCarryChecked[lowerDim] = index;
            continue;
          }
          auto mbConstantDiff = isConstantValue(diff);
          auto mbConstantIndex = isConstantValue(index);

          // If we get lucky, everything is constant and so we have a constant
          // result
          if (mbConstantIndex.hasValue() && mbConstantDiff.hasValue()) {
            int64_t index = mbConstantIndex.getValue();
            int64_t diff = mbConstantDiff.getValue();
            if (index < upperBound) {
              overflowOp = zeroConstantOp;
              lowerIndicesCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, index);
              lowerDiffsCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, diff);
            } else {
              int64_t carry = index / upperBound;
              int64_t newIndex = index % upperBound;
              int64_t newDiff = diff - (carry * upperBound);
              overflowOp = b.create<ConstantIndexOp>(loc, carry);
              lowerIndicesCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, newIndex);
              lowerDiffsCarryChecked[lowerDim] =
                  b.create<ConstantIndexOp>(loc, newDiff);
            }
            continue;
          }
          // No change -> no carry-out
          if (mbConstantDiff.getValueOr(-1L) == 0) {
            overflowOp = zeroConstantOp;
            lowerDiffsCarryChecked[lowerDim] = diff;
            lowerIndicesCarryChecked[lowerDim] = index;
            continue;
          }

          Value upperBoundOp = b.create<ConstantIndexOp>(loc, upperBound);
          Value carry = b.create<DivUIOp>(loc, index, upperBoundOp);
          Value newIndex = b.create<RemUIOp>(loc, index, upperBoundOp);
          // If the merge is, as is typical, near the end of the transformations
          // this computation should get hit by the dead code eleminator
          Value newDiff = b.create<SubIOp>(
              loc, diff, b.create<MulIOp>(loc, carry, upperBoundOp));

          overflowOp = carry;
          lowerDiffsCarryChecked[lowerDim] = newDiff;
          lowerIndicesCarryChecked[lowerDim] = newIndex;
        }

        assert(lowerDiffsCarryChecked.size() == lowerIndicesModified.size());
        assert(lowerIndicesCarryChecked.size() == lowerIndicesModified.size());
        lowerDiffs.clear();
        lowerIndicesModified.clear();
        for (uint32_t iter = 0; iter < q.size(); ++iter) {
          uint32_t lowerDim = q[iter];
          lowerDiffs.push_back(lowerDiffsCarryChecked[lowerDim]);
          lowerIndicesModified.push_back(lowerIndicesCarryChecked[lowerDim]);
        }
        assert(lowerDiffs.size() == q.size());
        assert(lowerIndicesModified.size() == q.size());
      }

      // Set lowerIndicesDiffMap and lowerIndicesUpdatedMap.
      for (uint32_t iter = 0; iter < q.size(); ++iter) {
        int64_t lowerDim = q[iter];
        lowerIndicesDiffMap[lowerDim] = lowerDiffs[iter];
        lowerIndicesUpdatedMap[lowerDim] = lowerIndicesModified[iter];
      }
    } else if (transformation == TransformType::AddDim) {
      // Do nothing - the dimension will be dropped by the code below
    }
  } // for (auto mapping : transforms.getOps())

  // Convert lowerIndicesDiffMap to lowerIndicesDiff.
  assert(lowerIndicesDiffMap.size() == lowerLayerShape.size());
  for (unsigned iter = 0; iter < lowerLayerShape.size(); ++iter)
    lowerIndicesDiff.push_back(lowerIndicesDiffMap[iter]);

  // Convert lowerIndicesUpdatedMap to lowerIndicesUpdated.
  assert(lowerIndicesUpdatedMap.size() == lowerLayerShape.size());
  for (unsigned iter = 0; iter < lowerLayerShape.size(); ++iter)
    lowerIndicesUpdated.push_back(lowerIndicesUpdatedMap[iter]);
}

//===----------------------------------------------------------------------===//
// Utility function to progressively use index diff map to compute the
// coordinate at the bottom most layer.
//===----------------------------------------------------------------------===//
void computeBottomIndicesWithIndexDiffMap(
    OpBuilder &b, Location loc,
    const ArrayRef<SmallVector<Value, 8>> layeredIndicesOrig,
    const ArrayRef<int64_t> loopIVs, const ArrayAttr transforms,
    SmallVectorImpl<Value> &bottomIndices) {
  SmallVector<Value, 8> topDiff;
  for (unsigned iter = 0; iter < loopIVs.size(); ++iter)
    topDiff.push_back(b.create<ConstantIndexOp>(loc, loopIVs[iter]));
  if (!transforms || transforms.size() == 0) {
    // No transformations mean that the updates to the lower coordinates are the
    // updates to the upper coordinates
    assert(layeredIndicesOrig.size() == 1 &&
           "With no transforms, the layered values are {upper (aka lower)}");
    const SmallVector<Value, 8> &bottomOrig = layeredIndicesOrig[0];
    assert(bottomOrig.size() == loopIVs.size() &&
           "One loop variable needed per upper index");
    bottomIndices.clear();
    for (auto pair : llvm::zip(bottomOrig, topDiff)) {
      bottomIndices.push_back(
          b.create<AddIOp>(loc, std::get<0>(pair), std::get<1>(pair)));
    }
  } else {
    SmallVector<Value, 8> upperDiffs = topDiff;
    SmallVector<Value, 8> lowerDiffs, lowerIndices;
    for (auto pair : llvm::zip(transforms.getAsRange<TransformMapAttr>(),
                               layeredIndicesOrig.slice(1))) {
      TransformMapAttr mapping = std::get<0>(pair);
      const SmallVector<Value, 8> &lowerOrig = std::get<1>(pair);
      lowerDiffs.clear();
      lowerIndices.clear();
      computeIndexDiffMap(b, loc, upperDiffs, mapping, lowerOrig, lowerDiffs,
                          lowerIndices);
      upperDiffs.clear();
      upperDiffs.append(lowerDiffs);
    }
    // Having applied all the transforms, output the result of the last one
    bottomIndices = lowerIndices;
  }
}

//===----------------------------------------------------------------------===//
// Utility function to repeatedly apply affine transformation to compute the
// coordinate for the next layer.
//===----------------------------------------------------------------------===//
void populateLayeredIndicesWithTransformMetadata(
    OpBuilder &b, Location loc,
    SmallVectorImpl<SmallVector<Value, 8>> &layeredIndices,
    const ArrayRef<Value> topIndices, const ArrayAttr transforms) {
  SmallVector<Value, 8> currentIndices;
  currentIndices.append(topIndices.begin(), topIndices.end());
  layeredIndices.push_back(currentIndices);

  if (!transforms || transforms.size() == 0) {
    // In case there is no metadata, simply return. The top layer indices have
    // recorded earlier.
    return;
  }
  // Go through each layer of transform metadata, fetch the map attribute
  // and apply it to obtain the indices for the next layer.
  for (auto mapping : transforms.getAsRange<TransformMapAttr>()) {
    AffineMap am = mapping.getMap().getAffineMap();
    SmallVector<Value, 8> nextLayerIndices =
        expandAffineMap(b, loc, am, currentIndices).getValue();
    layeredIndices.push_back(nextLayerIndices);

    currentIndices.clear();
    currentIndices = nextLayerIndices;
  }
}

//===----------------------------------------------------------------------===//
// Utility function to compute the uppermost layer and bottommost layer
// coorindates using affine map.
//===----------------------------------------------------------------------===//
void computeBottomIndicesWithAffineMap(OpBuilder &b, Location &loc,
                                       SmallVectorImpl<Value> &bottomIndices,
                                       const ArrayRef<Value> originalCoords,
                                       const ArrayRef<int64_t> loopIVs,
                                       Optional<AffineMap> map) {
  // Compute high-level coordinate.
  // index = (iv_0, iv_1, ...) + originalCoords
  SmallVector<Value, 8> topIndices;
  topIndices.append(originalCoords.begin(), originalCoords.end());

  for (unsigned iter = 0; iter < loopIVs.size(); ++iter) {
    auto loopIV = b.create<ConstantIndexOp>(loc, loopIVs[iter]);
    topIndices[iter] = b.create<AddIOp>(loc, loopIV, topIndices[iter]);
  }
  if (!map.hasValue() || !(map.getValue())) {
    bottomIndices.clear();
    bottomIndices.append(topIndices);
  } else {
    SmallVector<Value, 8> mapResults =
        expandAffineMap(b, loc, map.getValue(), topIndices).getValue();
    bottomIndices.clear();
    bottomIndices.append(mapResults);
  }
}

//===----------------------------------------------------------------------===//
// ThreadwiseCopy lowering.
//===----------------------------------------------------------------------===//

struct ThreadwiseCopyRewritePattern
    : public OpRewritePattern<ThreadwiseCopyOp> {
  using OpRewritePattern<ThreadwiseCopyOp>::OpRewritePattern;

  // NOTE: when extending this logic to support vectors
  // ensure the results of the non-xdlops gemm are stored in a vectorizable
  // layout. This'll likely require something analogous to the in_warp_transpose
  // call in the xdlops case
  LogicalResult matchAndRewrite(ThreadwiseCopyOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    int64_t globalArg = op.globalArg().getSExtValue();

    auto sourceElementType =
        op.source().getType().cast<MemRefType>().getElementType().cast<Type>();
    auto destElementType =
        op.dest().getType().cast<MemRefType>().getElementType().cast<Type>();

    auto sourceType = op.source().getType().cast<MemRefType>();
    auto destType = op.dest().getType().cast<MemRefType>();

    ArrayAttr boundsAttr = op->getAttr("bounds").dyn_cast_or_null<ArrayAttr>();
    Optional<ArrayAttr> mbBoundsAttr = llvm::None;
    if (boundsAttr) {
      mbBoundsAttr = boundsAttr;
    }
    // Debug switches.
    // true : use the slow but proven affine map.
    // false : use the faster index diff map.
    auto legacyLoadAttr = op->getAttr("legacy_load");
    auto legacyStoreAttr = op->getAttr("legacy_store");
    bool legacyLoad =
        (legacyLoadAttr &&
         legacyLoadAttr.template cast<BoolAttr>().getValue() == true);
    bool legacyStore =
        (legacyStoreAttr &&
         legacyStoreAttr.template cast<BoolAttr>().getValue() == true);

    ArrayAttr sourceTransforms = op.transforms()[0].cast<ArrayAttr>();
    ArrayAttr destTransforms = op.transforms()[1].cast<ArrayAttr>();

    Optional<AffineMap> composedSourceTransform;
    Optional<AffineMap> composedDestTransform;

    // Obtain coordinate lengths, as well as information of affine
    // transformations.
    uint32_t sourceCoordLength = obtainGenericTensorTransformationInfo(
        sourceType, sourceTransforms, composedSourceTransform, mbBoundsAttr);
    auto sourceCoord = op.sourceCoord();
    if (sourceCoordLength != sourceCoord.size()) {
      return op.emitOpError("Got " + Twine(sourceCoord.size()) +
                            " source coordinates but expected " +
                            Twine(sourceCoordLength));
    }

    uint32_t destCoordLength = obtainGenericTensorTransformationInfo(
        destType, destTransforms, composedDestTransform, mbBoundsAttr);
    auto destCoord = op.destCoord();
    if (destCoordLength != destCoord.size()) {
      return op.emitOpError("Got " + Twine(destCoord.size()) +
                            " dest coordinates but expected " +
                            Twine(destCoordLength));
    }

    // FIXME. XXX.
    legacyLoad = overrideLoadStoreHack(op.paddingInfo(), legacyLoad);
    legacyStore = overrideLoadStoreHack(op.paddingInfo(), legacyStore);

    // Determine if we need to emit codes for out-of-bound check, and which
    // dimensions need to dconduct such check.
    ArrayAttr boundCheckSourceAttr, boundCheckDestAttr;
    if (globalArg == 0) {
      boundCheckSourceAttr = op.oobDimsAttr();
    } else if (globalArg == 1) {
      boundCheckDestAttr = op.oobDimsAttr();
    } else {
      // No global memrefs means no oob loads/stores
    }
    SmallVector<uint32_t, 8> oobLoadCheckDims;
    bool toEmitOOBLoadCheckLogic = obtainOOBCheckInfo(
        composedSourceTransform, boundCheckSourceAttr, oobLoadCheckDims);
    SmallVector<uint32_t, 8> oobStoreCheckDims;
    bool toEmitOOBStoreCheckLogic = obtainOOBCheckInfo(
        composedDestTransform, boundCheckDestAttr, oobStoreCheckDims);

    // Distinguish between generic <-> naive v naive <-> naive tensors.
    auto NSliceRowAttr = op->getAttr("n_slice_row");
    auto NSliceColAttr = op->getAttr("n_slice_col");
    auto DataPerAccessAttr = op->getAttr("data_per_access");
    if (NSliceRowAttr && NSliceColAttr && DataPerAccessAttr) {
      // In cases where attributes n_slice_row/n_slice_col/data_per_access are
      // specified, source and dest memrefs are all on LDS or VGPR, use the
      // simpler algorithm because they are all naive tensors.
      int64_t NSliceRow = NSliceRowAttr.template cast<IntegerAttr>().getInt();
      int64_t NSliceCol = NSliceColAttr.template cast<IntegerAttr>().getInt();
      int64_t DataPerAccess =
          DataPerAccessAttr.template cast<IntegerAttr>().getInt();

      emitNaiveTensorCopyLogic(b, loc, NSliceRow, NSliceCol, DataPerAccess,
                               sourceCoord, destCoord, composedSourceTransform,
                               composedDestTransform, sourceElementType,
                               destElementType, op.source(), op.dest());
    } else {
      // Otherwise, employ the more elaborated algorithm.

      // llvm::errs() << "\nthreadwise_copy op:\n";
      // op.dump();
      // llvm::errs() << "\n";

      // Figure out the bounds of load/store loops.
      SmallVector<uint64_t, 6> sliceLengths;

      if (mbBoundsAttr) {
        computeSliceLengths(sliceLengths, boundsAttr);
      } else {
        computeSliceLengths(sliceLengths, sourceTransforms, destTransforms,
                            sourceType, destType);
      }

      // llvm::errs() << "slice lengths: ";
      // for (unsigned i = 0; i < sliceLengths.size(); ++i)
      //   llvm::errs() << sliceLengths[i] << " ";
      // llvm::errs() << "\n";

      // Copy coordinates into vectors so that we don't have issues
      // passing ArrayRef<Value>s
      SmallVector<Value, 8> srcUpperIndices;
      srcUpperIndices.append(sourceCoord.begin(), sourceCoord.end());
      SmallVector<Value, 8> destUpperIndices;
      destUpperIndices.append(destCoord.begin(), destCoord.end());

      SmallVector<Value, 8> srcLowerIndices;
      SmallVector<Value, 8> destLowerIndices;
      // Coordinates across the layers of transformations.
      // If the vector is of size n, 0 is the top layer, and
      // n-1 is the bottom layer.
      SmallVector<SmallVector<Value, 8>, 2> layeredSourceIndices;
      SmallVector<SmallVector<Value, 8>, 2> layeredDestIndices;

      // Obtain transform metadata and populate coordinates for all layers
      // wthe the metadata.
      // Only do such computation in the new approach where index diff maps
      // would be used.
      if (legacyLoad == false) {
        // Populate coorindates across the layers of transformations.
        populateLayeredIndicesWithTransformMetadata(
            b, loc, layeredSourceIndices, srcUpperIndices, sourceTransforms);

        // Fetch low-level coordinate.
        srcLowerIndices = layeredSourceIndices[layeredSourceIndices.size() - 1];
      }

      // Obtain transform metadata and populate coordinates for all layers
      // wthe the metadata.
      // Only do such computation in the new approach where index diff maps
      // would be used.
      if (legacyStore == false) {
        // Populate coorindates across the layers of transformations.
        populateLayeredIndicesWithTransformMetadata(
            b, loc, layeredDestIndices, destUpperIndices, destTransforms);

        // Fetch low-level coordinate.
        destLowerIndices = layeredDestIndices[layeredDestIndices.size() - 1];
      }

      // Emit fully unrolled loops for vector loads / stores.
      SmallVector<int64_t, 8> loopIVs;
      SmallVector<int64_t, 8> loopBounds;
      for (unsigned iter = 0; iter < srcUpperIndices.size(); ++iter) {
        loopIVs.push_back(0);
        loopBounds.push_back(sliceLengths[iter]);
      }

      // Main code emission loop.
      bool toExit = false;
      do {
        // Use the old logic in case "legacy_load" attribute is specified.
        if (legacyLoad == true) {
          computeBottomIndicesWithAffineMap(b, loc, srcLowerIndices,
                                            srcUpperIndices, loopIVs,
                                            composedSourceTransform);
        } else {
          // New approach. Use index diff map.
          // Progressively use index diff map to compute the coordinate at the
          // bottom most layer.
          computeBottomIndicesWithIndexDiffMap(b, loc, layeredSourceIndices,
                                               loopIVs, sourceTransforms,
                                               srcLowerIndices);
        }

        // Load from source.
        Value scalarValue = emitLoadLogic(
            b, loc, sourceType, sourceElementType, toEmitOOBLoadCheckLogic,
            oobLoadCheckDims, op.source(), srcLowerIndices);

        // Convert from sourceElementType to destElementType if necessary.
        Value convertedScalarValue = createTypeConversionOp(
            b, loc, scalarValue, sourceElementType, destElementType);

        // Use the old logic in case "legacy_store" attribute is specified.
        if (legacyStore == true) {
          computeBottomIndicesWithAffineMap(b, loc, destLowerIndices,
                                            destUpperIndices, loopIVs,
                                            composedDestTransform);
        } else {
          // New approach. Use index diff map.
          // Progressively use index diff map to compute the coordinate at the
          // bottom most layer.
          computeBottomIndicesWithIndexDiffMap(b, loc, layeredDestIndices,
                                               loopIVs, destTransforms,
                                               destLowerIndices);
        }

        // Store to dest.
        emitStoreLogic(op.paddingInfo().getBwdPaddingInfo(), b, loc, destType,
                       destElementType, toEmitOOBStoreCheckLogic,
                       oobStoreCheckDims, op.dest(), destLowerIndices,
                       convertedScalarValue);

        // increase IVs
        bool toIncreaseNextDigit = true;
        int iter = loopIVs.size() - 1;
        for (; toIncreaseNextDigit && iter >= 0; --iter) {
          if (++loopIVs[iter] == loopBounds[iter]) {
            loopIVs[iter] = 0;
            toIncreaseNextDigit = true;
          } else {
            toIncreaseNextDigit = false;
          }
        }

        // check if need to exit
        if (iter < 0 && toIncreaseNextDigit == true) {
          toExit = true;
        }
      } while (!toExit);
    }

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadwiseCopyV2 lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseCopyV2RewritePattern
    : public OpRewritePattern<ThreadwiseCopyV2Op> {
  using OpRewritePattern<ThreadwiseCopyV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseCopyV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto sourceElementType =
        op.source().getType().cast<VectorType>().getElementType().cast<Type>();
    auto destElementType =
        op.dest().getType().cast<MemRefType>().getElementType().cast<Type>();

    auto sourceType = op.source().getType().cast<VectorType>();
    auto destType = op.dest().getType().cast<MemRefType>();
    InMemoryDataOperation dataOpration = op.dataOperation();
    Value sourceOffsetOp =
        b.create<ConstantIndexOp>(loc, op.sourceOffset().getZExtValue());

    int64_t dataPerCopy =
        op->getAttrOfType<IntegerAttr>("dest_data_per_write").getInt();
    int64_t sourceDataPerRead =
        op->getAttrOfType<IntegerAttr>("source_data_per_read").getInt();
    assert(dataPerCopy == sourceDataPerRead &&
           "source and dest vector copy lengths are equal");

    int64_t upperVectorDim =
        op->getAttrOfType<IntegerAttr>("upper_vector_read_dim").getInt();
    int64_t lowerVectorDim =
        op->getAttrOfType<IntegerAttr>("vector_read_write_dim").getInt();
    assert((dataPerCopy == 1 || lowerVectorDim == 4) &&
           "Was asked to vectorize non-final dimension");

    auto sourceTransforms = op.transforms()[0].cast<ArrayAttr>();
    auto destTransforms = op.transforms()[1].cast<ArrayAttr>();

    Optional<AffineMap> composedSourceTransform;
    Optional<AffineMap> composedDestTransform;

    // Obtain coordinate lengths, as well as information of affine
    // transformations.
    uint32_t sourceCoordLength = obtainGenericTensorTransformationInfo(
        sourceType, sourceTransforms, composedSourceTransform, op.bounds());
    auto sourceCoord = op.sourceCoord();
    if (sourceCoordLength != sourceCoord.size()) {
      return op.emitOpError("Got " + Twine(sourceCoord.size()) +
                            " source coordinates but expected " +
                            Twine(sourceCoordLength));
    }

    uint32_t destCoordLength = obtainGenericTensorTransformationInfo(
        destType, destTransforms, composedDestTransform, op.bounds());
    auto destCoord = op.destCoord();
    if (destCoordLength != destCoord.size()) {
      return op.emitOpError("Got " + Twine(destCoord.size()) +
                            " dest coordinates but expected " +
                            Twine(destCoordLength));
    }

    // Determine if we need to emit codes for out-of-bound check, and which
    // dimensions need to dconduct such check.
    SmallVector<uint32_t, 8> oobStoreCheckDims;
    bool toEmitOOBStoreCheckLogic = obtainOOBCheckInfo(
        composedDestTransform, op.destOobDims(), oobStoreCheckDims);

    if (toEmitOOBStoreCheckLogic) {
      // TODO(kdrewnia) Work out if we can have stores that statically
      // won't OOB still be vectorized
      dataPerCopy = 1;
    }

    // llvm::errs() << "\nthreadwise_copy_v2 op:\n";
    // op.dump();
    // llvm::errs() << "\n";

    // Figure out the bounds of load/store loops.
    SmallVector<uint64_t, 5> sliceLengths;

    computeSliceLengths(sliceLengths, op.bounds());
    if (upperVectorDim >= 0) {
      sliceLengths[upperVectorDim] /= dataPerCopy;
      assert(sliceLengths[upperVectorDim] != 0);
    }

    // llvm::errs() << "slice lengths: ";
    // for (unsigned i = 0; i < sliceLengths.size(); ++i)
    //   llvm::errs() << sliceLengths[i] << " ";
    // llvm::errs() << "\n";
    // Compute high-level coordinate for source memref.
    SmallVector<Value, 8> srcUpperIndices;
    srcUpperIndices.append(sourceCoord.begin(), sourceCoord.end());

    // Coordinates across the layers of transformations.
    // If the vector is of size n, 0 is the top layer, and
    // n-1 is the bottom layer.
    SmallVector<SmallVector<Value, 8>, 2> layeredSourceIndices;

    // Populate coorindates across the layers of transformations.
    populateLayeredIndicesWithTransformMetadata(
        b, loc, layeredSourceIndices, srcUpperIndices, sourceTransforms);

    // Fetch low-level coordinate.
    SmallVector<Value, 8> srcLowerIndices =
        layeredSourceIndices[layeredSourceIndices.size() - 1];

    // Compute high-level coordinate for dest memref.
    SmallVector<Value, 8> destUpperIndices;
    destUpperIndices.append(destCoord.begin(), destCoord.end());

    // Coordinates across the layers of transformations.
    // If the vector is of size n, 0 is the top layer, and
    // n-1 is the bottom layer.
    SmallVector<SmallVector<Value, 8>, 2> layeredDestIndices;

    // Populate coorindates across the layers of transformations.
    populateLayeredIndicesWithTransformMetadata(
        b, loc, layeredDestIndices, destUpperIndices, destTransforms);

    // Fetch low-level coordinate.
    SmallVector<Value, 8> destLowerIndices =
        layeredDestIndices[layeredDestIndices.size() - 1];

    // Emit fully unrolled loops for vector loads / stores.
    SmallVector<int64_t, 8> loopIVs;
    SmallVector<int64_t, 8> loopBounds;
    for (unsigned iter = 0; iter < srcUpperIndices.size(); ++iter) {
      loopIVs.push_back(0);
      loopBounds.push_back(sliceLengths[iter]);
    }

    Type typeToLoad = sourceElementType;
    if (dataPerCopy > 1) {
      typeToLoad = VectorType::get({dataPerCopy}, typeToLoad);
    }
    Type typeToStore = destElementType;
    if (dataPerCopy > 1) {
      typeToStore = VectorType::get({dataPerCopy}, typeToStore);
    }

    bool toExit = false;
    do {
      // Load from source vector.

      // Progressively use index diff map to compute the coordinate at the
      // bottom most layer.
      computeBottomIndicesWithIndexDiffMap(b, loc, layeredSourceIndices,
                                           loopIVs, sourceTransforms,
                                           srcLowerIndices);

      // Add sourceOffset to derive the position in the vector.
      auto srcPosition =
          b.create<AddIOp>(loc, srcLowerIndices[0], sourceOffsetOp);

      // Load from source.
      Value loadedValue;
      if (dataPerCopy > 1) {
        loadedValue = createZeroConstantFloatOp(b, loc, typeToLoad);
        for (int64_t i = 0; i < dataPerCopy; ++i) {
          Value index = b.create<ConstantIndexOp>(loc, i);
          Value extracted = b.create<vector::ExtractElementOp>(
              loc, sourceElementType, op.source(),
              b.create<AddIOp>(loc, srcPosition, index));
          loadedValue = b.create<vector::InsertElementOp>(loc, extracted,
                                                          loadedValue, index);
        }
      } else {
        loadedValue = b.create<vector::ExtractElementOp>(
            loc, sourceElementType, op.source(), srcPosition);
      }

      // Convert from sourceElementType to destElementType if necessary.
      Value convertedValue =
          createTypeConversionOp(b, loc, loadedValue, typeToLoad, typeToStore);

      // Store to dest memref.

      // Progressively use index diff map to compute the coordinate at the
      // bottom most layer.
      computeBottomIndicesWithIndexDiffMap(b, loc, layeredDestIndices, loopIVs,
                                           destTransforms, destLowerIndices);

      // Store to dest.
      emitStoreLogic(op.paddingInfo().getBwdPaddingInfo(), b, loc, destType,
                     typeToStore, toEmitOOBStoreCheckLogic, oobStoreCheckDims,
                     op.dest(), destLowerIndices, convertedValue, dataOpration);
      // increase IVs
      bool toIncreaseNextDigit = true;
      int iter = loopIVs.size() - 1;
      for (; toIncreaseNextDigit && iter >= 0; --iter) {
        if (++loopIVs[iter] == loopBounds[iter]) {
          loopIVs[iter] = 0;
          toIncreaseNextDigit = true;
        } else {
          toIncreaseNextDigit = false;
        }
      }

      // check if need to exit
      if (iter < 0 && toIncreaseNextDigit == true) {
        toExit = true;
      }
    } while (!toExit);

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadwiseLoad lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseLoadRewritePattern
    : public OpRewritePattern<ThreadwiseLoadOp> {
  using OpRewritePattern<ThreadwiseLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseLoadOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    ArrayAttr transforms = op.transforms()[0].cast<ArrayAttr>();
    // The types in elements of the input are all the same.
    auto destElementType = op.getResult(0).getType().cast<Type>();

    auto sourceType = op.source().getType().cast<MemRefType>();
    auto destTypes = op.getResultTypes();

    // Debug switches.
    // true : use the slow but proven affine map.
    // false : use the faster index diff map.
    auto legacyLoadAttr = op->getAttr("legacy_load");
    bool legacyLoad =
        (legacyLoadAttr &&
         legacyLoadAttr.template cast<BoolAttr>().getValue() == true);

    Optional<AffineMap> composedSourceTransform;

    // Obtain coordinate lengths, as well as information of affine
    // transformations.
    uint32_t sourceCoordLength = obtainGenericTensorTransformationInfo(
        sourceType, transforms, composedSourceTransform, op.bounds());

    auto sourceCoord = op.sourceCoord();
    if (sourceCoordLength != sourceCoord.size()) {
      llvm::errs() << "INCORRECT source coordinates assigned!";
      return failure();
    }

    // FIXME. XXX.
    // Workaround to obtain gemmKExtra attribute.
    // And use it to override legacy load/store debug switch.
    legacyLoad = overrideLoadStoreHack(op.paddingInfo(), legacyLoad);

    // Determine if we need to emit codes for out-of-bound check, and which
    // dimensions need to dconduct such check.
    SmallVector<unsigned, 8> oobLoadCheckDims;
    bool toEmitOOBLoadCheckLogic = obtainOOBCheckInfo(
        composedSourceTransform, op.oobDims(), oobLoadCheckDims);

    // llvm::errs() << "\nthreadwise_load op:\n";
    // op.dump();
    // llvm::errs() << "\n";

    // --------------------------------

    auto srcDataPerRead = op->getAttr("source_data_per_read")
                              .template cast<IntegerAttr>()
                              .getInt();

    auto vectorReadWriteDim = op->getAttr("vector_read_write_dim")
                                  .template cast<IntegerAttr>()
                                  .getInt();

    // Figure out the bounds of load/store loops.
    SmallVector<uint64_t, 2> sliceLengths;

    computeSliceLengths(sliceLengths, op.bounds());

    // llvm::errs() << "slice lengths: ";
    // for (unsigned i = 0; i < sliceLengths.size(); ++i)
    //   llvm::errs() << sliceLengths[i] << " ";
    // llvm::errs() << "\n";

    // llvm::errs() << "vector dim: " << vectorReadWriteDim << "\n";
    // llvm::errs() << "source data per read: " << srcDataPerRead << "\n";

    sliceLengths[vectorReadWriteDim] /= srcDataPerRead;
    assert(sliceLengths[vectorReadWriteDim] != 0);

    // llvm::errs() << "modified lengths: ";
    // for (unsigned i = 0; i < sliceLengths.size(); ++i)
    //   llvm::errs() << sliceLengths[i] << " ";
    // llvm::errs() << "\n";

    // --------------------------------

    SmallVector<Value, 8> srcUpperIndices;
    srcUpperIndices.append(sourceCoord.begin(), sourceCoord.end());
    SmallVector<Value, 8> srcLowerIndices;
    // Coordinates across the layers of transformations.
    // If the vector is of size n, 0 is the top layer, and
    // n-1 is the bottom layer.
    SmallVector<SmallVector<Value, 8>, 2> layeredSourceIndices;

    // Obtain transform metadata and populate coordinates for all layers
    // wthe the metadata.
    // Only do such computation in the new approach where index diff maps
    // would be used.
    if (legacyLoad == false) {
      // Populate coorindates across the layers of transformations.
      populateLayeredIndicesWithTransformMetadata(b, loc, layeredSourceIndices,
                                                  srcUpperIndices, transforms);

      // Fetch low-level coordinate.
      srcLowerIndices = layeredSourceIndices[layeredSourceIndices.size() - 1];
    }

    // --------------------------------

    // Emit fully unrolled loops for vector loads / stores.
    SmallVector<int64_t, 8> loopIVs;
    SmallVector<int64_t, 8> loopBounds;
    for (unsigned iter = 0; iter < srcUpperIndices.size(); ++iter) {
      loopIVs.push_back(0);
      loopBounds.push_back(sliceLengths[iter]);
    }

    // --------------------------------

    // Main code emission loop.
    DenseMap<int64_t, Value> loadedValues;
    bool toExit = false;
    do {
      // llvm::errs() << "IVs: ";
      // for (auto v : loopIVs)
      //   llvm::errs() << v << " ";
      // llvm::errs() << "\n";

      // Use the old logic in case "legacy_load" attribute is specified.
      if (legacyLoad == true) {
        computeBottomIndicesWithAffineMap(b, loc, srcLowerIndices,
                                          srcUpperIndices, loopIVs,
                                          composedSourceTransform);
      } else {
        // New approach. Use index diff map.
        // Progressively use index diff map to compute the coordinate at the
        // bottom most layer.
        computeBottomIndicesWithIndexDiffMap(
            b, loc, layeredSourceIndices, loopIVs, transforms, srcLowerIndices);
      }

      // Determine the type to be loaded.
      // Construct a vector in case we can do vector load.
      Type loadedType = destElementType;
      if (srcDataPerRead > 1)
        loadedType = VectorType::get({srcDataPerRead}, destElementType);

      // Load from source.
      Value loadedValue =
          emitLoadLogic(b, loc, sourceType, loadedType, toEmitOOBLoadCheckLogic,
                        oobLoadCheckDims, op.source(), srcLowerIndices);

      // Compute the final index on the loadedValues, following IVs.
      int64_t inputsIndex = 0;
      int64_t stride = 1;
      int64_t vectorDimStride = 0;
      for (int64_t iter = loopIVs.size() - 1; iter >= 0; --iter) {
        inputsIndex += loopIVs[iter] * stride;
        if (iter == vectorReadWriteDim) {
          vectorDimStride = stride;
          stride *= (loopBounds[iter] * srcDataPerRead);
        } else {
          stride *= loopBounds[iter];
        }
      }
      // llvm::errs() << "inputsIndex: " << inputsIndex << "\n";

      // In case we do vector load, decompose the elements as the
      // results of threadwise_load only hold scalars.
      if (srcDataPerRead > 1) {
        assert(loadedValue.getType().isa<VectorType>());

        for (int64_t iter = 0; iter < srcDataPerRead; ++iter) {
          auto loadedElement = b.create<vector::ExtractElementOp>(
              loc, destElementType, loadedValue,
              b.create<ConstantIndexOp>(loc, iter));
          int64_t decomposedInputsIndex = inputsIndex + iter * vectorDimStride;
          // llvm::errs() << "decomposedInputsIndex: " << decomposedInputsIndex
          //              << "\n";

          loadedValues[decomposedInputsIndex] = loadedElement;
        }
      } else {
        loadedValues[inputsIndex] = loadedValue;
      }

      // increase IVs
      bool toIncreaseNextDigit = true;
      int iter = loopIVs.size() - 1;
      for (; toIncreaseNextDigit && iter >= 0; --iter) {
        loopIVs[iter] += 1;
        if (loopIVs[iter] >= loopBounds[iter]) {
          loopIVs[iter] = 0;
          toIncreaseNextDigit = true;
        } else {
          toIncreaseNextDigit = false;
        }
      }

      // check if need to exit
      if (iter < 0 && toIncreaseNextDigit == true) {
        toExit = true;
      }
    } while (!toExit);

    // --------------------------------

    // Extract the loaded values into a variadic results.
    assert(loadedValues.size() == destTypes.size());
    SmallVector<Value, 8> outputValues;
    for (int64_t iter = 0; iter < loadedValues.size(); ++iter) {
      assert(loadedValues.count(iter) == 1);
      outputValues.push_back(loadedValues[iter]);
    }
    op.replaceAllUsesWith(outputValues);
    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadwiseStore lowering.
//===----------------------------------------------------------------------===//

struct ThreadwiseStoreRewritePattern
    : public OpRewritePattern<ThreadwiseStoreOp> {
  using OpRewritePattern<ThreadwiseStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseStoreOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    ArrayAttr transforms = op.transforms()[0].cast<ArrayAttr>();
    auto sourceTypes = op.data().getTypes();
    // The types in the data arguments are all the same.
    auto sourceElementType = sourceTypes[0];

    auto destType = op.dest().getType().cast<MemRefType>();

    // This lowering is much simpler than other threadwise* ops since
    // threadwise_store cannot have a global tensor as an argument.
    // This means that it does not participate in the coordinate transformations
    // scheme.

    // Obtain coordinate lengths, as well as information of affine
    // transformations.
    Optional<AffineMap> composedDestTransform;
    uint32_t destCoordLength = obtainGenericTensorTransformationInfo(
        destType, transforms, composedDestTransform, op.bounds());

    auto destCoord = op.destCoord();
    if (destCoordLength != destCoord.size()) {
      llvm::errs() << "INCORRECT dest coordinates assigned!\n";
      return failure();
    }

    // llvm::errs() << "\nthreadwise_store op:\n";
    // op.dump();
    // llvm::errs() << "\n";

    // --------------------------------

    auto dstDataPerWrite = op->getAttr("dest_data_per_write")
                               .template cast<IntegerAttr>()
                               .getInt();

    auto vectorReadWriteDim = op->getAttr("vector_read_write_dim")
                                  .template cast<IntegerAttr>()
                                  .getInt();

    // Figure out the bounds of load/store loops.
    SmallVector<uint64_t, 5> sliceLengths;
    computeSliceLengths(sliceLengths, op.bounds());

    // llvm::errs() << "slice lengths: ";
    // for (unsigned i = 0; i < sliceLengths.size(); ++i)
    //   llvm::errs() << sliceLengths[i] << " ";
    // llvm::errs() << "\n";

    // llvm::errs() << "vector dim: " << vectorReadWriteDim << "\n";
    // llvm::errs() << "dest data per write: " << dstDataPerWrite << "\n";

    sliceLengths[vectorReadWriteDim] /= dstDataPerWrite;
    assert(sliceLengths[vectorReadWriteDim] != 0);

    // llvm::errs() << "modified lengths: ";
    // for (unsigned i = 0; i < sliceLengths.size(); ++i)
    //   llvm::errs() << sliceLengths[i] << " ";
    // llvm::errs() << "\n";

    // --------------------------------

    SmallVector<Value, 8> destUpperIndices;
    destUpperIndices.append(destCoord.begin(), destCoord.end());
    SmallVector<Value, 8> destLowerIndices;
    SmallVector<SmallVector<Value, 8>, 2> layeredDestIndices;

    populateLayeredIndicesWithTransformMetadata(b, loc, layeredDestIndices,
                                                destUpperIndices, transforms);

    // --------------------------------

    // Emit fully unrolled loops for vector loads / stores.
    SmallVector<int64_t, 8> loopIVs;
    SmallVector<int64_t, 8> loopBounds;
    for (unsigned iter = 0; iter < destUpperIndices.size(); ++iter) {
      loopIVs.push_back(0);
      loopBounds.push_back(sliceLengths[iter]);
    }

    // --------------------------------

    // Main code emission loop.
    bool toExit = false;
    do {
      // llvm::errs() << "IVs: ";
      // for (auto v : loopIVs)
      //   llvm::errs() << v << " ";
      // llvm::errs() << "\n";

      // Compute bottom indices from loop IVs
      computeBottomIndicesWithIndexDiffMap(b, loc, layeredDestIndices, loopIVs,
                                           transforms, destLowerIndices);

      // Determine the type to be stored.
      // Construct a vector in case we can do vector store.
      Type typeToStore = sourceElementType;
      if (dstDataPerWrite > 1)
        typeToStore = VectorType::get({dstDataPerWrite}, sourceElementType);

      // Compute the starting index inside the inputs, following IVs.
      int64_t inputsIndex = 0;
      int64_t stride = 1;
      int64_t vectorDimStride = 0;
      for (int iter = loopIVs.size() - 1; iter >= 0; --iter) {
        inputsIndex += loopIVs[iter] * stride;
        if (iter == vectorReadWriteDim) {
          vectorDimStride = stride;
          stride *= (loopBounds[iter] * dstDataPerWrite);
        } else {
          stride *= loopBounds[iter];
        }
      }
      // llvm::errs() << "inputsIndex: " << inputsIndex << "\n";

      // In case we do vector store, decompose the elements as the arguments
      // only hold scalars.
      Value valueToStore;
      if (dstDataPerWrite > 1) {
        assert(typeToStore.isa<VectorType>());

        valueToStore = createZeroConstantFloatOp(b, loc, typeToStore);
        for (int64_t iter = 0; iter < dstDataPerWrite; ++iter) {
          int64_t decomposedInputsIndex = inputsIndex + iter * vectorDimStride;
          // llvm::errs() << "decomposedInputsIndex: " << decomposedInputsIndex
          // << "\n";
          Value element = op.data()[decomposedInputsIndex];
          valueToStore = b.create<vector::InsertElementOp>(
              loc, typeToStore, element, valueToStore,
              b.create<ConstantIndexOp>(loc, iter));
        }
      } else {
        valueToStore = op.data()[inputsIndex];
      }

      // Store to dest. this part do not need backward data padding kernel
      // workaround just use "not applicable"
      BwdPaddingKernelInfo bwdPaddingInfo = BwdPaddingKernelInfo::NA;
      emitStoreLogic(bwdPaddingInfo, b, loc, destType, typeToStore,
                     /*toEmitOOBStoreLogic=*/false, /*oobStoreCheckDims=*/{},
                     op.dest(), destLowerIndices, valueToStore);

      // increase IVs
      bool toIncreaseNextDigit = true;
      int iter = loopIVs.size() - 1;
      for (; toIncreaseNextDigit && iter >= 0; --iter) {
        loopIVs[iter] += 1;
        if (loopIVs[iter] >= loopBounds[iter]) {
          loopIVs[iter] %= loopBounds[iter];
          toIncreaseNextDigit = true;
        } else {
          toIncreaseNextDigit = false;
        }
      }

      // check if need to exit
      if (iter < 0 && toIncreaseNextDigit == true) {
        toExit = true;
      }
    } while (!toExit);

    // --------------------------------

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InWarpTranspose lowering.
//===----------------------------------------------------------------------===//
constexpr size_t swizzleGroupSize = InWarpTransposeOp::swizzleGroupSize;
struct InWarpTransposeRewritePattern
    : public OpRewritePattern<InWarpTransposeOp> {
  using OpRewritePattern<InWarpTransposeOp>::OpRewritePattern;

  enum RotationDirection {
    // left rotation by 1 is a b c d -> b c d a
    // right rotation by 1 is a b c d -> d a b c
    Left,
    Right
  };

  // Emit the in-regester rotations needed for an in-register transpose
  //
  // This will rotate the values each line holds by `laneId % groupSize`
  // The emitted code uses a barrel rotator to enable performing these
  // `groupSize` different rotations in O(log(groupSize)) operations Arguments:
  // - `vector`: The vector of values to be rotated
  // - `laneId`: The current lane ID (thread ID % warpSize)
  // - `rotationDir`: whether to rotate left or right
  // - `groupSize` and `totalSize`: the size of the transpose
  // and the total vector length, respectively

  // - `lanePerm` : A mapping of physical lanes to logical lanes in each grou
  // That is, lanePerm[i] tells you where the value in lane i would "normally"
  // be, with all indices modulo the swizzle group size. If empty, the identity
  // map is used. For example, with lanePerm = [0, 2, 1, 3], lanes 1 and 3 will
  // rotate their values by 2 places, as opposed to lanes  2 and 3 Returns: The
  // vector of rotated values
  Value emitRotations(Location loc, PatternRewriter &b, Value vector,
                      Value laneId, RotationDirection dir, uint32_t groupSize,
                      uint32_t totalSize,
                      Optional<ArrayRef<uint32_t>> lanePerm) const {
    assert(totalSize % groupSize == 0 &&
           "block size is divisible by group size");

    uint32_t logGroupSize = llvm::Log2_32_Ceil(groupSize);

    int32_t base = 0, offset = 0, target = 0;
    switch (dir) {
    case Left:
      base = 0;
      offset = 1;
      target = logGroupSize;
      break;
    case Right:
      base = logGroupSize - 1;
      offset = -1;
      target = -1;
      break;
    }

    Value zeroConst = b.create<ConstantIndexOp>(loc, 0);

    llvm::SmallVector<Value, swizzleGroupSize> indexConsts;
    for (uint32_t i = 0; i < totalSize; ++i) {
      indexConsts.push_back(b.create<ConstantIndexOp>(loc, i));
    }

    Value laneInSwizzleGroup;
    if (lanePerm.hasValue()) {
      Value groupSizeConst = b.create<ConstantIndexOp>(loc, swizzleGroupSize);
      laneInSwizzleGroup = b.create<RemUIOp>(loc, laneId, groupSizeConst);
    }

    Value result = vector;

    for (int32_t logRotation = base; logRotation != target;
         logRotation += offset) {
      uint32_t rotation = 1 << logRotation;
      Value shouldParticipate;
      if (lanePerm.hasValue()) {
        // Non-standard arrangement of rows -> lanes, use longer test
        ArrayRef<uint32_t> theLanePerm = lanePerm.getValue();
        llvm::SmallVector<Value, swizzleGroupSize> comparisons;
        for (uint32_t i = 0; i < theLanePerm.size(); ++i) {
          if ((theLanePerm[i] & rotation) != 0) {
            Value toTest = b.create<ConstantIndexOp>(loc, i);
            comparisons.push_back(b.create<CmpIOp>(loc, CmpIPredicate::eq,
                                                   laneInSwizzleGroup, toTest));
          }
        }
        if (comparisons.empty()) {
          llvm_unreachable("Permutation on [0, 2^k) didn't have any entries "
                           "with some bit set");
        }
        shouldParticipate =
            std::accumulate(comparisons.begin() + 1, comparisons.end(),
                            comparisons[0], [&b, &loc](Value v1, Value v2) {
                              return b.create<OrIOp>(loc, v1, v2);
                            });
      } else { // The usual case
        Value maskConst = b.create<ConstantIndexOp>(loc, rotation);
        Value shouldParticipateVal = b.create<AndIOp>(loc, laneId, maskConst);
        shouldParticipate = b.create<CmpIOp>(loc, CmpIPredicate::ne,
                                             shouldParticipateVal, zeroConst);
      }

// TODO(kdrewnia): xplicitly emit selects until SWDEV-302607 and SWDEV-302609
// are fixed
#if 0
      scf::IfOp ifb = b.create<scf::IfOp>(
          loc, vector.getType(), shouldParticipate, /*withElseRegion=*/true);
      OpBuilder thenb = ifb.getThenBodyBuilder(b.getListener());

      Value thenResult = result;
      SmallVector<Value> extracted;
      for (uint32_t i = 0; i < totalSize; ++i) {
        extracted.push_back(thenb.create<vector::ExtractElementOp>(
            loc, thenResult, indexConsts[i]));
      }
      for (uint32_t group = 0; group < totalSize; group += groupSize) {
        for (uint32_t i = 0; i < groupSize; ++i) {
          uint32_t dest = 0xdeadbeef;
          switch (dir) {
          case Left:
            // We use groupSize - rotation to prevent underflow
            dest = (i + (groupSize - rotation)) % groupSize;
            break;
          case Right:
            dest = (i + rotation) % groupSize;
            break;
          }
          Value toInsert = extracted[group + i];
          thenResult = thenb.create<vector::InsertElementOp>(
              loc, toInsert, thenResult, indexConsts[group + dest]);
        }
      }
      thenb.create<scf::YieldOp>(loc, thenResult);

      OpBuilder elseb = ifb.getElseBodyBuilder(b.getListener());
      elseb.create<scf::YieldOp>(loc, result);

      result = ifb.getResult(0);
#endif

      SmallVector<Value> extracted;
      for (uint32_t i = 0; i < totalSize; ++i) {
        extracted.push_back(
            b.create<vector::ExtractElementOp>(loc, result, indexConsts[i]));
      }
      for (uint32_t group = 0; group < totalSize; group += groupSize) {
        for (uint32_t i = 0; i < groupSize; ++i) {
          uint32_t dest = 0xdeadbeef;
          switch (dir) {
          case Left:
            // We use groupSize - rotation to prevent underflow
            dest = (i + (groupSize - rotation)) % groupSize;
            break;
          case Right:
            dest = (i + rotation) % groupSize;
            break;
          }
          Value whenRotating = extracted[group + i];
          Value stable = extracted[group + dest];
          Value toInsert =
              b.create<SelectOp>(loc, shouldParticipate, whenRotating, stable);
          result = b.create<vector::InsertElementOp>(loc, toInsert, result,
                                                     indexConsts[group + dest]);
        }
      }
    }

    return result;
  }

  // Before calling this function, we will have emitted rotations so that the
  // group
  //  r[]: 0   1   2   3
  //  t0: 0,0 1,0 2,0 3,0
  //  t1: 0,1 1,1 2,1 3,1
  //  t2: 0,2 1,2 2,2 3,2
  //  t3: 0,3 1,3 2,3 3,3
  // will have become
  //  0,0 1,0 2,0 3,0
  //  3,1 0,1 1,1 2,1
  //  2,2 3,2 0,2 1,2
  //  1,3 2,3 3,3 0,3

  // (plus-minus size changes for other operations).
  // These rotations are the first step in the in-register transpose algorithm
  // as they allow the inter-lane shuffles to be permutation.

  // The goal of this function is to emit code that will lead to the result
  // state
  //  0,0 0,1 0,2 0,3
  //  1,3 1,0 1,1 1,2
  //  2,2 2,3 2,0 2,1
  //  3,1 3,2 3,3 3,0

  Value emitSwizzles(Location loc, PatternRewriter &b, Value vector,
                     uint32_t groupSize, uint32_t totalSize,
                     ArrayRef<uint32_t> inGroupPerm) const {

    llvm::SmallVector<ArrayAttr, swizzleGroupSize> swizzlePerms;

    llvm::SmallVector<int32_t, swizzleGroupSize> perm;
    llvm::SmallVector<uint32_t, swizzleGroupSize> have;
    llvm::SmallVector<uint32_t, swizzleGroupSize> want;
    for (uint32_t r = 0; r < groupSize; ++r) {
      perm.clear();
      have.clear();
      want.clear();

      for (uint32_t t = 0; t < swizzleGroupSize; ++t) {
        // Must correct for, say, 2x2 transpose being a 4 thread x 2 register
        // swizzle
        uint32_t smallGroupDup = groupSize * (t / groupSize);
        uint32_t preSwizzleI =
            (r + (groupSize - t)) % groupSize + smallGroupDup;
        uint32_t preSwizzleJ = t;

        uint32_t expectedThread = inGroupPerm[t];
        uint32_t postSwizzleI = expectedThread;
        uint32_t postSwizzleJ = (r + (groupSize - expectedThread)) % groupSize +
                                groupSize * (expectedThread / groupSize);
        uint32_t preSwizzleElem = preSwizzleJ + swizzleGroupSize * preSwizzleI;
        uint32_t postSwizzleElem =
            postSwizzleJ + swizzleGroupSize * postSwizzleI;
        /*         llvm::dbgs() << "//r = " << r << " t = " << t << ": " <<
           "have ("
                  << preSwizzleI << ", " << preSwizzleJ << ") = " <<
           preSwizzleElem
                  << " want (" << postSwizzleI << ", " << postSwizzleJ << ") = "
                  << postSwizzleElem << "\n"; */
        have.push_back(preSwizzleElem);
        want.push_back(postSwizzleElem);
      }

      for (uint32_t t = 0; t < swizzleGroupSize; ++t) {
        auto *srcElemIter = std::find(have.begin(), have.end(), want[t]);
        assert(srcElemIter != have.end() && "swizzle is not a permutation");
        auto readIdx = srcElemIter - have.begin();
        perm.push_back(readIdx);
      }

      if (perm[0] == 0 && perm[1] == 1 && perm[2] == 2 && perm[3] == 3) {
        swizzlePerms.push_back(b.getI32ArrayAttr({}));
      } else {
        swizzlePerms.push_back(b.getI32ArrayAttr(perm));
      }
    }

    Value result = b.create<vector::BitCastOp>(
        loc,
        VectorType::get(vector.getType().cast<VectorType>().getShape(),
                        b.getI32Type()),
        vector);
    // TODO(kdrewnia): Make this operation variadic and not just vector-valued
    SmallVector<Value> accessConsts;
    SmallVector<Value> initialRegisters;
    for (uint32_t i = 0; i < totalSize; ++i) {
      Value accessConst = b.create<ConstantIndexOp>(loc, i);
      initialRegisters.push_back(
          b.create<vector::ExtractElementOp>(loc, result, accessConst));
      accessConsts.push_back(accessConst);
    }

    SmallVector<Value> swizzledRegisters;
    for (uint32_t i = 0; i < totalSize; ++i) {
      ArrayAttr swizzleSelector = swizzlePerms[i % groupSize];
      if (0 == swizzleSelector.size()) {
        swizzledRegisters.push_back(initialRegisters[i]);
        continue;
      }
      Value swizzled = b.create<gpu::WarpSwizzleOp>(
          loc, b.getI32Type(), initialRegisters[i], swizzleSelector);
      swizzledRegisters.push_back(swizzled);
    }

    for (uint32_t i = 0; i < totalSize; ++i) {
      if (swizzledRegisters[i] != initialRegisters[i]) {
        result = b.create<vector::InsertElementOp>(loc, swizzledRegisters[i],
                                                   result, accessConsts[i]);
      }
    }

    result = b.create<vector::BitCastOp>(loc, vector.getType(), result);
    return result;
  }

  LogicalResult matchAndRewrite(InWarpTransposeOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value vector = op.vector();
    uint32_t totalSize = vector.getType().cast<VectorType>().getNumElements();

    Value laneId = op.laneId();
    uint32_t groupSize = op.size();

    ArrayAttr inGroupPermAttr = op.inGroupPerm();
    llvm::SmallVector<uint32_t, swizzleGroupSize> inGroupPerm;
    auto inGroupPermArr = inGroupPermAttr.getValue();
    // ::verify() ensures this is a permutation
    for (uint32_t i = 0; i < swizzleGroupSize; ++i) {
      inGroupPerm.push_back(inGroupPermArr[i]
                                .cast<mlir::IntegerAttr>()
                                .getValue()
                                .getZExtValue());
    }

    Optional<ArrayRef<uint32_t>> maybeInGroupPerm = llvm::None;
    if (inGroupPermAttr != b.getI32ArrayAttr({0, 1, 2, 3})) {
      maybeInGroupPerm = inGroupPerm;
    }

    Value rotatedRight = emitRotations(loc, b, vector, laneId, Right, groupSize,
                                       totalSize, llvm::None);
    Value swizzled =
        emitSwizzles(loc, b, rotatedRight, groupSize, totalSize, inGroupPerm);
    Value rotatedLeft = emitRotations(loc, b, swizzled, laneId, Left, groupSize,
                                      totalSize, maybeInGroupPerm);

    op.replaceAllUsesWith(rotatedLeft);
    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ThreadwiseGemm lowering.
//===----------------------------------------------------------------------===//
struct ThreadwiseGemmRewritePattern
    : public OpRewritePattern<ThreadwiseGemmOp> {
  using OpRewritePattern<ThreadwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ThreadwiseGemmOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto gemmA = op.matrixA();
    auto gemmB = op.matrixB();
    auto gemmC = op.matrixC();
    auto dataType =
        gemmA.getType().template cast<MemRefType>().getElementType();

    ArrayRef<int64_t> gemmAShape =
        gemmA.getType().cast<MemRefType>().getShape();
    ArrayRef<int64_t> gemmBShape =
        gemmB.getType().cast<MemRefType>().getShape();

    assert(gemmAShape.size() == gemmBShape.size());
    assert((gemmAShape.size() == 3) || (gemmAShape.size() == 4));
    if (gemmAShape.size() == 3) {
      // non-KPack path.
      auto loopG = b.create<AffineForOp>(loc, 0, gemmAShape[0]);
      auto lbG = loopG.getBody();
      b.setInsertionPointToStart(lbG);

      auto loopK = b.create<AffineForOp>(loc, 0, gemmAShape[1]);
      auto lbK = loopK.getBody();
      b.setInsertionPointToStart(lbK);

      auto loopM = b.create<AffineForOp>(loopK.getLoc(), 0, gemmAShape[2]);
      auto lbM = loopM.getBody();
      b.setInsertionPointToStart(lbM);

      auto loopN = b.create<AffineForOp>(loc, 0, gemmBShape[2]);
      auto lbN = loopN.getBody();
      b.setInsertionPointToStart(lbN);

      SmallVector<Value, 3> memIndicesKM;
      extractForInductionVars({loopG, loopK, loopM}, &memIndicesKM);
      auto gemmAKM = b.create<AffineLoadOp>(loc, gemmA, memIndicesKM);

      SmallVector<Value, 3> memIndicesKN;
      extractForInductionVars({loopG, loopK, loopN}, &memIndicesKN);
      auto gemmBKN = b.create<AffineLoadOp>(loc, gemmB, memIndicesKN);

      Value mul;
      if (dataType.isa<IntegerType>()) {
        mul = b.create<MulIOp>(loc, dataType, gemmAKM, gemmBKN);
      } else {
        mul = b.create<MulFOp>(loc, dataType, gemmAKM, gemmBKN);
      }
      SmallVector<Value, 3> memIndicesMN;
      extractForInductionVars({loopG, loopM, loopN}, &memIndicesMN);
      auto gemmCMN = b.create<AffineLoadOp>(loc, gemmC, memIndicesMN);

      Value add;
      if (dataType.isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, dataType, mul, gemmCMN);
      } else {
        add = b.create<AddFOp>(loc, dataType, mul, gemmCMN);
      }
      b.create<AffineStoreOp>(loc, add, gemmC, memIndicesMN);
    } else if (gemmAShape.size() == 4) {
      // KPack path.
      auto loopG = b.create<AffineForOp>(loc, 0, gemmAShape[0]);
      auto lbG = loopG.getBody();
      b.setInsertionPointToStart(lbG);

      auto loopK = b.create<AffineForOp>(loc, 0, gemmAShape[1]);
      auto lbK = loopK.getBody();
      b.setInsertionPointToStart(lbK);

      auto loopM = b.create<AffineForOp>(loopK.getLoc(), 0, gemmAShape[2]);
      auto lbM = loopM.getBody();
      b.setInsertionPointToStart(lbM);

      auto loopKPack = b.create<AffineForOp>(loc, 0, gemmAShape[3]);
      auto lbKPack = loopKPack.getBody();
      b.setInsertionPointToStart(lbKPack);

      auto loopN = b.create<AffineForOp>(loc, 0, gemmBShape[2]);
      auto lbN = loopN.getBody();
      b.setInsertionPointToStart(lbN);

      SmallVector<Value, 4> memIndicesKMKPack;
      extractForInductionVars({loopG, loopK, loopM, loopKPack},
                              &memIndicesKMKPack);
      auto gemmAKMKPack = b.create<AffineLoadOp>(loc, gemmA, memIndicesKMKPack);

      SmallVector<Value, 4> memIndicesKNKPack;
      extractForInductionVars({loopG, loopK, loopN, loopKPack},
                              &memIndicesKNKPack);
      auto gemmBKNKPack = b.create<AffineLoadOp>(loc, gemmB, memIndicesKNKPack);

      Value mul;
      if (dataType.isa<IntegerType>()) {
        mul = b.create<MulIOp>(loc, dataType, gemmAKMKPack, gemmBKNKPack);
      } else {
        mul = b.create<MulFOp>(loc, dataType, gemmAKMKPack, gemmBKNKPack);
      }
      SmallVector<Value, 4> memIndicesMN;
      extractForInductionVars({loopG, loopM, loopN}, &memIndicesMN);
      auto gemmCMN = b.create<AffineLoadOp>(loc, gemmC, memIndicesMN);

      Value add;
      if (dataType.isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, dataType, mul, gemmCMN);
      } else {
        add = b.create<AddFOp>(loc, dataType, mul, gemmCMN);
      }
      b.create<AffineStoreOp>(loc, add, gemmC, memIndicesMN);
    }

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XdlopsGemmV2 lowering.
//===----------------------------------------------------------------------===//
struct XdlopsGemmV2RewritePattern : public OpRewritePattern<XdlopsGemmV2Op> {
  using OpRewritePattern<XdlopsGemmV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(XdlopsGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain critical information.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t M = op->getAttr("m").template cast<IntegerAttr>().getInt();
    int64_t N = op->getAttr("n").template cast<IntegerAttr>().getInt();
    int64_t K = op->getAttr("k").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    // Obtain coordinate transforms for Matrix A and B.
    ArrayAttr transformsA = op.transforms()[0].cast<ArrayAttr>();
    ArrayAttr transformsB = op.transforms()[1].cast<ArrayAttr>();
    Optional<AffineMap> transformMatrixA, transformMatrixB;

    obtainGenericTensorTransformationInfo(op.matrixA().getType(), transformsA,
                                          transformMatrixA);
    obtainGenericTensorTransformationInfo(op.matrixB().getType(), transformsB,
                                          transformMatrixB);

    auto dataType =
        op.matrixA().getType().template cast<MemRefType>().getElementType();

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

    // Logic to do XDLOPS code selection.
    // llvm::errs() << "Invoke XDLOPS code selection logic:\n";
    // llvm::errs() << "dataType: "; dataType.dump(); llvm::errs() << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";

    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    StringRef mfmaInstr = xcs.mfmaInstr;
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;
    Type argType = xcs.argType;

    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t k_base = xcs.k_base;

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);

    // Original C++ logic.
    // const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
    // FloatA a[K * MRepeats];
    // FloatB b[K * NRepeats];
    // constexpr index_t KRepeats = sizeof(FloatA) / (sizeof(data_type) *
    // mfma_type.k_base); auto pa = reinterpret_cast<const data_type*>(&a); auto
    // pb = reinterpret_cast<const data_type*>(&b); constexpr index_t AStride =
    // K * KRepeats; constexpr index_t BStride = K * KRepeats;

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, wave_size));

    int64_t KRepeats = KPack / k_base;
    if (KRepeats == 0)
      KRepeats = 1;
    // llvm::errs() << "argVectorType: " << argType << "\n";
    // llvm::errs() << "k_base: " << k_base << "\n";
    // llvm::errs() << "KRepeats: " << KRepeats << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "bufferA type: " << op.bufferA().getType() << "\n";
    // llvm::errs() << "bufferB type: " << op.bufferB().getType() << "\n";

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);
    auto kBaseKRepeatsConstantOp =
        b.create<ConstantIndexOp>(loc, KRepeats * k_base);

    if (!IsKReduction) {
      // store bufferA logic.

      // Original C++ logic.
      // static_if<!IsKReduction>{}([&](auto) {
      //   for(index_t m_i = 0; m_i < MRepeats; ++m_i)
      //     for(index_t k_i      = 0; k_i < K; ++k_i)
      //       a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by waveOffsetA.

      auto outerLoopM = b.create<AffineForOp>(loc, 0, MRepeats);
      auto olmb = OpBuilder::atBlockTerminator(outerLoopM.getBody());
      auto olmiv = outerLoopM.getInductionVar();
      auto mOffset = olmb.create<MulIOp>(loc, MPerXdlopsConstantOp, olmiv);
      auto kOffsetA = olmb.create<MulIOp>(loc, olmiv, KConstantOp);

      auto innerLoopMK = olmb.create<AffineForOp>(loc, 0, K);
      auto ilmkb = OpBuilder::atBlockTerminator(innerLoopMK.getBody());
      auto ilmkiv = innerLoopMK.getInductionVar();

      //       a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by waveOffsetA.
      Value sourceOffsetBeforeTransformA = ilmkb.create<AddIOp>(
          loc, op.waveOffsetA(),
          ilmkb.create<AddIOp>(
              loc,
              ilmkb.create<AddIOp>(
                  loc, ilmkb.create<MulIOp>(loc, ilmkiv, MConstantOp), laneId),
              mOffset));

      if (KPack > 1)
        sourceOffsetBeforeTransformA =
            ilmkb.create<MulIOp>(loc, sourceOffsetBeforeTransformA,
                                 ilmkb.create<ConstantIndexOp>(loc, KPack));

      // FIXME: Move this to index diff maps?
      // Apply coord_transform for matrix A if necessarily.
      SmallVector<Value, 8> sourceOffsetA;
      if (transformMatrixA.hasValue() &&
          !transformMatrixA.getValue().isIdentity())
        sourceOffsetA =
            expandAffineMap(ilmkb, loc, transformMatrixA.getValue(),
                            ValueRange{sourceOffsetBeforeTransformA})
                .getValue();
      else
        sourceOffsetA.push_back(sourceOffsetBeforeTransformA);

      auto destOffsetA = ilmkb.create<AddIOp>(loc, ilmkiv, kOffsetA);

      Value valueA;
      if (KPack > 1) {
        valueA = emitLoadLogic(
            ilmkb, loc, op.matrixA().getType().template cast<MemRefType>(),
            op.bufferA().getType().template cast<MemRefType>().getElementType(),
            false, {}, op.matrixA(), sourceOffsetA);
      } else {
        valueA = ilmkb.create<memref::LoadOp>(loc, dataType, op.matrixA(),
                                              sourceOffsetA);
      }
      ilmkb.create<memref::StoreOp>(loc, valueA, op.bufferA(),
                                    ValueRange{destOffsetA});

      // store bufferB logic.

      // Original C++ logic.
      //   for(index_t n_i = 0; n_i < NRepeats; ++n_i)
      //     for(index_t k_i      = 0; k_i < K; ++k_i)
      //       b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by waveOffsetB.

      auto outerLoopN = b.create<AffineForOp>(loc, 0, NRepeats);
      auto olnb = OpBuilder::atBlockTerminator(outerLoopN.getBody());
      auto olniv = outerLoopN.getInductionVar();
      auto nOffset = olnb.create<MulIOp>(loc, NPerXdlopsConstantOp, olniv);
      auto kOffsetB = olnb.create<MulIOp>(loc, olniv, KConstantOp);

      auto innerLoopNK = olnb.create<AffineForOp>(loc, 0, K);
      auto ilnkb = OpBuilder::atBlockTerminator(innerLoopNK.getBody());
      auto ilnkiv = innerLoopNK.getInductionVar();

      //       b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by waveOffsetB.
      Value sourceOffsetBeforeTransformB = ilnkb.create<AddIOp>(
          loc, op.waveOffsetB(),
          ilnkb.create<AddIOp>(
              loc,
              ilnkb.create<AddIOp>(
                  loc, ilnkb.create<MulIOp>(loc, ilnkiv, NConstantOp), laneId),
              nOffset));

      if (KPack > 1)
        sourceOffsetBeforeTransformB =
            ilnkb.create<MulIOp>(loc, sourceOffsetBeforeTransformB,
                                 ilnkb.create<ConstantIndexOp>(loc, KPack));

      // Apply coord_transform for matrix B if necessarily.
      SmallVector<Value, 8> sourceOffsetB;
      if (transformMatrixB.hasValue() &&
          !transformMatrixB.getValue().isIdentity())
        sourceOffsetB =
            expandAffineMap(ilnkb, loc, transformMatrixB.getValue(),
                            ValueRange{sourceOffsetBeforeTransformB})
                .getValue();
      else
        sourceOffsetB.push_back(sourceOffsetBeforeTransformB);

      auto destOffsetB = ilnkb.create<AddIOp>(loc, ilnkiv, kOffsetB);

      Value valueB;
      if (KPack > 1) {
        valueB = emitLoadLogic(
            ilnkb, loc, op.matrixB().getType().template cast<MemRefType>(),
            op.bufferB().getType().template cast<MemRefType>().getElementType(),
            false, {}, op.matrixB(), sourceOffsetB);
      } else {
        valueB = ilnkb.create<memref::LoadOp>(loc, dataType, op.matrixB(),
                                              sourceOffsetB);
      }
      ilnkb.create<memref::StoreOp>(loc, valueB, op.bufferB(),
                                    ValueRange{destOffsetB});

      // Original C++ logic.
      //
      // for(index_t k_i = 0; k_i < K * KRepeats; ++k_i)
      // {
      //     p_c_thread = mfma_type.template run<MPerXdlops * MRepeats,
      //                                         NPerXdlops * NRepeats,
      //                                         AStride,
      //                                         BStride>(
      //         &pa[k_i * mfma_type.k_base], &pb[k_i * mfma_type.k_base],
      //         p_c_thread);
      // }

      // Rewrite as:
      //
      // for(index_t k_i = 0; k_i < K; ++k_i) {
      //   bufferAElement = a[k_i];
      //   bufferBElement = b[k_i];
      //   for(index_t ki_i = 0; ki_i < KRepeats; ++ki_i)
      //     argA = &bufferAElement[ki_i * mfma_type.k_base];
      //     argB = &bufferAElement[ki_i * mfma_type.k_base];
      //     p_c_thread = mfma_type.template run<MPerXlops * MRepeats,
      //                                         NPerXdlops * NRepeats,
      //                                         AStride,
      //                                         BStride>(argA, argB,
      //       p_c_thread);
      // }

      int64_t KForOuterLoop;
      if (KPack > 1) {
        KForOuterLoop = K;
      } else {
        KForOuterLoop = K / k_base;
        if (KForOuterLoop == 0) {
          // KForOuterLoop is too small. Reject lowering.
          return failure();
        }
      }
      auto outerLoop =
          b.create<AffineForOp>(loc, 0, KForOuterLoop, 1, op.vectorCs());
      auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
      auto outerLoopiv = outerLoop.getInductionVar();

      MemRefType bufferAType = op.bufferA().getType().cast<MemRefType>();
      MemRefType bufferBType = op.bufferA().getType().cast<MemRefType>();
      Type bufferAElementType = bufferAType.getElementType();
      Type bufferBElementType = bufferBType.getElementType();
      Value bufferAElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferAElementType, op.bufferA(), ValueRange{outerLoopiv});
      Value bufferBElement = outerLoopb.create<memref::LoadOp>(
          loc, bufferBElementType, op.bufferB(), ValueRange{outerLoopiv});

      auto innerLoop = outerLoopb.create<AffineForOp>(
          loc, 0, KRepeats, 1, outerLoop.getRegionIterArgs());
      auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
      auto innerLoopiv = innerLoop.getInductionVar();

      Value argA;
      Value argB;
      int64_t argTypeVectorLength =
          (argType.isa<VectorType>())
              ? argType.template cast<VectorType>().getShape()[0]
              : 1;
      if (argTypeVectorLength > 1) {
        Value zeroOp = createZeroConstantFloatOp(innerLoopb, loc, dataType);

        Value offset;
        if (KPack > 1) {
          offset = innerLoopb.create<MulIOp>(loc, innerLoopiv, KBaseConstantOp);
        } else {
          offset = innerLoopb.create<AddIOp>(
              loc, innerLoopb.create<MulIOp>(loc, outerLoopiv, KBaseConstantOp),
              innerLoopb.create<MulIOp>(loc, innerLoopiv, KBaseConstantOp));
        }
        if (bufferAElementType.isa<VectorType>()) {
          // bufferA/BElement loaded on LDS are vectors.
          // argA/B to be supplied to MFMA XDLOPS are also vectors.
          assert(bufferAElementType.isa<VectorType>());
          assert(bufferBElementType.isa<VectorType>());
          assert(bufferAElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferBElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferAElementType.cast<VectorType>().getShape()[0] %
                     argTypeVectorLength ==
                 0);
          assert(bufferBElementType.cast<VectorType>().getShape()[0] %
                     argTypeVectorLength ==
                 0);

          argA = innerLoopb.create<SplatOp>(loc, zeroOp, argType);
          argB = innerLoopb.create<SplatOp>(loc, zeroOp, argType);
          for (int64_t i = 0; i < argTypeVectorLength; ++i) {
            Value iConstantOp = innerLoopb.create<ConstantIndexOp>(loc, i);
            Value iConstantOp_i32 = innerLoopb.create<IndexCastOp>(
                loc, iConstantOp, innerLoopb.getIntegerType(32));
            Value iPlusOffsetConstantOp =
                innerLoopb.create<AddIOp>(loc, iConstantOp, offset);
            Value iPlusOffsetConstantOp_i32 = innerLoopb.create<IndexCastOp>(
                loc, iPlusOffsetConstantOp, innerLoopb.getIntegerType(32));

            Value elementA = innerLoopb.create<vector::ExtractElementOp>(
                loc, dataType, bufferAElement, iPlusOffsetConstantOp_i32);
            argA = innerLoopb.create<vector::InsertElementOp>(
                loc, argType, elementA, argA, iConstantOp_i32);
            Value elementB = innerLoopb.create<vector::ExtractElementOp>(
                loc, dataType, bufferBElement, iPlusOffsetConstantOp_i32);
            argB = innerLoopb.create<vector::InsertElementOp>(
                loc, argType, elementB, argB, iConstantOp_i32);
          }
        } else {
          // bufferA/BElement loaded on LDS are scalars.
          // argA/B to be supplied to MFMA XDLOPS are vectors.
          argA = innerLoopb.create<vector::TransferReadOp>(
              loc, argType.template cast<VectorType>(), op.bufferA(),
              ValueRange{offset});
          argB = innerLoopb.create<vector::TransferReadOp>(
              loc, argType.template cast<VectorType>(), op.bufferB(),
              ValueRange{offset});
        }
      } else {
        if (bufferAElementType.isa<VectorType>()) {
          // bufferA/BElement loaded on LDS are vectors.
          // argA/B to be supplied to MFMA XDLOPS are scalars.
          assert(bufferAElementType.isa<VectorType>());
          assert(bufferBElementType.isa<VectorType>());
          assert(bufferAElementType.cast<VectorType>().getShape().size() == 1);
          assert(bufferBElementType.cast<VectorType>().getShape().size() == 1);

          Value innerLoopiv_i32 = innerLoopb.create<IndexCastOp>(
              loc, innerLoopiv, innerLoopb.getIntegerType(32));
          argA = innerLoopb.create<vector::ExtractElementOp>(
              loc, dataType, bufferAElement, innerLoopiv_i32);
          argB = innerLoopb.create<vector::ExtractElementOp>(
              loc, dataType, bufferBElement, innerLoopiv_i32);
        } else {
          // bufferA/BElement loaded on LDS are scalars.
          // argA/B to be supplied to MFMA XDLOPS are also scalars.
          argA = bufferAElement;
          argB = bufferBElement;
        }
      }

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = innerLoop.getRegionIterArgs()[i];
        auto mfma =
            innerLoopb.create<MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma->setAttr("instr", innerLoopb.getStringAttr(mfmaInstr));
        mfma->setAttr("imm", innerLoopb.getArrayAttr(
                                 {innerLoopb.getI32IntegerAttr(imms[i][0]),
                                  innerLoopb.getI32IntegerAttr(imms[i][1]),
                                  innerLoopb.getI32IntegerAttr(imms[i][2])}));
        mfmas.push_back(mfma);
      }
      innerLoopb.create<AffineYieldOp>(loc, mfmas);

      outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());
      op.replaceAllUsesWith(outerLoop.results());
      op.erase();
    } else {
      // Original C++ logic.
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;

      auto NumThreadsBlkConstantOp =
          b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto blk_id = b.create<DivUIOp>(loc, laneId, NumThreadsBlkConstantOp);
      auto blk_td = b.create<RemUIOp>(loc, laneId, NumThreadsBlkConstantOp);

      // Original C++ logic.
      //     // load into registers
      //     for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks) {
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      //     }
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      // Instead loop to K, change loop bound to K / num_input_blks.
      auto loopKLoadIteration = K / num_input_blks;
      auto loopKLoad = b.create<AffineForOp>(loc, 0, loopKLoadIteration);

      auto NumInputBlksConstantOp =
          b.create<ConstantIndexOp>(loc, num_input_blks);

      auto lklb = OpBuilder::atBlockTerminator(loopKLoad.getBody());
      auto lkliv = loopKLoad.getInductionVar();

      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      // p_a_wave need to be offseted by waveOffsetA.
      // p_b_wave need to be offseted by waveOffsetB.

      // NOTICE: We times k_i by num_input_blks in MLIR path.
      Value sourceOffsetBeforeTransformA = lklb.create<AddIOp>(
          loc, op.waveOffsetA(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(
                  loc,
                  lklb.create<AddIOp>(
                      loc,
                      lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                      blk_id),
                  MConstantOp),
              blk_td));

      // Apply coord_transform for matrix A if necessarily.
      SmallVector<Value, 8> sourceOffsetA;
      if (transformMatrixA.hasValue() &&
          !transformMatrixA.getValue().isIdentity())
        sourceOffsetA =
            expandAffineMap(lklb, loc, transformMatrixA.getValue(),
                            ValueRange{sourceOffsetBeforeTransformA})
                .getValue();
      else
        sourceOffsetA.push_back(sourceOffsetBeforeTransformA);

      Value valueA;
      if (KPack > 1) {
        valueA = emitLoadLogic(
            lklb, loc, op.matrixA().getType().template cast<MemRefType>(),
            argType, false, {}, op.matrixA(), sourceOffsetA);
      } else {
        valueA = lklb.create<memref::LoadOp>(loc, dataType, op.matrixA(),
                                             ValueRange{sourceOffsetA});
      }
      lklb.create<memref::StoreOp>(loc, valueA, op.bufferA(),
                                   ValueRange{lkliv});

      // NOTICE: We times k_i by num_input_blks in MLIR path.
      Value sourceOffsetBeforeTransformB = lklb.create<AddIOp>(
          loc, op.waveOffsetB(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(
                  loc,
                  lklb.create<AddIOp>(
                      loc,
                      lklb.create<MulIOp>(loc, lkliv, NumInputBlksConstantOp),
                      blk_id),
                  NConstantOp),
              blk_td));

      // Apply coord_transform for matrix B if necessarily.
      SmallVector<Value, 8> sourceOffsetB;
      if (transformMatrixB.hasValue() &&
          !transformMatrixB.getValue().isIdentity())
        sourceOffsetB =
            expandAffineMap(lklb, loc, transformMatrixB.getValue(),
                            ValueRange{sourceOffsetBeforeTransformB})
                .getValue();
      else
        sourceOffsetB.push_back(sourceOffsetBeforeTransformB);

      Value valueB;
      if (KPack > 1) {
        valueB = emitLoadLogic(
            lklb, loc, op.matrixB().getType().template cast<MemRefType>(),
            argType, false, {}, op.matrixB(), sourceOffsetB);
      } else {
        valueB = lklb.create<memref::LoadOp>(loc, dataType, op.matrixB(),
                                             ValueRange{sourceOffsetB});
      }
      lklb.create<memref::StoreOp>(loc, valueB, op.bufferB(),
                                   ValueRange{lkliv});

      // Original C++ logic.
      // for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks)
      // {
      //     for(index_t i = 0; i < KRepeats; ++i)
      //         p_c_thread = mfma_type.template run<MPerXdlops, NPerXdlops,
      //         AStride, BStride>(
      //             &pa[(k_i * KRepeats + i) * mfma_type.k_base],
      //             &pb[(k_i * KRepeats + i) * mfma_type.k_base],
      //             p_c_thread);
      // }

      // Change loop bound to the same as loopKLoadIteration.
      // Instead of increasing num_input_blks, increase k_base.

      if (loopKLoadIteration == 0) {
        // K load iteration is too small. Reject lowering.
        return failure();
      }
      auto outerLoop = b.create<AffineForOp>(loc, 0, loopKLoadIteration, k_base,
                                             op.vectorCs());
      auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
      auto outerLoopiv = outerLoop.getInductionVar();
      auto outerOffset =
          outerLoopb.create<MulIOp>(loc, outerLoopiv, kBaseKRepeatsConstantOp);

      auto innerLoop = outerLoopb.create<AffineForOp>(
          loc, 0, KRepeats * k_base, k_base, outerLoop.getRegionIterArgs());
      auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
      auto innerLoopiv = innerLoop.getInductionVar();

      auto offset = innerLoopb.create<AddIOp>(loc, outerOffset, innerLoopiv);

      Value argA;
      Value argB;
      int64_t argTypeVectorLength =
          (argType.isa<VectorType>())
              ? argType.template cast<VectorType>().getShape()[0]
              : 1;
      if (argTypeVectorLength > 1) {
        argA = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.template cast<VectorType>(), op.bufferA(),
            ValueRange{offset});
        argB = innerLoopb.create<vector::TransferReadOp>(
            loc, argType.template cast<VectorType>(), op.bufferB(),
            ValueRange{offset});
      } else {
        argA = innerLoopb.create<memref::LoadOp>(loc, argType, op.bufferA(),
                                                 ValueRange{offset});
        argB = innerLoopb.create<memref::LoadOp>(loc, argType, op.bufferB(),
                                                 ValueRange{offset});
      }

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = innerLoop.getRegionIterArgs()[i];
        auto mfma =
            innerLoopb.create<MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma->setAttr("instr", innerLoopb.getStringAttr(mfmaInstr));
        mfma->setAttr("imm", innerLoopb.getArrayAttr(
                                 {innerLoopb.getI32IntegerAttr(imms[i][0]),
                                  innerLoopb.getI32IntegerAttr(imms[i][1]),
                                  innerLoopb.getI32IntegerAttr(imms[i][2])}));
        mfmas.push_back(mfma);
      }
      innerLoopb.create<AffineYieldOp>(loc, mfmas);

      outerLoopb.create<AffineYieldOp>(loc, innerLoop.results());

      op.replaceAllUsesWith(outerLoop.results());
      op.erase();
    }

    return success();
  }
};

void LowerMIOpenOpsStep4Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  patterns.insert<InWarpTransposeRewritePattern>(ctx);
  patterns.insert<ThreadwiseGemmRewritePattern>(ctx);
  patterns.insert<ThreadwiseCopyRewritePattern>(ctx);
  patterns.insert<ThreadwiseLoadRewritePattern>(ctx);
  patterns.insert<ThreadwiseStoreRewritePattern>(ctx);
  patterns.insert<ThreadwiseCopyV2RewritePattern>(ctx);
  patterns.insert<XdlopsGemmV2RewritePattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

void LowerMIOpenOpsStep5Pass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  OwningRewritePatternList patterns(ctx);
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep4Pass() {
  return std::make_unique<LowerMIOpenOpsStep4Pass>();
}

std::unique_ptr<Pass> mlir::miopen::createLowerMIOpenOpsStep5Pass() {
  return std::make_unique<LowerMIOpenOpsStep5Pass>();
}
