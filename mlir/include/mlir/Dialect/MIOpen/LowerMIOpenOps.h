//===- LowerMIOpenOps.h - MLIR to C++ for MIOpen conversion ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the lowering pass for the MLIR to MIOpen C++ conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_LOWERMIOPENOPS_H
#define MLIR_DIALECT_MIOPEN_LOWERMIOPENOPS_H

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MIOpen/AffineMapHelper.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "XdlopsCodeSelection.h"
#include "mlir/Dialect/MIOpen/Tuning/GridwiseGemmParams.h"
#include "mlir/Dialect/MIOpen/utility/common.h"
#include "mlir/Dialect/MIOpen/utility/loweringUtils.h"
#include "mlir/Dialect/MIOpen/utility/math.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::miopen;
using namespace mlir::arith;

// 2G ,INT MAX Value = 2147483647, use 2147483648 as offset and buffer
// store do nothing
static constexpr int kTwoGB = 2147483647;

//===----------------------------------------------------------------------===//
// FIXME. XXX.
// Force the use of affine maps over index maps in the presence of padding on
// GEMM during threadwise load/store/copy when the gemm is padded due to bugs in
// the index diff map implementation (or incompletenesses in it?)
//===----------------------------------------------------------------------===//
inline bool overrideLoadStoreHack(const PaddingInfoAttr paddingInfo,
                                  bool original) {
  if (paddingInfo.getExtraM() > 0 || paddingInfo.getExtraK() > 0 ||
      paddingInfo.getExtraN() > 0) {
    return true;
  }
  return original;
}

//===----------------------------------------------------------------------===//
// Utility function to determine the type to be loaded
//===----------------------------------------------------------------------===//
template <typename T>
inline std::tuple<Type, int, int, int>
computeLoadStoreTypeInfo(OpBuilder &b, T &gop, Type elementType,
                         SmallVectorImpl<Type> &variadicTypes, ArrayAttr dims,
                         bool isMatrixA, int64_t KPack = 1) {
  uint64_t loadLength = 1;
  uint64_t storeLength = 1;
  int vectorDim = dims.size() - 1;
  if (isMatrixA) {
    loadLength = gop->getAttr("matrix_a_source_data_per_read")
                     .template cast<IntegerAttr>()
                     .getInt();
    storeLength = gop->getAttr("matrix_a_dest_data_per_write_dim_m")
                      .template cast<IntegerAttr>()
                      .getInt();
    vectorDim = gop->getAttr("matrix_a_source_vector_read_dim")
                    .template cast<IntegerAttr>()
                    .getInt();
  } else {
    loadLength = gop->getAttr("matrix_b_source_data_per_read")
                     .template cast<IntegerAttr>()
                     .getInt();
    storeLength = gop->getAttr("matrix_b_dest_data_per_write_dim_n")
                      .template cast<IntegerAttr>()
                      .getInt();
    vectorDim = gop->getAttr("matrix_b_source_vector_read_dim")
                    .template cast<IntegerAttr>()
                    .getInt();
  }

  // In case KPack and vector load is used, and we vector load on GemmK
  // dimension (1), use the last dimension (GemmKPack) instead.
  if ((loadLength > 1) && (KPack > 1) && (vectorDim == GemmK)) {
    vectorDim = dims.size() - 1;
  }

  int64_t itemsToCopy = 1;
  for (llvm::APInt l : dims.getAsValueRange<IntegerAttr>())
    itemsToCopy *= l.getZExtValue();

  for (unsigned iter = 0; iter < itemsToCopy; ++iter)
    variadicTypes.push_back(elementType);

  return std::make_tuple(elementType, vectorDim, loadLength, storeLength);
}

//===----------------------------------------------------------------------===//
// Utility function to compute sliceLengths for threadwise_copy and
// threadwise_copy_v2 to determine the bounds of load/store loops.
//===----------------------------------------------------------------------===//
inline void computeSliceLengths(SmallVectorImpl<uint64_t> &sliceLengths,
                                const ArrayAttr bounds) {
  for (llvm::APInt v : bounds.getAsValueRange<IntegerAttr>()) {
    sliceLengths.push_back(v.getZExtValue());
  }
}

inline void computeSliceLengths(SmallVectorImpl<uint64_t> &sliceLengths,
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
// Utility function to emit constant float op. Returns a scalar.
//===----------------------------------------------------------------------===//
inline Value createConstantFloatOp(OpBuilder &b, Location loc, Type elementType,
                                   float value) {
  Value ret;
  if (elementType == b.getF32Type()) {
    ret = b.create<ConstantFloatOp>(loc, APFloat(value), b.getF32Type());
  } else if (elementType == b.getF16Type()) {
    bool lossy = false;
    APFloat constant(value);
    constant.convert(APFloat::IEEEhalf(), llvm::RoundingMode::TowardZero,
                     &lossy);
    ret = b.create<ConstantFloatOp>(loc, constant, b.getF16Type());
  } else if (elementType == b.getIntegerType(16)) {
    ret = b.create<ConstantIntOp>(loc, static_cast<int>(value),
                                  b.getIntegerType(16));
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// Utility function to emit constant zero op. Can return scalars or vectors.
//===----------------------------------------------------------------------===//
inline Value createZeroConstantFloatOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = type;
  if (type.isa<VectorType>())
    elementType = type.template cast<VectorType>().getElementType();
  auto semantics = static_cast<APFloat::Semantics>(-1);
  if (elementType == b.getF32Type()) {
    semantics = APFloat::S_IEEEsingle;
  } else if (elementType == b.getF16Type()) {
    semantics = APFloat::S_IEEEhalf;
  } else if (elementType == b.getIntegerType(16)) {
    semantics = APFloat::S_BFloat;
  } else {
    llvm_unreachable("Unexpected float semantics");
  }

  auto zero = APFloat::getZero(APFloat::EnumToSemantics(semantics));
  Value retValue;

  if (auto vecType = type.dyn_cast<VectorType>()) {
    Attribute constValue;
    if (auto intType = elementType.dyn_cast<IntegerType>()) {
      auto intZero = zero.bitcastToAPInt();
      assert(intType.getIntOrFloatBitWidth() == intZero.getBitWidth());
      constValue = b.getIntegerAttr(elementType, intZero);
    } else {
      constValue = b.getFloatAttr(elementType, zero);
    }
    llvm::SmallVector<Attribute> constValues;
    std::fill_n(std::back_inserter(constValues), vecType.getNumElements(),
                constValue);
    retValue = b.create<mlir::ConstantOp>(
        loc, DenseElementsAttr::get(vecType, constValues), type);
  } else {
    if (auto intType = elementType.dyn_cast<IntegerType>()) {
      auto intZero = zero.bitcastToAPInt();
      assert(intType.getIntOrFloatBitWidth() == intZero.getBitWidth());
      retValue =
          b.create<mlir::ConstantOp>(loc, b.getIntegerAttr(intType, intZero), type);
    } else {
      retValue =
          b.create<mlir::ConstantOp>(loc, b.getFloatAttr(elementType, zero), type);
    }
  }

  return retValue;
}

//===----------------------------------------------------------------------===//
// Utility function to emit load instructions with potentially OOB checks.
//===----------------------------------------------------------------------===//
inline Value emitLoadLogic(OpBuilder &b, Location loc, MemRefType sourceType,
                           Type loadedType, bool toEmitOOBLoadCheckLogic,
                           const ArrayRef<uint32_t> oobLoadCheckDims,
                           const Value source,
                           const ArrayRef<Value> srcLowerIndices) {
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
          auto loadedElement = b.create<memref::LoadOp>(loc, elementType, source,
                                                srcLowerIndicesUpdated);
          loadedVector = b.create<vector::InsertElementOp>(
              loc, loadedVectorType, loadedElement, loadedVector, iterIndex);
        }
        loadedValue = loadedVector;
      }
    } else {
      // Issue scalar load.
      loadedValue = b.create<memref::LoadOp>(loc, loadedType, source, srcLowerIndices);
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
inline void
emitStoreLogic(BwdPaddingKernelInfo bwdPaddingInfo, OpBuilder &b, Location loc,
               MemRefType destType, Type typeToStore,
               bool toEmitOOBStoreCheckLogic,
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
          b.create<memref::StoreOp>(loc, element, dest, destLowerIndicesUpdated);
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
inline bool obtainOOBCheckInfo(const Optional<AffineMap> composedTransform,
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
inline uint32_t obtainGenericTensorTransformationInfo(
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
inline Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
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
    // - fp16 -> fp32 : use fpext.
    // - fp32 -> fp16 : use fptrunc.
    // - fp16/fp32 -> bf16(i16) : use miopen.data_convert.
    // All these ops act elementwise on vectors
    // except the BFloat conversion
    if (sourceElemType == b.getF16Type() && destElemType == b.getF32Type()) {
      result = b.create<arith::ExtFOp>(loc, source, destType);
    } else if (sourceElemType == b.getF32Type() &&
               destElemType == b.getF16Type()) {
      result = b.create<arith::TruncFOp>(loc, source, destType);
    } else if (destElemType == b.getIntegerType(16)) {
      if (sourceElemType == sourceType) {
        result = b.create<miopen::DataConvertOp>(loc, destType, source);
      } else {
        result = createZeroConstantFloatOp(b, loc, destType);
        int64_t numElements = destType.cast<VectorType>().getNumElements();
        for (int64_t i = 0; i < numElements; ++i) {
          Value extracted = b.create<vector::ExtractElementOp>(
              loc, source, b.create<ConstantIndexOp>(loc, i));
          Value converted =
              b.create<miopen::DataConvertOp>(loc, destElemType, extracted);
          result = b.create<vector::InsertElementOp>(
              loc, converted, result, b.create<ConstantIndexOp>(loc, i));
        }
      }
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Utility function to emit the logic to copy between naive tensors.
// This function is used within the lowering logic of threadwise_copy.
//===----------------------------------------------------------------------===//
inline void emitNaiveTensorCopyLogic(
    OpBuilder &b, Location loc, int64_t nSliceRow, int64_t nSliceCol,
    int64_t dataPerAccess, const OperandRange sourceCoord,
    const OperandRange destCoord,
    const Optional<AffineMap> composedSourceTransform,
    const Optional<AffineMap> composedDestTransform, Type sourceElementType,
    Type destElementType, Value source, Value dest) {
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
      Value scalarValue =
          b.create<memref::LoadOp>(loc, sourceElementType, source, srcLowerIndices);
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
      b.create<memref::StoreOp>(loc, convertedScalarValue, dest, destLowerIndices);
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
inline Optional<int64_t> isConstantValue(Value v) {
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
inline void computeIndexDiffMap(OpBuilder &b, Location loc,
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
inline void computeBottomIndicesWithIndexDiffMap(
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
inline void populateLayeredIndicesWithTransformMetadata(
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
inline void computeBottomIndicesWithAffineMap(
    OpBuilder &b, Location &loc, SmallVectorImpl<Value> &bottomIndices,
    const ArrayRef<Value> originalCoords, const ArrayRef<int64_t> loopIVs,
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
// Conv2D (forward, backward) lowering.
//===----------------------------------------------------------------------===//

// Forward declare specialized lowerings that've been moved elsewhere
LogicalResult backwardWeightAtomicAdd(miopen::Conv2DBwdWeightOp op,
                                      PatternRewriter &b);
LogicalResult backwardData(miopen::Conv2DBwdDataOp op, PatternRewriter &b);
template <typename T>
LogicalResult checkNames(ArrayRef<StringRef> actual,
                         ArrayRef<StringRef> expected, StringRef argName,
                         T op) {
  if (actual.size() != expected.size()) {
    return op.emitOpError("Layout mismatch in ")
           << argName << " tensor: Expected " << expected.size()
           << " dimensions but have " << actual.size();
  }
  for (StringRef name : expected) {
    if (std::find(actual.begin(), actual.end(), name) == actual.end()) {
      return op.emitOpError("Layout mismatch in ")
             << argName << " tensor: Expected it to have a `" << name
             << "` dimension";
    }
  }
  return success();
}

template <typename T> struct Conv2DRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static miopen::ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    ConvolutionContext convContext = populateConvContext(op);

    bool isXdlops = false;
    auto xdlopsV2Attr = op->template getAttrOfType<BoolAttr>("xdlopsV2");
    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
      isXdlops = true;
    auto dataType =
        op.input().getType().template cast<MemRefType>().getElementType();
    if (miopen::ConvOpType::BwdData == convOpType) {
      return backwardData(cast<miopen::Conv2DBwdDataOp>(op), b);
    }
    auto loc = op.getLoc();

    auto archAttr = op->template getAttrOfType<StringAttr>("arch");
    auto numCuAttr = op->template getAttrOfType<IntegerAttr>("num_cu");

    auto KPackAttr = op->template getAttrOfType<IntegerAttr>("kpack");
    int64_t KPack = KPackAttr.getInt();

    auto filterLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        op->template getAttrOfType<ArrayAttr>("output_layout");

    auto dilationsAttr = op->template getAttrOfType<ArrayAttr>("dilations");
    auto stridesAttr = op->template getAttrOfType<ArrayAttr>("strides");
    auto paddingAttr = op->template getAttrOfType<ArrayAttr>("padding");

    // Get shape of filter tensor.
    auto filterType = op.filter().getType().template cast<MemRefType>();
    auto filterShape = filterType.getShape();

    // Get shape of input tensor.
    auto inputType = op.input().getType().template cast<MemRefType>();
    auto inputShape = inputType.getShape();

    // Get shape of output tensor.
    auto outputType = op.output().getType().template cast<MemRefType>();
    auto outputShape = outputType.getShape();

    // Obtain convolution parameters: padding / dialtion / stride.
    int64_t leftPadH =
        paddingAttr.getValue()[0].template cast<IntegerAttr>().getInt();
    int64_t leftPadW =
        paddingAttr.getValue()[2].template cast<IntegerAttr>().getInt();
    int64_t rightPadH =
        paddingAttr.getValue()[1].template cast<IntegerAttr>().getInt();
    int64_t rightPadW =
        paddingAttr.getValue()[3].template cast<IntegerAttr>().getInt();

    int64_t dilationH =
        dilationsAttr.getValue()[0].template cast<IntegerAttr>().getInt();
    int64_t dilationW =
        dilationsAttr.getValue()[1].template cast<IntegerAttr>().getInt();
    int64_t strideH =
        stridesAttr.getValue()[0].template cast<IntegerAttr>().getInt();
    int64_t strideW =
        stridesAttr.getValue()[1].template cast<IntegerAttr>().getInt();

    // get y, x, ho, wo, hi, wi, k, c, n
    int64_t y, x, ho, wo, hi, wi, k, c, n;
    y = x = ho = wo = hi = wi = k = c = n = 0;
    llvm::SmallVector<StringRef, 5> filterNames, inputNames, outputNames;
    for (uint32_t i = 0; i < filterLayoutAttr.size(); ++i) {
      auto filterAttr =
          filterLayoutAttr.getValue()[i].template cast<StringAttr>();
      auto inputAttr =
          inputLayoutAttr.getValue()[i].template cast<StringAttr>();
      auto outputAttr =
          outputLayoutAttr.getValue()[i].template cast<StringAttr>();

      filterNames.push_back(filterAttr.getValue());
      inputNames.push_back(inputAttr.getValue());
      outputNames.push_back(outputAttr.getValue());

      if (filterAttr.getValue() == "y") {
        y = filterShape[i];
      } else if (filterAttr.getValue() == "x") {
        x = filterShape[i];
      } else if (filterAttr.getValue() == "k") {
        k = filterShape[i];
      } else if (filterAttr.getValue() == "c") {
        c = filterShape[i];
      }

      if (inputAttr.getValue() == "hi") {
        hi = inputShape[i];
      } else if (inputAttr.getValue() == "wi") {
        wi = inputShape[i];
      } else if (inputAttr.getValue() == "ni") {
        n = inputShape[i];
      }

      if (outputAttr.getValue() == "ho") {
        ho = outputShape[i];
      } else if (outputAttr.getValue() == "wo") {
        wo = outputShape[i];
      }
    }

    if (failed(
            checkNames(filterNames, {"k", "g", "c", "y", "x"}, "filter", op)) ||
        failed(checkNames(inputNames, {"ni", "gi", "ci", "hi", "wi"}, "input",
                          op)) ||
        failed(checkNames(outputNames, {"no", "go", "ko", "ho", "wo"}, "output",
                          op))) {
      return failure();
    }

    int64_t gemmMSize, gemmNSize, gemmKSize;
    int64_t gemmMExtra, gemmNExtra, gemmKExtra;
    gemmMSize = gemmNSize = gemmKSize = 0;
    gemmMExtra = gemmNExtra = gemmKExtra = 0;
    // compute we should use extra padding kernel or not
    // c,k already / g ,so we can skip / g here
    switch (convOpType) {
    case miopen::ConvOpType::Fwd:
      gemmMSize = k;
      gemmKSize = c * y * x;
      gemmNSize = n * ho * wo;
      break;
    case miopen::ConvOpType::BwdData:
      gemmMSize = c;
      gemmKSize = k * y * x;
      gemmNSize = n * ho * wo;
      break;
    case miopen::ConvOpType::BwdWeight:
      gemmMSize = k;
      gemmKSize = n * ho * wo;
      gemmNSize = c * y * x;
      break;
    }

    bool needExtraPad = false;

    auto calculatePaddingKernelSize =
        [&needExtraPad, gemmMSize, gemmNSize, gemmKSize, &gemmMExtra,
         &gemmNExtra, &gemmKExtra, &convContext](auto populateParams) {
          auto configParams = populateParams.getTuningParameters(convContext);
          size_t numOfFailedConfigs = 0;
          for (auto &params : configParams) {
            if (gemmMSize % params.gemmMPerBlock == 0 &&
                gemmKSize % params.gemmKPerBlock == 0 &&
                gemmNSize % params.gemmNPerBlock == 0) {
              break;
            }
            numOfFailedConfigs++;
          }

          auto extraParams = populateParams.getUniversalParameters();
          if (numOfFailedConfigs == configParams.size()) {
            needExtraPad = true;
            int64_t gemmMRemain, gemmKRemain, gemmNRemain;

            gemmMRemain = gemmMSize % extraParams.gemmMPerBlock;
            if (gemmMRemain != 0)
              gemmMExtra = extraParams.gemmMPerBlock - gemmMRemain;

            gemmNRemain = gemmNSize % extraParams.gemmNPerBlock;
            if (gemmNRemain != 0)
              gemmNExtra = extraParams.gemmNPerBlock - gemmNRemain;

            gemmKRemain = gemmKSize % extraParams.gemmKPerBlock;
            if (gemmKRemain != 0)
              gemmKExtra = extraParams.gemmKPerBlock - gemmKRemain;

            // llvm::errs() << "gemmMExtra: " << gemmMExtra << "gemmNExtra: " <<
            // gemmNExtra << "gemmKExtra: " << gemmKExtra << "\n";
          }
        };

    if (!isXdlops) {
      PopulateParams populateParams;
      calculatePaddingKernelSize(populateParams);
    } else { // xdlops
      PopulateParamsXDL populateParamsXDL;
      calculatePaddingKernelSize(populateParamsXDL);
    }

    if (miopen::ConvOpType::BwdWeight == convOpType && isXdlops &&
        dataType == b.getF32Type() && needExtraPad == false) {
      // current backward weight with atomic_add can only run under xdlops +
      // fp32
      return backwardWeightAtomicAdd(cast<miopen::Conv2DBwdWeightOp>(op), b);
    }

    // Transform filter tensor.

    // filter dims need oob check
    OobCheckSet filterOobCheckDims;
    // set layout attribute.
    // Weight tensor transformation for Conv2DOp
    // - PassThrough G dimension to dimension 0, name it gemmG.
    // - Merge non-K dimensions to dimension 1, name it as gemmK.
    //   Optimization: If non-K dimensions are consequetive, apply unfold.
    // - PassThrough K dimension to dimension 2, name it as gemmM.
    //
    // Weight tensor transformation for Conv2DBwdWeightOp
    // - PassThrough G dimension to dimension 0, name it gemmG
    // - PassThrough K dimension to dimension 1, name it as gemmM.
    // - Merge non-K dimensions to dimension 2, name it as gemmN.
    SmallVector<StringRef, 5> filterNonKDims;
    for (StringRef name : filterNames)
      if (name != "g" && name != "k")
        filterNonKDims.push_back(name);

    BottomUpCTBuilder filterTransform(b, filterNames, filterShape, loc);
    filterTransform.passThrough({"gemmG"}, {0}, {"g"});
    bool isUnfold = filterTransform.startIndex("g") == 0 &&
                    (filterTransform.startIndex("k") == 1 ||
                     filterTransform.startIndex("k") == 4);
    switch (convOpType) {
    case miopen::ConvOpType::Fwd:
      filterTransform.merge("gemmK", 1, filterNonKDims, isUnfold);
      filterTransform.passThrough({"gemmM"}, {2}, {"k"});
      break;
    case miopen::ConvOpType::BwdWeight:
      filterTransform.passThrough({"gemmM"}, {1}, {"k"});
      filterTransform.merge("gemmN", 2, filterNonKDims, isUnfold);
      break;
    case miopen::ConvOpType::BwdData:
      llvm_unreachable("Backward data has been sent elsewhere");
      break;
    }

    TransformMapAttr filterTransformAttr = filterTransform.get();
    Value gemmFilter =
        b.create<miopen::TransformOp>(loc, op.filter(), filterTransformAttr);

    BottomUpCTBuilder padGemmFilterTransform = filterTransform;
    TransformMapAttr padGemmFilterTransformAttr = filterTransformAttr;
    Value gemmFilterPad = gemmFilter;

    // filter pad start
    // K:output channel, C:input channel,Y:filter height,X:filter width
    // filter dim : K & merge(C,Y,X) , if C*Y*X is under 64 or 32
    // we pad CYX to 32 or 64, then mlir can do gemm
    // we add more one transform to do pad
    bool filterCheckPadGemmM = false;
    bool filterCheckPadGemmK = false;
    bool filterCheckPadGemmN = false;
    filterCheckPadGemmM =
        (convOpType == miopen::ConvOpType::Fwd && gemmMExtra > 0) ||
        (convOpType == miopen::ConvOpType::BwdWeight && gemmMExtra > 0);
    filterCheckPadGemmK =
        (convOpType == miopen::ConvOpType::Fwd && gemmKExtra > 0);
    filterCheckPadGemmN =
        (convOpType == miopen::ConvOpType::BwdWeight && gemmNExtra > 0);
    bool isFilterPad = false;
    if (filterCheckPadGemmM || filterCheckPadGemmK || filterCheckPadGemmN) {
      isFilterPad = true;
      padGemmFilterTransform =
          BottomUpCTBuilder::above(filterTransform, filterTransformAttr);
      padGemmFilterTransform.passThrough("gemmG");

      // Note that, when padding a gemm dimension that came from the non-K
      // tensor dimensions, only the leading dimension is added to the oob check
      // set, as adding all the dimensions historically led to miscompilation
      if (filterCheckPadGemmK) {
        padGemmFilterTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
        filterOobCheckDims.insert(
            filterTransform.startIndex(filterNonKDims[0]));
      } else if (convOpType != miopen::ConvOpType::BwdWeight) {
        // Backward weight has no GemmK on its filter
        padGemmFilterTransform.passThrough("gemmK");
      }

      if (filterCheckPadGemmM) {
        padGemmFilterTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
        filterOobCheckDims.insert(filterTransform.startIndex("k"));
      } else {
        padGemmFilterTransform.passThrough("gemmM");
      }

      if (filterCheckPadGemmN) {
        padGemmFilterTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
        filterOobCheckDims.insert(
            filterTransform.startIndex(filterNonKDims[0]));
      } else if (convOpType != miopen::ConvOpType::Fwd) {
        padGemmFilterTransform.passThrough("gemmN");
      }

      padGemmFilterTransformAttr = padGemmFilterTransform.get();
      gemmFilterPad = b.create<miopen::TransformOp>(loc, gemmFilter,
                                                    padGemmFilterTransformAttr);
      // filter pad end
    }

    // KPack for filter tensor.
    Value gemmFilterKPack = gemmFilterPad;

    // FIXME. consider backward convolution.
    if ((KPack > 1) && (convOpType == miopen::ConvOpType::Fwd)) {
      if (!isFilterPad) {
        BottomUpCTBuilder kpackGemmFilterTransform =
            BottomUpCTBuilder::above(filterTransform, filterTransformAttr);
        int64_t gemmKLength = filterTransform.endSize("gemmK");
        kpackGemmFilterTransform.passThrough("gemmG");
        kpackGemmFilterTransform.passThrough("gemmM");
        kpackGemmFilterTransform.unmerge({"gemmK", "gemmKPack"}, {1, 3},
                                         "gemmK", {gemmKLength / KPack, KPack});
        TransformMapAttr kpackGemmFilterTransformAttr =
            kpackGemmFilterTransform.get();
        gemmFilterKPack = b.create<miopen::TransformOp>(
            loc, gemmFilter, kpackGemmFilterTransformAttr);
      } else {
        // isFilterPad == true.
        BottomUpCTBuilder kpackGemmFilterTransform = BottomUpCTBuilder::above(
            padGemmFilterTransform, padGemmFilterTransformAttr);
        int64_t gemmKLength = 0;
        if (filterCheckPadGemmK) {
          gemmKLength = padGemmFilterTransform.endSize("gemmKPad");
        } else {
          gemmKLength = padGemmFilterTransform.endSize("gemmK");
        }
        kpackGemmFilterTransform.passThrough("gemmG");
        if (filterCheckPadGemmM) {
          kpackGemmFilterTransform.passThrough("gemmMPad");
        } else {
          kpackGemmFilterTransform.passThrough("gemmM");
        }
        if (filterCheckPadGemmK) {
          kpackGemmFilterTransform.unmerge({"gemmKPad", "gemmKPack"}, {1, 3},
                                           "gemmKPad",
                                           {gemmKLength / KPack, KPack});
        } else {
          kpackGemmFilterTransform.unmerge({"gemmK", "gemmKPack"}, {1, 3},
                                           "gemmK",
                                           {gemmKLength / KPack, KPack});
        }
        TransformMapAttr kpackGemmFilterTransformAttr =
            kpackGemmFilterTransform.get();
        gemmFilterKPack = b.create<miopen::TransformOp>(
            loc, gemmFilterPad, kpackGemmFilterTransformAttr);
      }
    }

    // Transform input tensor.
    // Input tensor step 1: padded input.

    OobCheckSet inputOobCheckDims;
    // set layout attribute.
    // Padded input tensor transformation:
    // - Pass through ni, gi, and ci, not renaming them
    // - Padd hi and wi as specified in padding attributes, renaming them to
    // hipad and wipad
    BottomUpCTBuilder padInputTransform(b, inputNames, inputShape, loc);
    padInputTransform.passThrough("ni");
    padInputTransform.passThrough("gi");
    padInputTransform.passThrough("ci");

    llvm::SmallVector<int64_t, 4> padArgs = {leftPadH, rightPadH, leftPadW,
                                             rightPadW};
    llvm::SmallVector<uint32_t, 2> padOutDims = {
        padInputTransform.startIndex("hi"), padInputTransform.startIndex("wi")};
    padInputTransform.pad({"hipad", "wipad"}, padOutDims, {"hi", "wi"},
                          padArgs);
    if (leftPadH || rightPadH) {
      inputOobCheckDims.insert(padInputTransform.startIndex("hi"));
    }
    if (leftPadW || rightPadW) {
      inputOobCheckDims.insert(padInputTransform.startIndex("wi"));
    }

    TransformMapAttr padInputTransformAttr = padInputTransform.get();

    Value paddedInput =
        b.create<miopen::TransformOp>(loc, op.input(), padInputTransformAttr);

    // Input tensor step 2 : embedded input.
    // Embedded input tensor transformation:
    // - PassThrough gi, ni, and ci
    // - Embed hipad to y and ho with size filter y by output h and
    //   coefficients dilationH and strideH
    // - Embed wipad to x and wo with size filter x by output h and
    //   coefficients dilationW and strideW

    llvm::SmallVector<StringRef, 5> paddedInputNames;
    padInputTransform.getEndNames(paddedInputNames);
    llvm::StringMap<uint32_t> embeddedInputDims;
    for (uint32_t i = 0, newDim = 0, e = paddedInputNames.size(); i < e; ++i) {
      StringRef name = paddedInputNames[i];
      if (name == "hipad") {
        embeddedInputDims.insert({"y", newDim++});
        embeddedInputDims.insert({"ho", newDim++});
      } else if (name == "wipad") {
        embeddedInputDims.insert({"x", newDim++});
        embeddedInputDims.insert({"wo", newDim++});
      } else {
        embeddedInputDims.insert({name, newDim++});
      }
    }

    BottomUpCTBuilder embedInputTransform(
        b, paddedInputNames, padInputTransformAttr.getUpperBounds(), loc);
    embedInputTransform.passThrough({"ni", "gi", "ci"},
                                    {embeddedInputDims["ni"],
                                     embeddedInputDims["gi"],
                                     embeddedInputDims["ci"]},
                                    {"ni", "gi", "ci"});
    embedInputTransform.embed({"y", "ho"},
                              {embeddedInputDims["y"], embeddedInputDims["ho"]},
                              {y, ho}, "hipad", {dilationH, strideH});
    embedInputTransform.embed({"x", "wo"},
                              {embeddedInputDims["x"], embeddedInputDims["wo"]},
                              {x, wo}, "wipad", {dilationW, strideW});

    TransformMapAttr embedInputTransformAttr = embedInputTransform.get();
    Value embeddedInput = b.create<miopen::TransformOp>(
        loc, paddedInput, embedInputTransformAttr);

    // Input tensor step 3: GEMM'd input
    //
    // - PassThrough gi to dimension 0 and name it gemmG, then
    // For Conv2DOp:
    // - Merge ci, y, x dimensions to dimension 1, name it as gemmK.
    // - Merge ni, ho, wo dimensions to dimension 2, name it as gemmN.
    //
    // For Conv2DBwdWeightOp:
    // - Part 1: Merge ni, ho, wo dimensions to dimension 1, name it as gemmK.
    // - Part 2: Merge ci, y, x dimensions to dimension 2, name it as gemmN.

    auto gemmInputTransform =
        BottomUpCTBuilder::above(embedInputTransform, embedInputTransformAttr);
    gemmInputTransform.passThrough({"gemmG"}, {0}, {"gi"});

    llvm::SmallVector<StringRef, 3> nonNHWDims = {"ci", "y", "x"};
    std::sort(nonNHWDims.begin(), nonNHWDims.end(),
              [&gemmInputTransform](const StringRef &v1,
                                    const StringRef &v2) -> bool {
                return gemmInputTransform.startIndex(v1) <
                       gemmInputTransform.startIndex(v2);
              });

    llvm::SmallVector<StringRef, 3> mergeToK, mergeToN;
    switch (convOpType) {
    case miopen::ConvOpType::Fwd:
      mergeToK = std::move(nonNHWDims);
      mergeToN = {"ni", "ho", "wo"};
      break;
    case miopen::ConvOpType::BwdWeight:
      mergeToK = {"ni", "ho", "wo"};
      mergeToN = std::move(nonNHWDims);
      break;
    case miopen::ConvOpType::BwdData:
      llvm_unreachable("Backward data is in another function");
    }
    gemmInputTransform.merge("gemmK", 1, mergeToK);
    gemmInputTransform.merge("gemmN", 2, mergeToN);

    TransformMapAttr gemmInputTransformAttr = gemmInputTransform.get();
    Value gemmInput = b.create<miopen::TransformOp>(loc, embeddedInput,
                                                    gemmInputTransformAttr);

    BottomUpCTBuilder padGemmInputTransform = gemmInputTransform;
    TransformMapAttr padGemmInputTransformAttr = gemmInputTransformAttr;
    Value gemmInputPad = gemmInput;

    // input padding start
    // input : NHW & CRS , if CRS is under 64 or 32
    // we pad CRS to 32 or 64, then mlir can do gemm
    // we add more one transform to do pad

    // input forward : gemmK,gemmN
    // backward weights: gemmK,gemmN
    // so we don't need to pad gemmK
    bool inputCheckPadGemmK = false;
    bool inputCheckPadGemmN = false;
    inputCheckPadGemmK =
        (convOpType == miopen::ConvOpType::Fwd && gemmKExtra > 0) ||
        (convOpType == miopen::ConvOpType::BwdWeight && gemmKExtra > 0);
    inputCheckPadGemmN =
        (convOpType == miopen::ConvOpType::Fwd && gemmNExtra > 0) ||
        (convOpType == miopen::ConvOpType::BwdWeight && gemmNExtra > 0);
    bool isInputPad = false;
    if (inputCheckPadGemmK || inputCheckPadGemmN) {
      isInputPad = true;
      padGemmInputTransform = BottomUpCTBuilder::above(gemmInputTransform, gemmInputTransformAttr);

      padGemmInputTransform.passThrough("gemmG");
      if (inputCheckPadGemmK) {
        padGemmInputTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);

        // Forward convolution has a K made from c and parts of w and h
        // while backward weights have n and parts of w and h
        inputOobCheckDims.insert(padInputTransform.startIndex("hi"));
        inputOobCheckDims.insert(padInputTransform.startIndex("wi"));

        inputOobCheckDims.insert(padInputTransform.startIndex(
            convOpType == miopen::ConvOpType::Fwd ? "ci" : "ni"));
      } else {
        padGemmInputTransform.passThrough("gemmK");
      }

      if (inputCheckPadGemmN) {
        padGemmInputTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);

        // Forward convolution has a N made from n and parts of w and h
        // while backward weights have c and parts of w and h
        inputOobCheckDims.insert(padInputTransform.startIndex("hi"));
        inputOobCheckDims.insert(padInputTransform.startIndex("wi"));

        inputOobCheckDims.insert(padInputTransform.startIndex(
            convOpType == miopen::ConvOpType::Fwd ? "ni" : "ci"));
      } else {
        padGemmInputTransform.passThrough("gemmN");
      }

      TransformMapAttr padGemmInputTransformAttr = padGemmInputTransform.get();
      gemmInputPad = b.create<miopen::TransformOp>(loc, gemmInput,
                                                   padGemmInputTransformAttr);
      // input padding end
    }

    // KPack for input tensor.
    Value gemmInputKPack = gemmInputPad;

    // FIXME. consider backward convolution.
    if ((KPack > 1) && (convOpType == miopen::ConvOpType::Fwd)) {
      if (!isInputPad) {
        BottomUpCTBuilder kpackGemmInputTransform = BottomUpCTBuilder::above(gemmInputTransform, gemmInputTransformAttr);
        int64_t gemmKLength = gemmInputTransform.endSize("gemmK");
        kpackGemmInputTransform.passThrough("gemmG");
        kpackGemmInputTransform.passThrough("gemmN");
        kpackGemmInputTransform.unmerge({"gemmK", "gemmKPack"}, {1, 3}, "gemmK", {gemmKLength / KPack, KPack});
        TransformMapAttr kpackGemmInputTransformAttr = kpackGemmInputTransform.get();
        gemmInputKPack = b.create<miopen::TransformOp>(loc, gemmInput, kpackGemmInputTransformAttr);
      } else {
        // isInputPad == true
        BottomUpCTBuilder kpackGemmInputTransform = BottomUpCTBuilder::above(padGemmInputTransform, padGemmInputTransformAttr);
        int64_t gemmKLength = 0;
        if (inputCheckPadGemmK) {
          gemmKLength = padGemmInputTransform.endSize("gemmKPad");
        } else {
          gemmKLength = padGemmInputTransform.endSize("gemmK");
        }
        kpackGemmInputTransform.passThrough("gemmG");
        if (inputCheckPadGemmN) {
          kpackGemmInputTransform.passThrough("gemmNPad");
        } else {
          kpackGemmInputTransform.passThrough("gemmN");
        }
        if (inputCheckPadGemmK) {
          kpackGemmInputTransform.unmerge({"gemmKPad", "gemmKPack"}, {1, 3}, "gemmKPad", {gemmKLength / KPack, KPack});
        } else {
          kpackGemmInputTransform.unmerge({"gemmK", "gemmKPack"}, {1, 3}, "gemmK", {gemmKLength / KPack, KPack});
        }
        TransformMapAttr kpackGemmInputTransformAttr = kpackGemmInputTransform.get();
        gemmInputKPack = b.create<miopen::TransformOp>(loc, gemmInputPad, kpackGemmInputTransformAttr);
      }
    }

    // Transform output tensor.
    OobCheckSet outputOobCheckDims;
    // - PassThrough G to dimmension 0, name it gemmG, then
    // Output tensor transformation for Conv2DOp:
    // - PassThrough K dimension to dimension 1, named gemmM
    // - Merge non-K dimensions to dimension2, named gemmN

    // Output tensor transformation for backward weight:
    // - Merge non-K dimensions to dimension 1, named gemmK
    // - PassThrough K dimension to dimension 2, name it gemmM
    SmallVector<StringRef, 5> outputNonKDims;
    for (StringRef name : outputNames)
      if (name != "go" && name != "ko")
        outputNonKDims.push_back(name);

    BottomUpCTBuilder outputTransform(b, outputNames, outputShape, loc);
    outputTransform.passThrough({"gemmG"}, {0}, {"go"});
    switch (convOpType) {
    case miopen::ConvOpType::Fwd:
      outputTransform.passThrough({"gemmM"}, {1}, {"ko"});
      outputTransform.merge("gemmN", 2, outputNonKDims);
      break;
    case miopen::ConvOpType::BwdWeight:
      outputTransform.merge("gemmK", 1, outputNonKDims);
      outputTransform.passThrough({"gemmM"}, {2}, {"ko"});
      break;
    case miopen::ConvOpType::BwdData:
      llvm_unreachable("Backward data has been sent elsewhere");
      break;
    }

    TransformMapAttr outputTransformAttr = outputTransform.get();
    Value gemmOutput =
        b.create<miopen::TransformOp>(loc, op.output(), outputTransformAttr);
    Value gemmOutputPad = gemmOutput;

    // output padding start
    // output matrix dim: K & NHW
    // when backward weight , GEMMK = NHW
    // N:batch size, H:output height ,W:output width
    // If size of N*h*w is under 32 or 64 ,we pad it to 32 or 64
    // then mlir can do gemm
    // we just add more one transform to do it

    bool outputCheckPadGemmK = false;
    bool outputCheckPadGemmM = false;
    bool outputCheckPadGemmN = false;
    outputCheckPadGemmK =
        (convOpType == miopen::ConvOpType::BwdWeight && gemmKExtra > 0);
    outputCheckPadGemmM =
        (convOpType == miopen::ConvOpType::BwdWeight && gemmMExtra > 0) ||
        (convOpType == miopen::ConvOpType::Fwd && gemmMExtra > 0);
    outputCheckPadGemmN =
        (convOpType == miopen::ConvOpType::Fwd && gemmNExtra > 0);
    if (outputCheckPadGemmK || outputCheckPadGemmM || outputCheckPadGemmN) {
      auto padGemmOutputTransform =
          BottomUpCTBuilder::above(outputTransform, outputTransformAttr);
      padGemmOutputTransform.passThrough("gemmG");

      if (outputCheckPadGemmK) {
        padGemmOutputTransform.pad("gemmKPad", "gemmK", 0, gemmKExtra);
        // Unlike in the filter case, add all the dimensions to the oob check
        // since this is loading data during backward weights, not storing it
        outputOobCheckDims.insert(outputTransform.startIndex("no"));
        outputOobCheckDims.insert(outputTransform.startIndex("ho"));
        outputOobCheckDims.insert(outputTransform.startIndex("wo"));
      } else if (convOpType != miopen::ConvOpType::Fwd) {
        padGemmOutputTransform.passThrough("gemmK");
      }

      if (outputCheckPadGemmM) {
        padGemmOutputTransform.pad("gemmMPad", "gemmM", 0, gemmMExtra);
        // For the cases considered in this function, the m dimension of the
        // gemm is always the k dimension of the output tensor
        outputOobCheckDims.insert(outputTransform.startIndex("ko"));
      } else {
        padGemmOutputTransform.passThrough("gemmM");
      }

      if (outputCheckPadGemmN) {
        padGemmOutputTransform.pad("gemmNPad", "gemmN", 0, gemmNExtra);
        // As in the filter case with backward weight, set only the outermost
        // dimension of the output tensor for OOB checks or else extra 0s will
        // appear in the output
        outputOobCheckDims.insert(
            outputTransform.startIndex(outputNonKDims[0]));
      } else if (convOpType != miopen::ConvOpType::BwdWeight) {
        padGemmOutputTransform.passThrough("gemmN");
      }

      TransformMapAttr padGemmOutputTransformAttr =
          padGemmOutputTransform.get();
      gemmOutputPad = b.create<miopen::TransformOp>(loc, gemmOutput,
                                                    padGemmOutputTransformAttr);
      // output padding end
    }

    // Set attributes for gridwise_gemm op.
    llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
        b.getNamedAttr("arch", archAttr),
        b.getNamedAttr("num_cu", numCuAttr),
    };

    // xdlopsV2.
    if (isXdlops)
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

    if (convOpType == miopen::ConvOpType::BwdData) {
      gridwiseGemmAttrs.push_back(b.getNamedAttr(
          "kernel_algorithm", b.getStringAttr("backward_data_v1r1")));
    } else if (convOpType == miopen::ConvOpType::Fwd) {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kernel_algorithm", b.getStringAttr("v4r4")));
    } else if (convOpType == miopen::ConvOpType::BwdWeight) {
      gridwiseGemmAttrs.push_back(b.getNamedAttr(
          "kernel_algorithm", b.getStringAttr("backward_weight_v4r4")));
    }

    // Gather up OOB check attribute arrays
    ArrayAttr filterOobAttr =
        getBoundsCheckAttr(b, filterOobCheckDims, filterShape.size());
    ArrayAttr inputOobAttr =
        getBoundsCheckAttr(b, inputOobCheckDims, inputShape.size());
    ArrayAttr outputOobAttr =
        getBoundsCheckAttr(b, outputOobCheckDims, outputShape.size());

    SmallVector<Value, 3> arguments = {gemmFilterKPack, gemmInputKPack,
                                       gemmOutputPad};
    SmallVector<ArrayAttr, 3> oobs = {filterOobAttr, inputOobAttr,
                                      outputOobAttr};

    Value gemmA, gemmB, gemmC;
    ArrayAttr oobA, oobB, oobC;
    gemmA = arguments[fields.gridwiseGemmArgumentPosition[0]];
    oobA = oobs[fields.gridwiseGemmArgumentPosition[0]];
    gemmB = arguments[fields.gridwiseGemmArgumentPosition[1]];
    oobB = oobs[fields.gridwiseGemmArgumentPosition[1]];
    gemmC = arguments[fields.gridwiseGemmArgumentPosition[2]];
    oobC = oobs[fields.gridwiseGemmArgumentPosition[2]];

    // Create padding info attr
    PaddingInfoAttr paddingInfo =
        PaddingInfoAttr::get(b.getContext(), gemmMExtra, gemmKExtra, gemmNExtra,
                             BwdPaddingKernelInfo::NA);

    auto dataOp = InMemoryDataOperation::Set;
    // Emit miopen.gridwise_gemm op.
    // Emit miopen.gridwise_gemm_v2 if xdlopsV2 attribute is true.

    // Supply KPack information into gridwiseGemmAttrs.
    // FIXME. consider backward convolution.
    if ((KPack > 1) && (convOpType == miopen::ConvOpType::Fwd)) {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kpack", b.getI32IntegerAttr(KPack)));
    } else {
      // FIXME. Skip KPACK for backward passes for now.
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kpack", b.getI32IntegerAttr(1)));
    }

    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      auto gop = b.create<miopen::GridwiseGemmV2Op>(
          loc, gemmA, gemmB, gemmC, oobA, oobB, oobC, paddingInfo, dataOp,
          gridwiseGemmAttrs);
      affixGridwiseGemmAttributes(op, gop, b);
    } else {
      auto gop = b.create<miopen::GridwiseGemmOp>(loc, gemmA, gemmB, gemmC,
                                                  oobA, oobB, oobC, paddingInfo,
                                                  gridwiseGemmAttrs);
      affixGridwiseGemmAttributes(op, gop, b);
    }

    // Finally, erase the original Conv2D op.
    op.erase();

    return success();
  }
};

// Forward-declare field values to suppress warnings
template <> const ArgumentFields Conv2DRewritePattern<miopen::Conv2DOp>::fields;
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DOp>::convOpType;
template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::fields;
template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::convOpType;
template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::fields;
template <>
const miopen::ConvOpType
    Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::convOpType;
//===----------------------------------------------------------------------===//
// Assigning attributes.
//===----------------------------------------------------------------------===//

static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::GridwiseGemmOp gop,
                                          OpBuilder &b) {
  top->setAttr("vector_read_write_dim",
               gop->getAttr("matrix_c_dest_vector_write_dim"));
  top->setAttr("source_data_per_read", gop->getAttr("matrix_c_data_per_copy"));
  top->setAttr("dest_data_per_write", gop->getAttr("matrix_c_data_per_copy"));
}

static void affixThreadwiseCopyV2Attributes(miopen::ThreadwiseCopyV2Op top,
                                            miopen::GridwiseGemmV2Op gop,
                                            OpBuilder &b, bool isSwizzled) {
  // Account for split m/n dimension
  bool vectorStoreOverride = false;
  int64_t vectorGemmDim =
      gop->getAttrOfType<IntegerAttr>("matrix_c_source_vector_read_dim")
          .getInt();
  // Remap vectorized gemm dimensions to account for
  if (vectorGemmDim == gemmCDimM) {
    vectorGemmDim = gemmCSplitDimM2;
  } else if (vectorGemmDim == gemmCDimN) {
    if (isSwizzled) {
      vectorGemmDim = gemmCSplitDimN2;
    } else {
      vectorGemmDim = gemmCSplitDimN;
      // Need swizzles for this to be vector motion but swizzles are off
      vectorStoreOverride = true;
    }
  }
  Attribute dataPerCopy = gop->getAttr("matrix_c_data_per_copy");
  if (vectorStoreOverride) {
    dataPerCopy = b.getI32IntegerAttr(1);
  }
  top->setAttr("upper_vector_read_dim", b.getI32IntegerAttr(vectorGemmDim));
  top->setAttr("vector_read_write_dim",
               gop->getAttr("matrix_c_dest_vector_write_dim"));
  top->setAttr("source_data_per_read", dataPerCopy);
  top->setAttr("dest_data_per_write", dataPerCopy);
}

// XXX: Figure out a way to do away with isThreadwiseLoad parameter.
template <typename T, typename U>
static void affixThreadwiseCopyAttributes(T &top, U &bop, OpBuilder &b,
                                          bool isThreadwiseLoad) {
  if (isThreadwiseLoad) {
    top->setAttr("vector_read_write_dim",
                 bop->getAttr("source_vector_read_dim"));
    top->setAttr("source_data_per_read", bop->getAttr("source_data_per_read"));
    top->setAttr("dest_data_per_write", bop->getAttr("dest_data_per_write"));
  } else {
    top->setAttr("vector_read_write_dim",
                 bop->getAttr("dest_vector_write_dim"));
    top->setAttr("source_data_per_read", bop->getAttr("source_data_per_read"));
    top->setAttr("dest_data_per_write", bop->getAttr("dest_data_per_write"));
  }
}

// XXX: figure out a better way to get rid of isMatrixA parameter.
static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::BlockwiseGemmOp bop,
                                          OpBuilder &b, bool isMatrixA) {
  if (isMatrixA) {
    top->setAttr("n_slice_row", bop->getAttr("k_per_thread"));
    top->setAttr("n_slice_col", bop->getAttr("m_per_thread"));
    // XXX: TBD review how vector load/store attributes are passed down.
    // top->setAttr("data_per_access", bop->getAttr("m_per_thread"));
    top->setAttr("data_per_access", b.getI32IntegerAttr(1));
  } else {
    top->setAttr("n_slice_row", bop->getAttr("k_per_thread"));
    top->setAttr("n_slice_col", bop->getAttr("n_per_thread"));
    // XXX: TBD review how vector load/store attributes are passed down.
    // top->setAttr("data_per_access", bop->getAttr("n_per_thread"));
    top->setAttr("data_per_access", b.getI32IntegerAttr(1));
  }
}

template <typename T, typename U>
void affixBlockwiseCopyAttributes(T &bop, U &gop, OpBuilder &b, int vectorDim,
                                  int blockwiseLoadLength,
                                  int blockwiseStoreLength) {
  bop->setAttr("block_size", gop->getAttr("block_size"));

  bop->setAttr("source_vector_read_dim", b.getI32IntegerAttr(vectorDim));
  bop->setAttr("dest_vector_write_dim", b.getI32IntegerAttr(vectorDim));
  bop->setAttr("source_data_per_read",
               b.getI32IntegerAttr(blockwiseLoadLength));
  bop->setAttr("dest_data_per_write",
               b.getI32IntegerAttr(blockwiseStoreLength));
}

void affixGridwiseGemmAttributes(Operation *convOp, Operation *gop,
                                 OpBuilder &b);
//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

/// Utility function for constructing a subview that slices a buffer as a
/// TransformOp
inline Value sliceBufferSubview(OpBuilder &b, Location loc, Value buffer,
                                int64_t start, int64_t length) {
  auto bufferType = buffer.getType().cast<MemRefType>();
  assert(bufferType.getRank() == 1 && "Can't slice multidimensional buffer");
  ArrayRef<int64_t> shape = bufferType.getShape();

  int64_t end = start + length;
  BottomUpCTBuilder transform(b, {"buffer"}, shape, loc);
  transform.slice({"slice"}, {"buffer"}, {start}, {end});

  TransformMapAttr transformAttr = transform.get();
  Value subview = b.create<miopen::TransformOp>(
      loc, buffer, transformAttr, bufferType.getMemorySpaceAsInt());
  return subview;
}

// Utility function for creating a N-D reshaped view of a subview
inline Value reshapeBufferSubview(OpBuilder &b, Location loc, Value buffer,
                                  ArrayRef<int64_t> shape) {
  MemRefType bufferType = buffer.getType().cast<MemRefType>();
  ArrayRef<int64_t> outShape = bufferType.getShape();
  assert(outShape.size() == 1 && "Buffer being reshaped must start linear");

  SmallVector<int64_t> strides;
  strides.reserve(shape.size());
  int64_t stride = 1;
  for (int64_t v : llvm::reverse(shape)) {
    strides.push_back(stride);
    stride *= v;
  }
  std::reverse(strides.begin(), strides.end());
  assert(stride == outShape[0] && "Strides must multiply to buffer length");

  SmallVector<SmallString<4>, 4> names;
  SmallVector<StringRef, 4> nameRefs;
  for (size_t i = 0, e = shape.size(); i < e; ++i) {
    SmallString<4> name;
    (Twine("dim") + Twine(i)).toVector(name);
    names.push_back(name);
    nameRefs.push_back(StringRef(names[i]));
  }

  TopDownCTBuilder transform(b, nameRefs, shape, loc);
  transform.embed("slice", 0, outShape[0], nameRefs, strides);

  TransformMapAttr transformAttr = transform.get();
  Value ret = b.create<miopen::TransformOp>(loc, buffer, transformAttr,
                                            bufferType.getMemorySpaceAsInt());
  return ret;
}

struct GridwiseGemmRewritePattern : public OpRewritePattern<miopen::GridwiseGemmOp> {
  using OpRewritePattern<miopen::GridwiseGemmOp>::OpRewritePattern;

  void computeLDSBlockSizes(miopen::GridwiseGemmOp op, int64_t &a_block_space,
                            int64_t &b_block_space,
                            int64_t &block_space, int64_t KPack = 1) const {
    int64_t ABlockCopyDstDataPerWrite_M =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t BBlockCopyDstDataPerWrite_N =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t ThreadGemmAThreadCopySrcDataPerRead_M =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t ThreadGemmBThreadCopySrcDataPerRead_N =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();

    int64_t max_lds_align =
        math_util::lcm(ABlockCopyDstDataPerWrite_M, BBlockCopyDstDataPerWrite_N,
                  ThreadGemmAThreadCopySrcDataPerRead_M,
                  ThreadGemmBThreadCopySrcDataPerRead_N);

    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

    int64_t AlignedNPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t a_block_space =
    //    math_util::integer_least_multiple(a_k_m_block_desc.GetElementSpace(),
    //    max_lds_align);
    int64_t AlignedMPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);
    a_block_space = math_util::integer_least_multiple(KPerBlock * AlignedMPerBlock,
                                                 max_lds_align) * KPack;

    // B matrix in LDS memory, dst of blockwise copy
    //   be careful of LDS alignment
    // Original C++ logic:
    // constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
    //    Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
    // constexpr index_t b_block_space =
    //    math_util::integer_least_multiple(b_k_n_block_desc.GetElementSpace(),
    //    max_lds_align);
    b_block_space = math_util::integer_least_multiple(KPerBlock * AlignedNPerBlock,
                                                 max_lds_align) * KPack;

    block_space = a_block_space + b_block_space;

    // llvm::errs() << "a_block_space: " << a_block_space << "\n";
    // llvm::errs() << "b_block_space: " << b_block_space << "\n";
    // llvm::errs() << "double_block_space: " << double_block_space << "\n\n";
  }

  void affixBlockwiseGemmAttributes(miopen::BlockwiseGemmOp bop,
                                    miopen::GridwiseGemmOp gop,
                                    OpBuilder &b) const {
    bop->setAttr("block_size", gop->getAttr("block_size"));
    // Attributes used in non-xdlops lowering path.
    bop->setAttr("m_per_thread", gop->getAttr("m_per_thread"));
    bop->setAttr("n_per_thread", gop->getAttr("n_per_thread"));
    bop->setAttr("k_per_thread", gop->getAttr("k_per_thread"));
    bop->setAttr("m_level0_cluster", gop->getAttr("m_level0_cluster"));
    bop->setAttr("m_level1_cluster", gop->getAttr("m_level1_cluster"));
    bop->setAttr("n_level0_cluster", gop->getAttr("n_level0_cluster"));
    bop->setAttr("n_level1_cluster", gop->getAttr("n_level1_cluster"));

    if (gop->hasAttr("kpack"))
      bop->setAttr("kpack", gop->getAttr("kpack"));
  }

  LogicalResult matchAndRewrite(miopen::GridwiseGemmOp op, PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Determine the type used in the filter/input/output tensors.
    auto elementType = op.c()
                           .getType()
                           .cast<MemRefType>()
                           .getElementType()
                           .template cast<Type>();

    // Determine the type used on VGPR to act as accumulator.
    // f32: f32.
    // f16: f32 to prevent overflow from happening.
    // i16(bf16) : i16.
    Type accumulatorType = elementType;
    if (elementType == b.getF16Type()) {
      accumulatorType = b.getF32Type();
    }

    // Prepare some useful constants.
    Value zeroConstantFloatOp =
        createZeroConstantFloatOp(b, loc, accumulatorType);
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.a().getType().template cast<MemRefType>().getShape();
    bShape = op.b().getType().template cast<MemRefType>().getShape();
    cShape = op.c().getType().template cast<MemRefType>().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    if (bShape[0] != G || cShape[0] != G) {
      return op.emitOpError("Mismatched G dimensions in matrix multiply;")
             << " A[0] = " << G << " b[0] = " << bShape[0]
             << " C[0] = " << cShape[0];
    }
    if (cShape[1] != M) {
      return op.emitOpError("Mismatched M dimensions in matrix multiply:")
             << " A[2] = " << M << " C[1] = " << cShape[1];
    }
    if (bShape[1] != K) {
      return op.emitOpError("Mismatched K dimensions in matrix multiply:")
             << " A[1] = " << K << " B[1] = " << bShape[1];
    }

    if (cShape[2] != N) {
      return op.emitOpError("Mismatched N dimensions in matrix multiply:")
             << " B[2] = " << N << " C[2] = " << cShape[2];
    }

    Attribute aTransforms = op.transforms()[0];
    Attribute bTransforms = op.transforms()[1];
    Attribute cTransforms = op.transforms()[2];
    Attribute noTransforms = b.getArrayAttr({});

    // Obtain critical tuning parameters.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t BlockSize =
        op->getAttr("block_size").template cast<IntegerAttr>().getInt();
    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerThread =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t NPerThread =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();
    auto MPerThreadConstantOp = b.create<ConstantIndexOp>(loc, MPerThread);
    auto NPerThreadConstantOp = b.create<ConstantIndexOp>(loc, NPerThread);

    int64_t MLevel0Cluster =
        op->getAttr("m_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster =
        op->getAttr("m_level1_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster =
        op->getAttr("n_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster =
        op->getAttr("n_level1_cluster").template cast<IntegerAttr>().getInt();
    auto NLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel0Cluster);
    auto NLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel1Cluster);

    int64_t matrix_a_source_data_per_read =
        op->getAttr("matrix_a_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op->getAttr("matrix_b_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    auto matrix_a_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_a_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    auto matrix_b_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_b_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());

    // Get current workgroup ID.
    auto bid = b.create<miopen::WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    // llvm::errs() << "\ngridwise_gemm op:\n";
    // op.dump();
    // llvm::errs() << "\n";

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: " << N << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "BlockSize: " << BlockSize << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";

    auto NBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, NBlockWork);
    auto GStridOp = b.create<ConstantIndexOp>(loc, GStride);
    auto block_work_id_g =
        b.create<DivUIOp>(loc, bid, GStridOp); // id_g of coordinate
    auto block_work_rem = b.create<RemUIOp>(loc, bid, GStridOp);
    auto block_work_id_m =
        b.create<DivUIOp>(loc, block_work_rem, NBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<RemUIOp>(loc, block_work_rem, NBlockWorkConstantOp);
    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);

    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "matrix_a_source_data_per_read: "
    //              << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: "
    //              << matrix_b_source_data_per_read << "\n";

    // Compute ThreadSliceLengths for Matrix A.
    int64_t GemmABlockCopyNumberDataPerThread =
        MPerBlock * KPerBlock / BlockSize;

    int64_t GemmABlockCopyThreadSliceLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM;
    int64_t GemmABlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_a_source_vector_read_dim) {
    case GemmK:
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmKPack =
            matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread / KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmK = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmMorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM / KPack;
        GemmABlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM;
      }
      break;
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix A is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    uint64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmKPack << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyNumberDataPerThread =
        NPerBlock * KPerBlock / BlockSize;

    int64_t GemmBBlockCopyThreadSliceLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmKPack = 1;
    assert(matrix_b_source_vector_read_dim != GemmG);
    switch (matrix_b_source_vector_read_dim) {
    case GemmK:
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmKPack =
            matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread / KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmK = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmMorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN / KPack;
        GemmBBlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN;
      }
      break;
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix B is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix B.
    uint64_t GemmBBlockCopyClusterLengths_GemmKPack =
        KPack / GemmBBlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmBBlockCopyClusterLengths_GemmK =
        KPerBlock / GemmBBlockCopyThreadSliceLengths_GemmK;
    uint64_t GemmBBlockCopyClusterLengths_GemmN =
        NPerBlock / GemmBBlockCopyThreadSliceLengths_GemmN;

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmKPack << "\n";

    // Get current workitem ID.

    auto tid = b.create<miopen::WorkitemIdOp>(loc, b.getIndexType());

    // Compute thread_data_id_begin for Matrix A.
    // ClusterArrangeOrder for Matrix A is <1, 0>.
    // So divide by GemmABlockCopyClusterLengths_GemmK.
    auto GemmABlockCopyClusterLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyClusterLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmMConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmM);

    Value GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmABlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmK *
                                        GemmABlockCopyClusterLengths_GemmKPack);
      GemmABlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmKPack);

      GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmABlockCopyThreadClusterId_Y;
    Value GemmABlockCopyThreadClusterId_X;
    Value GemmAThreadDataIdBegin_Y;
    Value GemmAThreadDataIdBegin_X;

    Value GemmABlockCopyThreadClusterId_Z;
    Value GemmAThreadDataIdBegin_Z;
    if (KPack > 1) {
      GemmABlockCopyThreadClusterId_Z = b.create<RemUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);

      GemmAThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Z,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
      GemmAThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmABlockCopyThreadClusterId_X,
          GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmABlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_X,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
    }

    Value GemmABlockCopySourceCoord_Y;
    Value GemmABlockCopySourceCoord_X;

    Value GemmABlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmABlockCopySourceCoord_Z = GemmAThreadDataIdBegin_Z;
      GemmABlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_Y);
      GemmABlockCopySourceCoord_X = GemmAThreadDataIdBegin_X;
    } else {
      GemmABlockCopySourceCoord_Y = GemmAThreadDataIdBegin_Y;
      GemmABlockCopySourceCoord_X = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_X);
    }

    Value GemmABlockCopyDestCoord_Y;
    Value GemmABlockCopyDestCoord_X;

    Value GemmABlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmABlockCopyDestCoord_Z = GemmAThreadDataIdBegin_Z;
    }
    GemmABlockCopyDestCoord_Y = GemmAThreadDataIdBegin_Y;
    GemmABlockCopyDestCoord_X = GemmAThreadDataIdBegin_X;

    // Compute thread_data_id_begin for Matrix B.
    // ClusterArrangeOrder for Matrix B is <0, 1>
    // So divide by GemmBBlockCopyClusterLengths_GemmN.
    auto GemmBBlockCopyClusterLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyClusterLengths_GemmN);
    auto GemmBBlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmK);
    auto GemmBBlockCopyThreadSliceLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmN);

    Value GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmBBlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmK *
                                        GemmBBlockCopyClusterLengths_GemmKPack);
      GemmBBlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmKPack);

      GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmBBlockCopyThreadClusterId_Y;
    Value GemmBBlockCopyThreadClusterId_X;
    Value GemmBThreadDataIdBegin_Y;
    Value GemmBThreadDataIdBegin_X;

    Value GemmBBlockCopyThreadClusterId_Z;
    Value GemmBThreadDataIdBegin_Z;

    if (KPack > 1) {
      GemmBBlockCopyThreadClusterId_Z = b.create<DivUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);
      GemmBBlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);

      GemmBThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Z,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmBBlockCopyThreadClusterId_X,
          GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmBBlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_X,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
    }

    Value GemmBBlockCopySourceCoord_Y;
    Value GemmBBlockCopySourceCoord_X;

    Value GemmBBlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopySourceCoord_Z = GemmBThreadDataIdBegin_Z;
      GemmBBlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_Y);
      GemmBBlockCopySourceCoord_X = GemmBThreadDataIdBegin_X;
    } else {
      GemmBBlockCopySourceCoord_Y = GemmBThreadDataIdBegin_Y;
      GemmBBlockCopySourceCoord_X = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_X);
    }

    Value GemmBBlockCopyDestCoord_Y;
    Value GemmBBlockCopyDestCoord_X;

    Value GemmBBlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopyDestCoord_Z = GemmBThreadDataIdBegin_Z;
    }
    GemmBBlockCopyDestCoord_Y = GemmBThreadDataIdBegin_Y;
    GemmBBlockCopyDestCoord_X = GemmBThreadDataIdBegin_X;

    auto GemmDataIdBegin_G = block_work_id_g;
    auto GemmBlockCoord_G = GemmDataIdBegin_G;

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize, KPack);

    //llvm::errs() << "LDS block size:" << ldsBlockASize << " " << ldsBlockBSize
    //             << " " << ldsBlockSize << "\n";

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockAOffset = 0;
    auto ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);

    // Get matrix subviews.
    // Compute matrix A dimension from attributes.
    Value ldsMatrixASubviewOp;
    if (KPack > 1) {
      ldsMatrixASubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockASubviewOp, {1, KPerBlock, MPerBlock, KPack});
    } else {
      ldsMatrixASubviewOp = reshapeBufferSubview(b, loc, ldsBlockASubviewOp,
                                                 {1, KPerBlock, MPerBlock});
    }

    // Subviews for Matrix B.
    auto ldsBlockBOffset = ldsBlockASize;
    auto ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp;
    if (KPack > 1) {
      ldsMatrixBSubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockBSubviewOp, {1, KPerBlock, NPerBlock, KPack});
    } else {
      ldsMatrixBSubviewOp = reshapeBufferSubview(b, loc, ldsBlockBSubviewOp,
                                                 {1, KPerBlock, NPerBlock});
    }

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    int64_t GemmMRepeat = 0, GemmNRepeat = 0;

    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";

    GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

    // llvm::errs() << "GemmMRepeat: " << GemmMRepeat << "\n";
    // llvm::errs() << "GemmNRepeat: " << GemmNRepeat << "\n";

    auto threadCRegisterMemRefType = MemRefType::get(
        {1, GemmMRepeat * MPerThread, GemmNRepeat * NPerThread},
        accumulatorType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    Value registerMatrixCAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadCRegisterMemRefType);

    // Determine vector / scalar load type for Matrix A / B.
    ArrayAttr blockwiseCopyABounds;
    if (KPack > 1) {
      blockwiseCopyABounds =
          b.getIndexArrayAttr({1, GemmABlockCopyThreadSliceLengths_GemmK,
                               GemmABlockCopyThreadSliceLengths_GemmM,
                               GemmABlockCopyThreadSliceLengths_GemmKPack});
    } else {
      blockwiseCopyABounds =
          b.getIndexArrayAttr({1, GemmABlockCopyThreadSliceLengths_GemmK,
                               GemmABlockCopyThreadSliceLengths_GemmM});
    }
    Type blockwiseLoadAType;
    SmallVector<Type, 8> blockwiseLoadATypes;
    int blockwiseAVectorDim;
    int blockwiseLoadAVectorLength;
    int blockwiseStoreAVectorLength;

    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmK: "
    //              << GemmABlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmM: "
    //              << GemmABlockCopyThreadSliceLengths_GemmM << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy A bounds: ";
    // for (auto v : blockwiseCopyABounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    std::tie(blockwiseLoadAType, blockwiseAVectorDim,
             blockwiseLoadAVectorLength, blockwiseStoreAVectorLength) =
        computeLoadStoreTypeInfo(b, op, elementType, blockwiseLoadATypes,
                                 blockwiseCopyABounds, true, KPack);

    // llvm::errs() << "vector load dim: " << blockwiseAVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadAType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadAVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreAVectorLength << "\n";

    ArrayAttr blockwiseCopyBBounds;
    if (KPack > 1) {
      blockwiseCopyBBounds =
          b.getIndexArrayAttr({1, GemmBBlockCopyThreadSliceLengths_GemmK,
                               GemmBBlockCopyThreadSliceLengths_GemmN,
                               GemmBBlockCopyThreadSliceLengths_GemmKPack});
    } else {
      blockwiseCopyBBounds =
          b.getIndexArrayAttr({1, GemmBBlockCopyThreadSliceLengths_GemmK,
                               GemmBBlockCopyThreadSliceLengths_GemmN});
    }
    Type blockwiseLoadBType;
    SmallVector<Type, 8> blockwiseLoadBTypes;
    int blockwiseBVectorDim;
    int blockwiseLoadBVectorLength;
    int blockwiseStoreBVectorLength;

    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmK: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmN: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy B bounds: ";
    // for (auto v : blockwiseCopyBBounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    std::tie(blockwiseLoadBType, blockwiseBVectorDim,
             blockwiseLoadBVectorLength, blockwiseStoreBVectorLength) =
        computeLoadStoreTypeInfo(b, op, elementType, blockwiseLoadBTypes,
                                 blockwiseCopyBBounds, false, KPack);

    // llvm::errs() << "vector load dim: " << blockwiseBVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadBType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadBVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreBVectorLength << "\n";

    // Zero init Matrix C on registers.
    b.create<miopen::FillOp>(loc, registerMatrixCAllocOp, zeroConstantFloatOp);

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // Compute source and destination coordinates for BlockwiseCopy ops.
    // Matrix A: {0, 0, m_block_data_on_global}, {0, 0, 0}
    // Matrix B: {0, 0, n_block_data_on_global}, {0, 0, 0}

    Value mMyThreadOffsetA, mMyThreadOffsetB;
    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value m_thread_data_on_global, n_thread_data_on_global;

    // Compute c_thread_mtx_index for Matrix C.
    int64_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;
    auto ThreadPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, ThreadPerLevel0Cluster);
    auto level1_id =
        b.create<DivUIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
    auto level1_m_id =
        b.create<DivUIOp>(loc, level1_id, NLevel1ClusterConstantOp);
    auto level1_n_id =
        b.create<RemUIOp>(loc, level1_id, NLevel1ClusterConstantOp);

    auto level0_id =
        b.create<RemUIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
    auto level0_m_id =
        b.create<DivUIOp>(loc, level0_id, NLevel0ClusterConstantOp);
    auto level0_n_id =
        b.create<RemUIOp>(loc, level0_id, NLevel0ClusterConstantOp);

    int64_t MPerLevel0Cluster = MPerThread * MLevel0Cluster;
    int64_t NPerLevel0Cluster = NPerThread * NLevel0Cluster;
    auto MPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, MPerLevel0Cluster);
    auto NPerLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NPerLevel0Cluster);

    // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.row} = c_thread_mtx_index_row
    c_thread_mtx_index_row = b.create<AddIOp>(
        loc, b.create<MulIOp>(loc, level1_m_id, MPerLevel0ClusterConstantOp),
        b.create<MulIOp>(loc, level0_m_id, MPerThreadConstantOp));
    mMyThreadOffsetA = c_thread_mtx_index_row;

    // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.col} = c_thread_mtx_index_col
    c_thread_mtx_index_col = b.create<AddIOp>(
        loc, b.create<MulIOp>(loc, level1_n_id, NPerLevel0ClusterConstantOp),
        b.create<MulIOp>(loc, level0_n_id, NPerThreadConstantOp));
    mMyThreadOffsetB = c_thread_mtx_index_col;

    m_thread_data_on_global =
        b.create<AddIOp>(loc, m_block_data_on_global, c_thread_mtx_index_row);
    n_thread_data_on_global =
        b.create<AddIOp>(loc, n_block_data_on_global, c_thread_mtx_index_col);

    SmallVector<Value, 4> blockwiseLoadACoords;
    if (KPack > 1) {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Z,
                              GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    } else {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    }
    // Emit blockwise_load for matrix A.
    auto blockwiseLoadA = b.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadATypes, op.a(), blockwiseCopyABounds,
        b.getArrayAttr({aTransforms}), op.paddingInfo(), op.aOobDims(),
        blockwiseLoadACoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadA, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    SmallVector<Value, 4> blockwiseStoreACoords;
    if (KPack > 1) {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Z,
                               GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    } else {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    }
    // Emit blockwise_store for matrix A.
    auto blockwiseStoreA = b.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixASubviewOp, blockwiseCopyABounds, noTransformsArray(b, 1),
        blockwiseLoadA.getResults(), blockwiseStoreACoords);
    affixBlockwiseCopyAttributes(
        blockwiseStoreA, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    SmallVector<Value, 4> blockwiseLoadBCoords;
    if (KPack > 1) {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Z,
                              GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    } else {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    }
    // Emit blockwise_load for matrix B.
    auto blockwiseLoadB = b.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadBTypes, op.b(), blockwiseCopyBBounds,
        b.getArrayAttr({bTransforms}), op.paddingInfo(), op.bOobDims(),
        blockwiseLoadBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadB, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    SmallVector<Value, 4> blockwiseStoreBCoords;
    if (KPack > 1) {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Z,
                               GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    } else {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    }
    // Emit blockwise_store for matrix B.
    auto blockwiseStoreB = b.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixBSubviewOp, blockwiseCopyBBounds, noTransformsArray(b, 1),
        blockwiseLoadB.getResults(), blockwiseStoreBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseStoreB, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // Emit loop.
    // Compute loop iterations from attributes.

    auto KPerBlockConstantOp = b.create<ConstantIndexOp>(loc, KPerBlock);

    int64_t loopIteration = (K - KPerBlock) / KPerBlock;

    // Assign iter args.
    // 0: blockwise copy A src y coordinate.
    // 1: blockwise copy B src y coordinate.
    SmallVector<Value, 2> iterArgs = {blockwiseLoadA.sourceCoord()[1],
                                      blockwiseLoadB.sourceCoord()[1]};

    auto loopOp = b.create<AffineForOp>(loc, 0, loopIteration, 1, iterArgs);

    // inside the loop.
    auto lb = OpBuilder::atBlockBegin(loopOp.getBody());

    // LDS barrier.
    lb.create<miopen::LDSBarrierOp>(loc);

    // Emit blockwise GEMM.
    auto blockwiseGemmOp = lb.create<miopen::BlockwiseGemmOp>(
        loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCAllocOp,
        noTransformsArray(b, 2), mMyThreadOffsetA, mMyThreadOffsetB);
    affixBlockwiseGemmAttributes(blockwiseGemmOp, op, b);

    // LDS barrier.
    // This barrier prevents halo part of outputs having weird values.
    lb.create<miopen::LDSBarrierOp>(loc);

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    const auto &args = loopOp.getRegionIterArgs();
    Value blockwiseCopyASrcUpdated =
        lb.create<AddIOp>(loc, args[0], KPerBlockConstantOp);
    // Emit blockwise_load for matrix A.
    blockwiseLoadACoords[1] = blockwiseCopyASrcUpdated;
    auto blockwiseLoadATop = lb.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadATypes, op.a(), blockwiseLoadA.bounds(),
        blockwiseLoadA.transforms(), blockwiseLoadA.paddingInfo(),
        blockwiseLoadA.oobDims(), blockwiseLoadACoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadATop, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    Value blockwiseCopyBSrcUpdated =
        lb.create<AddIOp>(loc, args[1], KPerBlockConstantOp);
    blockwiseLoadBCoords[1] = blockwiseCopyBSrcUpdated;
    // Emit blockwise_load for matrix B.
    auto blockwiseLoadBTop = lb.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadBTypes, op.b(), blockwiseLoadB.bounds(),
        blockwiseLoadB.transforms(), blockwiseLoadB.paddingInfo(),
        blockwiseLoadB.oobDims(), blockwiseLoadBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadBTop, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).

    // Emit blockwise_store for matrix A.
    auto blockwiseStoreABottom = lb.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixASubviewOp, blockwiseStoreA.bounds(),
        blockwiseStoreA.transforms(), blockwiseLoadATop.getResults(),
        blockwiseStoreA.destCoord());
    affixBlockwiseCopyAttributes(
        blockwiseStoreABottom, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    // Emit blockwise_store for matrix B.
    auto blockwiseStoreBBottom = lb.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixBSubviewOp, blockwiseStoreB.bounds(),
        blockwiseStoreB.transforms(), blockwiseLoadBTop.getResults(),
        blockwiseStoreB.destCoord());
    affixBlockwiseCopyAttributes(
        blockwiseStoreBBottom, op, b,
        /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // update iter args.
    // blockwiseCopyASrcVector and blockwiseCopyBSrcVector are updated.
    iterArgs[0] = blockwiseCopyASrcUpdated;
    iterArgs[1] = blockwiseCopyBSrcUpdated;
    // emit loop yield so iter args can be passed to the next iteration.
    lb.create<AffineYieldOp>(loc, iterArgs);

    // outside the loop.

    // LDS barrier.
    b.create<miopen::LDSBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmTailOp = b.create<miopen::BlockwiseGemmOp>(
        loc, ldsMatrixASubviewOp, ldsMatrixBSubviewOp, registerMatrixCAllocOp,
        blockwiseGemmOp.transforms(), mMyThreadOffsetA, mMyThreadOffsetB);
    affixBlockwiseGemmAttributes(blockwiseGemmTailOp, op, b);

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    int64_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
    int64_t M0 = M / M1;
    int64_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;
    int64_t N0 = N / N1;

    auto M1ConstantOp = b.create<ConstantIndexOp>(loc, M1);
    auto N1ConstantOp = b.create<ConstantIndexOp>(loc, N1);

    // Build transformation that unsplits the output matrix for writing
    // by (g, m0, m1, n0, n1) -> (g, m0 * M1 + m1, n0 * N1, n1)
    TopDownCTBuilder cSplitTransform(b, {"G", "M0", "M1", "N0", "N1"},
                                     {G, M0, M1, N0, N1}, loc);
    cSplitTransform.passThrough({"gemmG"}, {0}, {"G"});
    cSplitTransform.embed("gemmM", 1, M1 * M0, {"M0", "M1"}, {M1, 1});
    cSplitTransform.embed("gemmN", 2, N1 * N0, {"N0", "N1"}, {N1, 1});

    TransformMapAttr cSplitTransformAttr = cSplitTransform.get();
    auto cTransformed =
        b.create<miopen::TransformOp>(loc, op.c(), cSplitTransformAttr);

    // Build transformation that maps the in-regester results to
    // three dimensions for writing with
    //  (g, m0, m1, n0, n1) -> (g, m0 * MPerThread + m1, n0 * NPerThread + n1)
    TopDownCTBuilder registerCTransform(
        b, {"g", "gemmMRepeat", "mPerThread", "gemmNRepeat", "nPerThread"},
        {1, GemmMRepeat, MPerThread, GemmNRepeat, NPerThread}, loc);
    registerCTransform.passThrough({"gemmG"}, {0}, {"g"});
    registerCTransform.embed("gemmM", 1, GemmMRepeat * MPerThread,
                             {"gemmMRepeat", "mPerThread"}, {MPerThread, 1});
    registerCTransform.embed("gemmN", 2, GemmNRepeat * NPerThread,
                             {"gemmNRepeat", "nPerThread"}, {NPerThread, 1});

    TransformMapAttr registerCTransformAttr = registerCTransform.get();
    Value registerCTransformed = b.create<miopen::TransformOp>(
        loc, registerMatrixCAllocOp, registerCTransformAttr,
        gpu::GPUDialect::getPrivateAddressSpace());

    SmallVector<Value, 5> matrixCThreadwiseCopySourceCoords;
    std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 5,
                zeroConstantOp.getResult());

    SmallVector<Value, 5> matrixCThreadwiseCopyDestCoords = {
        GemmDataIdBegin_G,
        b.create<DivUIOp>(loc, m_thread_data_on_global, M1ConstantOp),
        b.create<RemUIOp>(loc, m_thread_data_on_global, M1ConstantOp),
        b.create<DivUIOp>(loc, n_thread_data_on_global, N1ConstantOp),
        b.create<RemUIOp>(loc, n_thread_data_on_global, N1ConstantOp)};
    // g index

    auto threadwiseCopyCMatrixOp = b.create<miopen::ThreadwiseCopyOp>(
        loc, registerCTransformed, cTransformed,
        b.getArrayAttr({noTransforms, cTransforms}), op.paddingInfo(),
        op.cOobDims(), b.getIndexAttr(1), matrixCThreadwiseCopySourceCoords,
        matrixCThreadwiseCopyDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyCMatrixOp, op, b);

    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct GridwiseGemmV2RewritePattern
    : public OpRewritePattern<miopen::GridwiseGemmV2Op> {
  using OpRewritePattern<miopen::GridwiseGemmV2Op>::OpRewritePattern;

  void computeLDSBlockSizes(miopen::GridwiseGemmV2Op op, int64_t &a_block_space,
                            int64_t &b_block_space,
                            int64_t &total_block_space, int64_t KPack = 1) const {
    int64_t ABlockCopyDstDataPerWrite_M =
        op->getAttr("matrix_a_dest_data_per_write_dim_m")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t BBlockCopyDstDataPerWrite_N =
        op->getAttr("matrix_b_dest_data_per_write_dim_n")
            .template cast<IntegerAttr>()
            .getInt();

    int64_t max_lds_align =
        math_util::lcm(ABlockCopyDstDataPerWrite_M, BBlockCopyDstDataPerWrite_N);

    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

    int64_t AlignedNPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

    // A matrix in LDS memory, dst of blockwise copy
    int64_t AlignedMPerBlock =
        max_lds_align *
        math_util::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);

    // llvm::errs() << "MPerBlock : " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock : " << NPerBlock << "\n";
    // llvm::errs() << "max_lds_align : " << max_lds_align << "\n";
    // llvm::errs() << "AlignedMPerBlock : " << AlignedMPerBlock << "\n";
    // llvm::errs() << "AlignedNPerBlock : " << AlignedNPerBlock << "\n";

    a_block_space = math_util::integer_least_multiple(KPerBlock * AlignedMPerBlock,
                                                 max_lds_align) * KPack;

    // B matrix in LDS memory, dst of blockwise copy
    b_block_space = math_util::integer_least_multiple(KPerBlock * AlignedNPerBlock,
                                                 max_lds_align) * KPack;

    total_block_space = a_block_space + b_block_space;

    // llvm::errs() << "a_block_space: " << a_block_space << "\n";
    // llvm::errs() << "b_block_space: " << b_block_space << "\n";
    // llvm::errs() << "total_block_space: " << total_block_space << "\n\n";
  }

  void affixBlockwiseGemmV2Attributes(miopen::BlockwiseGemmV2Op bop,
                                      miopen::GridwiseGemmV2Op gop, int64_t m,
                                      int64_t k, int64_t n,
                                      OpBuilder &b) const {
    bop->setAttr("block_size", gop->getAttr("block_size"));

    int64_t MPerBlock =
        gop->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        gop->getAttr("n_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        gop->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        gop->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();
    int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;

    bop->setAttr("m_per_wave", gop->getAttr("m_per_wave"));
    bop->setAttr("n_per_wave", gop->getAttr("n_per_wave"));
    bop->setAttr("m_waves", b.getI32IntegerAttr(MWaves));
    bop->setAttr("n_waves", b.getI32IntegerAttr(NWaves));

    bop->setAttr("m", b.getI32IntegerAttr(m));
    bop->setAttr("n", b.getI32IntegerAttr(n));
    bop->setAttr("k", b.getI32IntegerAttr(k));

    if (gop->hasAttr("kpack"))
      bop->setAttr("kpack", gop->getAttr("kpack"));
  }

  LogicalResult matchAndRewrite(miopen::GridwiseGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain data type.
    auto elementType = op.c().getType().cast<MemRefType>().getElementType();

    // Prepare some useful constants.
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    // Obtain critical matrix dimensions.
    ArrayRef<int64_t> aShape, bShape, cShape;
    aShape = op.a().getType().template cast<MemRefType>().getShape();
    bShape = op.b().getType().template cast<MemRefType>().getShape();
    cShape = op.c().getType().template cast<MemRefType>().getShape();
    // Obtain critical matrix dimensions.
    int64_t G = aShape[0];
    int64_t K = aShape[1];
    int64_t M = aShape[2];
    int64_t N = bShape[2];

    if (bShape[0] != G || cShape[0] != G) {
      return op.emitOpError("Mismatched G dimensions in matrix multiply;")
             << " A[0] = " << G << " b[0] = " << bShape[0]
             << " C[0] = " << cShape[0];
    }
    if (cShape[1] != M) {
      return op.emitOpError("Mismatched M dimensions in matrix multiply:")
             << " A[2] = " << M << " C[1] = " << cShape[1];
    }
    if (bShape[1] != K) {
      return op.emitOpError("Mismatched K dimensions in matrix multiply:")
             << " A[1] = " << K << " B[1] = " << bShape[1];
    }
    if (cShape[2] != N) {
      return op.emitOpError("Mismatched N dimensions in matrix multiply:")
             << " B[2] = " << N << " C[2] = " << cShape[2];
    }

    Attribute aTransforms = op.transforms()[0];
    Attribute bTransforms = op.transforms()[1];
    Attribute cTransforms = op.transforms()[2];

    // Obtain critical tuning parameters.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t BlockSize =
        op->getAttr("block_size").template cast<IntegerAttr>().getInt();
    int64_t KPerBlock =
        op->getAttr("k_per_block").template cast<IntegerAttr>().getInt();
    int64_t MPerBlock =
        op->getAttr("m_per_block").template cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        op->getAttr("n_per_block").template cast<IntegerAttr>().getInt();

    int64_t matrix_a_source_data_per_read =
        op->getAttr("matrix_a_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op->getAttr("matrix_b_source_data_per_read")
            .template cast<IntegerAttr>()
            .getInt();
    auto matrix_a_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_a_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());
    auto matrix_b_source_vector_read_dim = static_cast<GemmDimensions>(
        op->getAttr("matrix_b_source_vector_read_dim")
            .template cast<IntegerAttr>()
            .getInt());

    // Obtain XDLOPS-related attributes.
    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();
    // int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;
    auto dataType =
        op.b().getType().template cast<MemRefType>().getElementType();

    auto MPerWaveConstantOp = b.create<ConstantIndexOp>(loc, MPerWave);
    auto NPerWaveConstantOp = b.create<ConstantIndexOp>(loc, NPerWave);
    auto NWavesConstantOp = b.create<ConstantIndexOp>(loc, NWaves);

    int64_t WaveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, WaveSize);

    // Get current workgroup ID.
    auto bid = b.create<miopen::WorkgroupIdOp>(loc, b.getIndexType());

    // Get current workitem ID.
    auto tid = b.create<miopen::WorkitemIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    int64_t GStride = MBlockWork * NBlockWork;

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: " << N << "\n";
    // llvm::errs() << "K: " << K << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";
    // llvm::errs() << "MWaves = MPerBlock / MPerWave: " << MWaves << "\n";
    // llvm::errs() << "NWaves = NPerBlock / NPerWave: " << NWaves << "\n";

    // llvm::errs() << "matrix_a_source_data_per_read: "
    //              << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: "
    //              << matrix_b_source_data_per_read << "\n";
    // llvm::errs() << "matrix_a_source_vector_read_dim: "
    //              << matrix_a_source_vector_read_dim << "\n";
    // llvm::errs() << "matrix_b_source_vector_read_dim: "
    //              << matrix_b_source_vector_read_dim << "\n";

    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto KPerBlockConstantOp = b.create<ConstantIndexOp>(loc, KPerBlock);
    auto MBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, MBlockWork);
    auto GStridOp = b.create<ConstantIndexOp>(loc, GStride);
    // -----

    // Compute the coordinate for the current workgroup on global memory.

    // Original C++ logic:
    // constexpr auto wkgrp_schd_order = NBlock1MBlock0;
    // constexpr auto block_work_sequence =
    //     make_batch_block_work_sequence<G, MBlockWork, NBlockWork,
    //     WorkgroupSchdOrder>{}.get();
    // constexpr auto block_work_desc =
    // make_cluster_descriptor(block_work_sequence); const auto block_work_id =
    // block_work_desc.CalculateClusterIndex(get_block_1d_id());

    // Result block_work_desc is <NBlockWorkd, MBlockWork>

    auto block_work_id_g = b.create<DivUIOp>(loc, bid, GStridOp);
    auto block_work_rem = b.create<RemUIOp>(loc, bid, GStridOp);
    auto block_work_id_m =
        b.create<RemUIOp>(loc, block_work_rem, MBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<DivUIOp>(loc, block_work_rem, MBlockWorkConstantOp);

    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);

    // -----

    // Logic to prepare parameters for blockwise_copy.

    // Compute ThreadSliceLengths for Matrix A.
    int64_t GemmABlockCopyNumberDataPerThread =
        MPerBlock * KPerBlock * KPack / BlockSize;

    // llvm::errs() << "GemmABlockCopyNumberDataPerThread: "
    //              << GemmABlockCopyNumberDataPerThread << "\n";

    int64_t GemmABlockCopyThreadSliceLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM;
    int64_t GemmABlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_a_source_vector_read_dim) {
    case GemmK:
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmKPack =
            matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread / KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmK = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmM =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmMorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM / KPack;
        GemmABlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmABlockCopyThreadSliceLengths_GemmM = matrix_a_source_data_per_read;
        GemmABlockCopyThreadSliceLengths_GemmK =
            GemmABlockCopyNumberDataPerThread /
            GemmABlockCopyThreadSliceLengths_GemmM;
      }
      break;
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmABlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmM == 0 ||
        GemmABlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix A is zero which "
                      "is invalid.\n";
      return failure();
    }

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmKPack =
        KPack / GemmABlockCopyThreadSliceLengths_GemmKPack;
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / GemmABlockCopyThreadSliceLengths_GemmK;
    // int64_t GemmABlockCopyClusterLengths_GemmM =
    //    MPerBlock / GemmABlockCopyThreadSliceLengths_GemmM;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmKPack << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyNumberDataPerThread =
        NPerBlock * KPerBlock * KPack / BlockSize;

    // llvm::errs() << "GemmBBlockCopyNumberDataPerThread: "
    //              << GemmBBlockCopyNumberDataPerThread << "\n";

    int64_t GemmBBlockCopyThreadSliceLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmKPack = 1;
    switch (matrix_b_source_vector_read_dim) {
    case GemmK:
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmKPack =
            matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            KPack / matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread / KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmK = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmN =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmK;
      }
      break;
    case GemmMorN:
      // TBD: FIXME. Review logic here.
      if (KPack > 1) {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN / KPack;
        GemmBBlockCopyThreadSliceLengths_GemmKPack = KPack;
      } else {
        GemmBBlockCopyThreadSliceLengths_GemmN = matrix_b_source_data_per_read;
        GemmBBlockCopyThreadSliceLengths_GemmK =
            GemmBBlockCopyNumberDataPerThread /
            GemmBBlockCopyThreadSliceLengths_GemmN;
      }
      break;
    case GemmG:
      llvm::errs() << "Vector loads/stores aren't possible in the G dimension "
                      "and should not haven been attempted";
      return failure();
    }

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";

    if (GemmBBlockCopyThreadSliceLengths_GemmK == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmN == 0 ||
        GemmBBlockCopyThreadSliceLengths_GemmKPack == 0) {
      llvm::errs() << "Blockwise copy slice lengths for matrix B is zero which "
                      "is invalid.\n";
      return failure();
    }

    assert(GemmBBlockCopyThreadSliceLengths_GemmK > 0);
    assert(GemmBBlockCopyThreadSliceLengths_GemmN > 0);
    assert(GemmBBlockCopyThreadSliceLengths_GemmKPack > 0);
    // Compute ThreadClusterLengths for Matrix B.
    uint64_t GemmBBlockCopyClusterLengths_GemmKPack =
        KPack / GemmBBlockCopyThreadSliceLengths_GemmKPack;
    uint64_t GemmBBlockCopyClusterLengths_GemmK =
        KPerBlock / GemmBBlockCopyThreadSliceLengths_GemmK;
    uint64_t GemmBBlockCopyClusterLengths_GemmN =
        NPerBlock / GemmBBlockCopyThreadSliceLengths_GemmN;

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmKPack << "\n";

    // Compute thread_data_id_begin for Matrix A.
    // ClusterArrangeOrder for Matrix A is <1, 0>.
    // So divide by GemmABlockCopyClusterLengths_GemmK.
    auto GemmABlockCopyClusterLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyClusterLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmMConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmM);

    Value GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmABlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmK *
                                        GemmABlockCopyClusterLengths_GemmKPack);
      GemmABlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyClusterLengths_GemmKPack);
      GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmABlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmABlockCopyThreadClusterId_Y;
    Value GemmABlockCopyThreadClusterId_X;
    Value GemmAThreadDataIdBegin_Y;
    Value GemmAThreadDataIdBegin_X;

    Value GemmABlockCopyThreadClusterId_Z;
    Value GemmAThreadDataIdBegin_Z;
    if (KPack > 1) {
      GemmABlockCopyThreadClusterId_Z = b.create<RemUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc,
          b.create<RemUIOp>(
              loc, tid,
              GemmABlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp),
          GemmABlockCopyClusterLengths_GemmKConstantOp);

      GemmAThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Z,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
      GemmAThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmABlockCopyThreadClusterId_X,
          GemmABlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmABlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmABlockCopyThreadClusterId_X = b.create<DivUIOp>(
          loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                           GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmAThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_X,
                           GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
    }

    Value GemmABlockCopySourceCoord_Y;
    Value GemmABlockCopySourceCoord_X;

    Value GemmABlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmABlockCopySourceCoord_Z = GemmAThreadDataIdBegin_Z;
      GemmABlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_Y);
      GemmABlockCopySourceCoord_X = GemmAThreadDataIdBegin_X;
    } else {
      GemmABlockCopySourceCoord_Y = GemmAThreadDataIdBegin_Y;
      GemmABlockCopySourceCoord_X = b.create<AddIOp>(
          loc, m_block_data_on_global, GemmAThreadDataIdBegin_X);
    }

    Value GemmABlockCopyDestCoord_Y;
    Value GemmABlockCopyDestCoord_X;

    Value GemmABlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmABlockCopyDestCoord_Z = GemmAThreadDataIdBegin_Z;
    }
    GemmABlockCopyDestCoord_Y = GemmAThreadDataIdBegin_Y;
    GemmABlockCopyDestCoord_X = GemmAThreadDataIdBegin_X;

    // Compute thread_data_id_begin for Matrix B.
    // ClusterArrangeOrder for Matrix B is <0, 1>
    // So divide by GemmBBlockCopyClusterLengths_GemmN.
    auto GemmBBlockCopyClusterLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyClusterLengths_GemmN);
    auto GemmBBlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmK);
    auto GemmBBlockCopyThreadSliceLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmN);

    Value GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp;
    Value GemmBBlockCopyClusterLengths_GemmKPackConstantOp;
    Value GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp;
    if (KPack > 1) {
      GemmBBlockCopyClusterLengths_GemmKTimesGemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmK *
                                        GemmBBlockCopyClusterLengths_GemmKPack);
      GemmBBlockCopyClusterLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyClusterLengths_GemmKPack);
      GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp =
          b.create<ConstantIndexOp>(loc,
                                    GemmBBlockCopyThreadSliceLengths_GemmKPack);
    }

    Value GemmBBlockCopyThreadClusterId_Y;
    Value GemmBBlockCopyThreadClusterId_X;
    Value GemmBThreadDataIdBegin_Y;
    Value GemmBThreadDataIdBegin_X;

    Value GemmBBlockCopyThreadClusterId_Z;
    Value GemmBThreadDataIdBegin_Z;

    if (KPack > 1) {
      GemmBBlockCopyThreadClusterId_Z = b.create<DivUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);
      GemmBBlockCopyThreadClusterId_Y = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc,
          b.create<DivUIOp>(loc, tid,
                            GemmBBlockCopyClusterLengths_GemmNConstantOp),
          GemmBBlockCopyClusterLengths_GemmKPackConstantOp);

      GemmBThreadDataIdBegin_Z =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Z,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_X = b.create<MulIOp>(
          loc, GemmBBlockCopyThreadClusterId_X,
          GemmBBlockCopyThreadSliceLengths_GemmKPackConstantOp);
    } else {
      GemmBBlockCopyThreadClusterId_Y = b.create<DivUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBBlockCopyThreadClusterId_X = b.create<RemUIOp>(
          loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
      GemmBThreadDataIdBegin_Y =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                           GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
      GemmBThreadDataIdBegin_X =
          b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_X,
                           GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
    }

    Value GemmBBlockCopySourceCoord_Y;
    Value GemmBBlockCopySourceCoord_X;

    Value GemmBBlockCopySourceCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopySourceCoord_Z = GemmBThreadDataIdBegin_Z;
      GemmBBlockCopySourceCoord_Y = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_Y);
      GemmBBlockCopySourceCoord_X = GemmBThreadDataIdBegin_X;
    } else {
      GemmBBlockCopySourceCoord_Y = GemmBThreadDataIdBegin_Y;
      GemmBBlockCopySourceCoord_X = b.create<AddIOp>(
          loc, n_block_data_on_global, GemmBThreadDataIdBegin_X);
    }

    Value GemmBBlockCopyDestCoord_Y;
    Value GemmBBlockCopyDestCoord_X;

    Value GemmBBlockCopyDestCoord_Z;
    if (KPack > 1) {
      GemmBBlockCopyDestCoord_Z = GemmBThreadDataIdBegin_Z;
    }
    GemmBBlockCopyDestCoord_Y = GemmBThreadDataIdBegin_Y;
    GemmBBlockCopyDestCoord_X = GemmBThreadDataIdBegin_X;

    auto GemmBlockCoord_G = block_work_id_g;
    // -----

    // Alocate LDS and create subviews.

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize, KPack);

    // llvm::errs() << "KPack: " << KPack << "\n";
    // llvm::errs() << "LDS block size:" << ldsBlockASize << " " << ldsBlockBSize
    //              << " " << ldsBlockSize << "\n";

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockAOffset = 0;

    auto ldsBlockASubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockAOffset, ldsBlockASize);

    // Get matrix subviews.
    // Compute matrix A dimension from attributes.
    Value ldsMatrixASubviewOp;
    if (KPack > 1) {
      ldsMatrixASubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockASubviewOp, {1, KPerBlock, MPerBlock, KPack});
    } else {
      ldsMatrixASubviewOp = reshapeBufferSubview(b, loc, ldsBlockASubviewOp,
                                                 {1, KPerBlock, MPerBlock});
    }

    // Subviews for Matrix B.
    auto ldsBlockBOffset = ldsBlockASize;
    auto ldsBlockBSubviewOp = sliceBufferSubview(
        b, loc, ldsGpuAllocOp, ldsBlockBOffset, ldsBlockBSize);

    // Get matrix subviews.
    // Compute matrix B dimension from attributes.
    Value ldsMatrixBSubviewOp;
    if (KPack > 1) {
      ldsMatrixBSubviewOp = reshapeBufferSubview(
          b, loc, ldsBlockBSubviewOp, {1, KPerBlock, NPerBlock, KPack});
    } else {
      ldsMatrixBSubviewOp = reshapeBufferSubview(b, loc, ldsBlockBSubviewOp,
                                                 {1, KPerBlock, NPerBlock});
    }

    ArrayAttr noTransforms1 = noTransformsArray(b, 1);
    // -----

    // Determine vector / scalar load type for Matrix A / B.
    ArrayAttr blockwiseCopyABounds;
    if (KPack > 1) {
      blockwiseCopyABounds =
          b.getIndexArrayAttr({1, GemmABlockCopyThreadSliceLengths_GemmK,
                               GemmABlockCopyThreadSliceLengths_GemmM,
                               GemmABlockCopyThreadSliceLengths_GemmKPack});
    } else {
      blockwiseCopyABounds =
          b.getIndexArrayAttr({1, GemmABlockCopyThreadSliceLengths_GemmK,
                               GemmABlockCopyThreadSliceLengths_GemmM});
    }
    Type blockwiseLoadAType;
    SmallVector<Type, 8> blockwiseLoadATypes;
    int blockwiseAVectorDim;
    int blockwiseLoadAVectorLength;
    int blockwiseStoreAVectorLength;

    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmK: "
    //              << GemmABlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmM: "
    //              << GemmABlockCopyThreadSliceLengths_GemmM << "\n";
    // llvm::errs() << "GemmABlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmABlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy A bounds: ";
    // for (auto v : blockwiseCopyABounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    std::tie(blockwiseLoadAType, blockwiseAVectorDim,
             blockwiseLoadAVectorLength, blockwiseStoreAVectorLength) =
        computeLoadStoreTypeInfo(b, op, elementType, blockwiseLoadATypes,
			         blockwiseCopyABounds, true, KPack);

    // llvm::errs() << "vector load dim: " << blockwiseAVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadAType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadAVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreAVectorLength << "\n";

    ArrayAttr blockwiseCopyBBounds;
    if (KPack > 1) {
      blockwiseCopyBBounds =
          b.getIndexArrayAttr({1, GemmBBlockCopyThreadSliceLengths_GemmK,
                               GemmBBlockCopyThreadSliceLengths_GemmN,
                               GemmBBlockCopyThreadSliceLengths_GemmKPack});
    } else {
      blockwiseCopyBBounds =
          b.getIndexArrayAttr({1, GemmBBlockCopyThreadSliceLengths_GemmK,
                               GemmBBlockCopyThreadSliceLengths_GemmN});
    }
    Type blockwiseLoadBType;
    SmallVector<Type, 8> blockwiseLoadBTypes;
    int blockwiseBVectorDim;
    int blockwiseLoadBVectorLength;
    int blockwiseStoreBVectorLength;

    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmK: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmK << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmN: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";
    // llvm::errs() << "GemmBBlockCopyThreadSliceLengths_GemmKPack: "
    //              << GemmBBlockCopyThreadSliceLengths_GemmKPack << "\n";
    // llvm::errs() << "blockwise copy B bounds: ";
    // for (auto v : blockwiseCopyBBounds)
    //   llvm::errs() << v << " ";
    // llvm::errs() << "\n";

    std::tie(blockwiseLoadBType, blockwiseBVectorDim,
             blockwiseLoadBVectorLength, blockwiseStoreBVectorLength) =
        computeLoadStoreTypeInfo(b, op, elementType, blockwiseLoadBTypes,
			         blockwiseCopyBBounds, false, KPack);

    // llvm::errs() << "vector load dim: " << blockwiseBVectorDim << "\n";
    // llvm::errs() << "element type: " << blockwiseLoadBType << "\n";
    // llvm::errs() << "load size: " << blockwiseLoadBVectorLength << "\n";
    // llvm::errs() << "store size: " << blockwiseStoreBVectorLength << "\n";

    // -----

    // Compute source and destination coordinates for BlockwiseCopy ops.
    // Matrix A: {0, 0, m_block_data_on_global}, {0, 0, 0}
    // Matrix B: {0, 0, n_block_data_on_global}, {0, 0, 0}

    // -----

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    SmallVector<Value, 4> blockwiseLoadACoords;
    if (KPack > 1) {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Z,
                              GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    } else {
      blockwiseLoadACoords = {GemmBlockCoord_G, GemmABlockCopySourceCoord_Y,
                              GemmABlockCopySourceCoord_X};
    }
    // Emit blockwise_load for matrix A.
    auto blockwiseLoadA = b.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadATypes, op.a(), blockwiseCopyABounds,
        b.getArrayAttr({aTransforms}), op.paddingInfo(), op.aOobDims(),
        blockwiseLoadACoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadA, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    SmallVector<Value, 4> blockwiseStoreACoords;
    if (KPack > 1) {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Z,
                               GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    } else {
      blockwiseStoreACoords = {zeroConstantOp, GemmABlockCopyDestCoord_Y,
                               GemmABlockCopyDestCoord_X};
    }
    // Emit blockwise_store for matrix A.
    auto blockwiseStoreA = b.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixASubviewOp, blockwiseCopyABounds, noTransforms1,
        blockwiseLoadA.getResults(), blockwiseStoreACoords);
    affixBlockwiseCopyAttributes(
        blockwiseStoreA, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    SmallVector<Value, 4> blockwiseLoadBCoords;
    if (KPack > 1) {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Z,
                              GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    } else {
      blockwiseLoadBCoords = {GemmBlockCoord_G, GemmBBlockCopySourceCoord_Y,
                              GemmBBlockCopySourceCoord_X};
    }
    // Emit blockwise_load for matrix B.
    auto blockwiseLoadB = b.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadBTypes, op.b(), blockwiseCopyBBounds,
        b.getArrayAttr({bTransforms}), op.paddingInfo(), op.bOobDims(),
        blockwiseLoadBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadB, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    SmallVector<Value, 4> blockwiseStoreBCoords;
    if (KPack > 1) {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Z,
                               GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    } else {
      blockwiseStoreBCoords = {zeroConstantOp, GemmBBlockCopyDestCoord_Y,
                               GemmBBlockCopyDestCoord_X};
    }
    // Emit blockwise_store for matrix B.
    auto blockwiseStoreB = b.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixBSubviewOp, blockwiseCopyBBounds, noTransforms1,
        blockwiseLoadB.getResults(), blockwiseStoreBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseStoreB, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // -----

    // Logic to do XDLOPS code selection.
    // llvm::errs() << "Invoke XDLOPS code selection logic:\n";
    // llvm::errs() << "dataType: "; dataType.dump(); llvm::errs() << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";

    XdlopsCodeSelection xcs =
        XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;
    VectorType vectorType = xcs.vectorType;
    int64_t vectorNumber = xcs.vectorNumber;
    SmallVector<SmallVector<unsigned, 3>, 2> imms = xcs.imms;

    int64_t group_size = xcs.group_size;
    int64_t num_groups_blk = xcs.num_groups_blk;
    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t m = xcs.m;
    int64_t n = xcs.n;

    // -----

    // Logic to setup blockwise_gemm_v2 parameters.
    //
    // Original C++ logic:
    // index_t mMyWaveOffsetA;
    // index_t mMyWaveOffsetB;
    // const index_t waveId   = get_thread_local_1d_id() / WaveSize;
    // const index_t waveId_m = waveId / GemmNWaves;
    // const index_t waveId_n = waveId % GemmNWaves;
    // mMyWaveOffsetA = waveId_m * GemmMPerWave;
    // mMyWaveOffsetB = waveId_n * GemmNPerWave;
    auto waveId = b.create<DivUIOp>(loc, tid, waveSizeConstantOp);
    auto waveId_m = b.create<DivUIOp>(loc, waveId, NWavesConstantOp);
    auto waveId_n = b.create<RemUIOp>(loc, waveId, NWavesConstantOp);

    Value mMyWaveOffsetA, mMyWaveOffsetB;
    mMyWaveOffsetA = b.create<MulIOp>(loc, waveId_m, MPerWaveConstantOp);
    mMyWaveOffsetB = b.create<MulIOp>(loc, waveId_n, NPerWaveConstantOp);

    // Logic to setup buffers for blockwise_gemm_v2.

    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);
    int64_t arrayASize = (!IsKReduction)
                             ? (KPerBlock * MRepeats)
                             : (KPerBlock / num_input_blks * MRepeats);
    int64_t arrayBSize = (!IsKReduction)
                             ? (KPerBlock * NRepeats)
                             : (KPerBlock / num_input_blks * NRepeats);
    Type arrayAType, arrayBType;
    if (KPack > 1) {
      arrayAType = MemRefType::get({arrayASize},
                                   VectorType::get({KPack}, dataType), {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType = MemRefType::get({arrayBSize},
                                   VectorType::get({KPack}, dataType), {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      arrayAType = MemRefType::get({arrayASize}, dataType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
      arrayBType = MemRefType::get({arrayBSize}, dataType, {},
                                   gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto arrayA = b.create<miopen::GpuAllocOp>(loc, arrayAType);
    auto arrayB = b.create<miopen::GpuAllocOp>(loc, arrayBType);

    // -----

    // Logic to allocate 0-initialized vectors for C.
    SmallVector<Value, 4> vectorCs;
    SmallVector<Type, 4> vectorCTypes;
    auto vectorZeroConst = createZeroConstantFloatOp(b, loc, vectorType);
    std::fill_n(std::back_inserter(vectorCs), vectorNumber, vectorZeroConst);
    std::fill_n(std::back_inserter(vectorCTypes), vectorNumber, vectorType);

    // -----

    // Emit loop.

    int64_t loopIteration = (K - KPerBlock) / KPerBlock;

    // Assign iter args.
    // 0: blockwise copy A src y coordinate.
    // 1: blockwise copy B src y coordinate.
    // 2-x : vectorCs.
    SmallVector<Value, 6> iterArgs = {blockwiseLoadA.sourceCoord()[1],
                                      blockwiseLoadB.sourceCoord()[1]};
    iterArgs.append(vectorCs);

    auto mfmaLoopOp = b.create<AffineForOp>(loc, 0, loopIteration, 1, iterArgs);

    // inside the loop.
    auto mfmalb = OpBuilder::atBlockBegin(mfmaLoopOp.getBody());

    const auto &mfmalArgs = mfmaLoopOp.getRegionIterArgs();
    // get vectorCs for this iteration.
    std::copy(mfmalArgs.begin() + 2, mfmalArgs.end(), vectorCs.begin());

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    Value blockwiseCopyASrcUpdated =
        mfmalb.create<AddIOp>(loc, mfmalArgs[0], KPerBlockConstantOp);
    blockwiseLoadACoords[1] = blockwiseCopyASrcUpdated;
    // Emit blockwise_load for matrix A.
    auto blockwiseLoadATop = mfmalb.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadATypes, op.a(), blockwiseLoadA.bounds(),
        blockwiseLoadA.transforms(), blockwiseLoadA.paddingInfo(),
        blockwiseLoadA.oobDims(), blockwiseLoadACoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadATop, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);

    Value blockwiseCopyBSrcUpdated =
        mfmalb.create<AddIOp>(loc, mfmalArgs[1], KPerBlockConstantOp);
    blockwiseLoadBCoords[1] = blockwiseCopyBSrcUpdated;
    // Emit blockwise_load for matrix B.
    auto blockwiseLoadBTop = mfmalb.create<miopen::BlockwiseLoadOp>(
        loc, blockwiseLoadBTypes, op.b(), blockwiseLoadB.bounds(),
        blockwiseLoadB.transforms(), blockwiseLoadB.paddingInfo(),
        blockwiseLoadB.oobDims(), blockwiseLoadBCoords);
    affixBlockwiseCopyAttributes(
        blockwiseLoadBTop, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // LDS barrier : guarantees LDS update completion before reading out to register.
    // requires LDS fence + barrier.
    mfmalb.create<miopen::LDSBarrierOp>(loc);

    // Emit blockwise V2 GEMM.
    // The xdlops gemms take a 1D buffer because reasons
    auto blockwiseGemmV2Op = mfmalb.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, ldsBlockASubviewOp, ldsBlockBSubviewOp,
        noTransformsArray(b, 2), mMyWaveOffsetA, mMyWaveOffsetB, arrayA, arrayB,
        vectorCs);
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2Op, op, MPerBlock, KPerBlock,
                                   NPerBlock, b);

    // LDS barrier.
    mfmalb.create<miopen::LDSBarrierOp>(loc);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    // Emit blockwise_store for matrix A.
    auto blockwiseStoreABottom = mfmalb.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixASubviewOp, blockwiseStoreA.bounds(),
        blockwiseStoreA.transforms(), blockwiseLoadATop.getResults(),
        blockwiseStoreA.destCoord());
    affixBlockwiseCopyAttributes(
        blockwiseStoreABottom, op, b, /*vectorDim=*/blockwiseAVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadAVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreAVectorLength);
    // Emit blockwise_store for matrix B.
    auto blockwiseStoreBBottom = mfmalb.create<miopen::BlockwiseStoreOp>(
        loc, ldsMatrixBSubviewOp, blockwiseStoreB.bounds(),
        blockwiseStoreB.transforms(), blockwiseLoadBTop.getResults(),
        blockwiseStoreB.destCoord());
    affixBlockwiseCopyAttributes(
        blockwiseStoreBBottom, op, b, /*vectorDim=*/blockwiseBVectorDim,
        /*blockwiseLoadLength=*/blockwiseLoadBVectorLength,
        /*blockwiseStoreLength=*/blockwiseStoreBVectorLength);

    // Update iter args.
    // blockwiseCopyASrcVector and blockwiseCopyBSrcVector are updated.
    iterArgs[0] = blockwiseCopyASrcUpdated;
    iterArgs[1] = blockwiseCopyBSrcUpdated;
    // blockwise_gemm_v2 updates iter args[4-].
    std::copy(blockwiseGemmV2Op.getResults().begin(),
              blockwiseGemmV2Op.getResults().end(), iterArgs.begin() + 2);

    // emit loop yield so iter args can be passed to the next iteration.
    mfmalb.create<AffineYieldOp>(loc, iterArgs);
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<miopen::LDSBarrierOp>(loc);

    // get vectorCs for loop tail.
    std::copy(mfmaLoopOp.getResults().begin() + 2,
              mfmaLoopOp.getResults().end(), vectorCs.begin());

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmV2TailOp = b.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, ldsBlockASubviewOp, ldsBlockBSubviewOp,
        blockwiseGemmV2Op.transforms(), mMyWaveOffsetA, mMyWaveOffsetB, arrayA,
        arrayB, vectorCs);
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2TailOp, op, MPerBlock,
                                   KPerBlock, NPerBlock, b);

    // -----

    // Matrix C write out logic.

    // Original C++ logic.
    // __device__ static constexpr index_t GetNumBlksPerXdlops() {
    //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
    // }
    //
    // struct OutputLayout {
    //     __device__ static constexpr index_t GetBlkSize() { return
    //     mfma_type.num_regs_blk; }
    //     __device__ static constexpr index_t GetNumBlks() {
    //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
    //     }
    // };
    // using CThreadCopySliceLengths = Sequence<M0, 1, M2, 1>;
    // constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
    // constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

    // int64_t BlkSize = xcs.num_regs_blk;
    int64_t NumBlksPerXdlops = (MPerXdlops * NPerXdlops) / (m * n);
    int64_t NumBlks = NumBlksPerXdlops * MRepeats * NRepeats;

    int64_t iterationsPerVectorC = NumBlks / vectorNumber;
    int64_t vectorCoffset = vectorType.getShape()[0] / iterationsPerVectorC;

    // llvm::errs() << "MPerXlops: " << MPerXdlops << "\n";
    // llvm::errs() << "NPerXlops: " << NPerXdlops << "\n";
    // llvm::errs() << "m: " << m << "\n";
    // llvm::errs() << "n: " << n << "\n";
    // llvm::errs() << "MRepeat: " << MRepeats << "\n";
    // llvm::errs() << "NRepeat: " << NRepeats << "\n\n";

    // llvm::errs() << "BlkSize: " << BlkSize << "\n";
    // llvm::errs() << "NumBlksPerXdlops: " << NumBlksPerXdlops << "\n";
    // llvm::errs() << "NumBlks: " << NumBlks << "\n\n";

    // llvm::errs() << "iterationsPerVectorC: " << iterationsPerVectorC << "\n";
    // llvm::errs() << "vectorCoffset: " << vectorCoffset << "\n";

    auto group_size_ConstantOp = b.create<ConstantIndexOp>(loc, group_size);
    auto wave_size_ConstantOp = b.create<ConstantIndexOp>(loc, wave_size);
    auto num_threads_blk_ConstantOp =
        b.create<ConstantIndexOp>(loc, num_threads_blk);

    // Threadwise copy from register (naive tensor) to global (generic tensor).

    int64_t M3 = num_groups_blk;
    int64_t M1 = num_input_blks;
    int64_t M2 = group_size;
    int64_t M0 = M / (M1 * M2);
    int64_t N1 = group_size;
    int64_t N0 = N / N1;
    // llvm::errs() << "M0: " << M0 << "\n";
    // llvm::errs() << "M1: num_input_blks: " << M1 << "\n";
    // llvm::errs() << "M2: group_size: " << M2 << "\n";
    // llvm::errs() << "M3: num_groups_blk: " << M3 << "\n\n";

    auto M2ConstantOp = b.create<ConstantIndexOp>(loc, M2);
    auto M2TimesM1Op = b.create<ConstantIndexOp>(loc, M2 * M1);
    auto N1ConstantOp = M2ConstantOp;

    auto laneId_xdlops_gemm = b.create<RemUIOp>(loc, tid, wave_size_ConstantOp);
    auto blk_id_xdlops_gemm =
        b.create<DivUIOp>(loc, laneId_xdlops_gemm, num_threads_blk_ConstantOp);
    auto blk_td_xdlops_gemm =
        b.create<RemUIOp>(loc, laneId_xdlops_gemm, num_threads_blk_ConstantOp);

    // emit vector swizzles
    auto gemmCVectorizedMatrixDim =
        op->getAttrOfType<IntegerAttr>("matrix_c_source_vector_read_dim");
    int64_t matrixCDataPerCopy =
        op->getAttrOfType<IntegerAttr>("matrix_c_data_per_copy").getInt();

    constexpr int64_t swizzleGroup = 4;
    // Ensure that the prerequisites are met
    // - The N dimension of the output will be stored vectorized
    // - The lowest level of splitting in registers is equal to swizzleGroup
    //    so transpose is well defined
    // - None of the larger dimensions of interest have overhangs that lead to
    //    incomplete transposes
    // - The writes will vectorize: if we're not getting vectorization
    //    due to HW % swizzleGroup != 0, then there's no point
    bool enableOutSwizzles =
        gemmCVectorizedMatrixDim.getInt() == gemmCDimN &&
        (matrixCDataPerCopy >= swizzleGroup) &&
        (M2 == swizzleGroup && (m % swizzleGroup == 0) &&
         (n % swizzleGroup == 0) && (MPerWave % swizzleGroup == 0) &&
         (NPerWave % swizzleGroup == 0));
    const auto &tailResults = blockwiseGemmV2TailOp->getResults();

    TransformMapAttr splitCTransformAttr, cVectorAccessTransformAttr;
    ArrayAttr copyBounds;
    llvm::SmallVector<Value, 4> vectors;
    vectors.reserve(tailResults.size());
    if (enableOutSwizzles) {
      // The swizzle operation doesn't fundamentally affect the mapping
      // of "expanded GEMM" (G x M0 X M1 X M2 X N) to GEMM (G X M X N)
      // space, just how we walk across it and where each thread starts.

      // However, because of the 4x4 transpose we'll be imposing
      // instead of holding N constant and walking up the M2 dimension,
      // we'll need to take 4 steps in the N dimension but hold the
      // divisible-by-4 part of the N coordinate constant. Therefore, we need to
      // break the N dimension into N0 and N1 The affine map remains otherwise
      // unchanged and becomes
      //  (d0, d1, d2, d3, d4, d5) ->
      //  (d0, d1 * M1 * M2 + d2 * M2 + d3, d4 * N1 + d5)
      TopDownCTBuilder splitCTransform(b, {"G", "M0", "M1", "M2", "N0", "N1"},
                                       {G, M0, M1, M2, N0, N1}, loc);
      splitCTransform.passThrough({"gemmG"}, {0}, {"G"});
      splitCTransform.embed("gemmM", 1, M, {"M0", "M1", "M2"},
                            {M1 * M2, M2, 1});
      splitCTransform.embed("gemmN", 2, N, {"N0", "N1"}, {N1, 1});

      splitCTransformAttr = splitCTransform.get();

      // Here is the first main effect of the swizzling transformation
      // Instead of having the fastest coordinate be the M2 dimension
      // it's now the N1 dimension, since each group of 4 values in a vector
      // corresponds to 4 successive N values after the transpose, as opposed
      // to 4 successive M values.
      // The source vector reading map is therefore
      //  (g, m0, m1, m2, n0, n1) -> (m0 * N1 + n1)
      TopDownCTBuilder cVectorAccessTransform(
          b, {"G", "M0", "M1", "M2", "N0", "N1"}, {G, M0, M1, M2, N0, N1}, loc);
      cVectorAccessTransform.embed("raw", 0, M3 * N1,
                                   {"G", "M0", "M1", "M2", "N0", "N1"},
                                   {M3 * N1, N1, N1, N1, N1, 1});
      cVectorAccessTransformAttr = cVectorAccessTransform.get();

      copyBounds = b.getIndexArrayAttr({1, M3, 1, 1, 1, N1});

      // Actually perform the swizzles
      for (const Value &result : tailResults) {
        auto swizzle = b.create<miopen::InWarpTransposeOp>(
            loc, result.getType(), result, laneId_xdlops_gemm,
            b.getI32IntegerAttr(group_size), b.getI32ArrayAttr({0, 1, 2, 3}));
        vectors.push_back(swizzle);
      }
    } else {
      // build affine expression: d0 = g
      // (d0, d1, d2, d3, d4) -> (d0, d1 * M1 * M2 + d2 * M2 + d3, d4)
      TopDownCTBuilder splitCTransform(b, {"G", "M0", "M1", "M2", "N"},
                                       {G, M0, M1, M2, N}, loc);
      splitCTransform.passThrough({"gemmG"}, {0}, {"G"});
      splitCTransform.embed("gemmM", 1, M, {"M0", "M1", "M2"},
                            {M1 * M2, M2, 1});
      splitCTransform.passThrough({"gemmN"}, {2}, {"N"});

      splitCTransformAttr = splitCTransform.get();

      // The source vector reading map is
      //  (g, m0, m1, m2, n) -> (m0 * M2 + m2)
      TopDownCTBuilder cVectorAccessTransform(b, {"G", "M0", "M1", "M2", "N"},
                                              {G, M0, M1, M2, N}, loc);
      cVectorAccessTransform.embed("raw", 0, M3 * M2,
                                   {"G", "M0", "M1", "M2", "N"},
                                   {M3 * M2, M2, M2, 1, 1});
      cVectorAccessTransformAttr = cVectorAccessTransform.get();

      copyBounds = b.getIndexArrayAttr({1, M3, 1, M2, 1});

      // No swizzles means we keep the original results
      std::copy(tailResults.begin(), tailResults.end(),
                std::back_inserter(vectors));
    }
    Value cTransformed =
        b.create<miopen::TransformOp>(loc, op.c(), splitCTransformAttr);
    // The transform for the destination memref will be copied in
    // by TransformOp lowering
    llvm::SmallVector<Attribute, 2> threadwiseCopyV2Transforms = {
        b.getArrayAttr({cVectorAccessTransformAttr}), cTransforms};
    ArrayAttr threadwiseCopyV2ArgTransform =
        b.getArrayAttr(threadwiseCopyV2Transforms);

    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value m_thread_data_on_global, n_thread_data_on_global;

    // emit unrolled loop.
    for (int64_t iter = 0; iter < NumBlks; ++iter) {
      // In gridwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // const auto c_thread_mtx_on_block =
      // blockwise_gemm.GetBeginOfThreadMatrixC(i); const index_t
      // m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      // compute thread_mtx_on_blk_row and thread_mtx_on_blk_col.

      // Original C++ logic.
      //
      // In xdlops_gemm.hpp:
      //
      // static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops;
      // }
      // __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i) {
      //     const index_t xdlops_i = i / GetNumBlksPerXdlops();
      //     const index_t j        = i % GetNumBlksPerXdlops();
      //     const index_t m_i = xdlops_i / NRepeats;
      //     const index_t n_i = xdlops_i % NRepeats;
      //     const index_t laneId = get_thread_local_1d_id() %
      //     mfma_type.wave_size; const index_t blk_id = laneId /
      //     mfma_type.num_threads_blk; const index_t blk_td = laneId %
      //     mfma_type.num_threads_blk; index_t col_blk = j %
      //     mfma_type.num_output_blks; index_t row_blk = j /
      //     mfma_type.num_output_blks; static_if<!IsABroadcast>{}([&](auto) {
      //         col_blk = j / mfma_type.num_output_blks;
      //         row_blk = j % mfma_type.num_output_blks;
      //     });
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size
      //     + m_i * MPerXdlops; return MatrixIndex{row, col};
      // }
      //
      int64_t xdlops_i_xdlops_gemm = iter / NumBlksPerXdlops;
      int64_t j_xdlops_gemm = iter % NumBlksPerXdlops;
      int64_t m_i_xdlops_gemm = xdlops_i_xdlops_gemm / NRepeats;
      int64_t n_i_xdlops_gemm = xdlops_i_xdlops_gemm % NRepeats;

      int64_t col_blk_xdlops_gemm, row_blk_xdlops_gemm;
      bool IsABroadcast = (NPerXdlops >= MPerXdlops);
      if (IsABroadcast) {
        col_blk_xdlops_gemm = j_xdlops_gemm % num_output_blks;
        row_blk_xdlops_gemm = j_xdlops_gemm / num_output_blks;
      } else {
        col_blk_xdlops_gemm = j_xdlops_gemm / num_output_blks;
        row_blk_xdlops_gemm = j_xdlops_gemm % num_output_blks;
      }

      // Within a group of elements, a non-swizzled loop will output
      // to (ignoring OOB) [(i, j), (i + 1, j), (i + 2, j), (i + 3, j)]
      // for some starting position (i, j) that's a function of coordinates
      // that very slower.

      // The swizzles mean that each thread instead outputs to
      //  [(i, j), (i, j+1), (i, j+2), (i, j+3)]
      // Therefore, in order to ensure that values remain output to the correct
      // place we must map the starting coordinates through
      //  (i, j) -> (i / 4 * 4 + j % 4, j / 4 + 4 + i % 4)
      Value threadMtxColInBlock;
      if (enableOutSwizzles) {
        // The starting coordinate remap means that we must start
        // at (blk_td / 4) * 4, since blk_td % 4 is moved to the
        // row coordinate by the transpose and nothing replaces it
        // (the unswizzled row coordinate is always a multiple of 4
        // in cases where swizzles are enabled)
        threadMtxColInBlock =
            b.create<MulIOp>(loc,
                             b.create<arith::DivUIOp>(loc, blk_td_xdlops_gemm,
                                                      group_size_ConstantOp),
                             group_size_ConstantOp);
      } else {
        // Original C++ logic.
        //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
        threadMtxColInBlock = blk_td_xdlops_gemm;
      }
      int64_t thread_mtx_on_blk_col_const =
          col_blk_xdlops_gemm * n + n_i_xdlops_gemm * NPerXdlops;
      Value thread_mtx_on_blk_col = b.create<AddIOp>(
          loc, threadMtxColInBlock,
          b.create<ConstantIndexOp>(loc, thread_mtx_on_blk_col_const));

      // Original C++ logic.
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size
      //     + m_i * MPerXdlops;
      Value threadMtxRowInBlock =
          b.create<MulIOp>(loc, blk_id_xdlops_gemm, group_size_ConstantOp);
      if (enableOutSwizzles) {
        // Here, we must incorporate the mod-4 parts of blk_td
        // since while, without swizzles, these four values
        // were stored on successive threads, now they're stored
        // in four consecutive vector entries on the same thread
        threadMtxRowInBlock =
            b.create<AddIOp>(loc, threadMtxRowInBlock,
                             b.create<arith::RemUIOp>(loc, blk_td_xdlops_gemm,
                                                      group_size_ConstantOp));
      }
      int64_t thread_mtx_on_blk_row_const =
          row_blk_xdlops_gemm * m + m_i_xdlops_gemm * MPerXdlops;
      auto thread_mtx_on_blk_row = b.create<AddIOp>(
          loc, threadMtxRowInBlock,
          b.create<ConstantIndexOp>(loc, thread_mtx_on_blk_row_const));

      // compute c_thread_mtx_index_row, c_thread_mtx_index_col.
      // compute c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32.

      // In blockwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      //  __device__ static constexpr index_t GetNumBlks()
      //      return GetNumBlksPerXdlops() * MRepeats * NRepeats;
      //
      // __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i) {
      //     const index_t waveId = get_thread_local_1d_id() / WaveSize;
      //     const index_t xdlops_i = i /
      //     XdlopsGemm.GetOutputLayout().GetNumBlks(); const index_t j        =
      //     i % XdlopsGemm.GetOutputLayout().GetNumBlks(); const index_t m =
      //     xdlops_i / NRepeats; const index_t n = xdlops_i % NRepeats; const
      //     auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(j); const
      //     index_t col =
      //         (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops +
      //         thread_mtx_on_blk.col;
      //     const index_t row =
      //         (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops +
      //         thread_mtx_on_blk.row;
      //     return MatrixIndex{row, col};
      // }

      int64_t xdlops_i_blockwise_gemm = iter / NumBlks;
      int64_t m_blockwise_gemm = xdlops_i_blockwise_gemm / NRepeats;
      int64_t n_blockwise_gemm = xdlops_i_blockwise_gemm % NRepeats;

      // Original C++ logic.
      // const index_t col = (waveId % GemmNWaves) * GemmNPerWave + n *
      // NPerXdlops + thread_mtx_on_blk.col;
      c_thread_mtx_index_col = b.create<AddIOp>(
          loc,
          b.create<AddIOp>(
              loc,
              b.create<MulIOp>(loc,
                               b.create<RemUIOp>(loc, waveId, NWavesConstantOp),
                               NPerWaveConstantOp),
              b.create<ConstantIndexOp>(loc, n_blockwise_gemm * NPerXdlops)),
          thread_mtx_on_blk_col);

      // Original C++ logic.
      // const index_t row = (waveId / GemmNWaves) * GemmMPerWave + m *
      // MPerXdlops + thread_mtx_on_blk.row;
      c_thread_mtx_index_row = b.create<AddIOp>(
          loc,
          b.create<AddIOp>(
              loc,
              b.create<MulIOp>(loc,
                               b.create<DivUIOp>(loc, waveId, NWavesConstantOp),
                               MPerWaveConstantOp),
              b.create<ConstantIndexOp>(loc, m_blockwise_gemm * MPerXdlops)),
          thread_mtx_on_blk_row);

      // In gridwise_gemm_xdlops.hpp:
      //
      // const auto c_thread_mtx_on_block =
      // blockwise_gemm.GetBeginOfThreadMatrixC(i); const index_t
      // m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      m_thread_data_on_global =
          b.create<AddIOp>(loc, m_block_data_on_global, c_thread_mtx_index_row);
      n_thread_data_on_global =
          b.create<AddIOp>(loc, n_block_data_on_global, c_thread_mtx_index_col);

      SmallVector<Value, 6> matrixCThreadwiseCopySourceCoords;
      SmallVector<Value, 6> matrixCThreadwiseCopyDestCoords;
      if (enableOutSwizzles) {
        std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 6,
                    zeroConstantOp.getResult());
        matrixCThreadwiseCopyDestCoords.append(
            {// g
             GemmBlockCoord_G,
             // m_thread_data_on_global / (M2 * M1)
             b.create<DivUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
             // m_thread_data_on_global % (M2 * M1) / M2
             b.create<DivUIOp>(
                 loc,
                 b.create<RemUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
                 M2ConstantOp),
             // m_thread_data_on_global % M2
             b.create<RemUIOp>(loc, m_thread_data_on_global, M2ConstantOp),
             // n_thread_data_on_global / N1
             b.create<DivUIOp>(loc, n_thread_data_on_global, N1ConstantOp),
             // n_thread-data_on_global % N1
             b.create<RemUIOp>(loc, n_thread_data_on_global, N1ConstantOp)});
      } else {
        std::fill_n(std::back_inserter(matrixCThreadwiseCopySourceCoords), 5,
                    zeroConstantOp.getResult());
        matrixCThreadwiseCopyDestCoords.append(
            {// g
             GemmBlockCoord_G,
             // m_thread_data_on_global / (M2 * M1)
             b.create<DivUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
             // m_thread_data_on_global % (M2 * M1) / M2
             b.create<DivUIOp>(
                 loc,
                 b.create<RemUIOp>(loc, m_thread_data_on_global, M2TimesM1Op),
                 M2ConstantOp),
             // m_thread_data_on_global % M2
             b.create<RemUIOp>(loc, m_thread_data_on_global, M2ConstantOp),
             // n_thread_data_on_global
             n_thread_data_on_global});
      }
      // Select which vector C to use, and offset.
      int64_t vectorCIndex = iter / iterationsPerVectorC;
      int64_t vectorCOffset = vectorCoffset * (iter % iterationsPerVectorC);
      auto vectorCOffsetConstantAttr = b.getIndexAttr(vectorCOffset);

      // Emit threadwise_copy_v2.
      auto threadwiseCopyV2CMatrixOp = b.create<miopen::ThreadwiseCopyV2Op>(
          loc, vectors[vectorCIndex], cTransformed, copyBounds,
          threadwiseCopyV2ArgTransform, op.paddingInfo(),
          op.storeOperationAttr(), op.cOobDims(), vectorCOffsetConstantAttr,
          matrixCThreadwiseCopySourceCoords, matrixCThreadwiseCopyDestCoords);
      affixThreadwiseCopyV2Attributes(threadwiseCopyV2CMatrixOp, op, b,
                                      enableOutSwizzles);
    }

    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmRewritePattern
    : public OpRewritePattern<miopen::BlockwiseGemmOp> {
  using OpRewritePattern<miopen::BlockwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::BlockwiseGemmOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Prepare some useful constants.
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);

    auto blockAType = op.matrixA().getType().cast<MemRefType>();

    auto elementType =
        op.matrixC().getType().cast<MemRefType>().getElementType();

    // Obtain critical matrix dimensions.
    int64_t K = blockAType.getShape()[1];

    auto transformsA = op.transforms()[0].cast<ArrayAttr>();
    auto transformsB = op.transforms()[1].cast<ArrayAttr>();

    ArrayAttr emptyArr = b.getArrayAttr({});
    auto noPadding =
        PaddingInfoAttr::get(b.getContext(), 0, 0, 0, BwdPaddingKernelInfo::NA);
    auto noOobDims = b.getArrayAttr({});
    IntegerAttr noGlobals = b.getIndexAttr(-1);
    // Non-xdlops path.

    // Obtain critical attributes.
    int64_t KPack =
        op->hasAttr("kpack")
            ? op->getAttr("kpack").template cast<IntegerAttr>().getInt()
            : 1;
    int64_t KPerThread =
        op->getAttr("k_per_thread").template cast<IntegerAttr>().getInt();
    int64_t MPerThread =
        op.matrixC().getType().template cast<MemRefType>().getShape()[1];
    int64_t NPerThread =
        op.matrixC().getType().template cast<MemRefType>().getShape()[2];
    int64_t MPerThreadSubC =
        op->getAttr("m_per_thread").template cast<IntegerAttr>().getInt();
    int64_t NPerThreadSubC =
        op->getAttr("n_per_thread").template cast<IntegerAttr>().getInt();

    // llvm::errs() << "MPerThread: " << MPerThread << "\n";
    // llvm::errs() << "MPerThreadSubC: " << MPerThreadSubC << "\n";
    // llvm::errs() << "NPerThread: " << NPerThread << "\n";
    // llvm::errs() << "NPerThreadSubC: " << NPerThreadSubC << "\n";

    auto MPerThreadSubCConstantOp =
        b.create<ConstantIndexOp>(loc, MPerThreadSubC);
    auto NPerThreadSubCConstantOp =
        b.create<ConstantIndexOp>(loc, NPerThreadSubC);

    int64_t MLevel0Cluster =
        op->getAttr("m_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster =
        op->getAttr("m_level1_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster =
        op->getAttr("n_level0_cluster").template cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster =
        op->getAttr("n_level1_cluster").template cast<IntegerAttr>().getInt();

    int64_t MPerLevel1Cluster =
        MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
    int64_t NPerLevel1Cluster =
        NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;
    auto MPerLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, MPerLevel1Cluster);
    auto NPerLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NPerLevel1Cluster);

    int64_t MRepeat = MPerThread / MPerThreadSubC;
    int64_t NRepeat = NPerThread / NPerThreadSubC;

    // Alloc register for thread_a and thread_b.
    Type threadARegisterMemRefType;
    if (KPack > 1) {
      threadARegisterMemRefType =
          MemRefType::get({1, KPerThread, MPerThread, KPack}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      threadARegisterMemRefType =
          MemRefType::get({1, KPerThread, MPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto threadAAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

    Type threadBRegisterMemRefType;
    if (KPack > 1) {
      threadBRegisterMemRefType =
          MemRefType::get({1, KPerThread, NPerThread, KPack}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    } else {
      threadBRegisterMemRefType =
          MemRefType::get({1, KPerThread, NPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
    }
    auto threadBAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Main loop.
    auto loopIteration = K / KPerThread;
    auto loopOp = b.create<AffineForOp>(loc, 0, loopIteration);

    // inside the main loop.
    auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

    auto iv = loopOp.getInductionVar();

    // read matrix A loop.
    auto loopReadMatrixAIteration = MRepeat;
    auto loopReadMatrixAOp =
        lb.create<AffineForOp>(loc, 0, loopReadMatrixAIteration);

    // inside read matrix A loop.
    auto lab = OpBuilder::atBlockTerminator(loopReadMatrixAOp.getBody());

    auto iva = loopReadMatrixAOp.getInductionVar();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixAThreadwiseCopySourceCoords;
    if (KPack > 1) {
      matrixAThreadwiseCopySourceCoords = {
          zeroConstantOp, zeroConstantOp, iv,
          lab.create<AddIOp>(
              loc, lab.create<MulIOp>(loc, iva, MPerLevel1ClusterConstantOp),
              op.threadOffsetA())};
    } else {
      matrixAThreadwiseCopySourceCoords = {
          zeroConstantOp, iv,
          lab.create<AddIOp>(
              loc, lab.create<MulIOp>(loc, iva, MPerLevel1ClusterConstantOp),
              op.threadOffsetA())};
    }

    SmallVector<Value, 4> matrixAThreadwiseCopyDestCoords;
    if (KPack > 1) {
      matrixAThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp, zeroConstantOp,
          lab.create<MulIOp>(loc, iva, MPerThreadSubCConstantOp)};
    } else {
      matrixAThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp,
          lab.create<MulIOp>(loc, iva, MPerThreadSubCConstantOp)};
    }

    // Emit threadwise_copy.
    auto threadwiseCopyAMatrixOp = lab.create<miopen::ThreadwiseCopyOp>(
        loc, op.matrixA(), threadAAllocOp,
        b.getArrayAttr({transformsA, emptyArr}), noPadding, noOobDims,
        noGlobals, matrixAThreadwiseCopySourceCoords,
        matrixAThreadwiseCopyDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyAMatrixOp, op, b,
                                  /*isMatrixA=*/true);

    // read matrix B loop.
    auto loopReadMatrixBIteration = NRepeat;
    auto loopReadMatrixBOp =
        lb.create<AffineForOp>(loc, 0, loopReadMatrixBIteration);

    // inside read matrix B loop.
    auto lbb = OpBuilder::atBlockTerminator(loopReadMatrixBOp.getBody());

    auto ivb = loopReadMatrixBOp.getInductionVar();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixBThreadwiseCopySourceCoords;
    if (KPack > 1) {
      matrixBThreadwiseCopySourceCoords = {
          zeroConstantOp, zeroConstantOp, iv,
          lbb.create<AddIOp>(
              loc, lbb.create<MulIOp>(loc, ivb, NPerLevel1ClusterConstantOp),
              op.threadOffsetB())};
    } else {
      matrixBThreadwiseCopySourceCoords = {
          zeroConstantOp, iv,
          lbb.create<AddIOp>(
              loc, lbb.create<MulIOp>(loc, ivb, NPerLevel1ClusterConstantOp),
              op.threadOffsetB())};
    }

    SmallVector<Value, 4> matrixBThreadwiseCopyDestCoords;
    if (KPack > 1) {
      matrixBThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp, zeroConstantOp,
          lbb.create<MulIOp>(loc, ivb, NPerThreadSubCConstantOp)};
    } else {
      matrixBThreadwiseCopyDestCoords = {
          zeroConstantOp, zeroConstantOp,
          lbb.create<MulIOp>(loc, ivb, NPerThreadSubCConstantOp)};
    }

    // Emit threadwise_copy.
    auto threadwiseCopyBMatrixOp = lbb.create<miopen::ThreadwiseCopyOp>(
        loc, op.matrixB(), threadBAllocOp,
        b.getArrayAttr({transformsB, emptyArr}), noPadding, noOobDims,
        noGlobals, matrixBThreadwiseCopySourceCoords,
        matrixBThreadwiseCopyDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyBMatrixOp, op, b,
                                  /*isMatrixA=*/false);

    lb.create<miopen::ThreadwiseGemmOp>(loc, threadAAllocOp, threadBAllocOp,
                                        op.matrixC());

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseLoad lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseLoadRewritePattern
    : public OpRewritePattern<miopen::BlockwiseLoadOp> {
  using OpRewritePattern<miopen::BlockwiseLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::BlockwiseLoadOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    TypeRange resultTypes = op.result().getTypes();

    // BlockwiseLoad only accepts the following data movement:
    // - 0 (global) -> 5 (register) : load

    // Threadwise copy from global (generic tensor) to register (naive
    // tensor).

    auto threadwiseLoadOp = b.create<miopen::ThreadwiseLoadOp>(
        loc, resultTypes, op.source(), op.bounds(), op.transforms(),
        op.paddingInfo(), op.oobDims(), op.sourceCoord());
    affixThreadwiseCopyAttributes(threadwiseLoadOp, op, b,
                                  /*isThreadwiseLoad=*/true);

    op.replaceAllUsesWith(threadwiseLoadOp.getResults());
    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseStore lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseStoreRewritePattern
    : public OpRewritePattern<miopen::BlockwiseStoreOp> {
  using OpRewritePattern<miopen::BlockwiseStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::BlockwiseStoreOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    // BlockwiseLoad only accepts the following data movement:
    // - 5 (register) -> 3 (LDS) : store

    // Threadwise copy from register (naive tensor) to LDS (naive tensor).
    auto threadwiseStoreOp = b.create<miopen::ThreadwiseStoreOp>(
        loc, op.dest(), op.bounds(), op.transforms(), op.data(),
        op.destCoord());
    affixThreadwiseCopyAttributes(threadwiseStoreOp, op, b,
                                  /*isThreadwiseLoad=*/false);

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpRewritePattern<miopen::FillOp> {
  using OpRewritePattern<miopen::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::FillOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto inputType = op.input().getType().cast<MemRefType>();
    auto inputShape = inputType.getShape();

    AffineForOp currentLoop;
    OpBuilder currentScope = b;
    std::vector<mlir::Value> range;

    for (unsigned i = 0; i < inputShape.size(); ++i) {
      // Rank 1 loop.
      currentLoop = currentScope.create<AffineForOp>(loc, 0, inputShape[i]);

      // collect current loop induction var for store indexes
      range.push_back(currentLoop.getInductionVar());

      // inside loop.
      currentScope = OpBuilder::atBlockTerminator(currentLoop.getBody());
    }

    // Store fill value
    currentScope.create<memref::StoreOp>(loc, op.value(), op.input(), range);

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// InWarpTranspose lowering.
//===----------------------------------------------------------------------===//
static constexpr size_t swizzleGroupSize = InWarpTransposeOp::swizzleGroupSize;
struct InWarpTransposeRewritePattern
    : public OpRewritePattern<miopen::InWarpTransposeOp> {
  using OpRewritePattern<miopen::InWarpTransposeOp>::OpRewritePattern;

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

  LogicalResult matchAndRewrite(miopen::InWarpTransposeOp op,
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
    : public OpRewritePattern<miopen::ThreadwiseGemmOp> {
  using OpRewritePattern<miopen::ThreadwiseGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::ThreadwiseGemmOp op,
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
// ThreadwiseCopy lowering.
//===----------------------------------------------------------------------===//

struct ThreadwiseCopyRewritePattern
    : public OpRewritePattern<miopen::ThreadwiseCopyOp> {
  using OpRewritePattern<miopen::ThreadwiseCopyOp>::OpRewritePattern;

  // NOTE: when extending this logic to support vectors
  // ensure the results of the non-xdlops gemm are stored in a vectorizable
  // layout. This'll likely require something analogous to the in_warp_transpose
  // call in the xdlops case
  LogicalResult matchAndRewrite(miopen::ThreadwiseCopyOp op,
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
// ThreadwiseLoad lowering.
//===----------------------------------------------------------------------===//

struct ThreadwiseLoadRewritePattern
    : public OpRewritePattern<miopen::ThreadwiseLoadOp> {
  using OpRewritePattern<miopen::ThreadwiseLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::ThreadwiseLoadOp op,
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
    : public OpRewritePattern<miopen::ThreadwiseStoreOp> {
  using OpRewritePattern<miopen::ThreadwiseStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::ThreadwiseStoreOp op,
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
// ThreadwiseCopyV2 lowering.
//===----------------------------------------------------------------------===//

struct ThreadwiseCopyV2RewritePattern
    : public OpRewritePattern<miopen::ThreadwiseCopyV2Op> {
  using OpRewritePattern<miopen::ThreadwiseCopyV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::ThreadwiseCopyV2Op op,
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
// Transform lowering.
//
// Gathers a chain of transformations and puts them into the appropriate index
// of the transforms attribute of the user of that chain.
//===----------------------------------------------------------------------===//

struct TransformRewritePattern : public OpRewritePattern<miopen::TransformOp> {
  using OpRewritePattern<miopen::TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::TransformOp op,
                                PatternRewriter &b) const override {
    // To cut down on the number of intermediate arrays we pull in,
    // we'll deal with entire chains of transform ops at once

    TransformOp lastTransform = op;
    TransformOp firstTransform = op;

    while (auto pred = dyn_cast_or_null<TransformOp>(
               firstTransform.input().getDefiningOp())) {
      firstTransform = pred;
    }

    bool userIsOneTransform = false;
    do {
      userIsOneTransform = false;
      if (lastTransform.output().hasOneUse()) {
        if (auto succ = dyn_cast<TransformOp>(
                *(lastTransform.output().getUsers().begin()))) {
          userIsOneTransform = true;
          lastTransform = succ;
        }
      }
    } while (userIsOneTransform);

    // Each successive transform op creates an upper view from a lower view
    // so the transforms must be composed with the last (uppermost) transform
    // at the front.
    SmallVector<Attribute, 5> transforms;
    SmallVector<TransformOp, 5> transformOpStack;
    {
      Operation *beforeFirstTransform = firstTransform.input().getDefiningOp();
      Operation *currentOp = lastTransform.getOperation();
      while (currentOp != beforeFirstTransform) {
        auto currentTransform = cast<TransformOp>(currentOp);
        transformOpStack.push_back(currentTransform);
        // verify() would've failed if we couldn't cast<> these
        for (Attribute attr : currentTransform.transforms()) {
          TransformMapAttr ta = attr.cast<TransformMapAttr>();
          transforms.push_back(ta);
        }
        currentOp = currentTransform.input().getDefiningOp();
      }
    }

    // Check module-level invariants before applying the rewrite
    for (OpOperand &use : lastTransform.output().getUses()) {
      uint32_t argNum = use.getOperandNumber();
      Operation *useOp = use.getOwner();
      if (auto useTransforms =
              useOp->getAttr("transforms").dyn_cast_or_null<ArrayAttr>()) {
        if (useTransforms.size() <= argNum && !isa<TransformOp>(useOp)) {
          useOp->emitOpError("The transforms attribute does not have an entry "
                             "for each transformed argument");
        }
      } else {
        return useOp->emitOpError(
            "Operation taking a miopen.transform()'ed argument does not have a "
            "transforms attribute in which to place the transforms");
      }
    }

    // And now actually do the amendments
    for (OpOperand &use : lastTransform.output().getUses()) {
      uint32_t argNum = use.getOperandNumber();
      Operation *useOp = use.getOwner();
      auto argTransforms = useOp->getAttrOfType<ArrayAttr>("transforms");

      // Edge case: A transformed value is transformed in multiple ways
      if (auto useTransform = dyn_cast<TransformOp>(useOp)) {
        // The new set of transformations is the composition we're removing
        // going below/after the transforms this operation defines
        llvm::SmallVector<Attribute, 5> newTransforms;
        std::copy(argTransforms.begin(), argTransforms.end(),
                  std::back_inserter(newTransforms));
        newTransforms.append(transforms);
        useTransform->setAttr("transforms", b.getArrayAttr(newTransforms));
      } else {
        ArrayAttr thisArgTransforms = argTransforms[argNum].cast<ArrayAttr>();
        ArrayAttr newTransformsAttr;
        if (thisArgTransforms.size() == 0) {
          newTransformsAttr = b.getArrayAttr(transforms);
        } else {
          // The set of transformations is those being applied to the output of
          // the output transformation followed by those this chain of
          // transformations is performing.
          llvm::SmallVector<Attribute, 5> newTransforms;
          std::copy(thisArgTransforms.begin(), thisArgTransforms.end(),
                    std::back_inserter(newTransforms));
          newTransforms.append(transforms);
          newTransformsAttr = b.getArrayAttr(newTransforms);
        }
        useOp->setAttr("transforms", argTransforms.replaceImmediateSubAttribute(
                                         {{argNum, newTransformsAttr}}));
      }
    }

    Value replacement = firstTransform.input();
    lastTransform.output().replaceAllUsesWith(replacement);
    b.replaceOp(lastTransform, {replacement});

    // The stack of transformations will now remove itself by canonicalization
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XdlopsGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct XdlopsGemmV2RewritePattern
    : public OpRewritePattern<miopen::XdlopsGemmV2Op> {
  using OpRewritePattern<miopen::XdlopsGemmV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::XdlopsGemmV2Op op,
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

    XdlopsCodeSelection xcs = XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

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
    // constexpr index_t KRepeats = sizeof(FloatA) / (sizeof(data_type) * mfma_type.k_base);
    // auto pa = reinterpret_cast<const data_type*>(&a);
    // auto pb = reinterpret_cast<const data_type*>(&b);
    // constexpr index_t AStride = K * KRepeats;
    // constexpr index_t BStride = K * KRepeats;

    auto tid = b.create<miopen::WorkitemIdOp>(loc, b.getIndexType());
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
        valueA = ilmkb.create<memref::LoadOp>(loc, dataType, op.matrixA(), sourceOffsetA);
      }
      ilmkb.create<memref::StoreOp>(loc, valueA, op.bufferA(), ValueRange{destOffsetA});

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
        valueB = ilnkb.create<memref::LoadOp>(loc, dataType, op.matrixB(), sourceOffsetB);
      }
      ilnkb.create<memref::StoreOp>(loc, valueB, op.bufferB(), ValueRange{destOffsetB});

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

      Type bufferAElementType =
          op.bufferA().getType().template cast<MemRefType>().getElementType();
      Type bufferBElementType =
          op.bufferB().getType().template cast<MemRefType>().getElementType();
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
        auto mfma = innerLoopb.create<miopen::MFMAV2Op>(loc, vectorType, argA,
                                                        argB, vectorC);

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

      auto NumThreadsBlkConstantOp = b.create<ConstantIndexOp>(loc, num_threads_blk);
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

      auto NumInputBlksConstantOp = b.create<ConstantIndexOp>(loc, num_input_blks);

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
        valueA = lklb.create<memref::LoadOp>(loc, dataType, op.matrixA(), ValueRange{sourceOffsetA});
      }
      lklb.create<memref::StoreOp>(loc, valueA, op.bufferA(), ValueRange{lkliv});

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
        valueB = lklb.create<memref::LoadOp>(loc, dataType, op.matrixB(), ValueRange{sourceOffsetB});
      }
      lklb.create<memref::StoreOp>(loc, valueB, op.bufferB(), ValueRange{lkliv});

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
        auto mfma = innerLoopb.create<miopen::MFMAV2Op>(loc, vectorType, argA,
                                                        argB, vectorC);

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

//===----------------------------------------------------------------------===//
// BlockwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmV2RewritePattern
    : public OpRewritePattern<miopen::BlockwiseGemmV2Op> {
  using OpRewritePattern<miopen::BlockwiseGemmV2Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::BlockwiseGemmV2Op op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    int64_t MPerWave =
        op->getAttr("m_per_wave").template cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op->getAttr("n_per_wave").template cast<IntegerAttr>().getInt();

    // Original C++ logic.
    // static constexpr index_t MRepeats = (GemmMPerWave > 64) ? (GemmMPerWave /
    // 64) : 1; static constexpr index_t NRepeats = (GemmNPerWave > 64) ?
    // (GemmNPerWave / 64) : 1; static constexpr index_t MPerXdlops =
    // (GemmMPerWave > 64) ? 64 : GemmMPerWave; static constexpr index_t
    // NPerXdlops = (GemmNPerWave > 64) ? 64 : GemmNPerWave;

    int64_t MRepeats = (MPerWave > 64) ? (MPerWave / 64) : 1;
    int64_t NRepeats = (NPerWave > 64) ? (NPerWave / 64) : 1;
    int64_t MPerXdlops = (MPerWave > 64) ? 64 : MPerWave;
    int64_t NPerXdlops = (NPerWave > 64) ? 64 : NPerWave;

    if (MRepeats == 1 && NRepeats == 1) {
      SmallVector<Type, 2> resultTypes;
      for (auto result : op.vectorDs()) {
        resultTypes.push_back(result.getType());
      }

      auto xdlopsGemmV2Op = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes, op.matrixA(), op.matrixB(), op.transforms(),
          op.waveOffsetA(), op.waveOffsetB(), op.bufferA(), op.bufferB(),
          op.vectorCs());

      xdlopsGemmV2Op->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op->setAttr("k", op->getAttr("k"));
      xdlopsGemmV2Op->setAttr("m_per_wave", op->getAttr("m_per_wave"));
      xdlopsGemmV2Op->setAttr("n_per_wave", op->getAttr("n_per_wave"));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(xdlopsGemmV2Op.vectorDs());
      op.erase();
    } else if (MRepeats == 2 && NRepeats == 1) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block + MPerXdlops, p_b_block, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.transforms(),
          op.waveOffsetA(), op.waveOffsetB(), op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

      xdlopsGemmV2Op0->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op0->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op0->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op0->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op0->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op0->setAttr("kpack", op->getAttr("kpack"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes1, op.matrixA(), op.matrixB(), op.transforms(),
          b.create<AddIOp>(loc, op.waveOffsetA(), MPerXdlopsConstantOp),
          op.waveOffsetB(), op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
    } else if (MRepeats == 1 && NRepeats == 2) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block,
      // p_b_block, p_c_thread.s.x.l); p_c_thread.s.y.l = XdlopsGemm.template
      // Run<M, N, K>(p_a_block, p_b_block + NPerXdlops, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.transforms(),
          op.waveOffsetA(), op.waveOffsetB(), op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

      xdlopsGemmV2Op0->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op0->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op0->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op0->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op0->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op0->setAttr("kpack", op->getAttr("kpack"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes1, op.matrixA(), op.matrixB(), op.transforms(),
          op.waveOffsetA(),
          b.create<AddIOp>(loc, op.waveOffsetB(), NPerXdlopsConstantOp),
          op.bufferA(), op.bufferB(),
          ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1->setAttr("m", op->getAttr("m"));
      xdlopsGemmV2Op1->setAttr("n", op->getAttr("n"));
      xdlopsGemmV2Op1->setAttr("k", op->getAttr("k"));
      // Hard-coded m_per_wave/n_per_wave as 64 when MRepeat>1 or NRepeat>1.
      // So each xdlops_gemm_v2 handles a 64x64 GEMM.
      xdlopsGemmV2Op1->setAttr("m_per_wave", b.getI32IntegerAttr(64));
      xdlopsGemmV2Op1->setAttr("n_per_wave", b.getI32IntegerAttr(64));
      if (op->hasAttr("kpack"))
        xdlopsGemmV2Op1->setAttr("kpack", op->getAttr("kpack"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
    }

    return success();
  }
};

#endif // MLIR_DIALECT_MIOPEN_LOWERMIOPENOPS_H
