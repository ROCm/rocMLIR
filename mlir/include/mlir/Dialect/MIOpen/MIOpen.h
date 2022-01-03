//===- MIOpenOps.h - MIOpen MLIR Dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines MIOpen memref attributes and operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_MIOPENOPS_OPS_H_
#define MLIR_MIOPENOPS_OPS_H_

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <type_traits>

//===----------------------------------------------------------------------===//
//  MIOpen Dialect
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.h.inc"
#include "mlir/Dialect/MIOpen/MIOpenTypes.h.inc"

template <> struct mlir::FieldParser<mlir::miopen::BwdPaddingKernelInfo> {
  static FailureOr<mlir::miopen::BwdPaddingKernelInfo>
  parse(mlir::AsmParser &parser) {
    std::string bits;
    llvm::SMLoc startLoc = parser.getCurrentLocation();
    if (parser.parseString(&bits)) {
      return failure();
    }
    auto ret = mlir::miopen::getBwdPaddingKernelInfoForBits(bits);
    if (!ret.hasValue()) {
      parser.emitError(startLoc, "Expected a | seperated list of status bits");
      return failure();
    }
    return ret.getValue();
  }
};

namespace mlir {
namespace miopen {
AsmPrinter &operator<<(AsmPrinter &printer, BwdPaddingKernelInfo v);
//===----------------------------------------------------------------------===//
// Utility method for creating an array attribute of n empty array attributes.
// We use this structure so transforms can be uniformly copied onto the final
// user(s) of the transformed value
//
// TODO(kdrewnia) See if this declaration should be elsewhere
//===----------------------------------------------------------------------===//
ArrayAttr noTransformsArray(Builder &b, size_t n);
} // end namespace miopen
} // end namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.h.inc"

namespace mlir {
namespace miopen {
/// Methods for building up coordinate transformations
///
/// A coordinate transformation is a description of how to map a set of upper
/// dimensions P to a set of lower dimensions Q through certain transformaton
/// functions. Each p in P is an input to exactly one transformation function,
/// and each q in Q is similarly the output of one such function.

/// For the methods that work the same whether you're working from the upper
/// shape down or the lower shape up, such as passThrough() or pad(), "start" is
/// the coordinate space you began building from and end is the coordinate space
/// you're building towards (whether that's the upper or lower coordinates)
class CoordTransformsBuilder {
public:
  virtual ~CoordTransformsBuilder() = default;

  // Get the TransformsAttr that's being built up by the builder
  TransformsAttr get();
  // Only valid after the transformation has been built.
  // The names live as long as the CoordTransformBuilder
  void getEndNames(SmallVectorImpl<StringRef> &names);

  SmallString<8> startName(uint32_t dim);
  SmallString<8> endName(uint32_t dim);
  uint32_t startIndex(StringRef name);
  uint32_t endIndex(StringRef name);

  int64_t startSize(StringRef name);
  int64_t startSize(uint32_t dim);
  int64_t endSize(StringRef name);
  int64_t endSize(uint32_t dim);

  void passThrough(StringRef name);
  void passThrough(StringRef outName, StringRef inName);
  void passThrough(ArrayRef<StringRef> outNames, ArrayRef<uint32_t> outDims,
                   ArrayRef<StringRef> inNames);

  // Parameters is the pre and post padding for each dimension in the order they
  // appear as arguments. For example, padding x with 2 on the left and 1 on the
  // right and then y with 1 above and 2 below would lead to pad({"x", "y"}, {2,
  // 1, 1, 2})
  void pad(ArrayRef<StringRef> names, ArrayRef<int64_t> params);
  void pad(StringRef outName, StringRef inName, int64_t left, int64_t right);
  void pad(ArrayRef<StringRef> outNames, ArrayRef<uint32_t> outDims,
           ArrayRef<StringRef> inNames, ArrayRef<int64_t> params);

protected:
  CoordTransformsBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                         ArrayRef<int64_t> startShape, mlir::Location loc);

  template <class T, typename = typename std::enable_if<std::is_base_of<
                         CoordTransformsBuilder, T>::value>::type>
  static T nextTransforms(T &previous, ArrayRef<int64_t> startShape) {
    llvm::SmallVector<StringRef, 8> previousNames;
    previous.getEndNames(previousNames);
    return T(previous.b, previousNames, startShape, previous.loc);
  }

  virtual void addTransform(TransformType type, ArrayRef<int64_t> params,
                            ArrayRef<StringRef> startNames,
                            ArrayRef<uint32_t> startDims,
                            ArrayRef<StringRef> endNames,
                            ArrayRef<uint32_t> endDims) = 0;
  virtual void extractBounds(SmallVectorImpl<int64_t> &upperBounds,
                             SmallVectorImpl<int64_t> &lowerBounds) = 0;

  virtual int64_t paddingSign() const = 0;

  uint32_t nStartDims();
  uint32_t nEndDims();
  void defineDim(StringRef name, uint32_t dim, int64_t size);

  mlir::Builder &b;
  // It's an invariant that these go upper to lower dimensions
  llvm::SmallVector<TransformAttr> result;
  mlir::Location loc;

private:
  llvm::StringMap<uint32_t> startIndices;
  llvm::SmallVector<SmallString<8>, 8> startNames;
  llvm::SmallVector<int64_t, 8> startShape;

  llvm::StringMap<uint32_t> endIndices;
  llvm::SmallMapVector<uint32_t, SmallString<8>, 8> endNames;
  llvm::SmallVector<int64_t, 8> endShape;

  bool frozen = false;
};

class TopDownCTBuilder : public CoordTransformsBuilder {
public:
  TopDownCTBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                   ArrayRef<int64_t> startShape)
      : TopDownCTBuilder(builder, startNames, startShape,
                         builder.getUnknownLoc()) {}

  TopDownCTBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                   ArrayRef<int64_t> startShape, mlir::Location loc)
      : CoordTransformsBuilder(builder, startNames, startShape, loc) {}

  static TopDownCTBuilder below(TopDownCTBuilder &previous,
                                TransformsAttr &result) {
    return CoordTransformsBuilder::nextTransforms(previous,
                                                  result.getLowerBounds());
  }

  // NOTE: There is no  builder for slice() as it isn't used in this context
  // If you find you need one, please add it, bearing in mind
  // that your start dimension is already sliced so you need to pass the full
  // length

  void ignore(StringRef dim);
  void embed(StringRef lowerName, uint32_t lowerDim, int64_t lowerSize,
             ArrayRef<StringRef> upperNames, ArrayRef<int64_t> coefficients);
  void unmerge(StringRef lowerName, uint32_t lowerDim,
               ArrayRef<StringRef> upperNames, ArrayRef<int64_t> lengths);

  void merge(ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims,
             StringRef upperName, ArrayRef<int64_t> sizes,
             bool isUnfold = false);

protected:
  void addTransform(TransformType type, ArrayRef<int64_t> params,
                    ArrayRef<StringRef> startNames,
                    ArrayRef<uint32_t> startDims, ArrayRef<StringRef> endNames,
                    ArrayRef<uint32_t> endDims) override final;
  void extractBounds(SmallVectorImpl<int64_t> &upperDims,
                     SmallVectorImpl<int64_t> &lowerDims) override final;
  int64_t paddingSign() const override final;
};

class BottomUpCTBuilder : public CoordTransformsBuilder {
public:
  BottomUpCTBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                    ArrayRef<int64_t> startShape)
      : BottomUpCTBuilder(builder, startNames, startShape,
                          builder.getUnknownLoc()) {}

  BottomUpCTBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                    ArrayRef<int64_t> startShape, mlir::Location loc)
      : CoordTransformsBuilder(builder, startNames, startShape, loc) {}

  static BottomUpCTBuilder above(BottomUpCTBuilder &previous,
                                 TransformsAttr &result) {
    return CoordTransformsBuilder::nextTransforms(previous,
                                                  result.getUpperBounds());
  }

  // Defines a dimension that is not mapped to any coordinates in the output
  void addDim(StringRef name, uint32_t dim, int64_t size);

  void slice(ArrayRef<StringRef> upperNames, ArrayRef<StringRef> lowerNames,
             ArrayRef<int64_t> begins, ArrayRef<int64_t> ends);

  void embed(ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
             ArrayRef<int64_t> upperSizes, StringRef lowerName,
             ArrayRef<int64_t> coefficients);
  void unmerge(ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
               StringRef lowerName, ArrayRef<int64_t> lengths);

  // The coefficients to the merge will automatically be the size of the lower
  // dimensions
  void merge(StringRef upperName, uint32_t upperDim,
             ArrayRef<StringRef> lowerNames, bool isUnfold = false);

protected:
  void addTransform(TransformType type, ArrayRef<int64_t> params,
                    ArrayRef<StringRef> startNames,
                    ArrayRef<uint32_t> startDims, ArrayRef<StringRef> endNames,
                    ArrayRef<uint32_t> endDims) override final;
  void extractBounds(SmallVectorImpl<int64_t> &upperDims,
                     SmallVectorImpl<int64_t> &lowerDims) override final;
  int64_t paddingSign() const override final;
};

TransformAttr getTransformAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, TransformType type, ArrayRef<int64_t> params,
    ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
    ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims);

TransformsAttr getTransformsAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    ArrayRef<int64_t> upperBounds, ArrayRef<int64_t> lowerBounds);

} // namespace miopen
} // namespace mlir
#endif // MLIR_MIOPENOPS_OPS_H_
