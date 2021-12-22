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
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
//  MIOpen Dialect
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/MIOpen/MIOpenOpsDialect.h.inc"

namespace mlir {

namespace miopen {

enum class TransformType {
  PassThrough,
  Pad,
  Slice,
  Embed,
  Unmerge,
  Merge,
  Unfold
};
llvm::Optional<TransformType> getTransformTypeForName(llvm::StringRef name);
const char *getNameForTransformType(const TransformType);

enum ConvOpType { Conv2DOpType, Conv2DBwdDataOpType, Conv2DBwdWeightOpType };

llvm::Optional<ConvOpType> getConvOpTypeForName(llvm::StringRef name);
const char *getNameForConvOpType(const ConvOpType);

} // end namespace miopen
} // end namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpen/MIOpenOps.h.inc"

namespace mlir {
namespace miopen {
class CoordTransformBuilder {
public:
  enum class Place { Top, Bottom };

  CoordTransformBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                        ArrayRef<int64_t> startShape);
  SmallVector<TransformAttr> get();

  SmallString<8> nameOf(unsigned dim, Place place);
  unsigned indexOf(StringRef name, Place place);
  int64_t sizeOf(StringRef name, Place place);
  int64_t sizeOf(unsigned dim, Place place);
  unsigned nDims(Place place);

  CoordTransformBuilder &passThrough(StringRef name, Place place,
                                     Optional<StringRef> newName);
  CoordTransformBuilder &passThrough(StringRef name, StringRef newName,
                                     unsigned newDim, Place place);

  // Parameters is the pre and post padding for each dimension in the order they
  // appear as arguments
  CoordTransformBuilder &pad(ArrayRef<StringRef> names,
                             ArrayRef<int64_t> parameters, Place place);
  CoordTransformBuilder &pad(ArrayRef<StringRef> newNames,
                             ArrayRef<unsigned> newDims,
                             ArrayRef<StringRef> oldNames,
                             ArrayRef<int64_t> parameters, Place place);

  CoordTransformBuilder &slice(ArrayRef<StringRef> newNames,
                               ArrayRef<StringRef> names,
                               ArrayRef<int64_t> begins, ArrayRef<int64_t> ends,
                               Place place);
  // This builder won't be used anywhere
  // CoordTransformBuilder& slice(ArrayRef<StringRef> newNames,
  // ArrayRef<unsigned> newDims, ArrayRef<StringRef> names, ArrayRef<int64_t>
  // begins, ArrayRef<int64_t> ends, Place place);

  CoordTransformBuilder &embedTop(StringRef newName, unsigned newDim,
                                  ArrayRef<StringRef> components);
  CoordTransformBuilder &embedBottom(ArrayRef<StringRef> newNames,
                                     ArrayRef<unsigned> newDims,
                                     StringRef wasEmbedded,
                                     ArrayRef<int64_t> coefficients);

  CoordTransformBuilder &unmergeTop(StringRef newName, unsigned newDim,
                                    ArrayRef<StringRef> components);
  CoordTransformBuilder &unmergeBottom(ArrayRef<StringRef> newNames,
                                       ArrayRef<unsigned> newDims,
                                       StringRef wasUnmerged,
                                       ArrayRef<int64_t> coefficients);

  CoordTransformBuilder &mergeTop(ArrayRef<StringRef> newNames,
                                  ArrayRef<unsigned> newDims,
                                  StringRef wasMerged,
                                  ArrayRef<int64_t> coefficients,
                                  bool isUnfold = false);
  CoordTransformBuilder &mergeBottom(StringRef newName, unsigned newDim,
                                     ArrayRef<StringRef> willBeMerged,
                                     bool isUnfold = false);

private:
  void dropDim(unsigned dim, Place place);
  void updateDim(unsigned dim, StringRef name, int64_t size, Place place);

  mlir::Builder &b;
  llvm::StringMap<unsigned> upperIndices;
  llvm::SmallVector<SmallString<8>, 8> upperNames;
  llvm::SmallVector<int64_t, 8> upperShape;
  llvm::SmallDenseSet<unsigned> upperUndef;

  llvm::StringMap<unsigned> lowerIndices;
  llvm::SmallVector<SmallString<8>, 8> lowerNames;
  llvm::SmallVector<int64_t, 8> lowerShape;
  llvm::SmallDenseSet<unsigned> lowerUndef;

  llvm::SmallVector<TransformAttr> result;
};
} // namespace miopen
} // namespace mlir
#endif // MLIR_MIOPENOPS_OPS_H_
