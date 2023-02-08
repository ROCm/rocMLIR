//===- TransformMapBuilder.cpp - Rock MLIR Operations
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/IR/Rock.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <iterator>

using namespace mlir;
using namespace mlir::rock;

static AffineMapAttr assembleMapFor(Builder &b,
                                    ArrayRef<TransformAttr> transforms,
                                    ArrayRef<int64_t> upperBounds,
                                    ArrayRef<int64_t> lowerBounds) {
  llvm::SmallMapVector<int64_t, AffineExpr, 8> affExprsMap;
  for (const TransformAttr transform : transforms) {
    TransformType type = transform.getType();
    ArrayRef<uint32_t> upperDims = transform.getUpperDims();
    ArrayRef<uint32_t> lowerDims = transform.getLowerDims();
    ArrayRef<int64_t> params = transform.getParams();
    if (type == TransformType::PassThrough) {
      for (auto pair : llvm::zip(upperDims, lowerDims)) {
        uint32_t upper, lower;
        std::tie(upper, lower) = pair;
        affExprsMap.insert({lower, b.getAffineDimExpr(upper)});
      }
    } else if (type == TransformType::Pad) {
      for (unsigned i = 0, e = upperDims.size(); i < e; ++i) {
        // example of h and w pad parameters [0, 2, 3, 1] :
        // leftpadH = 0 rightPadH = 2 leftpadW = 3 rightPadW = 1
        // first run leftPad = 0 rightPad = 2
        // second run leftPad = 3 rightPad = 1
        // if your pad is one dim , example of pad parameters [1,2]
        // leftPad = 1 rightPad = 2
        int64_t leftPad = params[i * 2];
        uint32_t upperDim = upperDims[i];
        uint32_t lowerDim = lowerDims[i];
        AffineExpr expr =
            b.getAffineDimExpr(upperDim) - b.getAffineConstantExpr(leftPad);
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::Slice) {
      // The params for slice are begin1 end1 begin2 end2... just line on pad
      for (uint32_t i = 0, e = upperDims.size(); i < e; ++i) {
        uint32_t upperDim = upperDims[i];
        uint32_t lowerDim = lowerDims[i];
        int64_t begin = params[i * 2];
        AffineExpr expr =
            b.getAffineDimExpr(upperDim) + b.getAffineConstantExpr(begin);
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::Embed) {
      ArrayRef<int64_t> coefficients = params;
      uint32_t lowerDim = lowerDims[0];
      AffineExpr expr = b.getAffineConstantExpr(0);
      for (auto pair : llvm::zip(upperDims, coefficients)) {
        uint32_t upperDim;
        int64_t coefficient;
        std::tie(upperDim, coefficient) = pair;
        expr = expr + (b.getAffineDimExpr(upperDim) *
                       b.getAffineConstantExpr(coefficient));
      }
      affExprsMap.insert({lowerDim, expr});
    } else if (type == TransformType::Unmerge) {
      ArrayRef<int64_t> lengths = params;
      AffineExpr expr = b.getAffineDimExpr(upperDims[0]);
      for (auto pair : llvm::zip(upperDims.slice(1), lengths.slice(1))) {
        uint32_t upperDim;
        int64_t length;
        std::tie(upperDim, length) = pair;
        expr = expr * b.getAffineConstantExpr(length) +
               b.getAffineDimExpr(upperDim);
      }
      affExprsMap.insert({lowerDims[0], expr});
    } else if (type == TransformType::Merge) {
      // Compute lower dimension strides.
      llvm::SmallVector<int64_t, 4> lowerDimStrides;
      int64_t totalStride = 1;
      lowerDimStrides.push_back(totalStride);
      for (unsigned i = params.size() - 1; i > 0; --i) {
        totalStride *= params[i];
        lowerDimStrides.push_back(totalStride);
      }
      totalStride *= params[0];
      std::reverse(lowerDimStrides.begin(), lowerDimStrides.end());

      // Build affine transformation expressions.
      AffineExpr remainder = b.getAffineDimExpr(upperDims[0]);
      for (uint32_t i = 0, e = lowerDims.size(); i < e; ++i) {
        // If the constant we're about to divide by is the same as the total
        // stride in the input dimension, output 0, as, if you're above
        // said stride, the result of this merge is undefined behavior.
        if (lowerDimStrides[i] == totalStride) {
          AffineExpr thisDim = b.getAffineConstantExpr(0);
          affExprsMap.insert({lowerDims[i], thisDim});
          continue;
        }
        // While, in general, (x mod N) floordiv N is not 0, because
        // for x < 0 it is instead -1, in our context, negative coordinates
        // are never produced within an indexing process, so we can make
        // that simplification.
        if (i > 0 && lowerDimStrides[i] == lowerDimStrides[i - 1]) {
          AffineExpr thisDim = b.getAffineConstantExpr(0);
          affExprsMap.insert({lowerDims[i], thisDim});
          continue;
        }
        AffineExpr stride = b.getAffineConstantExpr(lowerDimStrides[i]);
        AffineExpr thisDim = remainder.floorDiv(stride);
        remainder = remainder % stride;
        affExprsMap.insert({lowerDims[i], thisDim});
      }
    } else if (type == TransformType::AddDim) {
      assert(upperDims.size() == 1 && lowerDims.size() == 0 &&
             "Invalid AddDim");
      // dimension is ignored, do nothing
    } else if (type == TransformType::Broadcast) {
      // Compute lower dimension strides.
      for (auto tuple : llvm::zip(params, lowerDims, upperDims)) {
        int64_t param = std::get<0>(tuple);
        uint32_t lowerDim = std::get<1>(tuple);
        uint32_t upperDim = std::get<2>(tuple);
        AffineExpr expr =
            b.getAffineDimExpr(upperDim) % b.getAffineConstantExpr(param);
        affExprsMap.insert({lowerDim, expr});
      }
    } else if (type == TransformType::ConstDim) {
      for (unsigned i = 0, e = lowerDims.size(); i < e; ++i) {
        uint32_t lowerDim = lowerDims[i];
        int64_t constant = params[2 * i];
        AffineExpr expr = b.getAffineConstantExpr(constant);
        affExprsMap.insert({lowerDim, expr});
      }
    } else {
      llvm_unreachable("Handled all the cases in affine map building");
    }
  }

  llvm::SmallVector<AffineExpr, 8> affExprsVec;
  affExprsVec.reserve(affExprsMap.size());
  for (uint32_t i = 0, e = lowerBounds.size(); i < e; ++i) {
    assert(affExprsMap.count(i) == 1 &&
           "Lower dimension must have associated output expression");
    affExprsVec.push_back(affExprsMap[i]);
  }
  AffineMap ret =
      AffineMap::get(upperBounds.size(), 0, affExprsVec, b.getContext());
  return AffineMapAttr::get(ret);
}

/// Builder for when we know what we're doing.
TransformMapAttr TransformMapAttr::get(ArrayRef<TransformAttr> transforms,
                                       ArrayRef<int64_t> upperBounds,
                                       ArrayRef<int64_t> lowerBounds) {
  assert(!transforms.empty() && "This builder does not support the empty map");
  Builder b(transforms.front().getContext());
  AffineMapAttr map = assembleMapFor(b, transforms, upperBounds, lowerBounds);
  return TransformMapAttr::get(map.getContext(), transforms, map,
                               b.getDenseI64ArrayAttr(upperBounds),
                               b.getDenseI64ArrayAttr(lowerBounds));
}

/// Accessors and common infrastructure

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<StringRef> startNamesArg,
                                         ArrayRef<int64_t> startShapeArg,
                                         mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  assert(startNamesArg.size() == startShapeArg.size() &&
         "Start names and shape must have the same size");
  for (auto pair : llvm::enumerate(startNamesArg)) {
    uint32_t index = pair.index();
    StringRef value = pair.value();

    startNames.push_back(value);
    startIndices.insert_or_assign(value, index);
    startShape.push_back(startShapeArg[index]);
  }
}

TransformMapBuilder::TransformMapBuilder(mlir::Builder &builder,
                                         ArrayRef<int64_t> startShapeArg,
                                         mlir::Location loc)
    : b(builder), result(), loc(loc), startIndices(), startNames(),
      startShape(), endIndices(), endNames(), endShape() {
  for (auto pair : llvm::enumerate(startShapeArg)) {
    uint32_t index = pair.index();
    int64_t value = pair.value();

    SmallString<8> name;
    ("dim" + Twine(index)).toVector(name);

    startNames.push_back(name);
    startIndices.insert_or_assign(startNames.back(), index);

    startShape.push_back(value);
  }
}

TransformMapAttr TransformMapBuilder::get() {
  SmallVector<int64_t, 8> upperBounds, lowerBounds;
  extractBounds(upperBounds, lowerBounds);
  AffineMapAttr map = assembleMapFor(b, result, upperBounds, lowerBounds);
  auto errorEmitter = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error assembling transform map: ");
    if (b.getContext()->shouldPrintOpOnDiagnostic()) {
      err.attachNote(loc).append("The transforms were").appendRange(result);
    }
    return err;
  };
  frozen = true;
  return getTransformMapAttrChecked(errorEmitter, b.getContext(), result, map,
                                    b.getDenseI64ArrayAttr(upperBounds),
                                    b.getDenseI64ArrayAttr(lowerBounds));
}

void TransformMapBuilder::getEndNames(SmallVectorImpl<StringRef> &names) {
  uint32_t e = nEndDims();
  names.reserve(e);
  for (uint32_t i = 0; i < e; ++i) {
    names.emplace_back(endNames[i]);
  }
}

void TransformMapBuilder::getStartNames(SmallVectorImpl<StringRef> &names) {
  names.reserve(startNames.size());
  for (const auto &name : startNames) {
    names.emplace_back(name);
  }
}

StringRef TransformMapBuilder::startName(uint32_t dim) {
  return startNames[dim];
}

StringRef TransformMapBuilder::endName(uint32_t dim) {
  assert(endNames.count(dim) == 1 &&
         "Dimension not defined in ending dimension space");
  return endNames[dim];
}

uint32_t TransformMapBuilder::startIndex(StringRef name) {
  assert(startIndices.count(name) == 1 && "Key not in starting set of names");
  return startIndices[name];
}

uint32_t TransformMapBuilder::endIndex(StringRef name) {
  assert(endIndices.count(name) == 1 &&
         "Key has not yet been defined in the ending set of names");
  return endIndices[name];
}

int64_t TransformMapBuilder::startSize(StringRef name) {
  return startShape[startIndices[name]];
}

int64_t TransformMapBuilder::startSize(uint32_t dim) { return startShape[dim]; }

int64_t TransformMapBuilder::endSize(StringRef name) {
  return endShape[endIndices[name]];
}

uint32_t TransformMapBuilder::nStartDims() { return startShape.size(); }

uint32_t TransformMapBuilder::nEndDims() { return endShape.size(); }

int64_t TransformMapBuilder::endSize(uint32_t dim) { return endShape[dim]; }

void TransformMapBuilder::defineDim(StringRef name, uint32_t dim,
                                    int64_t size) {
  assert(!frozen && "It's a bug to add to a coordinate transform after "
                    "fetching the attribute");
  bool nameInsertResult = endIndices.insert({name, dim}).second;
  assert(nameInsertResult &&
         "Trying to redife a result name in a coordinate transformation");
  SmallString<8> nameCopy = name;
  bool dimInsertResult = endNames.insert({dim, nameCopy}).second;
  assert(dimInsertResult &&
         "Trying to redefine a result dimension in a coordinate transform");
  for (uint32_t e = endShape.size(); e <= dim; ++e) {
    endShape.push_back(0);
  }
  endShape[dim] = size;
}

/// Transformations that work basically the same in either direction
void TransformMapBuilder::passThrough(StringRef name) {
  uint32_t dim = startIndex(name);
  int64_t size = startSize(dim);
  defineDim(name, dim, size);
  addTransform(TransformType::PassThrough, {}, {name}, {dim}, {name}, {dim});
}

void TransformMapBuilder::passThrough(StringRef outName, StringRef inName) {
  uint32_t dim = startIndex(inName);
  int64_t size = startSize(dim);
  defineDim(outName, dim, size);
  addTransform(TransformType::PassThrough, {}, {inName}, {dim}, {outName},
               {dim});
}

void TransformMapBuilder::passThrough(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> dims;
  llvm::SmallVector<uint32_t> sizes;
  dims.reserve(names.size());
  sizes.reserve(names.size());
  for (const auto name : names) {
    uint32_t dim = startIndex(name);
    dims.push_back(dim);
    sizes.push_back(startSize(dim));
  }
  for (uint32_t i = 0, e = names.size(); i < e; ++i) {
    defineDim(names[i], dims[i], sizes[i]);
  }
  addTransform(TransformType::PassThrough, {}, names, dims, names, dims);
}

void TransformMapBuilder::passThrough(ArrayRef<StringRef> outNames,
                                      ArrayRef<uint32_t> outDims,
                                      ArrayRef<StringRef> inNames) {
  assert(outNames.size() == inNames.size() && "One output per input");
  assert(outNames.size() == outDims.size() && "One location per output");

  llvm::SmallVector<uint32_t> inDims;
  llvm::SmallVector<uint32_t> inSizes;
  inDims.reserve(inNames.size());
  inSizes.reserve(inNames.size());
  for (const auto name : inNames) {
    uint32_t dim = startIndex(name);
    inDims.push_back(dim);
    inSizes.push_back(startSize(dim));
  }
  for (uint32_t i = 0, e = outNames.size(); i < e; ++i) {
    defineDim(outNames[i], outDims[i], inSizes[i]);
  }
  addTransform(TransformType::PassThrough, {}, inNames, inDims, outNames,
               outDims);
}

void TransformMapBuilder::passThrough(ArrayRef<uint32_t> endIndices,
                                      ArrayRef<uint32_t> startIndices) {
  assert(endIndices.size() == startIndices.size() && "One output per input");

  llvm::SmallVector<StringRef> names;
  names.reserve(endIndices.size());
  for (auto tuple : llvm::zip(endIndices, startIndices)) {
    uint32_t index = std::get<1>(tuple);
    StringRef name = startNames[index];
    names.push_back(name);
    defineDim(name, std::get<0>(tuple), startSize(index));
  }
  addTransform(TransformType::PassThrough, {}, names, startIndices, names,
               endIndices);
}

void TransformMapBuilder::pad(ArrayRef<StringRef> names,
                              ArrayRef<int64_t> params) {
  llvm::SmallVector<uint32_t, 8> dims;
  dims.reserve(names.size());
  std::transform(names.begin(), names.end(), std::back_inserter(dims),
                 [&](StringRef s) -> uint32_t { return startIndex(s); });
  pad(names, dims, names, params);
}

void TransformMapBuilder::pad(StringRef outName, StringRef inName, int64_t left,
                              int64_t right) {
  uint32_t dim = startIndex(inName);
  SmallVector<int64_t, 2> params = {left, right};
  pad({outName}, {dim}, {inName}, params);
}

void TransformMapBuilder::pad(ArrayRef<StringRef> outNames,
                              ArrayRef<uint32_t> outDims,
                              ArrayRef<StringRef> inNames,
                              ArrayRef<int64_t> params) {
  assert(outNames.size() == outDims.size() &&
         "One name needed per dimension in padding");
  assert(outNames.size() == inNames.size() &&
         "Same number of output and input dimensions");
  assert(params.size() == 2 * outNames.size() &&
         "Two padding parameters given per dimension");
  llvm::SmallVector<uint32_t, 8> inDims;
  inDims.reserve(inNames.size());
  std::transform(inNames.begin(), inNames.end(), std::back_inserter(inDims),
                 [&](StringRef s) { return startIndex(s); });
  int64_t padSign = paddingSign();
  for (uint32_t i = 0, e = outNames.size(); i < e; ++i) {
    int64_t leftPad = params[i * 2];
    int64_t rightPad = params[i * 2 + 1];
    int64_t outSize =
        startSize(inDims[i]) + (padSign * leftPad) + (padSign * rightPad);
    defineDim(outNames[i], outDims[i], outSize);
  }
  addTransform(TransformType::Pad, params, inNames, inDims, outNames, outDims);
}

TransformMapBuilder &
TransformMapBuilder::operator=(const TransformMapBuilder &other) {
  if (this != &other) {
    b = other.b;
    result = other.result;
    loc = other.loc;

    startNames = other.startNames;
    startShape = other.startShape;
    endNames = other.endNames;
    endShape = other.endShape;
    frozen = other.frozen;

    startIndices.clear();
    for (uint32_t i = 0, e = startNames.size(); i < e; ++i)
      startIndices.insert({StringRef(startNames[i]), i});

    endIndices.clear();
    for (const auto &pair : endNames)
      endIndices.insert({StringRef(pair.second), pair.first});
  }
  return *this;
}

TransformMapBuilder::TransformMapBuilder(const TransformMapBuilder &other)
    : b(other.b), result(other.result), loc(other.loc), startIndices(),
      startNames(other.startNames), startShape(other.startShape), endIndices(),
      endNames(other.endNames), endShape(other.endShape), frozen(other.frozen) {
  for (uint32_t i = 0, e = startNames.size(); i < e; ++i)
    startIndices.insert({StringRef(startNames[i]), i});
  for (const auto &pair : endNames)
    endIndices.insert({StringRef(pair.second), pair.first});
}

/// Building from a defined set of upper dimensions
void TopDownTMBuilder::addTransform(TransformType type,
                                    ArrayRef<int64_t> params,
                                    ArrayRef<StringRef> startNames,
                                    ArrayRef<uint32_t> startDims,
                                    ArrayRef<StringRef> endNames,
                                    ArrayRef<uint32_t> endDims) {
  auto emitError = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error constructing coordinate transformation: ");
    err.attachNote(loc)
        .append("The operation type was ")
        .append(getNameForTransformType(type))
        .append("\n  Upper dimensions =")
        .appendRange(startNames)
        .append(" at ")
        .appendRange(startDims)
        .append("\n  Lower dimensions = ")
        .appendRange(endNames)
        .append(" at ")
        .appendRange(endDims)
        .append("\n  Parameters = ")
        .appendRange(params);
    return err;
  };
  TransformAttr attr =
      getTransformAttrChecked(emitError, b.getContext(), type, params,
                              startNames, startDims, endNames, endDims);
  if (!attr) {
    return;
  }
  result.push_back(attr);
}

void TopDownTMBuilder::extractBounds(SmallVectorImpl<int64_t> &upperBounds,
                                     SmallVectorImpl<int64_t> &lowerBounds) {
  uint32_t nStart = nStartDims(), nEnd = nEndDims();
  upperBounds.reserve(nStart);
  lowerBounds.reserve(nEnd);
  for (uint32_t i = 0; i < nStart; ++i) {
    upperBounds.push_back(startSize(i));
  }
  for (uint32_t i = 0; i < nEnd; ++i) {
    lowerBounds.push_back(endSize(i));
  }
}

int64_t TopDownTMBuilder::paddingSign() const {
  // When building top-down, the output size (lower dimension) is the input size
  // (upper dimension) minus padding
  return -1;
}

void TopDownTMBuilder::ignore(StringRef name) {
  uint32_t dim = startIndex(name);
  int64_t size = startSize(dim);
  addTransform(TransformType::AddDim, {size}, {name}, {dim}, {}, {});
}

void TopDownTMBuilder::constDim(StringRef lowerName, uint32_t lowerDim,
                                int64_t constantVal, int64_t lowerSize) {
  defineDim(lowerName, lowerDim, lowerSize);
  SmallVector<int64_t> params = {constantVal, lowerSize};
  addTransform(TransformType::ConstDim, params, {}, {}, {lowerName},
               {lowerDim});
}

void TopDownTMBuilder::constDim(ArrayRef<StringRef> lowerNames,
                                ArrayRef<uint32_t> lowerDims,
                                ArrayRef<int64_t> constantVals,
                                ArrayRef<int64_t> lowerSizes) {
  assert(constantVals.size() == lowerSizes.size() &&
         "must have equal number of constant values and dimension lengths");
  SmallVector<int64_t> params;
  params.reserve(2 * constantVals.size());
  for (const auto &[name, dim, val, size] :
       llvm::zip(lowerNames, lowerDims, constantVals, lowerSizes)) {
    params.emplace_back(val);
    params.emplace_back(size);
    defineDim(name, dim, size);
  }
  addTransform(TransformType::ConstDim, params, {}, {}, lowerNames, lowerDims);
}

void TopDownTMBuilder::embed(StringRef lowerName, uint32_t lowerDim,
                             int64_t lowerSize, ArrayRef<StringRef> upperNames,
                             ArrayRef<int64_t> coefficients) {
  assert(upperNames.size() == coefficients.size() &&
         "Must provide a coefficient for each dimension");
  SmallVector<uint32_t, 8> upperDims;
  upperDims.reserve(upperNames.size());
  for (const StringRef name : upperNames) {
    upperDims.push_back(startIndex(name));
  }

  defineDim(lowerName, lowerDim, lowerSize);
  addTransform(TransformType::Embed, coefficients, upperNames, upperDims,
               {lowerName}, {lowerDim});
}

void TopDownTMBuilder::unmerge(StringRef lowerName, uint32_t lowerDim,
                               ArrayRef<StringRef> upperNames,
                               ArrayRef<int64_t> lengths) {
  assert(upperNames.size() == lengths.size() &&
         "Must provide a length for each dimension");
  SmallVector<uint32_t, 8> upperDims;
  upperDims.reserve(upperNames.size());
  for (const StringRef name : upperNames) {
    upperDims.push_back(startIndex(name));
  }
  int64_t size = 1;
  for (auto length : lengths) {
    size *= length;
  }
  defineDim(lowerName, lowerDim, size);
  addTransform(TransformType::Unmerge, lengths, upperNames, upperDims,
               {lowerName}, {lowerDim});
}

void TopDownTMBuilder::merge(ArrayRef<StringRef> lowerNames,
                             ArrayRef<uint32_t> lowerDims, StringRef upperName,
                             ArrayRef<int64_t> sizes) {
  assert(lowerNames.size() == lowerDims.size() &&
         "One name per dimension required in merge");
  assert(lowerDims.size() == sizes.size() &&
         "One size per output dimension required in merge");

  uint32_t upperDim = startIndex(upperName);
  int64_t upperSize = startSize(upperDim);

  int64_t totalLowerSize = 1;
  for (const int64_t s : sizes) {
    totalLowerSize *= s;
  }
  assert(upperSize == totalLowerSize &&
         "Upper dimension to merge must have same size as combined lower "
         "dimensions");
  for (auto triple : llvm::zip(lowerNames, lowerDims, sizes)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Merge, sizes, {upperName}, {upperDim}, lowerNames,
               lowerDims);
}

llvm::SmallVector<uint32_t>
TopDownTMBottomDimsWrapper::toBottomDims(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> ret;
  ret.reserve(names.size());
  for (auto name : names) {
    ret.push_back(bottomDims[name]);
  }
  return ret;
}

void TopDownTMBottomDimsWrapper::passThrough(StringRef name) {
  b.passThrough(name, bottomDims[name], name);
}

void TopDownTMBottomDimsWrapper::passThrough(ArrayRef<StringRef> names) {
  b.passThrough(names, toBottomDims(names), names);
}

void TopDownTMBottomDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                     ArrayRef<StringRef> inNames,
                                     ArrayRef<int64_t> params) {
  b.pad(outNames, toBottomDims(outNames), inNames, params);
}

void TopDownTMBottomDimsWrapper::constDim(StringRef lowerName,
                                          int64_t constantVal,
                                          int64_t lowerSize) {
  b.constDim(lowerName, bottomDims[lowerName], constantVal, lowerSize);
}

void TopDownTMBottomDimsWrapper::constDim(ArrayRef<StringRef> lowerNames,
                                          ArrayRef<int64_t> constantVals,
                                          ArrayRef<int64_t> lowerSizes) {
  b.constDim(lowerNames, toBottomDims(lowerNames), constantVals, lowerSizes);
}

void TopDownTMBottomDimsWrapper::embed(StringRef lowerName, int64_t lowerSize,
                                       ArrayRef<StringRef> upperNames,
                                       ArrayRef<int64_t> coefficients) {
  b.embed(lowerName, bottomDims[lowerName], lowerSize, upperNames,
          coefficients);
}

void TopDownTMBottomDimsWrapper::unmerge(StringRef lowerName,
                                         ArrayRef<StringRef> upperNames,
                                         ArrayRef<int64_t> lengths) {
  b.unmerge(lowerName, bottomDims[lowerName], upperNames, lengths);
}

void TopDownTMBottomDimsWrapper::merge(ArrayRef<StringRef> lowerNames,
                                       StringRef upperName,
                                       ArrayRef<int64_t> sizes) {
  b.merge(lowerNames, toBottomDims(lowerNames), upperName, sizes);
}

/// Building from a defined set of lower dimensions
void BottomUpTMBuilder::addTransform(TransformType type,
                                     ArrayRef<int64_t> params,
                                     ArrayRef<StringRef> startNames,
                                     ArrayRef<uint32_t> startDims,
                                     ArrayRef<StringRef> endNames,
                                     ArrayRef<uint32_t> endDims) {
  auto emitError = [&]() -> InFlightDiagnostic {
    InFlightDiagnostic err =
        mlir::emitError(loc, "Error constructing coordinate transformation: ");
    err.attachNote(loc)
        .append("The operation type was ")
        .append(getNameForTransformType(type))
        .append("\n  Upper dimensions =")
        .appendRange(endNames)
        .append(" at ")
        .appendRange(endDims)
        .append("\n  Lower dimensions = ")
        .appendRange(startNames)
        .append(" at ")
        .appendRange(startDims)
        .append("\n  Parameters = ")
        .appendRange(params);
    return err;
  };
  TransformAttr attr =
      getTransformAttrChecked(emitError, b.getContext(), type, params, endNames,
                              endDims, startNames, startDims);
  if (!attr) {
    return;
  }
  result.push_back(attr);
}

void BottomUpTMBuilder::extractBounds(SmallVectorImpl<int64_t> &upperBounds,
                                      SmallVectorImpl<int64_t> &lowerBounds) {
  uint32_t nStart = nStartDims(), nEnd = nEndDims();
  upperBounds.reserve(nEnd);
  lowerBounds.reserve(nStart);
  for (uint32_t i = 0; i < nEnd; ++i) {
    upperBounds.push_back(endSize(i));
  }
  for (uint32_t i = 0; i < nStart; ++i) {
    lowerBounds.push_back(startSize(i));
  }
}

int64_t BottomUpTMBuilder::paddingSign() const {
  // When building bottom-up, the output size (upper dimension) is the input
  // size (bottom dimension) plus padding
  return 1;
}

void BottomUpTMBuilder::addDim(StringRef name, uint32_t dim, int64_t size) {
  defineDim(name, dim, size);
  addTransform(TransformType::AddDim, {size}, {}, {}, {name}, {dim});
}

void BottomUpTMBuilder::broadcast(ArrayRef<uint32_t> endDims,
                                  ArrayRef<int64_t> endSizes) {
  SmallVector<int64_t, 8> params;
  SmallVector<StringRef, 8> lowerNames;
  SmallVector<StringRef, 8> upperNames;
  for (auto tuple : llvm::zip(endDims, endSizes)) {
    uint32_t dim = std::get<0>(tuple);
    int64_t size = std::get<1>(tuple);
    auto name = startName(dim);
    params.push_back(startSize(dim));
    lowerNames.push_back(name);
    upperNames.push_back(name);
    defineDim(name, dim, size);
  }
  addTransform(TransformType::Broadcast, params, upperNames, endDims,
               lowerNames, endDims);
}

void BottomUpTMBuilder::slice(ArrayRef<StringRef> upperNames,
                              ArrayRef<StringRef> lowerNames,
                              ArrayRef<int64_t> begins,
                              ArrayRef<int64_t> ends) {
  assert(upperNames.size() == lowerNames.size() &&
         "Need same number of input and output dimensions in slice");
  assert(upperNames.size() == begins.size() &&
         "Need beginning of slice for each dimension");
  assert(upperNames.size() == ends.size() &&
         "Need end of slice for each dimension");

  uint32_t n = lowerNames.size();
  SmallVector<uint32_t, 4> dims;
  dims.reserve(n);

  SmallVector<int64_t, 8> params;
  params.reserve(2 * n);

  for (uint32_t i = 0; i < n; ++i) {
    uint32_t dim = startIndex(lowerNames[i]);
    dims.push_back(dim);
    int64_t begin = begins[i];
    int64_t end = ends[i];
    defineDim(upperNames[i], dim, end - begin);
    params.push_back(begin);
    params.push_back(end);
  }
  addTransform(TransformType::Slice, params, lowerNames, dims, upperNames,
               dims);
}

void BottomUpTMBuilder::embed(ArrayRef<StringRef> upperNames,
                              ArrayRef<uint32_t> upperDims,
                              ArrayRef<int64_t> upperSizes, StringRef lowerName,
                              ArrayRef<int64_t> coefficients) {
  assert(upperNames.size() == upperDims.size() &&
         "One name per upper dimension needed in merge");
  assert(upperDims.size() == coefficients.size() &&
         "One coefficient per upper dimension needed in merge");
  assert(upperDims.size() == upperSizes.size() &&
         "One size per upper dimension needed in merge");

  uint32_t lowerDim = startIndex(lowerName);
  for (auto triple : llvm::zip(upperNames, upperDims, upperSizes)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Embed, coefficients, {lowerName}, {lowerDim},
               upperNames, upperDims);
}

void BottomUpTMBuilder::unmerge(ArrayRef<StringRef> upperNames,
                                ArrayRef<uint32_t> upperDims,
                                StringRef lowerName,
                                ArrayRef<int64_t> lengths) {
  assert(upperNames.size() == upperDims.size() &&
         "One name needed per upper dimension in unmerge");
  assert(upperDims.size() == lengths.size() &&
         "One length needed per upper dimension in unmerge");

  uint32_t lowerDim = startIndex(lowerName);

  int64_t totalLength = startSize(lowerDim);
  int64_t lengthsProd = 1;
  for (int64_t length : lengths) {
    lengthsProd *= length;
  }
  assert(lengthsProd == totalLength &&
         "failed to partition unmerge length among upper dimensions");

  for (auto triple : llvm::zip(upperNames, upperDims, lengths)) {
    defineDim(std::get<0>(triple), std::get<1>(triple), std::get<2>(triple));
  }
  addTransform(TransformType::Unmerge, lengths, {lowerName}, {lowerDim},
               upperNames, upperDims);
}

void BottomUpTMBuilder::merge(StringRef upperName, uint32_t upperDim,
                              ArrayRef<StringRef> lowerNames) {
  uint32_t n = lowerNames.size();
  llvm::SmallVector<uint32_t, 4> lowerDims;
  lowerDims.reserve(n);
  llvm::SmallVector<int64_t, 4> lowerSizes;
  lowerSizes.reserve(n);

  int64_t upperSize = 1;
  for (const StringRef name : lowerNames) {
    uint32_t dim = startIndex(name);
    int64_t size = startSize(dim);
    upperSize *= size;
    lowerDims.push_back(dim);
    lowerSizes.push_back(size);
  }
  defineDim(upperName, upperDim, upperSize);
  addTransform(TransformType::Merge, lowerSizes, lowerNames, lowerDims,
               {upperName}, {upperDim});
}

void BottomUpTMTopDimsWrapper::passThrough(StringRef name) {
  b.passThrough({name}, {topDims[name]}, {name});
}

void BottomUpTMTopDimsWrapper::passThrough(ArrayRef<StringRef> names) {
  b.passThrough(names, toTopDims(names), names);
}

void BottomUpTMTopDimsWrapper::pad(ArrayRef<StringRef> outNames,
                                   ArrayRef<StringRef> inNames,
                                   ArrayRef<int64_t> params) {
  b.pad(outNames, toTopDims(outNames), inNames, params);
}

void BottomUpTMTopDimsWrapper::addDim(StringRef name, int64_t size) {
  b.addDim(name, topDims[name], size);
}

void BottomUpTMTopDimsWrapper::embed(ArrayRef<StringRef> upperNames,
                                     ArrayRef<int64_t> upperSizes,
                                     StringRef lowerName,
                                     ArrayRef<int64_t> coefficients) {
  b.embed(upperNames, toTopDims(upperNames), upperSizes, lowerName,
          coefficients);
}

void BottomUpTMTopDimsWrapper::unmerge(ArrayRef<StringRef> upperNames,
                                       StringRef lowerName,
                                       ArrayRef<int64_t> lengths) {
  b.unmerge(upperNames, toTopDims(upperNames), lowerName, lengths);
}

void BottomUpTMTopDimsWrapper::merge(StringRef upperName,
                                     ArrayRef<StringRef> lowerNames) {
  b.merge(upperName, topDims[upperName], lowerNames);
}

llvm::SmallVector<uint32_t>
BottomUpTMTopDimsWrapper::toTopDims(ArrayRef<StringRef> names) {
  llvm::SmallVector<uint32_t> ret;
  ret.reserve(names.size());
  for (auto name : names) {
    ret.push_back(topDims[name]);
  }
  return ret;
}

/// Utility methods

llvm::StringMap<uint32_t> mlir::rock::expandNamesInPlace(
    ArrayRef<StringRef> original,
    const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  uint32_t offset = 0;
  llvm::StringMap<uint32_t> ret;
  for (auto pair : llvm::enumerate(original)) {
    uint32_t origIndex = pair.index();
    StringRef origName = pair.value();
    if (expansion.count(origName) != 0) {
      for (auto newName : (*expansion.find(origName)).getValue()) {
        bool insertResult = ret.insert({newName, origIndex + offset}).second;
        assert(insertResult && "Duplicate dimension in dimension expansion");
        offset++;
      }
      offset--; // Handle extra count and dropping a dimension
    } else {
      bool insertResult = ret.insert({origName, origIndex + offset}).second;
      assert(insertResult && "Dimsion already defined by expansion");
    }
  }
  return ret;
}

llvm::StringMap<uint32_t> mlir::rock::expandNamesInPlace(
    TransformMapBuilder &builder,
    const llvm::StringMap<SmallVector<StringRef, 2>> expansion) {
  SmallVector<StringRef, 8> names;
  builder.getEndNames(names);
  return expandNamesInPlace(names, expansion);
}
