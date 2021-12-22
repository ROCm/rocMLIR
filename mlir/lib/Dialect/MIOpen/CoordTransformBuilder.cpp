//===- CoordTransformBuilder.cpp - MIOpen MLIR Operations
//-----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpen/MIOpen.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

using Place = mlir::miopen::CoordTransformBuilder::Place;

static AffineMapAttr getIdentityExcept(mlir::Builder &b, unsigned n, unsigned d,
                                       AffineExpr expr) {
  llvm::SmallVector<AffineExpr> exprs;
  for (unsigned i = 0; i < n; ++i) {
    if (i == d) {
      exprs.push_back(expr);
    } else {
      exprs.push_back(b.getAffineDimExpr(i));
    }
  }
  return AffineMapAttr::get(AffineMap::get(n, 0, exprs, b.getContext()));
}

static mlir::miopen::TransformAttr
getTransformAttr(mlir::Builder &b, mlir::miopen::TransformType type,
                 ArrayRef<int64_t> params, ArrayRef<StringRef> newNames,
                 ArrayRef<unsigned> newDims, ArrayRef<StringRef> oldNames,
                 ArrayRef<unsigned> oldDims, AffineMapAttr map, Place place) {

  if (place == Place::Top) {
    return miopen::TransformAttr::get(b.getContext(), type, params, newNames,
                                      newDims, oldNames, oldDims, map);
  }
  return miopen::TransformAttr::get(b.getContext(), type, params, oldNames,
                                    oldDims, newNames, newDims, map);
}

template <typename T, typename U>
inline static void addAtPlace(U &container, T &&elem, Place place) {
  if (place == Place::Top) {
    container.insert(0, elem);
  } else {
    container.push_back(elem);
  }
}

namespace mlir {
namespace miopen {

CoordTransformBuilder::CoordTransformBuilder(Builder &b,
                                             ArrayRef<StringRef> startNames,
                                             ArrayRef<int64_t> startShape)
    : b(b), upperIndices(), upperNames(), upperShape(), upperUndef(),
      lowerIndices(), lowerNames(), lowerShape(), lowerUndef(), result() {
  assert(startNames.size() == startShape.size() &&
         "No 1-1 map between shape elements and names");
  for (auto pair : llvm::enumerate(startNames)) {
    unsigned idx = pair.index();
    StringRef name = pair.value();

    upperIndices.insert_or_assign(name, idx);
    lowerIndices.insert_or_assign(name, idx);

    upperNames[idx] = name;
    lowerNames[idx] = name;
  }
}

void CoordTransformBuilder::dropDim(unsigned dim, Place place) {
  auto &indices = place == Place::Top ? upperIndices : lowerIndices;
  auto &names = place == Place::Top ? upperNames : lowerNames;
  auto &shape = place == Place::Top ? upperShape : lowerShape;
  auto &undef = place == Place::Top ? upperUndef : lowerUndef;

  if (dim == names.size() - 1) {
    undef.erase(dim);
    shape.pop_back();
    auto name = names.pop_back_val();
    indices.erase(name);

    if (dim > 0) {
      // Clean up dimensions that also need to be removed here
      for (unsigned probeDown = dim - 1; undef.contains(probeDown);
           --probeDown) {
        undef.erase(probeDown);
        shape.pop_back();
        auto name = names.pop_back_val();
        indices.erase(name);
      }
    }
  } else {
    // Mark this dimension undefined
    undef.insert(dim);
    shape[dim] = 0;
  }
}

void CoordTransformBuilder::updateDim(unsigned dim, StringRef name,
                                      int64_t size, Place place) {
  auto &indices = place == Place::Top ? upperIndices : lowerIndices;
  auto &names = place == Place::Top ? upperNames : lowerNames;
  auto &shape = place == Place::Top ? upperShape : lowerShape;
  auto &undef = place == Place::Top ? upperUndef : lowerUndef;

  for (unsigned e = names.size(); e < dim; ++e) {
    undef.insert(e);
    names.push_back(SmallString<8>{"[UNDEF]"});
    shape.push_back(0);
  }
  if (dim == names.size()) {
    names.push_back(name);
    shape.push_back(size);
  } else {
    if (!undef.contains(dim)) {
      auto &name = names[dim];
      indices.erase(name);
    }
    names[dim] = name;
    shape[dim] = size;
  }

  undef.erase(dim);
  indices.insert_or_assign(name, dim);
}

unsigned CoordTransformBuilder::indexOf(StringRef name, Place place) {
  return place == Place::Top ? upperIndices[name] : lowerIndices[name];
}

SmallString<8> CoordTransformBuilder::nameOf(unsigned dim, Place place) {
  return place == Place::Top ? upperNames[dim] : lowerNames[dim];
}

int64_t CoordTransformBuilder::sizeOf(StringRef name, Place place) {
  if (place == Place::Top) {
    return upperShape[upperIndices[name]];
  }
  if (place == Place::Bottom) {
    return lowerShape[lowerIndices[name]];
  }
  llvm_unreachable("Unknown enum variant");
}

int64_t CoordTransformBuilder::sizeOf(unsigned dim, Place place) {
  return place == Place::Top ? upperShape[dim] : lowerShape[dim];
}

unsigned CoordTransformBuilder::nDims(Place place) {
  return place == Place::Top ? upperNames.size() : lowerNames.size();
}

CoordTransformBuilder &
CoordTransformBuilder::passThrough(StringRef name, Place place,
                                   Optional<StringRef> newName) {
  unsigned dim = indexOf(name, place);
  SmallString<8> oldName = nameOf(dim, place);
  int64_t size = sizeOf(dim, place);
  unsigned n = nDims(place);
  StringRef newNameVal = newName.getValueOr(name);

  updateDim(dim, newNameVal, size, place);
  auto map = AffineMapAttr::get(b.getMultiDimIdentityMap(n));
  TransformAttr attr =
      getTransformAttr(b, TransformType::PassThrough, {}, {newNameVal}, {dim},
                       {name}, {dim}, map, place);
  addAtPlace(result, attr, place);
  return *this;
}

CoordTransformBuilder &CoordTransformBuilder::passThrough(StringRef name,
                                                          StringRef newName,
                                                          unsigned dim,
                                                          Place place) {
  unsigned oldDim = indexOf(name, place);
  int64_t size = sizeOf(oldDim, place);
  unsigned nOld = nDims(place);

  if (dim != oldDim) {
    dropDim(oldDim, place);
  }
  updateDim(dim, newName, size, place);

  unsigned nNew = nDims(place);
  AffineMapAttr map = getIdentityExcept(
      b, std::max(nOld, nNew), place == Place::Top ? dim : oldDim,
      b.getAffineDimExpr(place == Place::Top ? oldDim : dim));
  auto attr = getTransformAttr(b, TransformType::PassThrough, {}, {newName},
                               {dim}, {name}, {oldDim}, map, place);
  addAtPlace(result, attr, place);
  return *this;
}

CoordTransformBuilder &CoordTransformBuilder::pad(ArrayRef<StringRef> names,
                                                  ArrayRef<int64_t> parameters,
                                                  Place place) {
  SmallVector<unsigned, 2> dims;
  SmallVector<int64_t, 2> oldSizes;
  for (const auto &name : names) {
    dims.push_back(indexOf(name, place));
  }
  return pad(names, dims, names, parameters, place);
}

CoordTransformBuilder &CoordTransformBuilder::pad(ArrayRef<StringRef> newNames,
                                                  ArrayRef<unsigned> newDims,
                                                  ArrayRef<StringRef> oldNames,
                                                  ArrayRef<int64_t> parameters,
                                                  Place place) {
  assert(newNames.size() == oldNames.size() &&
         "Padding can't add/remove dimensions");
  assert(parameters.size() == 2 * newNames.size() &&
         "Must supply left and right padding for each padded dimension");
  SmallVector<unsigned, 2> oldDims;
  SmallVector<int64_t, 2> oldSizes;
  for (const auto &name : oldNames) {
    oldDims.push_back(indexOf(name, place));
    oldSizes.push_back(sizeOf(name, place));
  }
  unsigned nOld = nDims(place);

  llvm::SmallVector<AffineExpr, 8> affineOuts;

  for (unsigned i = 0, e = newNames.size(); i < e; ++i) {
    int64_t newSize;
    if (place == Place::Top) {
      // The input is the output with padding on the dimensions
      newSize = oldSizes[i] + parameters[2 * i] + parameters[2 * i + 1];
    } else {
      // The output is the input with its padding removed
      newSize = oldSizes[i] - parameters[2 * i] - parameters[2 * i + 1];
    }
    dropDim(oldDims[i], place);
    updateDim(newDims[i], newNames[i], newSize, place);

    if (place == Place::Top) {
      for (unsigned j = affineOuts.size(); j <= oldDims[i]; ++j) {
        affineOuts.push_back(b.getAffineDimExpr(j));
      }
      affineOuts[oldDims[i]] = b.getAffineDimExpr(newDims[i]) -
                               b.getAffineConstantExpr(parameters[2 * i]);
    } else {
      for (unsigned j = affineOuts.size(); j <= newDims[i]; ++j) {
        affineOuts.push_back(b.getAffineDimExpr(j));
      }
      affineOuts[newDims[i]] = b.getAffineDimExpr(oldDims[i]) -
                               b.getAffineConstantExpr(parameters[2 * i]);
    }
  }
  unsigned nNew = nDims(place);

  auto map = AffineMapAttr::get(AffineMap::get(
      place == Place::Top ? nNew : nOld, 0, affineOuts, b.getContext()));
  auto attr = getTransformAttr(b, TransformType::Pad, parameters, newNames,
                               newDims, oldNames, oldDims, map, place);
  addAtPlace(result, attr, place);
  return *this;
}

} // namespace miopen
} // namespace mlir
