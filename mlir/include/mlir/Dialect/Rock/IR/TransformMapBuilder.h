//===- TransformMapBuilder.h - Rock transform map builder -*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the transform map builder interface.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ROCK_TRANSFORMMAPBUILDER_H
#define MLIR_DIALECT_ROCK_TRANSFORMMAPBUILDER_H

#include "mlir/Dialect/Rock/IR/Rock.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace rock {
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
class TransformMapBuilder {
public:
  virtual ~TransformMapBuilder() = default;

  TransformMapBuilder(const TransformMapBuilder &other);
  TransformMapBuilder(TransformMapBuilder &&other) = delete;
  TransformMapBuilder &operator=(const TransformMapBuilder &other);

  // Get the TransformsAttr that's being built up by the builder
  TransformMapAttr get();
  // Only valid after the transformation has been built.
  // The names live as long as the TransformMapBuilder
  void getEndNames(SmallVectorImpl<StringRef> &names);
  void getStartNames(SmallVectorImpl<StringRef> &names);

  StringRef startName(uint32_t dim);
  StringRef endName(uint32_t dim);
  uint32_t startIndex(StringRef name);
  uint32_t endIndex(StringRef name);

  int64_t startSize(StringRef name);
  int64_t startSize(uint32_t dim);
  int64_t endSize(StringRef name);
  int64_t endSize(uint32_t dim);

  void passThrough(StringRef name);
  void passThrough(StringRef outName, StringRef inName);
  void passThrough(ArrayRef<StringRef> names);
  void passThrough(ArrayRef<StringRef> outNames, ArrayRef<uint32_t> outDims,
                   ArrayRef<StringRef> inNames);
  void passThrough(ArrayRef<uint32_t> endIndices,
                   ArrayRef<uint32_t> startIndices);

  // Parameters is the pre and post padding for each dimension in the order they
  // appear as arguments. For example, padding x with 2 on the left and 1 on the
  // right and then y with 1 above and 2 below would lead to pad({"x", "y"}, {2,
  // 1, 1, 2})
  void pad(ArrayRef<StringRef> names, ArrayRef<int64_t> params);
  void pad(StringRef outName, StringRef inName, int64_t left, int64_t right);
  void pad(ArrayRef<StringRef> outNames, ArrayRef<uint32_t> outDims,
           ArrayRef<StringRef> inNames, ArrayRef<int64_t> params);

protected:
  TransformMapBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                      ArrayRef<int64_t> startShape, mlir::Location loc);
  TransformMapBuilder(mlir::Builder &builder, ArrayRef<int64_t> startShape,
                      mlir::Location loc);

  template <class T, typename = typename std::enable_if<
                         std::is_base_of<TransformMapBuilder, T>::value>::type>
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

/// Builds a coordinate transformation from the top (upper) layer down.
///
/// This should be used in cases where the shape of the upper layer is already
/// available and easier to work with. For example, when describing the
/// transformation from xdlops-space indices into gemmM (M0, M1, M2)
/// to the gemmM dimension, it's more straightforward to start with the
/// upper shape and simply define an Embed without having to go through and
/// write the sizes again as one would have to if working bottom up.
class TopDownTMBuilder : public TransformMapBuilder {
public:
  TopDownTMBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                   ArrayRef<int64_t> startShape)
      : TopDownTMBuilder(builder, startNames, startShape,
                         builder.getUnknownLoc()) {}

  TopDownTMBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                   ArrayRef<int64_t> startShape, mlir::Location loc)
      : TransformMapBuilder(builder, startNames, startShape, loc) {}

  TopDownTMBuilder(mlir::Builder &builder, ArrayRef<int64_t> startShape,
                   mlir::Location loc)
      : TransformMapBuilder(builder, startShape, loc) {}

  static TopDownTMBuilder below(TopDownTMBuilder &previous,
                                TransformMapAttr &result) {
    return TransformMapBuilder::nextTransforms(previous,
                                               result.getLowerBounds());
  }

  // NOTE: There is no  builder for slice() as it isn't used in this context
  // If you find you need one, please add it, bearing in mind
  // that your start dimension is already sliced so you need to pass the full
  // length

  // Drop `dim`, making it disappear from the underlying view.
  void ignore(StringRef dim);

  // Defines dimension(s) that have a constan value and some particular size.
  void constDim(StringRef lowerName, uint32_t lowerDim, int64_t constantVal,
                int64_t lowerSize);
  void constDim(ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims,
                ArrayRef<int64_t> constantVals, ArrayRef<int64_t> lowerSizes);

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

/// A wrapper around a TopDownTMBuilder that looks up end dimensions in a
/// provided map, used for cases such as when merge()ing will create extra
/// dimensions. This takes the builder by reference and does modefiations there
/// and thus doesn't expose its own get() method. Everything is defined here
/// to increase inlineability
struct TopDownTMBottomDimsWrapper {
  TopDownTMBuilder &b;
  llvm::StringMap<uint32_t> bottomDims;

  TopDownTMBottomDimsWrapper(TopDownTMBuilder &b,
                             llvm::StringMap<uint32_t> bottomDims)
      : b(b), bottomDims(bottomDims) {}
  void passThrough(StringRef name);
  void passThrough(ArrayRef<StringRef> names);

  void pad(ArrayRef<StringRef> outNames, ArrayRef<StringRef> inNames,
           ArrayRef<int64_t> params);

  void constDim(StringRef lowerName, int64_t constantVal, int64_t lowerSize);
  void constDim(ArrayRef<StringRef> lowerNames, ArrayRef<int64_t> constantVals,
                ArrayRef<int64_t> lowerSizes);

  void embed(StringRef lowerName, int64_t lowerSize,
             ArrayRef<StringRef> upperNames, ArrayRef<int64_t> coefficients);

  void unmerge(StringRef lowerName, ArrayRef<StringRef> upperNames,
               ArrayRef<int64_t> lengths);

  void merge(ArrayRef<StringRef> lowerNames, StringRef upperName,
             ArrayRef<int64_t> sizes, bool isUnfold = false);

  llvm::SmallVector<uint32_t> toBottomDims(ArrayRef<StringRef> names);
};

/// Builds a coordinate transformation from the bottom (lower) layer up.
///
/// A bottom-up builder can be used when you know the shape of the untransformed
/// and should be used when computing the result shape would require work
/// equivalent to working through the transformation (multiplying sizes together
/// etc.). For example, most transformations from the convolution arguments to
/// GEMM arguments are described bottom-up.
class BottomUpTMBuilder : public TransformMapBuilder {
public:
  BottomUpTMBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                    ArrayRef<int64_t> startShape)
      : BottomUpTMBuilder(builder, startNames, startShape,
                          builder.getUnknownLoc()) {}

  BottomUpTMBuilder(mlir::Builder &builder, ArrayRef<StringRef> startNames,
                    ArrayRef<int64_t> startShape, mlir::Location loc)
      : TransformMapBuilder(builder, startNames, startShape, loc) {}

  BottomUpTMBuilder(mlir::Builder &builder, ArrayRef<int64_t> startShape,
                    mlir::Location loc)
      : TransformMapBuilder(builder, startShape, loc) {}

  static BottomUpTMBuilder above(BottomUpTMBuilder &previous,
                                 TransformMapAttr &result) {
    return TransformMapBuilder::nextTransforms(previous,
                                               result.getUpperBounds());
  }

  // Defines a dimension that is not mapped to any coordinates in the output
  void addDim(StringRef name, uint32_t dim, int64_t size);

  // NOTE: there is no builder for constDim but you can add one if you really
  // want to. If you do so, put some sort of warning in the name, like
  // assumeDimIsConst(), because, when working from the bottom up,
  // that transformation is an assertion that a given dimension has a particular
  // constant value.

  void broadcast(ArrayRef<uint32_t> endDims, ArrayRef<int64_t> endSizes);

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

/// A wrapper around a BottomUpTMBuilder that looks up end dimensions in a
/// provided map, used for cases such as when embed()ing will create extra
/// dimensions. This takes the builder by reference and does modefiations there
/// and thus doesn't expose its own get() method. Everything is defined here
/// to increase inlineability
struct BottomUpTMTopDimsWrapper {
  BottomUpTMBuilder &b;
  llvm::StringMap<uint32_t> topDims;

  BottomUpTMTopDimsWrapper(BottomUpTMBuilder &b,
                           llvm::StringMap<uint32_t> topDims)
      : b(b), topDims(topDims) {}
  void passThrough(StringRef name);
  void passThrough(ArrayRef<StringRef> names);

  void pad(ArrayRef<StringRef> outNames, ArrayRef<StringRef> inNames,
           ArrayRef<int64_t> params);

  void addDim(StringRef name, int64_t size);

  void embed(ArrayRef<StringRef> upperNames, ArrayRef<int64_t> upperSizes,
             StringRef lowerName, ArrayRef<int64_t> coefficients);

  void unmerge(ArrayRef<StringRef> upperNames, StringRef lowerName,
               ArrayRef<int64_t> lengths);

  void merge(StringRef upperName, ArrayRef<StringRef> lowerNames,
             bool isUnfold = false);

  llvm::SmallVector<uint32_t> toTopDims(ArrayRef<StringRef> names);
};

/// Create a map of dimension names to indices in a destination coordinate space
/// using the expansion map [original name] -> [expanded names] to
/// replace one dimension with multiple ones.
/// For example, expandNamesInPlace(["a", "b", "c"], {"b": ["x", "y"]})
/// will return the mapping {"a": 0, "x": 1, "y": 2, "c": 3}
llvm::StringMap<uint32_t>
expandNamesInPlace(ArrayRef<StringRef> original,
                   const llvm::StringMap<SmallVector<StringRef, 2>> expansion);
llvm::StringMap<uint32_t>
expandNamesInPlace(TransformMapBuilder &builder,
                   const llvm::StringMap<SmallVector<StringRef, 2>> expansion);
} // namespace rock
} // namespace mlir
#endif
