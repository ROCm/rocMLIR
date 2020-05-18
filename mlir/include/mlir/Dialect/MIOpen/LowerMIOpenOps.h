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

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MIOpen/MIOpenOps.h"
#include "mlir/Dialect/MIOpen/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conv2D (forward, backward) lowering.
//===----------------------------------------------------------------------===//

// The ArgumentFields keep track of differences between conv operations
struct ArgumentFields {
  int gridwiseGemmArgumentPosition[3];
  StringRef gemmTargetCharName[3];
};

template <typename T>
struct Conv2DRewritePattern : public OpRewritePattern<T> {
  const static ArgumentFields fields;
  const static miopen::ConvOpType convOpType;
  using OpRewritePattern<T>::OpRewritePattern;

  LogicalResult matchAndRewrite(T op, PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto filterLayoutAttr =
        op.template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr = op.template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        op.template getAttrOfType<ArrayAttr>("output_layout");

    auto dilationsAttr = op.template getAttrOfType<ArrayAttr>("dilations");
    auto stridesAttr = op.template getAttrOfType<ArrayAttr>("strides");
    auto paddingAttr = op.template getAttrOfType<ArrayAttr>("padding");

    // Get shape of output tensor.
    auto outputType = op.output().getType().template dyn_cast<MemRefType>();
    auto outputShape = outputType.getShape();
    // HO/WO dimension for output tensor.
    int64_t outputHDim, outputWDim;

    // Find Ho/Wo dimension for output tensor. They will be used in
    // transforming input tensor.
    for (unsigned i = 0; i < outputLayoutAttr.size(); ++i) {
      if (auto strAttr =
              outputLayoutAttr.getValue()[i].template dyn_cast<StringAttr>()) {
        if (strAttr.getValue() == "ho") {
          outputHDim = i;
        } else if (strAttr.getValue() == "wo") {
          outputWDim = i;
        }
      }
    }

    // Transform filter tensor.
    auto filterType = op.filter().getType().template dyn_cast<MemRefType>();
    auto filterShape = filterType.getShape();
    auto filterElementType = filterType.getElementType();
    // Y/X dimension for filter tensor.
    int64_t filterYDim, filterXDim;

    llvm::SmallVector<int64_t, 2> transformedFilterShape;

    llvm::SmallVector<NamedAttribute, 3> transformedFilterAttrs;

    SmallString<4> arg0TargetLayoutName0("gemm");
    arg0TargetLayoutName0.append(fields.gemmTargetCharName[0].substr(0, 1));
    SmallString<4> arg0TargetLayoutName1("gemm");
    arg0TargetLayoutName1.append(fields.gemmTargetCharName[0].substr(1, 1));

    // set layout attribute.
    // Weight tensor transformation for Conv2DOp
    // - Part 1: Merge non-K dimensions to dimension 0, name it as gemmK.
    //           Optimization: If non-K dimensions are consequetive, apply
    //           unfold.
    // - Part 2: PassThrough K dimension to dimension 1, name it as gemmM.
    //
    // Weight tensor transformation for Conv2DBwdDataOp
    // - Part 1: Merge K dimensions to dimension 0, name it as gemmK.
    // - Part 2: PassThrough non-K dimension to dimension 1, name it as gemmM.
    //           Optimization: If non-K dimensions are consequetive, apply
    //           unfold.
    {
      llvm::SmallVector<IntegerAttr, 3> nonKDims;
      IntegerAttr kDim;
      llvm::SmallVector<StringAttr, 3> nonKDimNames;
      StringAttr kDimName;
      for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
        if (auto strAttr =
                filterLayoutAttr.getValue()[i].template dyn_cast<StringAttr>()) {
          if (strAttr.getValue() == "k") {
            kDim = b.getI32IntegerAttr(i);
            kDimName = strAttr;
          } else {
            // Register filter Y/X dimension to be used later when transforming
            // input tensor.
            if (strAttr.getValue() == "y") {
              filterYDim = i;
            } else if (strAttr.getValue() == "x") {
              filterXDim = i;
            }
            nonKDims.push_back(b.getI32IntegerAttr(i));
            nonKDimNames.push_back(strAttr);
          }
        }
      }

      // Compute transformed filter shape dimension.
      int64_t nonKDimSize = 1;
      for (unsigned i = 0; i < filterShape.size(); ++i) {
        if (i != kDim.getInt()) {
          nonKDimSize *= filterShape[i];
        }
      }
      transformedFilterShape.push_back(nonKDimSize);
      transformedFilterShape.push_back(filterShape[kDim.getInt()]);

      llvm::SmallVector<NamedAttribute, 2> sourceNonKDimAttr{
          b.getNamedAttr("source_dimensions",
                         b.getArrayAttr(ArrayRef<Attribute>(nonKDims.begin(),
                                                            nonKDims.end()))),
          b.getNamedAttr("source_names",
                         b.getArrayAttr(ArrayRef<Attribute>(
                             nonKDimNames.begin(), nonKDimNames.end())))};
      if (kDim.getInt() != 0 && kDim.getInt() != 3) {
        sourceNonKDimAttr.push_back(
            b.getNamedAttr("transformation", b.getStringAttr("Merge")));
      } else {
        sourceNonKDimAttr.push_back(
            b.getNamedAttr("transformation", b.getStringAttr("Unfold")));
      }

      llvm::SmallVector<NamedAttribute, 3> sourceKDimAttr{
          b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
          b.getNamedAttr("source_dimensions", b.getArrayAttr({kDim})),
          b.getNamedAttr("source_names", b.getArrayAttr({kDimName}))};

      llvm::SmallVector<NamedAttribute, 3> targetKDimAttr{
          b.getNamedAttr("dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(0)})),
          b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                      arg0TargetLayoutName0)}))};

      llvm::SmallVector<NamedAttribute, 3> targetNonKDimAttr{
          b.getNamedAttr("dimensions",
                         b.getArrayAttr({b.getI32IntegerAttr(1)})),
          b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                      arg0TargetLayoutName1)}))};

      llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
      llvm::SmallVector<NamedAttribute, 0> layoutAttr1;

      if (convOpType == miopen::ConvOpType::Conv2DBwdDataOpType) {
        layoutAttr0.append(targetKDimAttr.begin(), targetKDimAttr.end());
        layoutAttr0.append(sourceKDimAttr.begin(), sourceKDimAttr.end());
        layoutAttr1.append(targetNonKDimAttr.begin(), targetNonKDimAttr.end());
        layoutAttr1.append(sourceNonKDimAttr.begin(), sourceNonKDimAttr.end());
      } else if (convOpType == miopen::ConvOpType::Conv2DOpType) {
        layoutAttr0.append(targetKDimAttr.begin(), targetKDimAttr.end());
        layoutAttr0.append(sourceNonKDimAttr.begin(), sourceNonKDimAttr.end());
        layoutAttr1.append(targetNonKDimAttr.begin(), targetNonKDimAttr.end());
        layoutAttr1.append(sourceKDimAttr.begin(), sourceKDimAttr.end());
      }

      transformedFilterAttrs.push_back(b.getNamedAttr(
          "layout", b.getArrayAttr({
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr0.begin(), layoutAttr0.end())}),

                        // Part 2: Passthrough part.
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr1.begin(), layoutAttr1.end())}),
                    })));
    }

    // set source_layout attribute.
    transformedFilterAttrs.push_back(
        b.getNamedAttr("source_layout", filterLayoutAttr));
    // set output_layout attribute.
    transformedFilterAttrs.push_back(b.getNamedAttr(
        "output_layout",
        b.getArrayAttr({b.getStringAttr(arg0TargetLayoutName0),
                        b.getStringAttr(arg0TargetLayoutName1)})));
    // set gridwise_gemm_argument_pos attribute.
    transformedFilterAttrs.push_back(b.getNamedAttr(
        "gridwise_gemm_argument_position",
        b.getI32IntegerAttr(fields.gridwiseGemmArgumentPosition[0])));

    auto transformedFilterMemRefType =
        MemRefType::get(transformedFilterShape, filterElementType);
    auto gemmA = b.create<miopen::TransformOp>(
        loc, transformedFilterMemRefType, op.filter(), transformedFilterAttrs);

    // Transform input tensor.
    // Input tensor step 1: padded input.
    auto inputType = op.input().getType().template dyn_cast<MemRefType>();
    auto inputShape = inputType.getShape();
    auto inputElementType = inputType.getElementType();

    llvm::SmallVector<int64_t, 4> paddedInputShape;

    llvm::SmallVector<NamedAttribute, 3> paddedInputAttrs;

    // reorderedPaddedInputDimNames would be used by the next stage.
    llvm::SmallVector<StringAttr, 4> reorderedPaddedInputDimNames;

    // set layout attribute.
    // Padded input tensor transformation:
    // - Part 1: PassThrough ni dimension to its original dimension, name it as
    // ni.
    // - Part 2: PassThrough ci dimension to its original dimension, name it as
    // ci.
    // - Part 3: Pad hi/wi dimensions to their original dimensions, name it as
    // hipad/wipad.
    {
      IntegerAttr nDim, cDim;
      StringAttr nDimName, cDimName;
      llvm::SmallVector<IntegerAttr, 2> hwDims;
      llvm::SmallVector<StringAttr, 2> hwDimNames;
      for (unsigned i = 0; i < inputLayoutAttr.size(); ++i) {
        if (auto strAttr =
                inputLayoutAttr.getValue()[i].template dyn_cast<StringAttr>()) {
          if (strAttr.getValue() == "ni") {
            nDim = b.getI32IntegerAttr(i);
            nDimName = strAttr;
          } else if (strAttr.getValue() == "ci") {
            cDim = b.getI32IntegerAttr(i);
            cDimName = strAttr;
          } else {
            hwDims.push_back(b.getI32IntegerAttr(i));
            hwDimNames.push_back(strAttr);
          }
        }
      }

      llvm::SmallVector<StringAttr, 2> hwPaddedDimNames;
      for (auto strAttr : hwDimNames) {
        hwPaddedDimNames.push_back(
            b.getStringAttr((strAttr.getValue() + "pad").str()));
      }

      for (unsigned i = 0, j = 0; i < inputLayoutAttr.size(); ++i) {
        if (APInt(32, i) == nDim.getValue()) {
          reorderedPaddedInputDimNames.push_back(nDimName);
          paddedInputShape.push_back(inputShape[nDim.getInt()]);
        } else if (APInt(32, i) == cDim.getValue()) {
          reorderedPaddedInputDimNames.push_back(cDimName);
          paddedInputShape.push_back(inputShape[cDim.getInt()]);
        } else {
          // TBD: padding parameters.
          paddedInputShape.push_back(inputShape[hwDims[j].getInt()]);

          reorderedPaddedInputDimNames.push_back(hwPaddedDimNames[j++]);
        }
      }

      paddedInputAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr({
              // Part 1: Passthrough for ni dimension.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions", b.getArrayAttr({nDim})),
                  b.getNamedAttr("names", b.getArrayAttr({nDimName})),
                  b.getNamedAttr("transformation",
                                 b.getStringAttr("PassThrough")),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({nDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({nDimName})),
              }),

              // Part 2: Passthrough for ci dimension.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions", b.getArrayAttr({cDim})),
                  b.getNamedAttr("names", b.getArrayAttr({cDimName})),
                  b.getNamedAttr("transformation",
                                 b.getStringAttr("PassThrough")),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({cDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({cDimName})),
              }),

              // Part 3: Pad for h/w dimensions.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions",
                                 b.getArrayAttr(ArrayRef<Attribute>(
                                     hwDims.begin(), hwDims.end()))),
                  b.getNamedAttr("names", b.getArrayAttr(ArrayRef<Attribute>(
                                              hwPaddedDimNames.begin(),
                                              hwPaddedDimNames.end()))),
                  b.getNamedAttr("transformation", b.getStringAttr("Pad")),
                  // TBD: padding parmeters.
                  b.getNamedAttr("parameters", b.getArrayAttr({
                                                   b.getI32IntegerAttr(0),
                                                   b.getI32IntegerAttr(0),
                                               })),
                  b.getNamedAttr("source_dimensions",
                                 b.getArrayAttr(ArrayRef<Attribute>(
                                     hwDims.begin(), hwDims.end()))),
                  b.getNamedAttr("source_names",
                                 b.getArrayAttr(ArrayRef<Attribute>(
                                     hwDimNames.begin(), hwDimNames.end()))),
              }),
          })));
    }
    // set source_layout attribute.
    paddedInputAttrs.push_back(
        b.getNamedAttr("source_layout", inputLayoutAttr));
    // set output_layout attribute.
    paddedInputAttrs.push_back(b.getNamedAttr(
        "output_layout", b.getArrayAttr(ArrayRef<Attribute>(
                             reorderedPaddedInputDimNames.begin(),
                             reorderedPaddedInputDimNames.end()))));
    auto paddedInputMemRefType =
        MemRefType::get(paddedInputShape, inputElementType);
    auto paddedInput = b.create<miopen::TransformOp>(
        loc, paddedInputMemRefType, op.input(), paddedInputAttrs);

    // Input tensor step 2 : embedded input.
    llvm::SmallVector<int64_t, 6> embeddedInputShape;

    llvm::SmallVector<NamedAttribute, 3> embeddedInputAttrs;

    // reorderedEmbeddedInputDimNames would be used by the next stage.
    llvm::SmallVector<StringAttr, 6> reorderedEmbeddedInputDimNames;

    // Embedded input tensor transformation:
    // - Part 1: PassThrough ni dimension to its original dimension, name it as
    // ni.
    // - Part 2: PassThrough ci dimension to its original dimension, name it as
    // ci.
    // - Part 3: Embed hipad dimension to 2 dimensions, name them as: y, ho.
    // - Part 4: Embed wipad dimension to 2 dimensions, name them as: x, wo.
    {
      IntegerAttr nDim, cDim;
      StringAttr nDimName, cDimName;
      IntegerAttr hDim, wDim;
      StringAttr hDimName, wDimName;
      // reorder dimensions from 4 to 6.
      // ex: (ni, ci, hipad, wipad) -> (ni, ci, y, ho, x, wo).
      IntegerAttr reorderedNDim, reorderedCDim;
      llvm::SmallVector<IntegerAttr, 2> reorderedYHoDim;
      llvm::SmallVector<IntegerAttr, 2> reorderedXWoDim;
      unsigned dimCtr = 0;
      for (unsigned i = 0; i < reorderedPaddedInputDimNames.size(); ++i) {
        auto strAttr = reorderedPaddedInputDimNames[i];
        if (strAttr.getValue() == "ni") {
          nDim = b.getI32IntegerAttr(i);
          nDimName = strAttr;

          reorderedNDim = b.getI32IntegerAttr(dimCtr++);

          reorderedEmbeddedInputDimNames.push_back(strAttr);

          embeddedInputShape.push_back(inputShape[nDim.getInt()]);
        } else if (strAttr.getValue() == "ci") {
          cDim = b.getI32IntegerAttr(i);
          cDimName = strAttr;

          reorderedCDim = b.getI32IntegerAttr(dimCtr++);

          reorderedEmbeddedInputDimNames.push_back(strAttr);

          embeddedInputShape.push_back(inputShape[cDim.getInt()]);
        } else if (strAttr.getValue() == "hipad") {
          hDim = b.getI32IntegerAttr(i);
          hDimName = strAttr;

          reorderedYHoDim.push_back(b.getI32IntegerAttr(dimCtr++));
          reorderedYHoDim.push_back(b.getI32IntegerAttr(dimCtr++));

          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("y"));
          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("ho"));

          embeddedInputShape.push_back(filterShape[filterYDim]);
          embeddedInputShape.push_back(outputShape[outputHDim]);
        } else if (strAttr.getValue() == "wipad") {
          wDim = b.getI32IntegerAttr(i);
          wDimName = strAttr;

          reorderedXWoDim.push_back(b.getI32IntegerAttr(dimCtr++));
          reorderedXWoDim.push_back(b.getI32IntegerAttr(dimCtr++));

          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("x"));
          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("wo"));

          embeddedInputShape.push_back(filterShape[filterXDim]);
          embeddedInputShape.push_back(outputShape[outputWDim]);
        }
      }

      embeddedInputAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr({
              // Part 1: Passthrough for ni dimension.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions", b.getArrayAttr({reorderedNDim})),
                  b.getNamedAttr("names", b.getArrayAttr({nDimName})),
                  b.getNamedAttr("transformation",
                                 b.getStringAttr("PassThrough")),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({nDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({nDimName})),
              }),

              // Part 2: Passthrough for ci dimension.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions", b.getArrayAttr({reorderedCDim})),
                  b.getNamedAttr("names", b.getArrayAttr({cDimName})),
                  b.getNamedAttr("transformation",
                                 b.getStringAttr("PassThrough")),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({cDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({cDimName})),
              }),

              // Part 3: Embed for y, ho dimensions.
              b.getDictionaryAttr({
                  b.getNamedAttr(
                      "dimensions",
                      b.getArrayAttr(ArrayRef<Attribute>(
                          reorderedYHoDim.begin(), reorderedYHoDim.end()))),
                  b.getNamedAttr("names", b.getArrayAttr({
                                              b.getStringAttr("y"),
                                              b.getStringAttr("ho"),
                                          })),
                  b.getNamedAttr("transformation", b.getStringAttr("Embed")),
                  // TBD: padding parmeters.
                  b.getNamedAttr("parameters", b.getArrayAttr({
                                                   b.getI32IntegerAttr(2),
                                                   b.getI32IntegerAttr(1),
                                                   b.getI32IntegerAttr(1),
                                                   b.getI32IntegerAttr(0),
                                               })),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({hDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({hDimName})),
              }),

              // Part 4: Embed for x, wo dimensions.
              b.getDictionaryAttr({
                  b.getNamedAttr(
                      "dimensions",
                      b.getArrayAttr(ArrayRef<Attribute>(
                          reorderedXWoDim.begin(), reorderedXWoDim.end()))),
                  b.getNamedAttr("names", b.getArrayAttr({
                                              b.getStringAttr("x"),
                                              b.getStringAttr("wo"),
                                          })),
                  b.getNamedAttr("transformation", b.getStringAttr("Embed")),
                  // TBD: embed parmeters.
                  b.getNamedAttr("parameters", b.getArrayAttr({
                                                   b.getI32IntegerAttr(2),
                                                   b.getI32IntegerAttr(1),
                                                   b.getI32IntegerAttr(1),
                                                   b.getI32IntegerAttr(0),
                                               })),
                  b.getNamedAttr("source_dimensions", b.getArrayAttr({wDim})),
                  b.getNamedAttr("source_names", b.getArrayAttr({wDimName})),
              }),
          })));
    }
    // set intermediate_layout attribute.
    embeddedInputAttrs.push_back(b.getNamedAttr(
        "intermediate_layout", b.getArrayAttr(ArrayRef<Attribute>(
                                   reorderedPaddedInputDimNames.begin(),
                                   reorderedPaddedInputDimNames.end()))));
    // set output_layout attribute.
    embeddedInputAttrs.push_back(b.getNamedAttr(
        "output_layout", b.getArrayAttr(ArrayRef<Attribute>(
                             reorderedEmbeddedInputDimNames.begin(),
                             reorderedEmbeddedInputDimNames.end()))));
    auto embeddedInputMemRefType =
        MemRefType::get(embeddedInputShape, inputElementType);
    auto embeddedInput = b.create<miopen::TransformOp>(
        loc, embeddedInputMemRefType, ArrayRef<Value>(paddedInput),
        embeddedInputAttrs);

    // Input tensor step 3: transformed input.
    llvm::SmallVector<int64_t, 2> transformedInputShape;

    llvm::SmallVector<NamedAttribute, 3> transformedInputAttrs;

    SmallString<4> arg1TargetLayoutName0("gemm");
    arg1TargetLayoutName0.append(fields.gemmTargetCharName[1].substr(0, 1));
    SmallString<4> arg1TargetLayoutName1("gemm");
    arg1TargetLayoutName1.append(fields.gemmTargetCharName[1].substr(1, 1));

    // set layout attribute.
    // Transformed input tensor transformation:
    // - Part 1: Merge ci, y, x dimensions to dimension 0, name it as gemmK.
    // - Part 2: Merge ni, ho, wo dimensions to dimension 1, name it as gemmN.
    {
      IntegerAttr nDim, cDim;
      StringAttr nDimName, cDimName;
      IntegerAttr hDim, wDim;
      StringAttr hDimName, wDimName;
      IntegerAttr yDim, xDim;
      StringAttr yDimName, xDimName;
      // reorder dimensions from 6 to 2.
      // ex: (ni, ci, y, ho, x, wo) -> ((ci, y, x), (ni, ho, wo)).
      for (unsigned i = 0; i < reorderedEmbeddedInputDimNames.size(); ++i) {
        auto strAttr = reorderedEmbeddedInputDimNames[i];
        if (strAttr.getValue() == "ni") {
          nDim = b.getI32IntegerAttr(i);
          nDimName = strAttr;
        } else if (strAttr.getValue() == "ci") {
          cDim = b.getI32IntegerAttr(i);
          cDimName = strAttr;
        } else if (strAttr.getValue() == "ho") {
          hDim = b.getI32IntegerAttr(i);
          hDimName = strAttr;
        } else if (strAttr.getValue() == "wo") {
          wDim = b.getI32IntegerAttr(i);
          wDimName = strAttr;
        } else if (strAttr.getValue() == "y") {
          yDim = b.getI32IntegerAttr(i);
          yDimName = strAttr;
        } else if (strAttr.getValue() == "x") {
          xDim = b.getI32IntegerAttr(i);
          xDimName = strAttr;
        }
      }

      llvm::SmallVector<StringAttr, 3> mergedPart1DimNames;
      llvm::SmallVector<IntegerAttr, 3> mergedPart1Dims;
      // Assume yDim is always less than xDim.
      if (cDim.getInt() < yDim.getInt()) {
        mergedPart1DimNames.push_back(cDimName);
        mergedPart1DimNames.push_back(yDimName);
        mergedPart1DimNames.push_back(xDimName);
        mergedPart1Dims.push_back(cDim);
        mergedPart1Dims.push_back(yDim);
        mergedPart1Dims.push_back(xDim);
      } else {
        mergedPart1DimNames.push_back(yDimName);
        mergedPart1DimNames.push_back(xDimName);
        mergedPart1DimNames.push_back(cDimName);
        mergedPart1Dims.push_back(yDim);
        mergedPart1Dims.push_back(xDim);
        mergedPart1Dims.push_back(cDim);
      }
      llvm::SmallVector<StringAttr, 3> mergedPart2DimNames;
      llvm::SmallVector<IntegerAttr, 3> mergedPart2Dims;
      // Assume hDim is always less than wDim.
      if (nDim.getInt() < hDim.getInt()) {
        mergedPart2DimNames.push_back(nDimName);
        mergedPart2DimNames.push_back(hDimName);
        mergedPart2DimNames.push_back(wDimName);
        mergedPart2Dims.push_back(nDim);
        mergedPart2Dims.push_back(hDim);
        mergedPart2Dims.push_back(wDim);
      } else {
        mergedPart2DimNames.push_back(hDimName);
        mergedPart2DimNames.push_back(wDimName);
        mergedPart2DimNames.push_back(nDimName);
        mergedPart2Dims.push_back(hDim);
        mergedPart2Dims.push_back(wDim);
        mergedPart2Dims.push_back(nDim);
      }
      transformedInputShape.push_back(embeddedInputShape[cDim.getInt()] * embeddedInputShape[yDim.getInt()] * embeddedInputShape[xDim.getInt()]);
      transformedInputShape.push_back(embeddedInputShape[hDim.getInt()] * embeddedInputShape[wDim.getInt()] * embeddedInputShape[nDim.getInt()]);

      transformedInputAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr({
              // Part 1: Merge ci, y, x dimensions.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions",
                                 b.getArrayAttr({b.getI32IntegerAttr(0)})),
                  b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                              arg1TargetLayoutName0)})),
                  b.getNamedAttr("transformation", b.getStringAttr("Merge")),
                  b.getNamedAttr(
                      "source_dimensions",
                      b.getArrayAttr(ArrayRef<Attribute>(
                          mergedPart1Dims.begin(), mergedPart1Dims.end()))),
                  b.getNamedAttr("source_names",
                                 b.getArrayAttr(ArrayRef<Attribute>(
                                     mergedPart1DimNames.begin(),
                                     mergedPart1DimNames.end()))),
              }),

              // Part 2: Merge ni, ho, wo dimensions.
              b.getDictionaryAttr({
                  b.getNamedAttr("dimensions",
                                 b.getArrayAttr({b.getI32IntegerAttr(1)})),
                  b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                              arg1TargetLayoutName1)})),
                  b.getNamedAttr("transformation", b.getStringAttr("Merge")),
                  b.getNamedAttr(
                      "source_dimensions",
                      b.getArrayAttr(ArrayRef<Attribute>(
                          mergedPart2Dims.begin(), mergedPart2Dims.end()))),
                  b.getNamedAttr("source_names",
                                 b.getArrayAttr(ArrayRef<Attribute>(
                                     mergedPart2DimNames.begin(),
                                     mergedPart2DimNames.end()))),
              }),
          })));
    }
    // set intermediate_layout attribute.
    transformedInputAttrs.push_back(b.getNamedAttr(
        "intermediate_layout", b.getArrayAttr(ArrayRef<Attribute>(
                                   reorderedEmbeddedInputDimNames.begin(),
                                   reorderedEmbeddedInputDimNames.end()))));
    // set output_layout attribute.
    transformedInputAttrs.push_back(b.getNamedAttr(
        "output_layout",
        b.getArrayAttr({b.getStringAttr(arg1TargetLayoutName0),
                        b.getStringAttr(arg1TargetLayoutName1)})));
    // set gridwise_gemm_argument_pos attribute.
    transformedInputAttrs.push_back(b.getNamedAttr(
        "gridwise_gemm_argument_position",
        b.getI32IntegerAttr(fields.gridwiseGemmArgumentPosition[1])));
    auto transformedInputMemRefType =
        MemRefType::get(transformedInputShape, inputElementType);
    auto gemmB = b.create<miopen::TransformOp>(loc, transformedInputMemRefType,
                                               ArrayRef<Value>(embeddedInput),
                                               transformedInputAttrs);

    // Transform output tensor.
    auto outputElementType = outputType.getElementType();
    llvm::SmallVector<int64_t, 2> transformedOutputShape;

    llvm::SmallVector<NamedAttribute, 3> transformedOutputAttrs;

    SmallString<4> arg2TargetLayoutName0("gemm");
    arg2TargetLayoutName0.append(fields.gemmTargetCharName[2].substr(0, 1));
    SmallString<4> arg2TargetLayoutName1("gemm");
    arg2TargetLayoutName1.append(fields.gemmTargetCharName[2].substr(1, 1));

    // set layout attribute.
    // Output tensor transformation:
    // - Part 1: PassThrough K dimension to dimension 0, name it as gemmM.
    // - Part 2: Merge non-K dimensions to dimension 1, name it as gemmN.
    {
      llvm::SmallVector<IntegerAttr, 3> nonKDims;
      IntegerAttr kDim;
      llvm::SmallVector<StringAttr, 3> nonKDimNames;
      StringAttr kDimName;
      for (unsigned i = 0; i < outputLayoutAttr.size(); ++i) {
        if (auto strAttr =
                outputLayoutAttr.getValue()[i].template dyn_cast<StringAttr>()) {
          if (strAttr.getValue() == "ko") {
            kDim = b.getI32IntegerAttr(i);
            kDimName = strAttr;
          } else {
            nonKDims.push_back(b.getI32IntegerAttr(i));
            nonKDimNames.push_back(strAttr);
          }
        }
      }

      // Compute transformed filter shape dimension.
      int64_t nonKDimSize = 1;
      for (unsigned i = 0; i < outputShape.size(); ++i) {
        if (i != kDim.getInt()) {
          nonKDimSize *= outputShape[i];
        }
      }
      transformedOutputShape.push_back(outputShape[kDim.getInt()]);
      transformedOutputShape.push_back(nonKDimSize);
 
      transformedOutputAttrs.push_back(b.getNamedAttr(
          "layout",
          b.getArrayAttr(
              {// Part 1: Passthrough.
               b.getDictionaryAttr({
                   b.getNamedAttr("dimensions",
                                  b.getArrayAttr({b.getI32IntegerAttr(0)})),
                   b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                               arg2TargetLayoutName0)})),
                   b.getNamedAttr("transformation",
                                  b.getStringAttr("PassThrough")),
                   b.getNamedAttr("source_dimensions", b.getArrayAttr({kDim})),
                   b.getNamedAttr("source_names", b.getArrayAttr({kDimName})),
               }),

               // Part 2: Merge.
               b.getDictionaryAttr({
                   b.getNamedAttr("dimensions",
                                  b.getArrayAttr({b.getI32IntegerAttr(1)})),
                   b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
                                               arg2TargetLayoutName1)})),
                   b.getNamedAttr("transformation", b.getStringAttr("Merge")),
                   b.getNamedAttr("source_dimensions",
                                  b.getArrayAttr(ArrayRef<Attribute>(
                                      nonKDims.begin(), nonKDims.end()))),
                   b.getNamedAttr(
                       "source_names",
                       b.getArrayAttr(ArrayRef<Attribute>(nonKDimNames.begin(),
                                                          nonKDimNames.end()))),
               })})));
    }

    // set source_layout attribute.
    transformedOutputAttrs.push_back(
        b.getNamedAttr("source_layout", outputLayoutAttr));
    // set output_layout attribute.
    transformedOutputAttrs.push_back(b.getNamedAttr(
        "output_layout", b.getArrayAttr({
                             b.getStringAttr(arg2TargetLayoutName0),
                             b.getStringAttr(arg2TargetLayoutName1),
                         })));
    // set gridwise_gemm_argument_pos attribute.
    transformedOutputAttrs.push_back(b.getNamedAttr(
        "gridwise_gemm_argument_position",
        b.getI32IntegerAttr(fields.gridwiseGemmArgumentPosition[2])));
    auto transformedOutputMemRefType =
        MemRefType::get(transformedOutputShape, outputElementType);
    auto gemmC = b.create<miopen::TransformOp>(
        loc, transformedOutputMemRefType, op.output(), transformedOutputAttrs);

    // compute right padding parameters.
    auto leftPadH = paddingAttr.getValue()[0].template dyn_cast<IntegerAttr>().getInt();
    auto leftPadW = paddingAttr.getValue()[1].template dyn_cast<IntegerAttr>().getInt();
    auto dilationH =
        dilationsAttr.getValue()[0].template dyn_cast<IntegerAttr>().getInt();
    auto dilationW =
        dilationsAttr.getValue()[1].template dyn_cast<IntegerAttr>().getInt();
    auto strideH = stridesAttr.getValue()[0].template dyn_cast<IntegerAttr>().getInt();
    auto strideW = stridesAttr.getValue()[1].template dyn_cast<IntegerAttr>().getInt();

    // get y, x, ho, wo, hi, wi
    int64_t y, x, ho, wo, hi, wi;
    y = x = ho = wo = hi = wi = 0;
    for (unsigned i = 0; i < 4; ++i) {
      auto filterAttr = filterLayoutAttr.getValue()[i].template dyn_cast<StringAttr>();
      auto inputAttr = inputLayoutAttr.getValue()[i].template dyn_cast<StringAttr>();
      auto outputAttr = outputLayoutAttr.getValue()[i].template dyn_cast<StringAttr>();

      if (filterAttr.getValue() == "y") {
        y = filterShape[i];
      } else if (filterAttr.getValue() == "x") {
        x = filterShape[i];
      }

      if (inputAttr.getValue() == "hi") {
        hi = inputShape[i];
      } else if (inputAttr.getValue() == "wi") {
        wi = inputShape[i];
      }

      if (outputAttr.getValue() == "ho") {
        ho = outputShape[i];
      } else if (outputAttr.getValue() == "wo") {
        wo = outputShape[i];
      }
    }

    auto hiPadded = 1 + (y - 1) * dilationH + (ho - 1) * strideH;
    auto wiPadded = 1 + (x - 1) * dilationW + (wo - 1) * strideW;
    auto rightPadH =
        hiPadded > (leftPadH + hi) ? hiPadded - (leftPadH + hi) : 0;
    auto rightPadW =
        wiPadded > (leftPadW + wi) ? wiPadded - (leftPadW + wi) : 0;

    // Set attributes for gridwise_gemm op.
    llvm::SmallVector<NamedAttribute, 8> gridwiseGemmAttrs{
        b.getNamedAttr("filter_layout", filterLayoutAttr),
        b.getNamedAttr("filter_dimension", b.getI64ArrayAttr(filterShape)),
        b.getNamedAttr("input_layout", inputLayoutAttr),
        b.getNamedAttr("input_dimension", b.getI64ArrayAttr(inputShape)),
        b.getNamedAttr("output_layout", outputLayoutAttr),
        b.getNamedAttr("output_dimension", b.getI64ArrayAttr(outputShape)),
        b.getNamedAttr("dilations", dilationsAttr),
        b.getNamedAttr("strides", stridesAttr),
        b.getNamedAttr(
            "padding",
            b.getArrayAttr(
                {paddingAttr, b.getI32ArrayAttr({rightPadH, rightPadW})})),
    };

    if (convOpType == miopen::ConvOpType::Conv2DBwdDataOpType) {
      gridwiseGemmAttrs.push_back(b.getNamedAttr(
          "kernel_algorithm", b.getStringAttr("backward_data_v1r1")));
    } else if (convOpType == miopen::ConvOpType::Conv2DOpType) {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kernel_algorithm", b.getStringAttr("v4r4")));
    }

    // Emit miopen.gridwise_gemm op.
    auto arguments = std::array<miopen::TransformOp, 3>{gemmA, gemmB, gemmC};
    b.create<miopen::GridwiseGemmOp>(
        loc, ArrayRef<Type>{},
        ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                   arguments[fields.gridwiseGemmArgumentPosition[1]],
                   arguments[fields.gridwiseGemmArgumentPosition[2]]},
        gridwiseGemmAttrs);

    // Finally, erase the original Conv2D op.
    op.erase();

    return success();
  }
};

// High level convolution operation always have
// [filter, input, output]
// as the convolution argument. The only difference between different
// hight level convolution operations is the argument sequence. For
// simplicity, we always arrange the first two arguments to be input
// and the last argument to be output
template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DOp>::fields = {
    {0, 1, 2},
    {"KM", "KN", "MN"},
};
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DOp>::convOpType =
    miopen::ConvOpType::Conv2DOpType;

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::fields = {
    {0, 2, 1},
    {"KM", "MN", "KN"},
};
template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DBwdDataOp>::convOpType =
    miopen::ConvOpType::Conv2DBwdDataOpType;

// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<miopen::Conv2DOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdDataOp>;

//===----------------------------------------------------------------------===//
// Assigning attributes.
//===----------------------------------------------------------------------===//

static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top, miopen::GridwiseGemmOp gop, PatternRewriter &b) {
  // Add attributes from C++ template arguments and ctor arguments.
  //
  // in gridwise_gemm:
  // 
  // ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_n0_n1_thread_desc),              - source memref
  //                                       decltype(c_m0_m1_n0_n1_global_desc),              - dest memref
  //                                       decltype(c_m0_m1_n0_n1_thread_desc.GetLengths()), - source memref
  //                                       CThreadCopySrcDstAccessOrder,                     - Sequence<0, 1, 2, 3>
  //                                       CThreadCopySrcDstVectorReadWriteDim,              - matrix_c_source_dest_vector_read_write_dim attribute
  //                                       1,                                                - 1
  //                                       CThreadCopyDstDataPerWrite,                       - matrix_c_dest_data_per_write attribute
  //                                       AddressSpace::Vgpr,                               - addrspace on source memref
  //                                       AddressSpace::Global,                             - addrspace on dest memref
  //                                       CGlobalMemoryDataOperation>(                      - NOT USED

  // XXX. we only use 2D coordinates in storing VGPR to global VRAM now.
  top.setAttr("dim_access_order", b.getArrayAttr({
                                      b.getI32IntegerAttr(0),
                                      b.getI32IntegerAttr(1),
                                  }));
  top.setAttr("vector_read_write_dim",
              gop.getAttr("matrix_c_source_dest_vector_read_write_dim"));
  top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
  top.setAttr("dest_data_per_write", gop.getAttr("matrix_c_dest_data_per_write"));
}

// XXX: Figure out a way to do away with isThreadwiseLoad parameter.
static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::BlockwiseCopyOp bop,
                                          PatternRewriter &b,
                                          bool isThreadwiseLoad) {
  // Add attributes from C++ template arguments and ctor arguments.
  //
  // in blockwise_copy:
  //
  // using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,              - source memref
  //                                                              ThreadBufferDesc,          - dest memref
  //                                                              ThreadSliceLengths,        - source memref
  //                                                              SrcDimAccessOrder,         - Sequence<1, 0>
  //                                                              SrcVectoReadDim,           - source_vector_read_dim attribute
  //                                                              SrcDataPerRead,            - source_data_per_read attribute
  //                                                              1,                         - 1
  //                                                              SrcAddressSpace,           - addrspace on source memref
  //                                                              ThreadBufferAddressSpace,  - addrspace on dest memref
  //                                                              InMemoryDataOperation::Set>- NOT USED
  //
  // using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,         - source memref
  //                                                               BlockDstDesc,             - dest memref
  //                                                               ThreadSliceLengths,       - source memref
  //                                                               DstDimAccessOrder,        - Sequence<0, 1>
  //                                                               DstVectorWriteDim,        - dest_vector_write_dim attribute
  //                                                               1,                        - 1
  //                                                               DstDataPerWrite,          - dest_data_per_write attribute
  //                                                               ThreadBufferAddressSpace, - addrspace of source memref
  //                                                               DstAddressSpace,          - addrspace of dest memref
  //                                                               DstInMemOp>;              - NOT USE

  if (isThreadwiseLoad) {
    top.setAttr("dim_access_order", bop.getAttr("source_dim_access_order"));
    top.setAttr("vector_read_write_dim", bop.getAttr("source_vector_read_dim"));
    top.setAttr("source_data_per_read", bop.getAttr("source_data_per_read"));
    top.setAttr("dest_data_per_write", b.getI32IntegerAttr(1));
  } else {
    top.setAttr("dim_access_order", bop.getAttr("dest_dim_access_order"));
    // XXX. Figure this out. Symmetry is somehow lost here.
    // top.setAttr("vector_read_write_dim",
    // bop.getAttr("dest_vector_write_dim"));
    top.setAttr("vector_read_write_dim", bop.getAttr("source_vector_read_dim"));
    top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
    top.setAttr("dest_data_per_write", bop.getAttr("dest_data_per_write"));
  }
}

// XXX: figure out a better way to get rid of isMatrixA parameter.
static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::BlockwiseGemmOp bop,
                                          PatternRewriter &b, bool isMatrixA) {
  // in blockwise_gemm:
  //
  // constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixA,                  - source memref
  //                                                          decltype(a_thread_mtx),        - dest memref
  //                                                          KPerThreadLoop,                - k_per_thread attribute
  //                                                          MPerThreadSubC,                - m_per_thread attribute
  //                                                          ThreadGemmADataPerRead_M>{};   - m_per_thread attribute
  // constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixB,                  - source memref
  //                                                          decltype(b_thread_mtx),        - dest memref
  //                                                          KPerThreadLoop,                - k_per_thread attribute
  //                                                          NPerThreadSubC,                - n_per_thread attribute
  //                                                          ThreadGemmBDataPerRead_N>{};   - n_per_thread attribute

  if (isMatrixA) {
    top.setAttr("n_slice_row", bop.getAttr("k_per_thread"));
    top.setAttr("n_slice_col", bop.getAttr("m_per_thread"));
    top.setAttr("data_per_access", bop.getAttr("m_per_thread"));
  } else {
    top.setAttr("n_slice_row", bop.getAttr("k_per_thread"));
    top.setAttr("n_slice_col", bop.getAttr("n_per_thread"));
    top.setAttr("data_per_access", bop.getAttr("n_per_thread"));
  }
}

//===----------------------------------------------------------------------===//
// GridwiseGemm lowering.
//===----------------------------------------------------------------------===//

namespace math {

// greatest common divisor, aka highest common factor
template <typename T>
T gcd(T x, T y)
{
    if(x == y || x == 0)
    {
        return y;
    }
    else if(y == 0)
    {
        return x;
    }
    else if(x > y)
    {
        return gcd(x - y, y);
    }
    else
    {
        return gcd(x, y - x);
    }
}

template <typename X, typename... Ys>
auto gcd(X x, Ys... ys)
{
    return gcd(x, ys...);
}

// least common multiple
template <typename T>
T lcm(T x, T y)
{
    return (x * y) / gcd(x, y);
}

template <typename X, typename... Ys>
auto lcm(X x, Ys... ys)
{
    return lcm(x, lcm(ys...));
}

template <class X, class Y>
auto integer_divide_ceil(X x, Y y)
{
    return (x + y - 1) / y;
}

template <class X, class Y>
auto integer_least_multiple(X x, Y y)
{
    return y * integer_divide_ceil(x, y);
}



} // namespace math

struct GridwiseGemmRewritePattern : public OpRewritePattern<miopen::GridwiseGemmOp> {
  using OpRewritePattern<miopen::GridwiseGemmOp>::OpRewritePattern;

  void computeLDSBlockSizes(miopen::GridwiseGemmOp op, int64_t &a_block_space, int64_t &b_block_space, int64_t &double_block_space) const {
     int64_t ABlockCopyDstDataPerWrite_M = op.getAttr("matrix_a_dest_data_per_write_dim_m").template dyn_cast<IntegerAttr>().getInt();
     int64_t BBlockCopyDstDataPerWrite_N = op.getAttr("matrix_b_dest_data_per_write_dim_n").template dyn_cast<IntegerAttr>().getInt();
     int64_t ThreadGemmAThreadCopySrcDataPerRead_M = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
     int64_t ThreadGemmBThreadCopySrcDataPerRead_N = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();

     int64_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_M,
                                    BBlockCopyDstDataPerWrite_N,
                                    ThreadGemmAThreadCopySrcDataPerRead_M,
                                    ThreadGemmBThreadCopySrcDataPerRead_N);

     int64_t KPerBlock = op.getAttr("k_per_block").template dyn_cast<IntegerAttr>().getInt();
     int64_t MPerBlock = op.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
     int64_t NPerBlock = op.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();

     int64_t AlignedNPerBlock = max_lds_align * math::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

     // A matrix in LDS memory, dst of blockwise copy
     //   be careful of LDS alignment
     // Original C++ logic:
     //constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
     //    Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
     //constexpr index_t a_block_space =
     //    math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_lds_align);
     int64_t AlignedMPerBlock = max_lds_align * math::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);
     a_block_space = math::integer_least_multiple(KPerBlock * AlignedMPerBlock, max_lds_align);

     // B matrix in LDS memory, dst of blockwise copy
     //   be careful of LDS alignment
     // Original C++ logic:
     //constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
     //    Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
     //constexpr index_t b_block_space =
     //    math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_lds_align);
     b_block_space = math::integer_least_multiple(KPerBlock * AlignedNPerBlock, max_lds_align);

     double_block_space = 2 * (a_block_space + b_block_space);
  }

  // XXX. Figure out a way to do away with isMatrixA parameter.
  void affixBlockwiseCopyAttributes(miopen::BlockwiseCopyOp bop,
                                    miopen::GridwiseGemmOp gop,
                                    PatternRewriter &b, bool isMatrixA) const {
    // Add attributes from C++ template arguments and ctor arguments.
    // a_blockwise_copy:
    // BlockSize                           - block_size attribute
    // a_k_m_global_desc                   - source memref
    // a_k_m_block_desc                    - dest memref
    // a_k_m_block_desc.getLengths()       - dest memref
    // ABlockCopyThreadSliceLengths_K_M    - logic specified in -miopen-affix-params pass
    // ABlockCopyThreadClusterLengths_K_M  - logic specified in -miopen-affix-params pass
    // ABlockCopyThreadClusterArrangeOrder - Sequence<1, 0>
    // ABlockCopySrcAccessOrder            - Sequence<1, 0>
    // Sequence<0, 1>                      - Sequence<0, 1>
    // ABlockCopySrcVectorReadDim          - matrix_a_source_vector_read_dim attribute
    // 1                                   - 1
    // ABlockCopySrcDataPerRead            - matrix_a_source_data_per_read attribute
    // ABlockCopyDstDataPerWrite_M         - matrix_a_dest_data_per_write_dim_m attribute

    // b_blockwise_copy:
    // BlockSize                           - block_size attribute
    // b_k_n_global_desc                   - source memref
    // b_k_n_block_desc                    - dest memref
    // b_k_n_block_desc.getLengths()       - dest memref
    // BBlockCopyThreadSliceLengths_K_N    - logic specified in -miopen-affix-params pass
    // BBlockCopyThreadClusterLengths_K_N  - logic specified in -miopen-affix-params pass
    // BBlockCopyThreadClusterArrangeOrder - Sequence<1, 0>
    // BBlockCopySrcAccessOrder            - Sequence<1, 0>
    // Sequence<0, 1>                      - Seuquence<0, 1>
    // BBlockCopySrcVectorReadDim          - matrix_b_source_read_dim attribute
    // 1                                   - 1
    // BBlockCopySrcDataPerRead            - matrix_b_source_data_per_read attribute
    // BBlockCopyDstDataPerWrite_N         - matrix_b_dest_data_per_write_dim_n attribute
    bop.setAttr("block_size", gop.getAttr("block_size"));

    if (isMatrixA) {
      bop.setAttr("source_dim_access_order", b.getArrayAttr({
                                                 b.getI32IntegerAttr(1),
                                                 b.getI32IntegerAttr(0),
                                             }));
      bop.setAttr("dest_dim_access_order", b.getArrayAttr({
                                               b.getI32IntegerAttr(0),
                                               b.getI32IntegerAttr(1),
                                           }));
      bop.setAttr("source_vector_read_dim",
                  gop.getAttr("matrix_a_source_vector_read_dim"));
      bop.setAttr("dest_vector_write_dim", b.getI32IntegerAttr(1));

      bop.setAttr("source_data_per_read",
                  gop.getAttr("matrix_a_source_data_per_read"));
      bop.setAttr("dest_data_per_write",
                  gop.getAttr("matrix_a_dest_data_per_write_dim_m"));
    } else {
      bop.setAttr("source_dim_access_order", b.getArrayAttr({
                                                 b.getI32IntegerAttr(0),
                                                 b.getI32IntegerAttr(1),
                                             }));
      bop.setAttr("dest_dim_access_order", b.getArrayAttr({
                                               b.getI32IntegerAttr(0),
                                               b.getI32IntegerAttr(1),
                                           }));
      bop.setAttr("source_vector_read_dim",
                  gop.getAttr("matrix_b_source_vector_read_dim"));
      bop.setAttr("dest_vector_write_dim", b.getI32IntegerAttr(1));

      bop.setAttr("source_data_per_read",
                  gop.getAttr("matrix_b_source_data_per_read"));
      bop.setAttr("dest_data_per_write",
                  gop.getAttr("matrix_b_dest_data_per_write_dim_n"));
    }
  }

  void affixBlockwiseGemmAttributes(miopen::BlockwiseGemmOp bop, miopen::GridwiseGemmOp gop) const {
    // Add attributes from C++ template arguments and ctor arguments.
    //const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
    //    BlockSize,                                - block_size attribute
    //    decltype(a_k_m_block_mtx_desc),           - matrixA memref
    //    decltype(b_k_n_block_mtx_desc),           - matrixB memref
    //    decltype(c_m0m1_n0n1_thread_mtx_desc),    - matrixC memref
    //    MPerThread,                               - m_per_thread attribute
    //    NPerThread,                               - n_per_thread attribute
    //    MLevel0Cluster,                           - m_level0_cluster attribute
    //    NLevel0Cluster,                           - n_level0_cluster attribute
    //    MLevel1Cluster,                           - m_level1_cluster attribute
    //    NLevel1Cluster,                           - n_level1_cluster attribute
    //    KPerThread,                               - k_per_thread attribute
    //    ThreadGemmAThreadCopySrcDataPerRead_M,    - m_per_thread attribute
    //    ThreadGemmBThreadCopySrcDataPerRead_N>{}; - n_per_thread attribute
    bop.setAttr("block_size", gop.getAttr("block_size"));
    bop.setAttr("m_per_thread", gop.getAttr("m_per_thread"));
    bop.setAttr("n_per_thread", gop.getAttr("n_per_thread"));
    bop.setAttr("k_per_thread", gop.getAttr("k_per_thread"));
    bop.setAttr("m_level0_cluster", gop.getAttr("m_level0_cluster"));
    bop.setAttr("m_level1_cluster", gop.getAttr("m_level1_cluster"));
    bop.setAttr("n_level0_cluster", gop.getAttr("n_level0_cluster"));
    bop.setAttr("n_level1_cluster", gop.getAttr("n_level1_cluster"));
  }

  template <typename T>
  MemRefType computeSubviewResultType(T op, MemRefType inputType,
                                      unsigned offset,
                                      ArrayRef<int64_t> outputShape,
                                      Type outputElementType) const {
    auto inputAffineMaps = inputType.getAffineMaps();

    auto outputRank = outputShape.size();

    auto expr = getAffineConstantExpr(offset, op.getContext());
    unsigned stride = 1;
    for (int i = outputRank - 1; i >= 0; --i) {
      expr = expr + getAffineDimExpr(i, op.getContext()) *
                        getAffineConstantExpr(stride, op.getContext());
      stride *= outputShape[i];
    }

    AffineMap transformAffineMap = AffineMap::get(
        outputRank, 0, ArrayRef<AffineExpr>{expr}, op.getContext());
    AffineMap outputAffineMap;
    if (inputAffineMaps.size() != 0) {
      auto inputAffineMap = inputAffineMaps[0];
      outputAffineMap = inputAffineMap.compose(transformAffineMap);
    } else {
      outputAffineMap = transformAffineMap;
    }
    auto transformedOutputType =
        MemRefType::get(outputShape, outputElementType, {outputAffineMap},
                        inputType.getMemorySpace());
    return transformedOutputType;
  }

  LogicalResult matchAndRewrite(miopen::GridwiseGemmOp op, PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Prepare some useful constants.
    auto zeroConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));

    auto zeroConstantIndexOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantIndexOp = b.create<ConstantIndexOp>(loc, 1);

    auto ldsMemorySpace = 3;
    auto registerMemorySpace = 5;

    // Obtain critical matrix dimensions.
    int64_t K = op.filter().getType().template dyn_cast<MemRefType>().getShape()[0];
    int64_t M = op.filter().getType().template dyn_cast<MemRefType>().getShape()[1];
    int64_t N = op.input().getType().template dyn_cast<MemRefType>().getShape()[1];

    // Obtain critical tuning parameters.
    int64_t BlockSize =
        op.getAttr("block_size").template dyn_cast<IntegerAttr>().getInt();
    int64_t KPerBlock = op.getAttr("k_per_block").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerBlock = op.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerBlock = op.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerThread = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerThread = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
    auto MPerThreadConstantIndexOp = b.create<ConstantIndexOp>(loc, MPerThread);
    auto NPerThreadConstantIndexOp = b.create<ConstantIndexOp>(loc, NPerThread);

    int64_t MLevel0Cluster = op.getAttr("m_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster = op.getAttr("m_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster = op.getAttr("n_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster = op.getAttr("n_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
    auto NLevel0ClusterConstantIndexOp =
        b.create<ConstantIndexOp>(loc, NLevel0Cluster);
    auto NLevel1ClusterConstantIndexOp =
        b.create<ConstantIndexOp>(loc, NLevel1Cluster);

    int64_t matrix_a_source_data_per_read =
        op.getAttr("matrix_a_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op.getAttr("matrix_b_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();

    // Get current workgroup ID.
    auto bid = b.create<miopen::WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;
    auto MBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, MBlockWork);
    auto NBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, NBlockWork);
    auto block_work_id_m =
        b.create<SignedDivIOp>(loc, bid, NBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<SignedRemIOp>(loc, bid, NBlockWorkConstantOp);
    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MBlockWorkConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NBlockWorkConstantOp);
    auto m_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, m_block_data_on_global, b.getIntegerType(32));
    auto n_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, n_block_data_on_global, b.getIntegerType(32));

    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "matrix_a_source_data_per_read: " <<
    // matrix_a_source_data_per_read << "\n"; llvm::errs() <<
    // "matrix_b_source_data_per_read: " << matrix_b_source_data_per_read <<
    // "\n";

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock / matrix_a_source_data_per_read;
    int64_t GemmABlockCopyClusterLengths_GemmM =
        MPerBlock /
        ((MPerBlock * KPerBlock / BlockSize) / matrix_a_source_data_per_read);
    // Compute ThreadSliceLengths for Matrix A.
    int64_t GemmABlockCopyThreadSliceLengths_GemmK =
        KPerBlock / GemmABlockCopyClusterLengths_GemmK;
    int64_t GemmABlockCopyThreadSliceLengths_GemmM =
        MPerBlock / GemmABlockCopyClusterLengths_GemmM;

    // llvm::errs() << "slice lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyThreadSliceLengths_GemmM << "\n";

    // Compute ThreadClusterLengths for Matrix B.
    int64_t GemmBBlockCopyClusterLengths_GemmK =
        KPerBlock /
        ((NPerBlock * KPerBlock / BlockSize) / matrix_b_source_data_per_read);
    int64_t GemmBBlockCopyClusterLengths_GemmN =
        NPerBlock / matrix_b_source_data_per_read;
    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyThreadSliceLengths_GemmK =
        KPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN =
        NPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    // llvm::errs() << "slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";

    // Get current workitem ID.
    auto tid = b.create<miopen::WorkitemIdOp>(loc, b.getIndexType());

    // Compute thread_data_id_begin for Matrix A.
    // ClusterArrangeOrder for Matrix A is <1, 0>.
    // So divide by GemmABlockCopyClusterLengths_GemmK.
    auto GemmABlockCopyClusterLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyClusterLengths_GemmK);
    auto GemmABlockCopyClusterLengths_GemmMConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyClusterLengths_GemmM);
    auto GemmABlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmK);
    auto GemmABlockCopyThreadSliceLengths_GemmMConstantOp =
        b.create<ConstantIndexOp>(loc, GemmABlockCopyThreadSliceLengths_GemmM);

    auto GemmABlockCopyThreadClusterId_Y = b.create<SignedDivIOp>(
        loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
    auto GemmABlockCopyThreadClusterId_X = b.create<SignedRemIOp>(
        loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
    auto GemmAThreadDataIdBegin_Y =
        b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_Y,
                         GemmABlockCopyThreadSliceLengths_GemmKConstantOp);
    auto GemmAThreadDataIdBegin_X =
        b.create<MulIOp>(loc, GemmABlockCopyThreadClusterId_X,
                         GemmABlockCopyThreadSliceLengths_GemmMConstantOp);
    auto GemmAThreadDataIdBegin_Y_i32 = b.create<IndexCastOp>(
        loc, GemmAThreadDataIdBegin_Y, b.getIntegerType(32));
    auto GemmAThreadDataIdBegin_X_i32 = b.create<IndexCastOp>(
        loc, GemmAThreadDataIdBegin_X, b.getIntegerType(32));

    auto GemmABlockCopySourceCoord_Y_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmAThreadDataIdBegin_Y_i32);
    auto GemmABlockCopySourceCoord_X_i32 = b.create<AddIOp>(
        loc, m_block_data_on_global_i32, GemmAThreadDataIdBegin_X_i32);
    auto GemmABlockCopyDestCoord_Y_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmAThreadDataIdBegin_Y_i32);
    auto GemmABlockCopyDestCoord_X_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmAThreadDataIdBegin_X_i32);

    // Compute thread_data_id_begin for Matrix B.
    // ClusterArrangeOrder for Matrix B is <0, 1>
    // So divide by GemmBBlockCopyClusterLengths_GemmN.
    auto GemmBBlockCopyClusterLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyClusterLengths_GemmK);
    auto GemmBBlockCopyClusterLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyClusterLengths_GemmN);
    auto GemmBBlockCopyThreadSliceLengths_GemmKConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmK);
    auto GemmBBlockCopyThreadSliceLengths_GemmNConstantOp =
        b.create<ConstantIndexOp>(loc, GemmBBlockCopyThreadSliceLengths_GemmN);

    auto GemmBBlockCopyThreadClusterId_Y = b.create<SignedDivIOp>(
        loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
    auto GemmBBlockCopyThreadClusterId_X = b.create<SignedRemIOp>(
        loc, tid, GemmBBlockCopyClusterLengths_GemmNConstantOp);
    auto GemmBThreadDataIdBegin_Y =
        b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_Y,
                         GemmBBlockCopyThreadSliceLengths_GemmKConstantOp);
    auto GemmBThreadDataIdBegin_X =
        b.create<MulIOp>(loc, GemmBBlockCopyThreadClusterId_X,
                         GemmBBlockCopyThreadSliceLengths_GemmNConstantOp);
    auto GemmBThreadDataIdBegin_Y_i32 = b.create<IndexCastOp>(
        loc, GemmBThreadDataIdBegin_Y, b.getIntegerType(32));
    auto GemmBThreadDataIdBegin_X_i32 = b.create<IndexCastOp>(
        loc, GemmBThreadDataIdBegin_X, b.getIntegerType(32));

    auto GemmBBlockCopySourceCoord_Y_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmBThreadDataIdBegin_Y_i32);
    auto GemmBBlockCopySourceCoord_X_i32 = b.create<AddIOp>(
        loc, n_block_data_on_global_i32, GemmBThreadDataIdBegin_X_i32);
    auto GemmBBlockCopyDestCoord_Y_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmBThreadDataIdBegin_Y_i32);
    auto GemmBBlockCopyDestCoord_X_i32 =
        b.create<AddIOp>(loc, zeroConstantI32Op, GemmBThreadDataIdBegin_X_i32);

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    computeLDSBlockSizes(op, ldsBlockASize, ldsBlockBSize, ldsBlockSize);

    auto elementType = op.output().getType().cast<MemRefType>().getElementType();

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {}, ldsMemorySpace);
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockADoubleSize = ldsBlockASize * 2;
    auto ldsBlockAOffset = 0;

    auto ldsBlockAOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOffset);
    auto ldsBlockADoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockAOffset,
                                 {ldsBlockADoubleSize}, elementType);
    auto ldsBlockADoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockADoubleMemRefType, ldsGpuAllocOp,
        ldsBlockAOffsetConstantIndexOp);

    auto ldsBlockAEvenOffset = 0;
    auto ldsBlockAEvenOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAEvenOffset);
    auto ldsBlockAEvenMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAEvenOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAEvenMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAEvenOffsetConstantIndexOp);

    auto ldsBlockAOddOffset = ldsBlockADoubleSize / 2;
    auto ldsBlockAOddOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOddOffset);
    auto ldsBlockAOddMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAOddOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAOddMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAOddOffsetConstantIndexOp);

    // Get 2D subviews.
    // Compute matrix A dimension from attributes.
    // Original C++ logic.
    // // A matrix in LDS memory, dst of blockwise copy
    // //   be careful of LDS alignment
    // constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
    //     Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
    auto lds2DMatrixAEvenMemRefType = computeSubviewResultType(
        op, ldsBlockAEvenMemRefType, 0, {KPerBlock, MPerBlock}, elementType);

    auto lds2DMatrixAOddMemRefType = computeSubviewResultType(
        op, ldsBlockAOddMemRefType, 0, {KPerBlock, MPerBlock}, elementType);

    auto lds2DMatrixAEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, lds2DMatrixAEvenMemRefType, ldsBlockAEvenSubviewOp,
        zeroConstantIndexOp);
    auto lds2DMatrixAOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixAOddMemRefType,
                                    ldsBlockAOddSubviewOp, zeroConstantIndexOp);

    // Subviews for Matrix B.
    auto ldsBlockBDoubleSize = ldsBlockBSize * 2;
    auto ldsBlockBOffset = ldsBlockSize - ldsBlockADoubleSize;

    auto ldsBlockBOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOffset);
    auto ldsBlockBDoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockBOffset,
                                 {ldsBlockBDoubleSize}, elementType);
    auto ldsBlockBDoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBDoubleMemRefType, ldsGpuAllocOp,
        ldsBlockBOffsetConstantIndexOp);

    auto ldsBlockBEvenOffset = 0;
    auto ldsBlockBEvenOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBEvenOffset);
    auto ldsBlockBEvenMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBEvenOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBEvenMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBEvenOffsetConstantIndexOp);

    auto ldsBlockBOddOffset = ldsBlockBDoubleSize / 2;
    auto ldsBlockBOddOffsetConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOddOffset);
    auto ldsBlockBOddMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBOddOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBOddMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBOddOffsetConstantIndexOp);

    // Get 2D subviews.
    // Compute matrix B dimension from attributes.
    // Original C++ logic.
    // // B matrix in LDS memory, dst of blockwise copy
    // //   be careful of LDS alignment
    // constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
    //     Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
    auto lds2DMatrixBEvenMemRefType = computeSubviewResultType(
        op, ldsBlockBEvenMemRefType, 0, {KPerBlock, NPerBlock}, elementType);

    auto lds2DMatrixBOddMemRefType = computeSubviewResultType(
        op, ldsBlockBOddMemRefType, 0, {KPerBlock, NPerBlock}, elementType);

    auto lds2DMatrixBEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, lds2DMatrixBEvenMemRefType, ldsBlockBEvenSubviewOp,
        zeroConstantIndexOp);
    auto lds2DMatrixBOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixBOddMemRefType,
                                    ldsBlockBOddSubviewOp, zeroConstantIndexOp);

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    // Original C++ logic.
    // constexpr index_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    // constexpr index_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);
    // constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
    //     Number<GemmMRepeat * MPerThread>{}, Number<GemmNRepeat * NPerThread>{});
    int64_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    int64_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

    auto threadCRegisterMemRefType =
        MemRefType::get({GemmMRepeat * MPerThread, GemmNRepeat * NPerThread}, elementType, {}, registerMemorySpace);
    auto register2DMatrixCAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadCRegisterMemRefType);

    // Alloc for Matrix A / B on registers.
    auto threadARegisterMemRefType =
        MemRefType::get({GemmABlockCopyThreadSliceLengths_GemmK,
                         GemmABlockCopyThreadSliceLengths_GemmM},
                        elementType, {}, registerMemorySpace);
    auto threadAEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);
    auto threadAOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get({GemmBBlockCopyThreadSliceLengths_GemmK,
                         GemmBBlockCopyThreadSliceLengths_GemmN},
                        elementType, {}, registerMemorySpace);
    auto threadBEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);
    auto threadBOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Zero init Matrix C on registers.
    b.create<miopen::FillOp>(loc, register2DMatrixCAllocOp,
                             zeroConstantFloatOp);

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // Compute source and destination coordinates for BlockwiseCopy ops.
    auto blockwiseCopyCoordType =
              MemRefType::get({2}, b.getIntegerType(32), {}, registerMemorySpace);

    // Matrix A: {0, m_block_data_on_global}, {0, 0}
    auto blockwiseCopyASrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_Y_i32, blockwiseCopyASrc,
                      ValueRange{zeroConstantIndexOp});
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_X_i32, blockwiseCopyASrc,
                      ValueRange{oneConstantIndexOp});

    auto blockwiseCopyADst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_Y_i32, blockwiseCopyADst,
                      ValueRange{zeroConstantIndexOp});
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_X_i32, blockwiseCopyADst,
                      ValueRange{oneConstantIndexOp});

    // Matrix B: {0, n_block_data_on_global}, {0, 0}
    auto blockwiseCopyBSrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_Y_i32, blockwiseCopyBSrc,
                      ValueRange{zeroConstantIndexOp});
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_X_i32, blockwiseCopyBSrc,
                      ValueRange{oneConstantIndexOp});

    auto blockwiseCopyBDst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_Y_i32, blockwiseCopyBDst,
                      ValueRange{zeroConstantIndexOp});
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_X_i32, blockwiseCopyBDst,
                      ValueRange{oneConstantIndexOp});

    // Compute c_thread_mtx_index for Matrix C.
    int64_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;
    auto ThreadPerLevel0ClusterConstantIndexOp =
        b.create<ConstantIndexOp>(loc, ThreadPerLevel0Cluster);
    auto level1_id =
        b.create<SignedDivIOp>(loc, tid, ThreadPerLevel0ClusterConstantIndexOp);
    auto level1_m_id =
        b.create<SignedDivIOp>(loc, level1_id, NLevel1ClusterConstantIndexOp);
    auto level1_n_id =
        b.create<SignedRemIOp>(loc, level1_id, NLevel1ClusterConstantIndexOp);

    auto level0_id =
        b.create<SignedRemIOp>(loc, tid, ThreadPerLevel0ClusterConstantIndexOp);
    auto level0_m_id =
        b.create<SignedDivIOp>(loc, level0_id, NLevel1ClusterConstantIndexOp);
    auto level0_n_id =
        b.create<SignedRemIOp>(loc, level0_id, NLevel1ClusterConstantIndexOp);

    int64_t MPerLevel0Cluster = MPerThread * MLevel0Cluster;
    int64_t NPerLevel0Cluster = NPerThread * NLevel0Cluster;
    auto MPerLevel0ClusterConstantIndexOp =
        b.create<ConstantIndexOp>(loc, MPerLevel0Cluster);
    auto NPerLevel0ClusterConstantIndexOp =
        b.create<ConstantIndexOp>(loc, NPerLevel0Cluster);

    // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0, c_thread_mtx_index.row} = c_thread_mtx_index_row
    auto c_thread_mtx_index_row = b.create<AddIOp>(
        loc,
        b.create<MulIOp>(loc, level1_m_id, MPerLevel0ClusterConstantIndexOp),
        b.create<MulIOp>(loc, level0_m_id, MPerThreadConstantIndexOp));
    auto c_thread_mtx_index_row_i32 = b.create<IndexCastOp>(
        loc, c_thread_mtx_index_row, b.getIntegerType(32));

    // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0, c_thread_mtx_index.col} = c_thread_mtx_index_col
    auto c_thread_mtx_index_col = b.create<AddIOp>(
        loc,
        b.create<MulIOp>(loc, level1_n_id, NPerLevel0ClusterConstantIndexOp),
        b.create<MulIOp>(loc, level0_n_id, NPerThreadConstantIndexOp));
    auto c_thread_mtx_index_col_i32 = b.create<IndexCastOp>(
        loc, c_thread_mtx_index_col, b.getIntegerType(32));

    auto m_thread_data_on_global_i32 = b.create<AddIOp>(
        loc, m_block_data_on_global_i32, c_thread_mtx_index_row_i32);
    auto n_thread_data_on_global_i32 = b.create<AddIOp>(
        loc, n_block_data_on_global_i32, c_thread_mtx_index_col_i32);

    // Emit BlockwiseCopy ops.
    auto blockwiseCopyA = b.create<miopen::BlockwiseCopyOp>(
        loc, op.filter(), lds2DMatrixAEvenSubviewOp, blockwiseCopyASrc,
        blockwiseCopyADst, threadAOddAllocOp);
    affixBlockwiseCopyAttributes(blockwiseCopyA, op, b, /*isMatrixA=*/true);
    auto blockwiseCopyB = b.create<miopen::BlockwiseCopyOp>(
        loc, op.input(), lds2DMatrixBEvenSubviewOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, threadBOddAllocOp);
    affixBlockwiseCopyAttributes(blockwiseCopyB, op, b, /*isMatrixA=*/false);

    // Emit loop.
    // Compute loop iterations from attributes.
    auto loopIteration = K / (KPerBlock * 2);
    auto loopIterationConstantIndexOp =
        b.create<ConstantIndexOp>(loc, loopIteration);
    auto loopOp =
        b.create<scf::ForOp>(loc, zeroConstantIndexOp,
                             loopIterationConstantIndexOp, oneConstantIndexOp);

    // inside the loop.
    auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

    // LDS barrier.
    lb.create<miopen::WorkgroupBarrierOp>(loc);

    auto KPerBlockConstantI32Op =
        b.create<ConstantIntOp>(loc, KPerBlock, b.getIntegerType(32));

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    lb.create<miopen::MovePosOp>(
        loc, blockwiseCopyASrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpAEven = lb.create<miopen::BlockwiseCopyOp>(
        loc, op.filter(), threadAEvenAllocOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAEven, op, b,
                                 /*isMatrixA=*/true);
    lb.create<miopen::MovePosOp>(
        loc, blockwiseCopyBSrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpBEven = lb.create<miopen::BlockwiseCopyOp>(
        loc, op.input(), threadBEvenAllocOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBEven, op, b,
                                 /*isMatrixA=*/false);

    // Emit blockwise GEMM.
    auto blockwiseGemmEvenOp = lb.create<miopen::BlockwiseGemmOp>(
        loc, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
        register2DMatrixCAllocOp, c_thread_mtx_index_row,
        c_thread_mtx_index_col);
    affixBlockwiseGemmAttributes(blockwiseGemmEvenOp, op);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    auto blockwiseCopyOpAOdd = lb.create<miopen::BlockwiseCopyOp>(
        loc, threadAEvenAllocOp, lds2DMatrixAOddSubviewOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAOdd, op, b,
                                 /*isMatrixA=*/true);
    auto blockwiseCopyOpBOdd = lb.create<miopen::BlockwiseCopyOp>(
        loc, threadBEvenAllocOp, lds2DMatrixBOddSubviewOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBOdd, op, b,
                                 /*isMatrixA=*/false);

    // LDS barrier.
    lb.create<miopen::WorkgroupBarrierOp>(loc);

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    lb.create<miopen::MovePosOp>(
        loc, blockwiseCopyASrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpAOddSecondIteration =
        lb.create<miopen::BlockwiseCopyOp>(loc, op.filter(), threadAOddAllocOp,
                                           blockwiseCopyASrc, blockwiseCopyADst,
                                           /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAOddSecondIteration, op, b,
                                 /*isMatrixA=*/true);
    lb.create<miopen::MovePosOp>(
        loc, blockwiseCopyBSrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpBOddSecondIteration =
        lb.create<miopen::BlockwiseCopyOp>(loc, op.input(), threadBOddAllocOp,
                                           blockwiseCopyBSrc, blockwiseCopyBDst,
                                           /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBOddSecondIteration, op, b,
                                 /*isMatrixA=*/false);

    // Emit blockwise GEMM.
    auto blockwiseGemmOddOp = lb.create<miopen::BlockwiseGemmOp>(
        loc, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
        register2DMatrixCAllocOp, c_thread_mtx_index_row,
        c_thread_mtx_index_col);
    affixBlockwiseGemmAttributes(blockwiseGemmOddOp, op);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    auto blockwiseCopyAEvenSecondIteration = lb.create<miopen::BlockwiseCopyOp>(
        loc, threadAOddAllocOp, lds2DMatrixAEvenSubviewOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyAEvenSecondIteration, op, b,
                                 /*isMatrixA=*/true);
    auto blockwiseCopyBEvenSecondIteration = lb.create<miopen::BlockwiseCopyOp>(
        loc, threadBOddAllocOp, lds2DMatrixBEvenSubviewOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyBEvenSecondIteration, op, b,
                                 /*isMatrixA=*/false);

    // outside the loop.

    // LDS barrier.
    b.create<miopen::WorkgroupBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    if (loopIteration % 2) {
      auto blockwiseGemmTailEvenOp = b.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
          register2DMatrixCAllocOp, c_thread_mtx_index_row,
          c_thread_mtx_index_col);
      affixBlockwiseGemmAttributes(blockwiseGemmTailEvenOp, op);
    } else {
      auto blockwiseGemmTailOddOp = b.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
          register2DMatrixCAllocOp, c_thread_mtx_index_row,
          c_thread_mtx_index_col);
      affixBlockwiseGemmAttributes(blockwiseGemmTailOddOp, op);
    }

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    // Add attributes from C++ template arguments and ctor arguments.
    // ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_n0_n1_thread_desc),
    //                                       decltype(c_m0_m1_n0_n1_global_desc),
    //                                       decltype(c_m0_m1_n0_n1_thread_desc.GetLengths()),
    //                                       CThreadCopySrcDstAccessOrder,
    //                                       CThreadCopySrcDstVectorReadWriteDim,
    //                                       1,
    //                                       CThreadCopyDstDataPerWrite,
    //                                       AddressSpace::Vgpr,
    //                                       AddressSpace::Global,
    //                                       CGlobalMemoryDataOperation>(
    //     {0, 0, 0, 0},
    //     {m_thread_data_on_global / M1,
    //      m_thread_data_on_global % M1,
    //      n_thread_data_on_global / N1,
    //      n_thread_data_on_global % N1})
    //     .Run(p_c_thread, p_c_global);
    // XXX. Use 2D coordinate only.
    SmallVector<Value, 4> matrixCThreadwiseCopySourceAndDestCoords;
    matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
    matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
    matrixCThreadwiseCopySourceAndDestCoords.push_back(
        m_thread_data_on_global_i32);
    matrixCThreadwiseCopySourceAndDestCoords.push_back(
        n_thread_data_on_global_i32);
    auto threadwiseCopyCMatrixOp = b.create<miopen::ThreadwiseCopyOp>(
        loc, register2DMatrixCAllocOp, op.output(),
        matrixCThreadwiseCopySourceAndDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyCMatrixOp, op, b);

    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmRewritePattern : public OpRewritePattern<miopen::BlockwiseGemmOp> {
  using OpRewritePattern<miopen::BlockwiseGemmOp>::OpRewritePattern;

  LogicalResult naiveRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const {
    auto loc = op.getLoc();

    // Prepare some useful constants.
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));
    auto zeroConstantIndexOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantIndexOp = b.create<ConstantIndexOp>(loc, 1);

    auto registerMemorySpace = 5;

    auto blockAType = op.matrixA().getType().cast<MemRefType>();
    auto blockBType = op.matrixA().getType().cast<MemRefType>();

    auto elementType = op.matrixC().getType().cast<MemRefType>().getElementType();

    // Obtain critical matrix dimensions.
    int64_t K = blockAType.getShape()[0];
    int64_t M = blockAType.getShape()[1];
    int64_t N = blockBType.getShape()[1];

    // Obtain critical attributes.
    int64_t KPerThread = op.getAttr("k_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerThread =
        op.matrixC().getType().template dyn_cast<MemRefType>().getShape()[0];
    int64_t NPerThread =
        op.matrixC().getType().template dyn_cast<MemRefType>().getShape()[1];
    int64_t MPerThreadSubC = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerThreadSubC = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
    auto MPerThreadSubCConstantI32Op =
        b.create<ConstantIntOp>(loc, MPerThreadSubC, b.getIntegerType(32));
    auto NPerThreadSubCConstantI32Op =
        b.create<ConstantIntOp>(loc, NPerThreadSubC, b.getIntegerType(32));

    int64_t MLevel0Cluster = op.getAttr("m_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster = op.getAttr("m_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster = op.getAttr("n_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster = op.getAttr("n_level1_cluster").template dyn_cast<IntegerAttr>().getInt();

    int64_t MPerLevel1Cluster = MPerThread * MLevel0Cluster * MLevel1Cluster;
    int64_t NPerLevel1Cluster = NPerThread * NLevel0Cluster * NLevel1Cluster;
    auto MPerLevel1ClusterConstantI32Op =
        b.create<ConstantIntOp>(loc, MPerLevel1Cluster, b.getIntegerType(32));
    auto NPerLevel1ClusterConstantI32Op =
        b.create<ConstantIntOp>(loc, NPerLevel1Cluster, b.getIntegerType(32));

    int64_t MRepeat = MPerThread / MPerThreadSubC;
    int64_t NRepeat = NPerThread / NPerThreadSubC;
    auto MRepeatConstantI32Op =
        b.create<ConstantIntOp>(loc, MRepeat, b.getIntegerType(32));
    auto NRepeatConstantI32Op =
        b.create<ConstantIntOp>(loc, NRepeat, b.getIntegerType(32));

    // Alloc register for thread_a and thread_b.
    auto threadARegisterMemRefType =
        MemRefType::get({KPerThread, MPerThread}, elementType, {}, registerMemorySpace);
    auto threadAAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get({KPerThread, NPerThread}, elementType, {}, registerMemorySpace);
    auto threadBAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Main loop.
    auto loopIteration = K / KPerThread;
    auto loopIterationConstantIndexOp =
        b.create<ConstantIndexOp>(loc, loopIteration);
    auto loopOp =
        b.create<scf::ForOp>(loc, zeroConstantIndexOp,
                             loopIterationConstantIndexOp, oneConstantIndexOp);

    // inside the main loop.
    auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

    auto iv = loopOp.getInductionVar();
    auto iv_i32 = lb.create<IndexCastOp>(loc, iv, lb.getIntegerType(32));

    // read matrix A loop.
    auto loopReadMatrixAIteration = MRepeat;
    auto loopReadMatrixAIterationConstantIndexOp =
        lb.create<ConstantIndexOp>(loc, loopReadMatrixAIteration);
    auto loopReadMatrixAOp = lb.create<scf::ForOp>(
        loc, zeroConstantIndexOp, loopReadMatrixAIterationConstantIndexOp,
        oneConstantIndexOp);

    // inside read matrix A loop.
    auto lab = OpBuilder::atBlockTerminator(loopReadMatrixAOp.getBody());

    auto iva = loopReadMatrixAOp.getInductionVar();
    auto iva_i32 = lab.create<IndexCastOp>(loc, iva, lab.getIntegerType(32));

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixAThreadwiseCopySourceAndDestCoords;
    // a_thread_copy.Run(
    //   p_a_block + a_block_mtx.CalculateOffset(k_begin, m_repeat *
    //   MPerLevel1Cluster) + mMyThreadOffsetA),
    // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.row} = c_thread_mtx_index_row
    matrixAThreadwiseCopySourceAndDestCoords.push_back(iv_i32);
    matrixAThreadwiseCopySourceAndDestCoords.push_back(lab.create<AddIOp>(
        loc, lab.create<MulIOp>(loc, iva_i32, MPerLevel1ClusterConstantI32Op),
        lab.create<IndexCastOp>(loc, op.threadOffsetA(),
                                lab.getIntegerType(32))));

    //   p_a_thread + a_thread_mtx.CalculateOffset(0, m_repeat * MPerThreadSubC));
    matrixAThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
    matrixAThreadwiseCopySourceAndDestCoords.push_back(
        lab.create<MulIOp>(loc, iva_i32, MPerThreadSubCConstantI32Op));

    // Emit threadwise_copy.
    auto threadwiseCopyAMatrixOp = lab.create<miopen::ThreadwiseCopyOp>(
        loc, op.matrixA(), threadAAllocOp,
        matrixAThreadwiseCopySourceAndDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyAMatrixOp, op, b,
                                  /*isMatrixA=*/true);

    // read matrix B loop.
    auto loopReadMatrixBIteration = NRepeat;
    auto loopReadMatrixBIterationConstantIndexOp =
        lb.create<ConstantIndexOp>(loc, loopReadMatrixBIteration);
    auto loopReadMatrixBOp = lb.create<scf::ForOp>(
        loc, zeroConstantIndexOp, loopReadMatrixBIterationConstantIndexOp,
        oneConstantIndexOp);

    // inside read matrix A loop.
    auto lbb = OpBuilder::atBlockTerminator(loopReadMatrixBOp.getBody());

    auto ivb = loopReadMatrixBOp.getInductionVar();
    auto ivb_i32 = lbb.create<IndexCastOp>(loc, ivb, lbb.getIntegerType(32));

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).

    // Set copy sorce and dest coordinate acoording to original C++ logic:
    SmallVector<Value, 4> matrixBThreadwiseCopySourceAndDestCoords;
    // b_thread_copy.Run(
    //   p_b_block + b_block_mtx.CalculateOffset(k_begin, n_repeat *
    //   NPerLevel1Cluster) + mMyThreadOffsetB),
    // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0,
    // c_thread_mtx_index.col} = c_thread_mtx_index_col
    matrixBThreadwiseCopySourceAndDestCoords.push_back(iv_i32);
    matrixBThreadwiseCopySourceAndDestCoords.push_back(lbb.create<AddIOp>(
        loc, lbb.create<MulIOp>(loc, ivb_i32, NPerLevel1ClusterConstantI32Op),
        lbb.create<IndexCastOp>(loc, op.threadOffsetB(),
                                lbb.getIntegerType(32))));

    //   p_b_thread + b_thread_mtx.CalculateOffset(0, n_repeat * NPerThreadSubC));
    matrixBThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
    matrixBThreadwiseCopySourceAndDestCoords.push_back(
        lbb.create<MulIOp>(loc, ivb_i32, NPerThreadSubCConstantI32Op));

    // Emit threadwise_copy.
    auto threadwiseCopyBMatrixOp = lbb.create<miopen::ThreadwiseCopyOp>(
        loc, op.matrixB(), threadBAllocOp,
        matrixBThreadwiseCopySourceAndDestCoords);
    affixThreadwiseCopyAttributes(threadwiseCopyBMatrixOp, op, b,
                                  /*isMatrixA=*/false);

    lb.create<miopen::ThreadwiseGemmOp>(loc, threadAAllocOp, threadBAllocOp,
                                        op.matrixC());

    op.erase();
    return success();
  }

  LogicalResult twoByTwoPipelinedRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const {
    // TBD implement 2x2 pipelined version.
    op.erase();
    return success();
  }

  LogicalResult matchAndRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const override {
    // TBD condition upon attributes.
    if (true) {
      return naiveRewrite(op, b);
    } else {
      return twoByTwoPipelinedRewrite(op, b);
    }
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseCopy lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseCopyRewritePattern : public OpRewritePattern<miopen::BlockwiseCopyOp> {
  using OpRewritePattern<miopen::BlockwiseCopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::BlockwiseCopyOp op, PatternRewriter &b) const override {
    bool rewritten = true;

    auto loc = op.getLoc();

    MemRefType sourceType = op.source().getType().cast<MemRefType>();
    MemRefType destType = op.dest().getType().cast<MemRefType>();
    MemRefType bufferType;
    if (op.buffer())
      bufferType = op.buffer().getType().cast<MemRefType>();

    auto elementType = destType.getElementType();

    // Prepare some useful constants.
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));

    // Check the address spaces of source and destination values and determine
    // lowering logic.
    // - 0 (global) -> 3 (LDS) : load + store
    // - 0 (global) -> 5 (register) : load 
    // - 5 (register) -> 3 (LDS) : store
    if (sourceType.getMemorySpace() == 0 && destType.getMemorySpace() == 3) {
      // Threadwise copy from global (generic tensor) to register (naive
      // tensor).
      SmallVector<Value, 4> ThreadwiseCopySourceAndBufferCoords;
      for (unsigned i = 0; i < sourceType.getRank(); ++i) {
        auto indexConstantOp = b.create<ConstantIndexOp>(loc, i);
        auto coord = b.create<LoadOp>(loc, op.sourceCoord(),
                                      ValueRange{indexConstantOp});
        ThreadwiseCopySourceAndBufferCoords.push_back(coord);
      }
      for (unsigned i = 0; i < bufferType.getRank(); ++i)
        ThreadwiseCopySourceAndBufferCoords.push_back(zeroConstantI32Op);

      auto threadwiseCopyLoadOp = b.create<miopen::ThreadwiseCopyOp>(
          loc, op.source(), op.buffer(), ThreadwiseCopySourceAndBufferCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyLoadOp, op, b,
                                    /*isThreadwiseLoad=*/true);

      // Threadwise copy from register (naive tensor) to LDS (naive tensor).
      SmallVector<Value, 4> ThreadwiseCopyBufferAndDestCoords;
      for (unsigned i = 0; i < bufferType.getRank(); ++i)
        ThreadwiseCopyBufferAndDestCoords.push_back(zeroConstantI32Op);
      for (unsigned i = 0; i < destType.getRank(); ++i) {
        auto indexConstantOp = b.create<ConstantIndexOp>(loc, i);
        auto coord =
            b.create<LoadOp>(loc, op.destCoord(), ValueRange{indexConstantOp});
        ThreadwiseCopyBufferAndDestCoords.push_back(coord);
      }

      auto threadwiseCopyStoreOp = b.create<miopen::ThreadwiseCopyOp>(
          loc, op.buffer(), op.dest(), ThreadwiseCopyBufferAndDestCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyStoreOp, op, b,
                                    /*isThreadwiseLoad=*/false);
    } else if (sourceType.getMemorySpace() == 0 && destType.getMemorySpace() == 5) {
      // Threadwise copy from global (generic tensor) to register (naive
      // tensor).
      SmallVector<Value, 4> ThreadwiseCopySourceAndDestCoords;
      for (unsigned i = 0; i < sourceType.getRank(); ++i) {
        auto indexConstantOp = b.create<ConstantIndexOp>(loc, i);
        auto coord = b.create<LoadOp>(loc, op.sourceCoord(),
                                      ValueRange{indexConstantOp});
        ThreadwiseCopySourceAndDestCoords.push_back(coord);
      }
      for (unsigned i = 0; i < destType.getRank(); ++i)
        ThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);

      auto threadwiseCopyLoadOp = b.create<miopen::ThreadwiseCopyOp>(
          loc, op.source(), op.dest(), ThreadwiseCopySourceAndDestCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyLoadOp, op, b,
                                    /*isThreadwiseLoad=*/true);
    } else if (sourceType.getMemorySpace() == 5 && destType.getMemorySpace() == 3) {
      // Threadwise copy from register (naive tensor) to LDS (naive tensor).
      SmallVector<Value, 4> ThreadwiseCopySourceAndDestCoords;
      for (unsigned i = 0; i < sourceType.getRank(); ++i)
        ThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      for (unsigned i = 0; i < destType.getRank(); ++i) {
        auto indexConstantOp = b.create<ConstantIndexOp>(loc, i);
        auto coord =
            b.create<LoadOp>(loc, op.destCoord(), ValueRange{indexConstantOp});
        ThreadwiseCopySourceAndDestCoords.push_back(coord);
      }

      auto threadwiseCopyStoreOp = b.create<miopen::ThreadwiseCopyOp>(
          loc, op.source(), op.dest(), ThreadwiseCopySourceAndDestCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyStoreOp, op, b,
                                    /*isThreadwiseLoad=*/false);
    } else {
      llvm::errs() << "UNSUPPORTED ThreadwiseCopyOp\n";
      rewritten = false;
    }

    if (rewritten)
      op.erase();
    return success();
  }
}; 

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpRewritePattern<miopen::FillOp> {
  using OpRewritePattern<miopen::FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::FillOp op, PatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto inputType = op.input().getType().cast<MemRefType>();
    auto inputShape = inputType.getShape();

    auto zero = b.create<ConstantIndexOp>(loc, 0);
    auto one = b.create<ConstantIndexOp>(loc, 1);

    if (inputShape.size() == 1) {
      // Rank 1 loop.
      auto loopIteration = b.create<ConstantIndexOp>(loc, inputShape[0]);
      auto loopOp = b.create<scf::ForOp>(loc, zero, loopIteration, one);

      // inside loop.
      auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

      for (unsigned i = 0; i < inputShape[0]; ++i) {
        auto iter = b.create<ConstantIndexOp>(loc, i);
        lb.create<StoreOp>(loc, op.value(), op.input(), ValueRange{iter});
      }
    } else if (inputShape.size() == 2) {
      // Rank 2 loop.
      auto loop0Iteration = b.create<ConstantIndexOp>(loc, inputShape[0]);
      auto loop0Op = b.create<scf::ForOp>(loc, zero, loop0Iteration, one);

      // inside outer loop.
      auto l0b = OpBuilder::atBlockTerminator(loop0Op.getBody());

      for (unsigned i = 0; i < inputShape[0]; ++i) {
        auto iter0 = b.create<ConstantIndexOp>(loc, i);

        auto loop1Iteration = b.create<ConstantIndexOp>(loc, inputShape[1]);
        auto loop1Op = l0b.create<scf::ForOp>(loc, zero, loop1Iteration, one);

        // inside inner loop.
        auto l1b = OpBuilder::atBlockTerminator(loop1Op.getBody());

        for (unsigned j = 0; j < inputShape[1]; ++j) {
          auto iter1 = b.create<ConstantIndexOp>(loc, j);

          l1b.create<StoreOp>(loc, op.value(), op.input(), ValueRange{iter0, iter1});
        }
      }
    }

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MovePos lowering.
//===----------------------------------------------------------------------===//

struct MovePosRewritePattern : public OpRewritePattern<miopen::MovePosOp> {
  using OpRewritePattern<miopen::MovePosOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::MovePosOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto memrefType = op.memref().getType().cast<MemRefType>();
    for (unsigned i = 0; i < memrefType.getShape()[0]; ++i) {
      auto iter = b.create<ConstantIndexOp>(loc, i);
      // load
      auto load = b.create<LoadOp>(loc, op.memref(), ValueRange{iter});
      // add
      Value add;
      if (memrefType.getElementType().isa<IntegerType>()) {
        add = b.create<AddIOp>(loc, load, op.getOperand(1 + i));
      } else {
        add = b.create<AddFOp>(loc, load, op.getOperand(1 + i));
      }
      // store
      auto store = b.create<StoreOp>(loc, add, op.memref(), ValueRange{iter});
    }
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

    ArrayRef<int64_t> gemmAShape =
        gemmA.getType().dyn_cast<MemRefType>().getShape();
    ArrayRef<int64_t> gemmBShape =
        gemmB.getType().dyn_cast<MemRefType>().getShape();

    auto loopK = b.create<AffineForOp>(loc, 0, gemmAShape[0], 1);
    auto lbK = loopK.getBody();
    b.setInsertionPointToStart(lbK);

    auto loopM = b.create<AffineForOp>(loopK.getLoc(), 0, gemmAShape[1], 1);
    auto lbM = loopM.getBody();
    b.setInsertionPointToStart(lbM);

    auto loopN = b.create<AffineForOp>(loc, 0, gemmBShape[1], 1);
    auto lbN = loopN.getBody();
    b.setInsertionPointToStart(lbN);

    SmallVector<Value, 2> memIndicesKM;
    extractForInductionVars({loopK, loopM}, &memIndicesKM);
    auto gemmAKM = b.create<AffineLoadOp>(loc, gemmA, memIndicesKM);

    SmallVector<Value, 2> memIndicesKN;
    extractForInductionVars({loopK, loopN}, &memIndicesKN);
    auto gemmBKN = b.create<AffineLoadOp>(loc, gemmB, memIndicesKN);
    auto mul = b.create<MulFOp>(loc, b.getF32Type(), gemmAKM, gemmBKN);

    SmallVector<Value, 2> memIndicesMN;
    extractForInductionVars({loopM, loopN}, &memIndicesMN);
    auto gemmCMN = b.create<AffineLoadOp>(loc, gemmC, memIndicesMN);

    auto add = b.create<AddFOp>(loc, b.getF32Type(), mul, gemmCMN);
    auto store = b.create<AffineStoreOp>(loc, add, gemmC, memIndicesMN);

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

  LogicalResult matchAndRewrite(miopen::ThreadwiseCopyOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto zeroConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    auto zeroConstantIndexOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantIndexOp = b.create<ConstantIndexOp>(loc, 1);

    auto sourceType = op.source().getType().cast<MemRefType>();
    auto destType = op.dest().getType().cast<MemRefType>();

    // Get source and dest coordinates.
    //
    // 1. For memrefs with no externally defined affine maps in coord_transforms
    //    attribute, or embedded affine maps. Use its rank.
    // 2. For memrefs with externally defined maps, use its input rank.
    // 3. For memrefs with embedded maps, use its input rank.
    auto sourceAndDestCoord = op.sourceAndDestCoord();
    auto sourceTypeAffineMaps = sourceType.getAffineMaps();
    auto destTypeAffineMaps = destType.getAffineMaps();
    auto coordTransformsAttr =
        op.getAttr("coord_transforms").template cast<ArrayAttr>();

    unsigned sourceCoordLength = sourceType.getRank();
    unsigned destCoordLength = destType.getRank();

    bool sourceEmbeddedTransform = false;
    bool destEmbeddedTransform = false;
    bool sourceExternalTransform = false;
    bool destExternalTransform = false;
    AffineMap sourceTransform;
    AffineMap destTransform;

    if (sourceTypeAffineMaps.size()) {
      // Use the first affine map in the attribute array.
      sourceCoordLength = sourceTypeAffineMaps[0].getNumInputs();
      sourceEmbeddedTransform = true;
      sourceTransform = sourceTypeAffineMaps[0];
    }
    if (destTypeAffineMaps.size()) {
      // Use the first affine map in the attribute array.
      destCoordLength = destTypeAffineMaps[0].getNumInputs();
      destEmbeddedTransform = true;
      destTransform = destTypeAffineMaps[0];
    }
    if (coordTransformsAttr) {
      for (auto attr : coordTransformsAttr) {
        auto dictAttr = attr.template cast<DictionaryAttr>();
        auto operandIndex =
            dictAttr.get("operand").template cast<IntegerAttr>().getInt();
        auto transforms = dictAttr.get("transforms").template cast<ArrayAttr>();
        // Use the first affine map in the transforms array.
        auto affineMap = transforms[0].template cast<AffineMapAttr>();
        if (operandIndex == 0) {
          sourceCoordLength = affineMap.getValue().getNumInputs();
          sourceExternalTransform = true;
          sourceTransform = affineMap.getValue();
        } else {
          destCoordLength = affineMap.getValue().getNumInputs();
          destExternalTransform = true;
          destTransform = affineMap.getValue();
        }
      }
    }

    if (sourceCoordLength + destCoordLength != sourceAndDestCoord.size()) {
      llvm::errs() << "INCORRECT source and dest coordinates assigned!";
      return failure();
    }

    llvm::SmallVector<Value, 2> sourceCoord;
    llvm::SmallVector<Value, 2> destCoord;
    for (unsigned i = 0; i < sourceCoordLength; ++i) {
      sourceCoord.push_back(sourceAndDestCoord[i]);
    }
    for (unsigned i = sourceCoordLength;
         i < sourceCoordLength + destCoordLength; ++i) {
      destCoord.push_back(sourceAndDestCoord[i]);
    }

    // Distinguish between generic <-> naive v naive <-> naive tensors.
    //
    // In cases where attributes n_slice_row/n_slice_col/data_per_access are
    // specified, source and dest memrefs are all on LDS or VGPR, use the
    // simpler algorithm because they are all naive tensors.
    //
    // Otherwise, employ the more elaborated algorithm.
    auto NSliceRowAttr = op.getAttr("n_slice_row");
    auto NSliceColAttr = op.getAttr("n_slice_col");
    auto DataPerAccessAttr = op.getAttr("data_per_access");
    if (NSliceRowAttr && NSliceColAttr && DataPerAccessAttr) {
      auto NSliceRow = NSliceRowAttr.template cast<IntegerAttr>().getInt();
      auto NSliceCol = NSliceColAttr.template cast<IntegerAttr>().getInt();
      auto DataPerAccess =
          DataPerAccessAttr.template cast<IntegerAttr>().getInt();

      // Original C++ logic:
      // template <typename SrcMatrix,
      //           typename DstMatrix,
      //           index_t NSliceRow,
      //           index_t NSliceCol,
      //           index_t DataPerAccess>
      // struct ThreadwiseMatrixSliceCopy
      // {
      //     __device__ constexpr ThreadwiseMatrixSliceCopy()
      //     {
      //         static_assert(SrcMatrix::RowStride() % DataPerAccess == 0 &&
      //                           DstMatrix::RowStride() % DataPerAccess == 0,
      //                       "wrong! wrong alignment");
      //         static_assert(NSliceCol % DataPerAccess == 0,
      //                       "wrong! should be NSliceCol % DataPerAccess ==
      //                       0");
      //     }
      //
      //     template <typename Data>
      //     __device__ static void Run(const Data* p_src, Data* p_dst)
      //     {
      //         using vector_t = typename vector_type<Data,
      //         DataPerAccess>::MemoryType;
      //
      //         for(index_t i = 0; i < NSliceRow; ++i)
      //         {
      //             for(index_t j = 0; j < NSliceCol; j += DataPerAccess)
      //             {
      //                 const index_t src_index = SrcMatrix::CalculateOffset(i,
      //                 j); const index_t dst_index =
      //                 DstMatrix::CalculateOffset(i, j);
      //
      //                 *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
      //                     *reinterpret_cast<const
      //                     vector_t*>(&p_src[src_index]);
      //             }
      //         }
      //     }
      // };
      auto NSliceRowConstantIndexOp = b.create<ConstantIndexOp>(loc, NSliceRow);
      auto NSliceColConstantIndexOp = b.create<ConstantIndexOp>(loc, NSliceCol);
      auto DataPerAccessConstantIndexOp =
          b.create<ConstantIndexOp>(loc, DataPerAccess);

      // outer loop.
      auto outerLoopOp =
          b.create<scf::ForOp>(loc, zeroConstantIndexOp,
                               NSliceRowConstantIndexOp, oneConstantIndexOp);

      // inside the outer loop.
      auto lob = OpBuilder::atBlockTerminator(outerLoopOp.getBody());
      auto ivo = outerLoopOp.getInductionVar();
      auto ivo_i32 = lob.create<IndexCastOp>(loc, ivo, b.getIntegerType(32));

      // inner loop
      auto innerLoopOp = lob.create<scf::ForOp>(loc, zeroConstantIndexOp,
                                                NSliceColConstantIndexOp,
                                                DataPerAccessConstantIndexOp);

      // inside the inner loop.
      auto lib = OpBuilder::atBlockTerminator(innerLoopOp.getBody());
      auto ivi = innerLoopOp.getInductionVar();
      auto ivi_i32 = lib.create<IndexCastOp>(loc, ivi, b.getIntegerType(32));

      // Compute high-level coordinate for source memref.
      // src_index = (ivo_i32, ivi_i32) + sourceCoord
      SmallVector<Value, 8> srcUpperIndices;
      srcUpperIndices.push_back(lib.create<IndexCastOp>(
          loc, lib.create<AddIOp>(loc, ivo_i32, sourceCoord[0]),
          b.getIndexType()));
      srcUpperIndices.push_back(lib.create<IndexCastOp>(
          loc, lib.create<AddIOp>(loc, ivi_i32, sourceCoord[1]),
          b.getIndexType()));

      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> srcLowerIndices;
      if (sourceExternalTransform || sourceEmbeddedTransform)
        srcLowerIndices =
            expandAffineMap(lib, loc, sourceTransform, srcUpperIndices)
                .getValue();
      else
        srcLowerIndices = srcUpperIndices;

      // Load from source.
      auto vectorType =
          VectorType::get(DataPerAccess, sourceType.getElementType());
      auto srcExpr =
          getAffineDimExpr(sourceType.getRank() - 1, op.getContext());
      auto srcProjection = AffineMap::get(sourceType.getRank(), 0, srcExpr);
      auto srcProjectionAttr = AffineMapAttr::get(srcProjection);
      auto vectorValue = lib.create<vector::TransferReadOp>(
          loc, vectorType, op.source(), srcLowerIndices, srcProjectionAttr,
          zeroConstantFloatOp);

      // Compute high-level coordinate for dest memref.
      // dst_index = (ivo_i32, ivi_i32) + destCoord
      SmallVector<Value, 8> destUpperIndices;
      destUpperIndices.push_back(lib.create<IndexCastOp>(
          loc, lib.create<AddIOp>(loc, ivo_i32, destCoord[0]),
          b.getIndexType()));
      destUpperIndices.push_back(lib.create<IndexCastOp>(
          loc, lib.create<AddIOp>(loc, ivi_i32, destCoord[1]),
          b.getIndexType()));

      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> destLowerIndices;
      if (destExternalTransform || destEmbeddedTransform)
        destLowerIndices =
            expandAffineMap(lib, loc, destTransform, destUpperIndices)
                .getValue();
      else
        destLowerIndices = destUpperIndices;

      // Store to dest.
      auto dstExpr = getAffineDimExpr(destType.getRank() - 1, op.getContext());
      auto dstProjection = AffineMap::get(destType.getRank(), 0, dstExpr);
      auto dstProjectionAttr = AffineMapAttr::get(dstProjection);
      lib.create<vector::TransferWriteOp>(loc, vectorValue, op.dest(),
                                          destLowerIndices, dstProjectionAttr);

    } else {
      // The more elaborated algorithm.
      // Refer to ThreadwiseGenericTensorSliceCopy_v4r2::Run() for the original
      // C++ implementation.

      // llvm::errs() << "\nthreadwise_copy op:\n";
      // op.dump();
      // llvm::errs() << "\n";

      auto dimAccessOrder =
          op.getAttr("dim_access_order").template cast<ArrayAttr>();
      auto vectorAccessDim = op.getAttr("vector_read_write_dim")
                                 .template cast<IntegerAttr>()
                                 .getInt();
      auto srcDataPerRead = op.getAttr("source_data_per_read")
                                .template cast<IntegerAttr>()
                                .getInt();
      auto destDataPerWrite = op.getAttr("dest_data_per_write")
                                  .template cast<IntegerAttr>()
                                  .getInt();

      auto longVectorSize = math::lcm(srcDataPerRead, destDataPerWrite);

      // llvm::errs() << "vector_read_write_dim: " << vectorAccessDim << "\n";
      // llvm::errs() << "source_data_per_read: " << srcDataPerRead << "\n";
      // llvm::errs() << "dest_data_per_write: " << destDataPerWrite << "\n";
      // llvm::errs() << "longVectorSize: " << longVectorSize << "\n";

      // Figure out which memref is the one without affine transformations.
      SmallVector<int64_t, 2> sliceLengths;
      if (sourceExternalTransform || sourceEmbeddedTransform) {
        if (destExternalTransform || destEmbeddedTransform) {
          // Couldn't happen.
          llvm::errs()
              << "Unsupported case: both memrefs have affine transforms!\n";
          return failure();
        } else {
          for (auto dim : destType.getShape())
            sliceLengths.push_back(dim);
        }
      } else {
        if (sourceExternalTransform || sourceEmbeddedTransform) {
          // Couldn't happen.
          llvm::errs()
              << "Unsupported case: both memrefs have affine transforms!\n";
          return failure();
        } else
          for (auto dim : sourceType.getShape())
            sliceLengths.push_back(dim);
      }
      // llvm::errs() << "slice lengths: ";
      // for (unsigned i = 0; i < sliceLengths.size(); ++i)
      //   llvm::errs() << sliceLengths[i] << " ";
      // llvm::errs() << "\n";

      // Modify slice lenths per vector access dim.
      sliceLengths[vectorAccessDim] =
          sliceLengths[vectorAccessDim] / longVectorSize;
      SmallVector<Value, 2> loopBounds;
      for (unsigned iter = 0; iter < sliceLengths.size(); ++iter)
        loopBounds.push_back(
            b.create<ConstantIndexOp>(loc, sliceLengths[iter]));

      // llvm::errs() << "modified slice lengths: ";
      // for (unsigned i = 0; i < sliceLengths.size(); ++i)
      //   llvm::errs() << sliceLengths[i] << " ";
      // llvm::errs() << "\n";

      // Emit loops for vector loads / stores.
      SmallVector<scf::ForOp, 2> loopOps;
      SmallVector<OpBuilder, 2> loopBuilders;
      SmallVector<Value, 2> loopIVs;
      SmallVector<Value, 2> loopIV_i32s;
      for (unsigned iter = 0; iter < dimAccessOrder.size(); ++iter) {
        auto dim = dimAccessOrder[iter].template cast<IntegerAttr>().getInt();
        auto loopBuilder = (iter == 0) ? b : loopBuilders[iter - 1];

        auto loopOp = loopBuilder.create<scf::ForOp>(
            loc, zeroConstantIndexOp, loopBounds[dim], oneConstantIndexOp);
        loopOps.push_back(loopOp);
        auto loopOpBuilder = OpBuilder::atBlockTerminator(loopOp.getBody());
        loopBuilders.push_back(loopOpBuilder);
        auto loopIV = loopOp.getInductionVar();
        loopIVs.push_back(loopIV);
        auto loopIV_i32 = loopOpBuilder.create<IndexCastOp>(
            loc, loopIV, b.getIntegerType(32));
        loopIV_i32s.push_back(loopIV_i32);
      }

      // Emit loop body.
      auto innerLoopBuilder = loopBuilders[loopBuilders.size() - 1];

      // Compute high-level coordinate for source memref.
      // src_index = (iv_0, iv_1, ...) + sourceCoord
      SmallVector<Value, 8> srcUpperIndices;
      for (unsigned iter = 0; iter < loopIV_i32s.size(); ++iter)
        srcUpperIndices.push_back(innerLoopBuilder.create<IndexCastOp>(
            loc,
            innerLoopBuilder.create<AddIOp>(loc, loopIV_i32s[iter],
                                            sourceCoord[iter]),
            b.getIndexType()));

      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> srcLowerIndices;
      if (sourceExternalTransform || sourceEmbeddedTransform)
        srcLowerIndices = expandAffineMap(innerLoopBuilder, loc,
                                          sourceTransform, srcUpperIndices)
                              .getValue();
      else
        srcLowerIndices = srcUpperIndices;

      // Load from source.
      auto sourceVectorType =
          VectorType::get(srcDataPerRead, sourceType.getElementType());
      auto srcExpr =
          getAffineDimExpr(sourceType.getRank() - 1, op.getContext());
      auto srcProjection = AffineMap::get(sourceType.getRank(), 0, srcExpr);
      auto srcProjectionAttr = AffineMapAttr::get(srcProjection);
      auto vectorValue = innerLoopBuilder.create<vector::TransferReadOp>(
          loc, sourceVectorType, op.source(), srcLowerIndices,
          srcProjectionAttr, zeroConstantFloatOp);

      // Compute high-level coordinate for dest memref.
      // dst_index = (iv_0, iv_1, ...) + destCoord
      SmallVector<Value, 8> destUpperIndices;
      for (unsigned iter = 0; iter < loopIV_i32s.size(); ++iter)
        destUpperIndices.push_back(innerLoopBuilder.create<IndexCastOp>(
            loc,
            innerLoopBuilder.create<AddIOp>(loc, loopIV_i32s[iter],
                                            destCoord[iter]),
            b.getIndexType()));

      // Apply affine transformations to compute the low-level coordinate.
      SmallVector<Value, 8> destLowerIndices;
      if (destExternalTransform || destEmbeddedTransform)
        destLowerIndices = expandAffineMap(innerLoopBuilder, loc, destTransform,
                                           destUpperIndices)
                               .getValue();
      else
        destLowerIndices = destUpperIndices;

      // Store to dest.
      auto dstExpr = getAffineDimExpr(destType.getRank() - 1, op.getContext());
      auto dstProjection = AffineMap::get(destType.getRank(), 0, dstExpr);
      auto dstProjectionAttr = AffineMapAttr::get(dstProjection);
      innerLoopBuilder.create<vector::TransferWriteOp>(
          loc, vectorValue, op.dest(), destLowerIndices, dstProjectionAttr);
    }

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Subview lowering.
//===----------------------------------------------------------------------===//

struct SubviewRewritePattern : public OpRewritePattern<miopen::SubviewOp> {
  using OpRewritePattern<miopen::SubviewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::SubviewOp op,
                                PatternRewriter &b) const override {
    auto outputType = op.output().getType().cast<MemRefType>();

    // Pass the output affine map to users of this op.
    for (auto user : op.output().getUsers()) {
      unsigned userOperandIndex = 0;
      for (userOperandIndex = 0; userOperandIndex < user->getNumOperands(); ++userOperandIndex)
        if (user->getOperand(userOperandIndex) == op.output())
          break;

      auto coordTransformAttrs = user->getAttr("coord_transforms");
      if (!coordTransformAttrs)
        user->setAttr("coord_transforms",
                      b.getArrayAttr({
                        b.getDictionaryAttr({
                          b.getNamedAttr("operand", b.getI32IntegerAttr(userOperandIndex)),
                          b.getNamedAttr("transforms", b.getAffineMapArrayAttr(outputType.getAffineMaps()))
                        })
                      }));
    }

    // Pass the input to uses of this op.
    op.replaceAllUsesWith(op.input());

    op.erase();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Transform lowering.
//===----------------------------------------------------------------------===//

struct TransformRewritePattern : public OpRewritePattern<miopen::TransformOp> {
  using OpRewritePattern<miopen::TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::TransformOp op,
                                PatternRewriter &b) const override {
    auto outputType = op.output().getType().cast<MemRefType>();

    // Pass the output affine map to users of this op.
    if (outputType.getAffineMaps().size() > 0)
      for (auto user : op.output().getUsers()) {
        unsigned userOperandIndex = 0;
        for (userOperandIndex = 0; userOperandIndex < user->getNumOperands();
             ++userOperandIndex)
          if (user->getOperand(userOperandIndex) == op.output())
            break;

        auto coordTransformAttrs = user->getAttr("coord_transforms");
        if (!coordTransformAttrs)
          user->setAttr(
              "coord_transforms",
              b.getArrayAttr({b.getDictionaryAttr(
                  {b.getNamedAttr("operand",
                                  b.getI32IntegerAttr(userOperandIndex)),
                   b.getNamedAttr("transforms",
                                  b.getAffineMapArrayAttr(
                                      outputType.getAffineMaps()))})}));
        else {
          // create a deep-copy of existing attributes, and amend the new one.
          // need to figure out if there's a better way than this.
          auto arrayAttr = coordTransformAttrs.cast<ArrayAttr>();
          llvm::SmallVector<Attribute, 2> augmentedArrayAttr;

          for (unsigned idx = 0; idx < arrayAttr.size(); ++idx) {
            auto dictAttr = arrayAttr.getValue()[idx].cast<DictionaryAttr>();
            auto operandIndex =
                dictAttr.get("operand").cast<IntegerAttr>().getInt();

            if (operandIndex != userOperandIndex) {
              augmentedArrayAttr.push_back(dictAttr);
            } else {
              auto existingTransforms =
                  dictAttr.get("transforms").cast<ArrayAttr>();
              llvm::SmallVector<Attribute, 4> augmentedTransforms;
              augmentedTransforms.append(existingTransforms.begin(),
                                         existingTransforms.end());
              augmentedTransforms.push_back(
                  AffineMapAttr::get(outputType.getAffineMaps()[0]));

              augmentedArrayAttr.push_back(b.getDictionaryAttr(
                  {b.getNamedAttr("operand",
                                  b.getI32IntegerAttr(userOperandIndex)),
                   b.getNamedAttr("transforms",
                                  b.getArrayAttr(augmentedTransforms))}));
            }
          }
          user->setAttr("coord_transforms", b.getArrayAttr(augmentedArrayAttr));
        }
      }

    // Pass the input to uses of this op.
    op.replaceAllUsesWith(op.input());

    op.erase();
    return success();
  }
};
