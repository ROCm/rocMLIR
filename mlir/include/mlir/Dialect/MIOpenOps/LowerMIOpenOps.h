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

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/MIOpenOps/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
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

  PatternMatchResult matchAndRewrite(T op, PatternRewriter &b) const override {
    auto filterLayoutAttr =
        op.template getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr = op.template getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr =
        op.template getAttrOfType<ArrayAttr>("output_layout");

    auto dilationsAttr = op.template getAttrOfType<ArrayAttr>("dilations");
    auto stridesAttr = op.template getAttrOfType<ArrayAttr>("strides");
    auto paddingAttr = op.template getAttrOfType<ArrayAttr>("padding");

    // Get shape of output tensor.
    auto outputType = op.output().getType().dyn_cast<MemRefType>();
    auto outputShape = outputType.getShape();
    // HO/WO dimension for output tensor.
    int64_t outputHDim, outputWDim;

    // Find Ho/Wo dimension for output tensor. They will be used in
    // transforming input tensor.
    for (unsigned i = 0; i < outputLayoutAttr.size(); ++i) {
      if (auto strAttr =
              outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
        if (strAttr.getValue() == "ho") {
          outputHDim = i;
        } else if (strAttr.getValue() == "wo") {
          outputWDim = i;
        }
      }
    }

    // Transform filter tensor.
    auto filterType = op.filter().getType().dyn_cast<MemRefType>();
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
                filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
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
    auto gemmA =
        b.create<miopen::TransformOp>(op.getLoc(), transformedFilterMemRefType,
                                      op.filter(), transformedFilterAttrs);

    // Transform input tensor.
    // Input tensor step 1: padded input.
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
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
                inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
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
        op.getLoc(), paddedInputMemRefType, op.input(), paddedInputAttrs);

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
        op.getLoc(), embeddedInputMemRefType, ArrayRef<Value>(paddedInput),
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
    auto gemmB = b.create<miopen::TransformOp>(
        op.getLoc(), transformedInputMemRefType, ArrayRef<Value>(embeddedInput),
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
                outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
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
    auto gemmC =
        b.create<miopen::TransformOp>(op.getLoc(), transformedOutputMemRefType,
                                      op.output(), transformedOutputAttrs);

    // compute right padding parameters.
    auto leftPadH = paddingAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
    auto leftPadW = paddingAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
    auto dilationH =
        dilationsAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
    auto dilationW =
        dilationsAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();
    auto strideH = stridesAttr.getValue()[0].dyn_cast<IntegerAttr>().getInt();
    auto strideW = stridesAttr.getValue()[1].dyn_cast<IntegerAttr>().getInt();

    // get y, x, ho, wo, hi, wi
    int64_t y, x, ho, wo, hi, wi;
    y = x = ho = wo = hi = wi = 0;
    for (unsigned i = 0; i < 4; ++i) {
      auto filterAttr = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>();
      auto inputAttr = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>();
      auto outputAttr = outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>();

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
        op.getLoc(), ArrayRef<Type>{},
        ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                   arguments[fields.gridwiseGemmArgumentPosition[1]],
                   arguments[fields.gridwiseGemmArgumentPosition[2]]},
        gridwiseGemmAttrs);

    // Finally, erase the original Conv2D op.
    op.erase();

    return this->matchSuccess();
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

  std::tuple<int64_t, int64_t, int64_t> computeLDSBlockByteSizes(miopen::GridwiseGemmOp op) const {
     int64_t ABlockCopyDstDataPerWrite_M = op.getAttr("matrix_a_dest_data_per_write_dim_m").dyn_cast<IntegerAttr>().getInt();
     int64_t BBlockCopyDstDataPerWrite_N = op.getAttr("matrix_b_dest_data_per_write_dim_n").dyn_cast<IntegerAttr>().getInt();
     int64_t ThreadGemmAThreadCopySrcDataPerRead_M = op.getAttr("m_per_thread").dyn_cast<IntegerAttr>().getInt();
     int64_t ThreadGemmBThreadCopySrcDataPerRead_N = op.getAttr("n_per_thread").dyn_cast<IntegerAttr>().getInt();

     int64_t max_lds_align = math::lcm(ABlockCopyDstDataPerWrite_M,
                                    BBlockCopyDstDataPerWrite_N,
                                    ThreadGemmAThreadCopySrcDataPerRead_M,
                                    ThreadGemmBThreadCopySrcDataPerRead_N);

     int64_t KPerBlock = op.getAttr("k_per_block").dyn_cast<IntegerAttr>().getInt();
     int64_t MPerBlock = op.getAttr("m_per_block").dyn_cast<IntegerAttr>().getInt();
     int64_t NPerBlock = op.getAttr("n_per_block").dyn_cast<IntegerAttr>().getInt();

     int64_t AlignedNPerBlock = max_lds_align * math::integer_divide_ceil<int64_t>(NPerBlock, max_lds_align);

     // A matrix in LDS memory, dst of blockwise copy
     //   be careful of LDS alignment
     // Original C++ logic:
     //constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
     //    Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
     //constexpr index_t a_block_space =
     //    math::integer_least_multiple(a_k_m_block_desc.GetElementSpace(), max_lds_align);
     int64_t AlignedMPerBlock = max_lds_align * math::integer_divide_ceil<int64_t>(MPerBlock, max_lds_align);
     int64_t a_block_space = math::integer_least_multiple(KPerBlock * AlignedMPerBlock, max_lds_align);

     // B matrix in LDS memory, dst of blockwise copy
     //   be careful of LDS alignment
     // Original C++ logic:
     //constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
     //    Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
     //constexpr index_t b_block_space =
     //    math::integer_least_multiple(b_k_n_block_desc.GetElementSpace(), max_lds_align);
     int64_t b_block_space = math::integer_least_multiple(KPerBlock * AlignedNPerBlock, max_lds_align);

     FloatType opElementType = op.getOperand(0).getType().dyn_cast<MemRefType>().getElementType().dyn_cast<FloatType>();
     unsigned opElementTypeWidthInByte = opElementType.getWidth() / 8;

     return std::make_tuple<int64_t, int64_t, int64_t>(
       a_block_space * opElementTypeWidthInByte,
       b_block_space * opElementTypeWidthInByte,
       2 * (a_block_space + b_block_space) * opElementTypeWidthInByte);
  }

  void affixBlockwiseGemmAttributes(miopen::BlockwiseGemmOp bop, miopen::GridwiseGemmOp gop) const {
    // Add attributes from C++ template arguments and ctor arguments.
    //const auto blockwise_gemm = BlockwiseGemmBlockABlockBThreadCTransANormalBNormalC_v2<
    //    BlockSize,
    //    decltype(a_k_m_block_mtx_desc),
    //    decltype(b_k_n_block_mtx_desc),
    //    decltype(c_m0m1_n0n1_thread_mtx_desc),
    //    MPerThread,
    //    NPerThread,
    //    MLevel0Cluster,
    //    NLevel0Cluster,
    //    MLevel1Cluster,
    //    NLevel1Cluster,
    //    KPerThread,
    //    ThreadGemmAThreadCopySrcDataPerRead_M,
    //    ThreadGemmBThreadCopySrcDataPerRead_N>{};
    bop.setAttr("block_size", gop.getAttr("block_size"));
    bop.setAttr("m_per_thread", gop.getAttr("m_per_thread"));
    bop.setAttr("n_per_thread", gop.getAttr("n_per_thread"));
    bop.setAttr("k_per_thread", gop.getAttr("k_per_thread"));
    bop.setAttr("m_level0_cluster", gop.getAttr("m_level0_cluster"));
    bop.setAttr("m_level1_cluster", gop.getAttr("m_level1_cluster"));
    bop.setAttr("n_level0_cluster", gop.getAttr("n_level0_cluster"));
    bop.setAttr("n_level0_cluster", gop.getAttr("n_level1_cluster"));
    bop.setAttr("matrix_a_source_vector_read_dim", gop.getAttr("matrix_a_source_vector_read_dim"));
    bop.setAttr("matrix_b_source_vector_read_dim", gop.getAttr("matrix_b_source_vector_read_dim"));
    bop.setAttr("matrix_a_source_data_per_read", gop.getAttr("matrix_a_source_data_per_read"));
    bop.setAttr("matrix_b_source_data_per_read", gop.getAttr("matrix_b_source_data_per_read"));
  }

  PatternMatchResult matchAndRewrite(miopen::GridwiseGemmOp op, PatternRewriter &b) const override {
    // Prepare some useful constants.
    auto zeroConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 0);
    auto oneConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 1);
    auto twoConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 2);

    auto ldsMemorySpace = 3;
    auto ldsMemorySpaceConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsMemorySpace);
    auto registerMemorySpace = 5;
    auto registerMemorySpaceConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), registerMemorySpace);

    // Obtain critical matrix dimensions.
    int64_t K = op.getOperand(0).getType().dyn_cast<MemRefType>().getShape()[0];
    int64_t M = op.getOperand(0).getType().dyn_cast<MemRefType>().getShape()[1];
    int64_t N = op.getOperand(1).getType().dyn_cast<MemRefType>().getShape()[1];

    // Obtain critical tuning parameters.
    int64_t KPerBlock = op.getAttr("k_per_block").dyn_cast<IntegerAttr>().getInt();
    int64_t MPerBlock = op.getAttr("m_per_block").dyn_cast<IntegerAttr>().getInt();
    int64_t NPerBlock = op.getAttr("n_per_block").dyn_cast<IntegerAttr>().getInt();
    int64_t MPerThread = op.getAttr("m_per_thread").dyn_cast<IntegerAttr>().getInt();
    int64_t NPerThread = op.getAttr("n_per_thread").dyn_cast<IntegerAttr>().getInt();
    int64_t MLevel0Cluster = op.getAttr("m_level0_cluster").dyn_cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster = op.getAttr("m_level1_cluster").dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster = op.getAttr("n_level0_cluster").dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster = op.getAttr("n_level1_cluster").dyn_cast<IntegerAttr>().getInt();

    // Compute required LDS sizes.
    int64_t ldsBlockASize, ldsBlockBSize, ldsBlockSize;
    std::tie(ldsBlockASize, ldsBlockBSize, ldsBlockSize) = computeLDSBlockByteSizes(op);

    // Allocate LDS.
    auto ldsBlockSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockSize);
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, b.getIntegerType(8), {}, ldsMemorySpace);
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), ldsMemRefType, ldsBlockSizeConstantIndexOp, ldsMemorySpaceConstantIndexOp);

    // Subviews for Matrix A.
    auto ldsBlockADoubleSize = ldsBlockASize * 2;
    auto ldsBlockAOffset = 0;

    auto ldsBlockAOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockAOffset);
    auto ldsBlockADoubleMemRefType =
        MemRefType::get({ldsBlockADoubleSize}, b.getIntegerType(8), {}, ldsMemorySpace);
    auto ldsBlockADoubleSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockADoubleMemRefType, ldsGpuAllocOp, ldsBlockAOffsetConstantIndexOp);

    auto ldsBlockAEvenOffset = 0;
    auto ldsBlockAOddOffset = ldsBlockADoubleSize / 2;

    auto ldsBlockAEvenOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockAEvenOffset);
    auto ldsBlockAOddOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockAOddOffset);
    auto ldsBlockAMemRefType =
        MemRefType::get({ldsBlockASize}, b.getIntegerType(8), {}, ldsMemorySpace);
    auto ldsBlockAEvenSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockAMemRefType, ldsBlockADoubleSubviewOp, ldsBlockAEvenOffsetConstantIndexOp);
    auto ldsBlockAOddSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockAMemRefType, ldsBlockADoubleSubviewOp, ldsBlockAOddOffsetConstantIndexOp);

    // Get 2D subviews.
    // Compute matrix A dimension from attributes.
    // Original C++ logic.
    // // A matrix in LDS memory, dst of blockwise copy
    // //   be careful of LDS alignment
    // constexpr auto a_k_m_block_desc = make_native_tensor_descriptor_aligned(
    //     Sequence<KPerBlock, MPerBlock>{}, Number<max_lds_align>{});
    auto lds2DMatrixAHeight = KPerBlock;
    auto lds2DMatrixAWidth = MPerBlock;
    auto lds2DMatrixAMemRefType =
        MemRefType::get({lds2DMatrixAHeight, lds2DMatrixAWidth}, b.getF32Type(), {}, ldsMemorySpace);

    llvm::SmallVector<int64_t, 2> lds2DMatrixADim {lds2DMatrixAHeight, lds2DMatrixAWidth};
    llvm::SmallVector<NamedAttribute, 8> lds2DMatrixADimAttr {
        b.getNamedAttr("dimensions", b.getI64ArrayAttr(lds2DMatrixADim)),
    };
 
    auto lds2DMatrixAEvenSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), lds2DMatrixAMemRefType, ldsBlockAEvenSubviewOp, zeroConstantIndexOp);
    lds2DMatrixAEvenSubviewOp.setAttrs(lds2DMatrixADimAttr);
    auto lds2DMatrixAOddSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), lds2DMatrixAMemRefType, ldsBlockAOddSubviewOp, zeroConstantIndexOp);
    lds2DMatrixAOddSubviewOp.setAttrs(lds2DMatrixADimAttr);
 

    // Subviews for Matrix B.
    auto ldsBlockBDoubleSize = ldsBlockBSize * 2;
    auto ldsBlockBOffset = ldsBlockSize - ldsBlockADoubleSize;

    auto ldsBlockBOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockBOffset);
    auto ldsBlockBDoubleMemRefType =
        MemRefType::get({ldsBlockBDoubleSize}, b.getIntegerType(8), {}, ldsMemorySpace);
    auto ldsBlockBDoubleSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockBDoubleMemRefType, ldsGpuAllocOp, ldsBlockBOffsetConstantIndexOp);

    auto ldsBlockBEvenOffset = 0;
    auto ldsBlockBOddOffset = ldsBlockBDoubleSize / 2;

    auto ldsBlockBEvenOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockBEvenOffset);
    auto ldsBlockBOddOffsetConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), ldsBlockBOddOffset);
    auto ldsBlockBMemRefType =
        MemRefType::get({ldsBlockBSize}, b.getIntegerType(8), {}, ldsMemorySpace);
    auto ldsBlockBEvenSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockBMemRefType, ldsBlockBDoubleSubviewOp, ldsBlockBEvenOffsetConstantIndexOp);
    auto ldsBlockBOddSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), ldsBlockBMemRefType, ldsBlockBDoubleSubviewOp, ldsBlockBOddOffsetConstantIndexOp);

    // Get 2D subviews.
    // Compute matrix B dimension from attributes.
    // Original C++ logic.
    // // B matrix in LDS memory, dst of blockwise copy
    // //   be careful of LDS alignment
    // constexpr auto b_k_n_block_desc = make_native_tensor_descriptor_aligned(
    //     Sequence<KPerBlock, NPerBlock>{}, Number<max_lds_align>{});
    auto lds2DMatrixBHeight = KPerBlock;
    auto lds2DMatrixBWidth = NPerBlock;
    auto lds2DMatrixBMemRefType =
        MemRefType::get({lds2DMatrixBHeight, lds2DMatrixBWidth}, b.getF32Type(), {}, ldsMemorySpace);

    llvm::SmallVector<int64_t, 2> lds2DMatrixBDim {lds2DMatrixBHeight, lds2DMatrixBWidth};
    llvm::SmallVector<NamedAttribute, 1> lds2DMatrixBDimAttr {
        b.getNamedAttr("dimensions", b.getI64ArrayAttr(lds2DMatrixBDim)),
    };

    auto lds2DMatrixBEvenSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), lds2DMatrixBMemRefType, ldsBlockBEvenSubviewOp, zeroConstantIndexOp);
    lds2DMatrixBEvenSubviewOp.setAttrs(lds2DMatrixBDimAttr);
    auto lds2DMatrixBOddSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), lds2DMatrixBMemRefType, ldsBlockBOddSubviewOp, zeroConstantIndexOp);
    lds2DMatrixBOddSubviewOp.setAttrs(lds2DMatrixBDimAttr);


    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    // Original C++ logic.
    // constexpr index_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    // constexpr index_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);
    // constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
    //     Number<GemmMRepeat * MPerThread>{}, Number<GemmNRepeat * NPerThread>{});
    int64_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
    int64_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

    auto threadCRegisterSize = (GemmMRepeat * MPerThread) * (GemmNRepeat * NPerThread);
    auto threadCRegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadCRegisterSize);
    auto threadCRegisterMemRefType =
        MemRefType::get({threadCRegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
    auto threadCAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadCRegisterMemRefType, threadCRegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

    // Subviews for Matrix C.
    // Compute matrix C dimension from attributes.
    auto register2DMatrixCHeight = (GemmMRepeat * MPerThread);
    auto register2DMatrixCWidth = (GemmNRepeat * NPerThread);
    auto register2DMatrixCMemRefType =
        MemRefType::get({register2DMatrixCHeight, register2DMatrixCWidth}, b.getF32Type(), {}, registerMemorySpace);

    llvm::SmallVector<int64_t, 2> register2DMatrixCDim {register2DMatrixCHeight, register2DMatrixCWidth};
    llvm::SmallVector<NamedAttribute, 1> register2DMatrixCDimAttr {
        b.getNamedAttr("dimensions", b.getI64ArrayAttr(register2DMatrixCDim)),
    };

    auto register2DMatrixCSubviewOp = b.create<miopen::SubviewOp>(op.getLoc(), register2DMatrixCMemRefType, threadCAllocOp, zeroConstantIndexOp);
    register2DMatrixCSubviewOp.setAttrs(register2DMatrixCDimAttr);
 

    // Alloc for Matrix A / B on registers.
    // TBD. compute thread A / B on registers from attributes.
    auto threadARegisterSize = 1024;
    auto threadARegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadARegisterSize);
    auto threadARegisterMemRefType =
        MemRefType::get({threadARegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
    auto threadAEvenAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadARegisterMemRefType, threadARegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);
    auto threadAOddAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadARegisterMemRefType, threadARegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

    auto threadBRegisterSize = 1024;
    auto threadBRegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadBRegisterSize);
    auto threadBRegisterMemRefType =
        MemRefType::get({threadBRegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
    auto threadBEvenAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadBRegisterMemRefType, threadBRegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);
    auto threadBOddAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadBRegisterMemRefType, threadBRegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

    // Zero init Matrix C on registers.
    b.create<miopen::FillOp>(op.getLoc(), threadCAllocOp, zeroConstantIndexOp);

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // TBD add attributes from C++ template arguments and ctor arguments.
    //// A matrix blockwise copy
    //auto a_blockwise_copy =
    //    BlockwiseGenericTensorSliceCopy_v4<BlockSize,
    //                                       decltype(a_k_m_global_desc),
    //                                       decltype(a_k_m_block_desc),
    //                                       decltype(a_k_m_block_desc.GetLengths()),
    //                                       ABlockCopyThreadSliceLengths_K_M,
    //                                       -> GemmABlockCopyThreadSliceLengths_GemmK_GemmM
    //                                       -> Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>
    //                                          -> GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK
    //                                          -> GemmKPerBlock / (GemmKPerBlock / matrix_a_source_data_per_read)
    //                                          -> GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM
    //                                          -> GemmMPerBlock / (GemmMPerBlock / (MPerBlock * KPerBlock / BlockSize) / matrix_a_source_data_per_read)

    //                                       ABlockCopyThreadClusterLengths_K_M,
    //                                       -> GemmABlockCopyThreadClusterLengths_GemmK_GemmM
    //                                       -> Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>
    //                                          -> GemmKPerBlock / matrix_a_source_data_per_read
    //                                          -> GemmMPerBlock / (MPerBlock * KPerBlock / BlockSize) / matrix_a_source_data_per_read)

    //                                       ABlockCopyThreadClusterArrangeOrder,
    //                                       -> Sequence<1, 0>

    //                                       ABlockCopySrcAccessOrder,
    //                                       -> Sequence<1, 0>

    //                                       Sequence<0, 1>,

    //                                       ABlockCopySrcVectorReadDim,
    //                                       -> from op attribute

    //                                       1,

    //                                       ABlockCopySrcDataPerRead,
    //                                       -> from op attribute

    //                                       ABlockCopyDstDataPerWrite_M,
    //                                       -> from op attribute

    //                                       AddressSpace::Global,
    //                                       AddressSpace::Vgpr,
    //                                       AddressSpace::Lds,
    //                                       InMemoryDataOperation::Set>(
    //        {0, m_block_data_on_global}, {0, 0});

    //// B matrix blockwise copy
    //auto b_blockwise_copy =
    //    BlockwiseGenericTensorSliceCopy_v4<BlockSize,
    //                                       decltype(b_k_n_global_desc),
    //                                       decltype(b_k_n_block_desc),
    //                                       decltype(b_k_n_block_desc.GetLengths()),
    //                                       BBlockCopyThreadSliceLengths_K_N,
    //                                       BBlockCopyThreadClusterLengths_K_N,
    //                                       BBlockCopyThreadClusterArrangeOrder,
    //                                       BBlockCopySrcAccessOrder,
    //                                       Sequence<0, 1>,
    //                                       BBlockCopySrcVectorReadDim,
    //                                       1,
    //                                       BBlockCopySrcDataPerRead,
    //                                       BBlockCopyDstDataPerWrite_N,
    //                                       AddressSpace::Global,
    //                                       AddressSpace::Vgpr,
    //                                       AddressSpace::Lds,
    //                                       InMemoryDataOperation::Set>(
    //        {0, n_block_data_on_global}, {0, 0});
    b.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(0), ldsBlockAEvenSubviewOp);
    b.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(1), ldsBlockBEvenSubviewOp);

    // Emit loop.
    // Compute loop iterations from attributes.
    auto loopIteration = K / (KPerBlock * 2);
    auto loopIterationConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), loopIteration);
    auto loopOp = b.create<loop::ForOp>(op.getLoc(), zeroConstantIndexOp, loopIterationConstantIndexOp, oneConstantIndexOp);

    // inside the loop.
    auto lb = loopOp.getBodyBuilder();
    // LDS barrier.
    lb.create<miopen::LdsBarrierOp>(op.getLoc());

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    auto blockwiseCopyOpAEven = lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(0), threadAEvenAllocOp);
    // Compute block_slice_copy_steps and set in the attribute.
    blockwiseCopyOpAEven.setAttr("move_source_slice_window", b.getI32IntegerAttr(KPerBlock));
    auto blockwiseCopyOpBEven = lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(1), threadBEvenAllocOp);
    // Compute block_slice_copy_steps and set in the attribute.
    blockwiseCopyOpBEven.setAttr("move_source_slice_window", b.getI32IntegerAttr(KPerBlock));

    // Emit blockwise GEMM.
    auto blockwiseGemmEvenOp = lb.create<miopen::BlockwiseGemmOp>(op.getLoc(), lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp, register2DMatrixCSubviewOp);
    affixBlockwiseGemmAttributes(blockwiseGemmEvenOp, op);

    // Blockwise copy from reigster (naitve tensor) to LDS (naive tensor).
    lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), threadAEvenAllocOp, ldsBlockAOddSubviewOp);
    lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), threadBEvenAllocOp, ldsBlockBOddSubviewOp);

    // LDS barrier.
    lb.create<miopen::LdsBarrierOp>(op.getLoc());

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    auto blockwiseCopyOpAOdd = lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(0), threadAOddAllocOp);
    // Compute block_slice_copy_steps and set in the attribute.
    blockwiseCopyOpAOdd.setAttr("move_source_slice_window", b.getI32IntegerAttr(KPerBlock));
    auto blockwiseCopyOpBOdd = lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), op.getOperand(1), threadBOddAllocOp);
    // Compute block_slice_copy_steps and set in the attribute.
    blockwiseCopyOpBOdd.setAttr("move_source_slice_window", b.getI32IntegerAttr(KPerBlock));

    // Emit blockwise GEMM.
    auto blockwiseGemmOddOp = lb.create<miopen::BlockwiseGemmOp>(op.getLoc(), lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp, register2DMatrixCSubviewOp);
    affixBlockwiseGemmAttributes(blockwiseGemmOddOp, op);

    // Blockwise copy from reigster (naitve tensor) to LDS (naive tensor).
    lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), threadAOddAllocOp, ldsBlockAEvenSubviewOp);
    lb.create<miopen::BlockwiseCopyOp>(op.getLoc(), threadBOddAllocOp, ldsBlockBEvenSubviewOp);

    // outside the loop.

    
    // LDS barrier.
    b.create<miopen::LdsBarrierOp>(op.getLoc());

    // Emit blockwise GEMM for the loop tail.
    if (loopIteration % 2) {
      auto blockwiseGemmTailEvenOp = b.create<miopen::BlockwiseGemmOp>(op.getLoc(), lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp, register2DMatrixCSubviewOp);
      affixBlockwiseGemmAttributes(blockwiseGemmTailEvenOp, op);
    } else {
      auto blockwiseGemmTailOddOp = b.create<miopen::BlockwiseGemmOp>(op.getLoc(), lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp, register2DMatrixCSubviewOp);
      affixBlockwiseGemmAttributes(blockwiseGemmTailOddOp, op);
    }

    // Threadwise copy from register (naive tensor) to global (generic tensor).
    // TBD add attributes from C++ template arguments and ctor arguments.
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
    b.create<miopen::ThreadwiseCopyOp>(op.getLoc(), register2DMatrixCSubviewOp, op.getOperand(2));

    op.erase();

    return matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmRewritePattern : public OpRewritePattern<miopen::BlockwiseGemmOp> {
  using OpRewritePattern<miopen::BlockwiseGemmOp>::OpRewritePattern;

  PatternMatchResult naiveRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const {
    // Prepare some useful constants.
    auto zeroConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 0);
    auto oneConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 1);
    auto twoConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), 2);

    auto registerMemorySpace = 5;
    auto registerMemorySpaceConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), registerMemorySpace);

    // Alloc register for thread_a and thread_b.
    // TBD compute actual size from attributes.
    auto threadARegisterSize = 1024;
    auto threadARegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadARegisterSize);
    auto threadARegisterMemRefType =
        MemRefType::get({threadARegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
    auto threadAAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadARegisterMemRefType, threadARegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

    auto threadBRegisterSize = 1024;
    auto threadBRegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadBRegisterSize);
    auto threadBRegisterMemRefType =
        MemRefType::get({threadARegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
    auto threadBAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadBRegisterMemRefType, threadBRegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

    // Main loop.
    // TBD. compute loop iterations from attributes.
    auto loopIteration = 15;
    auto loopIterationConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), loopIteration);
    auto loopOp = b.create<loop::ForOp>(op.getLoc(), zeroConstantIndexOp, loopIterationConstantIndexOp, oneConstantIndexOp);

    // inside the main loop.
    auto lb = loopOp.getBodyBuilder();
 
    // read matrix A loop.
    // TBD. compute loop iterations from attributes.
    auto loopReadMatrixAIteration = 15;
    auto loopReadMatrixAIterationConstantIndexOp = lb.create<ConstantIndexOp>(op.getLoc(), loopReadMatrixAIteration);
    auto loopReadMatrixAOp = lb.create<loop::ForOp>(op.getLoc(), zeroConstantIndexOp, loopReadMatrixAIterationConstantIndexOp, oneConstantIndexOp);

    // inside read matrix A loop.
    auto lab = loopReadMatrixAOp.getBodyBuilder();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).
    // TBD. add attribute from C++ template arguments.
    //constexpr auto a_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixA,
    //                                                         decltype(a_thread_mtx),
    //                                                         KPerThreadLoop,
    //                                                         MPerThreadSubC,
    //                                                         ThreadGemmADataPerRead_M>{};
    lab.create<miopen::ThreadwiseCopyOp>(op.getLoc(), op.getOperand(0), threadAAllocOp);

    // read matrix B loop.
    // TBD. compute loop iterations from attributes.
    auto loopReadMatrixBIteration = 15;
    auto loopReadMatrixBIterationConstantIndexOp = lb.create<ConstantIndexOp>(op.getLoc(), loopReadMatrixBIteration);
    auto loopReadMatrixBOp = lb.create<loop::ForOp>(op.getLoc(), zeroConstantIndexOp, loopReadMatrixBIterationConstantIndexOp, oneConstantIndexOp);

    // inside read matrix A loop.
    auto lbb = loopReadMatrixBOp.getBodyBuilder();

    // Threadwise copy from LDS (naive tensor) to register (generic tensor).
    // TBD. add attribute from C++ template arguments.
    //constexpr auto b_thread_copy = ThreadwiseMatrixSliceCopy<BlockMatrixB,
    //                                                         decltype(b_thread_mtx),
    //                                                         KPerThreadLoop,
    //                                                         NPerThreadSubC,
    //                                                         ThreadGemmBDataPerRead_N>{};
    lbb.create<miopen::ThreadwiseCopyOp>(op.getLoc(), op.getOperand(1), threadBAllocOp);

    // Emit threadwise GEMM.
    // TBD add attributes.
    // constexpr auto threadwise_gemm =
    //     ThreadwiseGemmTransANormalBNormalC<decltype(a_thread_mtx),
    //                                        decltype(b_thread_mtx),
    //                                        decltype(c_thread_mtx)>{};
    lb.create<miopen::ThreadwiseGemmOp>(op.getLoc(), threadAAllocOp, threadBAllocOp, op.getOperand(2));

    op.erase();
    return matchSuccess();
  }

  PatternMatchResult twoByTwoPipelinedRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const {
    // TBD implement 2x2 pipelined version.
    op.erase();
    return matchSuccess();
  }

  PatternMatchResult matchAndRewrite(miopen::BlockwiseGemmOp op, PatternRewriter &b) const override {
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

  PatternMatchResult matchAndRewrite(miopen::BlockwiseCopyOp op, PatternRewriter &b) const override {
    bool rewritten = true;

    auto source = op.getOperand(0);
    auto sourceType = source.getType().dyn_cast<MemRefType>();
    auto dest = op.getOperand(1);
    auto destType = dest.getType().dyn_cast<MemRefType>();

    // Check the address spaces of source and destination values and determine
    // lowering logic.
    // - 0 (global) -> 3 (LDS) : load + store
    // - 0 (global) -> 5 (register) : load 
    // - 5 (register) -> 3 (LDS) : store
    if (sourceType.getMemorySpace() == 0 && destType.getMemorySpace() == 3) {
      // TBD. compute register size from attributes and operands.
      auto registerMemorySpace = 5;
      auto registerMemorySpaceConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), registerMemorySpace);
      auto threadRegisterSize = 1024;
      auto threadRegisterSizeConstantIndexOp = b.create<ConstantIndexOp>(op.getLoc(), threadRegisterSize);
      auto threadRegisterMemRefType =
          MemRefType::get({threadRegisterSize}, b.getIntegerType(8), {}, registerMemorySpace);
      auto threadAllocOp = b.create<miopen::GpuAllocOp>(op.getLoc(), threadRegisterMemRefType, threadRegisterSizeConstantIndexOp, registerMemorySpaceConstantIndexOp);

      // Threadwise copy from global (generic tensor) to register (naive tensor).
      // TBD add attributes from C++ template arguments and ctor arguments.
      // using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));
      //
      // using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
      //                                                              ThreadBufferDesc,
      //                                                              ThreadSliceLengths,
      //                                                              SrcDimAccessOrder,
      //                                                              SrcVectoReadDim,
      //                                                              SrcDataPerRead,
      //                                                              1,
      //                                                              SrcAddressSpace,
      //                                                              ThreadBufferAddressSpace,
      //                                                              InMemoryDataOperation::Set>;
      //
      // constexpr auto thread_cluster_desc =
      //     make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});
      // const auto thread_cluster_id =
      //     thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());
      // const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};
      // mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
      // mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());
      b.create<miopen::ThreadwiseCopyOp>(op.getLoc(), source, threadAllocOp);

      // Threadwise copy from register (naive tensor) to LDS (naive tensor).
      // TBD add attributes from C++ template arguments and ctor arguments.
      // using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));
      //
      // using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
      //                                                               BlockDstDesc,
      //                                                               ThreadSliceLengths,
      //                                                               DstDimAccessOrder,
      //                                                               DstVectorWriteDim,
      //                                                               1,
      //                                                               DstDataPerWrite,
      //                                                               ThreadBufferAddressSpace,
      //                                                               DstAddressSpace,
      //                                                               DstInMemOp>;
      //
      // constexpr auto thread_cluster_desc =
      //     make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});
      // const auto thread_cluster_id =
      //     thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());
      // const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};
      // mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
      // mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
      b.create<miopen::ThreadwiseCopyOp>(op.getLoc(), threadAllocOp, dest);
    } else if (sourceType.getMemorySpace() == 0 && destType.getMemorySpace() == 5) {
      // Threadwise copy from global (generic tensor) to register (naive tensor).
      // TBD add attributes from C++ template arguments and ctor arguments.
      // using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));
      //
      // using ThreadwiseLoad = ThreadwiseGenericTensorSliceCopy_v4r2<BlockSrcDesc,
      //                                                              ThreadBufferDesc,
      //                                                              ThreadSliceLengths,
      //                                                              SrcDimAccessOrder,
      //                                                              SrcVectoReadDim,
      //                                                              SrcDataPerRead,
      //                                                              1,
      //                                                              SrcAddressSpace,
      //                                                              ThreadBufferAddressSpace,
      //                                                              InMemoryDataOperation::Set>;
      //
      // constexpr auto thread_cluster_desc =
      //     make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});
      // const auto thread_cluster_id =
      //     thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());
      // const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};
      // mThreadwiseLoad.SetSrcSliceOrigin(src_block_slice_origin + thread_data_id_begin);
      // mThreadwiseLoad.SetDstSliceOrigin(make_zero_array<index_t, nDim>());
      auto threadwiseCopyOp = b.create<miopen::ThreadwiseCopyOp>(op.getLoc(), source, dest);
      // Pass blockwise-level attribute to threadwise op.
      if (op.getAttr("move_source_slice_window")) {
        threadwiseCopyOp.setAttr("move_source_slice_window", op.getAttr("move_source_slice_window"));
      }
    } else if (sourceType.getMemorySpace() == 5 && destType.getMemorySpace() == 3) {
      // Threadwise copy from register (naive tensor) to LDS (naive tensor).
      // TBD add attributes from C++ template arguments and ctor arguments.
      // using ThreadBufferDesc = decltype(make_native_tensor_descriptor_packed(ThreadSliceLengths{}));
      //
      // using ThreadwiseStore = ThreadwiseGenericTensorSliceCopy_v4r2<ThreadBufferDesc,
      //                                                               BlockDstDesc,
      //                                                               ThreadSliceLengths,
      //                                                               DstDimAccessOrder,
      //                                                               DstVectorWriteDim,
      //                                                               1,
      //                                                               DstDataPerWrite,
      //                                                               ThreadBufferAddressSpace,
      //                                                               DstAddressSpace,
      //                                                               DstInMemOp>;
      //
      // constexpr auto thread_cluster_desc =
      //     make_cluster_descriptor(ThreadClusterLengths{}, ThreadClusterArrangeOrder{});
      // const auto thread_cluster_id =
      //     thread_cluster_desc.CalculateClusterIndex(get_thread_local_1d_id());
      // const auto thread_data_id_begin = thread_cluster_id * ThreadSliceLengths{};
      // mThreadwiseStore.SetSrcSliceOrigin(make_zero_array<index_t, nDim>());
      // mThreadwiseStore.SetDstSliceOrigin(dst_block_slice_origin + thread_data_id_begin);
      b.create<miopen::ThreadwiseCopyOp>(op.getLoc(), source, dest);
    } else {
      llvm::errs() << "UNSUPPORTED ThreadwiseCopyOp\n";
      rewritten = false;
    }

    if (rewritten)
      op.erase();
    return matchSuccess();
  }
}; 

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpRewritePattern<miopen::FillOp> {
  using OpRewritePattern<miopen::FillOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(miopen::FillOp op, PatternRewriter &b) const override {
    auto loc = op.getLoc();
    auto inputType = op.input().getType().cast<MemRefType>();
    auto inputShape = inputType.getShape();

    auto value = op.value().getDefiningOp()->getAttr("value").dyn_cast<IntegerAttr>().getInt();
    auto valueOp = b.create<ConstantIntOp>(loc, value, 8);

    auto zero = b.create<ConstantIndexOp>(loc, 0);
    auto one = b.create<ConstantIndexOp>(loc, 1);
    auto loopIteration = b.create<ConstantIndexOp>(loc, inputShape[0]);
    auto loopOp = b.create<loop::ForOp>(loc, zero, loopIteration, one);

    // inside loop.
    auto lb = loopOp.getBodyBuilder();

    for (unsigned i = 0; i < inputShape[0]; ++i) {
      auto iter = b.create<ConstantIndexOp>(loc, i);
      auto storeOp = lb.create<StoreOp>(loc, valueOp, op.input(), ValueRange{iter});
    }

    op.erase();
    return matchSuccess();
  }
};
 
