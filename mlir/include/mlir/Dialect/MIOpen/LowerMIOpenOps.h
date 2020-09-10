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
#include "mlir/Dialect/GPU/GPUDialect.h"
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

#include "XdlopsCodeSelection.h"

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
    //
    // Weight tensor transformation for Conv2DBwdWeightOp
    // - Part 1: Merge non-K dimensions to dimension 1, name it as gemmN.
    // - Part 2: PassThrough K dimension to dimension 0, name it as gemmM.
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

      llvm::SmallVector<NamedAttribute, 2> sourceProbCYXDimAttr{
	      b.getNamedAttr("source_dimensions",
			      b.getArrayAttr(ArrayRef<Attribute>(nonKDims.begin(),
					      nonKDims.end()))),
		      b.getNamedAttr("source_names",
				      b.getArrayAttr(ArrayRef<Attribute>(
						      nonKDimNames.begin(), nonKDimNames.end())))};
      if (kDim.getInt() != 0 && kDim.getInt() != 3) {
	      sourceProbCYXDimAttr.push_back(
			      b.getNamedAttr("transformation", b.getStringAttr("Merge")));
      } else {
	      sourceProbCYXDimAttr.push_back(
			      b.getNamedAttr("transformation", b.getStringAttr("Unfold")));
      }

      llvm::SmallVector<NamedAttribute, 3> sourceProbKDimAttr{
	      b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
		      b.getNamedAttr("source_dimensions", b.getArrayAttr({kDim})),
		      b.getNamedAttr("source_names", b.getArrayAttr({kDimName}))};

      llvm::SmallVector<NamedAttribute, 3> targetGemm0DimAttr{
	      b.getNamedAttr("dimensions",
			      b.getArrayAttr({b.getI32IntegerAttr(0)})),
		      b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
						      arg0TargetLayoutName0)}))};

      llvm::SmallVector<NamedAttribute, 3> targetGemm1DimAttr{
	      b.getNamedAttr("dimensions",
			      b.getArrayAttr({b.getI32IntegerAttr(1)})),
		      b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
						      arg0TargetLayoutName1)}))};

      llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
      llvm::SmallVector<NamedAttribute, 0> layoutAttr1;

      if (convOpType == miopen::ConvOpType::Conv2DOpType) {
        layoutAttr0.append(targetGemm0DimAttr.begin(),
                           targetGemm0DimAttr.end());
        layoutAttr0.append(sourceProbCYXDimAttr.begin(),
                           sourceProbCYXDimAttr.end());
        layoutAttr1.append(targetGemm1DimAttr.begin(),
                           targetGemm1DimAttr.end());
        layoutAttr1.append(sourceProbKDimAttr.begin(),
                           sourceProbKDimAttr.end());
      } else {
        layoutAttr0.append(targetGemm0DimAttr.begin(),
                           targetGemm0DimAttr.end());
        layoutAttr0.append(sourceProbKDimAttr.begin(),
                           sourceProbKDimAttr.end());
        layoutAttr1.append(targetGemm1DimAttr.begin(),
                           targetGemm1DimAttr.end());
        layoutAttr1.append(sourceProbCYXDimAttr.begin(),
                           sourceProbCYXDimAttr.end());
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
                                                   b.getI32IntegerAttr(1),
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
                                                   b.getI32IntegerAttr(1),
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
    //
    // input tensor transformation for Conv2DBwdWeightOp
    // - Part 1: Merge ni, ho, wo dimensions to dimension 0, name it as gemmK.
    // - Part 2: Merge ci, y, x dimensions to dimension 1, name it as gemmN.
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

      if (convOpType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
	      // Assume hDim is always less than wDim.
	      if (nDim.getInt() < hDim.getInt()) {
		      mergedPart1DimNames.push_back(nDimName);
		      mergedPart1DimNames.push_back(hDimName);
		      mergedPart1DimNames.push_back(wDimName);
		      mergedPart1Dims.push_back(nDim);
		      mergedPart1Dims.push_back(hDim);
		      mergedPart1Dims.push_back(wDim);
	      } else {
		      mergedPart1DimNames.push_back(hDimName);
		      mergedPart1DimNames.push_back(wDimName);
		      mergedPart1DimNames.push_back(nDimName);
		      mergedPart1Dims.push_back(hDim);
		      mergedPart1Dims.push_back(wDim);
		      mergedPart1Dims.push_back(nDim);
	      }
      } else {
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
      }

      llvm::SmallVector<StringAttr, 3> mergedPart2DimNames;
      llvm::SmallVector<IntegerAttr, 3> mergedPart2Dims;

      if (convOpType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
	      // Assume yDim is always less than xDim.
	      if (cDim.getInt() < yDim.getInt()) {
		      mergedPart2DimNames.push_back(cDimName);
		      mergedPart2DimNames.push_back(yDimName);
		      mergedPart2DimNames.push_back(xDimName);
		      mergedPart2Dims.push_back(cDim);
		      mergedPart2Dims.push_back(yDim);
		      mergedPart2Dims.push_back(xDim);
	      } else {
		      mergedPart2DimNames.push_back(yDimName);
		      mergedPart2DimNames.push_back(xDimName);
		      mergedPart2DimNames.push_back(cDimName);
		      mergedPart2Dims.push_back(yDim);
		      mergedPart2Dims.push_back(xDim);
		      mergedPart2Dims.push_back(cDim);
	      }
      } else {
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
    //
    // Output tensor transformation for backward weight:
    // - Part 1: Merge non-K dimensions to dimension 0, name it as gemmK.
    // - Part 2: PassThrough K dimension to dimension 1, name it as gemmM.
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

      llvm::SmallVector<NamedAttribute, 3> sourceProbNHoWoDimAttr{
	      b.getNamedAttr("source_dimensions",
			      b.getArrayAttr(ArrayRef<Attribute>(nonKDims.begin(),
					      nonKDims.end()))),
		      b.getNamedAttr("source_names",
				      b.getArrayAttr(ArrayRef<Attribute>(
						      nonKDimNames.begin(), nonKDimNames.end()))),
		      b.getNamedAttr("transformation", b.getStringAttr("Merge"))};

      llvm::SmallVector<NamedAttribute, 3> sourceProbKDimAttr{
	      b.getNamedAttr("transformation", b.getStringAttr("PassThrough")),
		      b.getNamedAttr("source_dimensions", b.getArrayAttr({kDim})),
		      b.getNamedAttr("source_names", b.getArrayAttr({kDimName}))};

      llvm::SmallVector<NamedAttribute, 3> targetGemm0DimAttr{
	      b.getNamedAttr("dimensions",
			      b.getArrayAttr({b.getI32IntegerAttr(0)})),
		      b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
						      arg2TargetLayoutName0)}))};

      llvm::SmallVector<NamedAttribute, 3> targetGemm1DimAttr{
	      b.getNamedAttr("dimensions",
			      b.getArrayAttr({b.getI32IntegerAttr(1)})),
		      b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr(
						      arg2TargetLayoutName1)}))};

      llvm::SmallVector<NamedAttribute, 0> layoutAttr0;
      llvm::SmallVector<NamedAttribute, 0> layoutAttr1;

      if (convOpType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
	      layoutAttr0.append(targetGemm0DimAttr.begin(),
			      targetGemm0DimAttr.end());
	      layoutAttr0.append(sourceProbNHoWoDimAttr.begin(),
			      sourceProbNHoWoDimAttr.end());
	      layoutAttr1.append(targetGemm1DimAttr.begin(),
			      targetGemm1DimAttr.end());
	      layoutAttr1.append(sourceProbKDimAttr.begin(),
			      sourceProbKDimAttr.end());
      } else {
	      layoutAttr0.append(targetGemm0DimAttr.begin(),
			      targetGemm0DimAttr.end());
	      layoutAttr0.append(sourceProbKDimAttr.begin(),
			      sourceProbKDimAttr.end());
	      layoutAttr1.append(targetGemm1DimAttr.begin(),
			      targetGemm1DimAttr.end());
	      layoutAttr1.append(sourceProbNHoWoDimAttr.begin(),
			      sourceProbNHoWoDimAttr.end());
      }
 
      transformedOutputAttrs.push_back(b.getNamedAttr(
          "layout", b.getArrayAttr({
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr0.begin(), layoutAttr0.end())}),
                        b.getDictionaryAttr({ArrayRef<NamedAttribute>(
                            layoutAttr1.begin(), layoutAttr1.end())}),
                    })));
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

    // xdlops.
    auto xdlopsAttr = op.template getAttrOfType<BoolAttr>("xdlops");
    if (xdlopsAttr && xdlopsAttr.getValue() == true)
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("xdlops", b.getBoolAttr(true)));

    // xdlopsV2.
    auto xdlopsV2Attr = op.template getAttrOfType<BoolAttr>("xdlopsV2");
    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("xdlopsV2", b.getBoolAttr(true)));

    if (convOpType == miopen::ConvOpType::Conv2DBwdDataOpType) {
      gridwiseGemmAttrs.push_back(b.getNamedAttr(
          "kernel_algorithm", b.getStringAttr("backward_data_v1r1")));
    } else if (convOpType == miopen::ConvOpType::Conv2DOpType) {
      gridwiseGemmAttrs.push_back(
          b.getNamedAttr("kernel_algorithm", b.getStringAttr("v4r4")));
    } else if (convOpType == miopen::ConvOpType::Conv2DBwdWeightOpType) {
      gridwiseGemmAttrs.push_back(b.getNamedAttr(
          "kernel_algorithm", b.getStringAttr("backward_weight_v4r4")));
    }

    // Emit miopen.gridwise_gemm op.
    // Emit miopen.gridwise_gemm_v2 if xdlopsV2 attribute is true.
    auto arguments = std::array<miopen::TransformOp, 3>{gemmA, gemmB, gemmC};

    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      b.create<miopen::GridwiseGemmV2Op>(
          loc, ArrayRef<Type>{},
          ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                     arguments[fields.gridwiseGemmArgumentPosition[1]],
                     arguments[fields.gridwiseGemmArgumentPosition[2]]},
          gridwiseGemmAttrs);
    } else {
      b.create<miopen::GridwiseGemmOp>(
          loc, ArrayRef<Type>{},
          ValueRange{arguments[fields.gridwiseGemmArgumentPosition[0]],
                     arguments[fields.gridwiseGemmArgumentPosition[1]],
                     arguments[fields.gridwiseGemmArgumentPosition[2]]},
          gridwiseGemmAttrs);
    }

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

template <>
const ArgumentFields Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::fields = {
	{2, 1, 0},
	{"MN", "KN", "KM"},
};

template <>
const miopen::ConvOpType Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>::convOpType =
miopen::ConvOpType::Conv2DBwdWeightOpType;


// Explicitly instantiate the template to operation type
template struct Conv2DRewritePattern<miopen::Conv2DOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdDataOp>;
template struct Conv2DRewritePattern<miopen::Conv2DBwdWeightOp>;

//===----------------------------------------------------------------------===//
// Assigning attributes.
//===----------------------------------------------------------------------===//

static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top, miopen::GridwiseGemmOp gop, OpBuilder &b) {
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

  top.setAttr("dim_access_order", b.getArrayAttr({
                                      b.getI32IntegerAttr(0),
                                      b.getI32IntegerAttr(1),
                                      b.getI32IntegerAttr(2),
                                      b.getI32IntegerAttr(3),
                                  }));
  top.setAttr("vector_read_write_dim",
              gop.getAttr("matrix_c_source_dest_vector_read_write_dim"));
  top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
  top.setAttr("dest_data_per_write", gop.getAttr("matrix_c_dest_data_per_write"));
}

static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top, miopen::GridwiseGemmV2Op gop, OpBuilder &b) {
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

  top.setAttr("dim_access_order", b.getArrayAttr({
                                      b.getI32IntegerAttr(0),
                                      b.getI32IntegerAttr(1),
                                      b.getI32IntegerAttr(2),
                                      b.getI32IntegerAttr(3),
                                  }));
  top.setAttr("vector_read_write_dim",
              gop.getAttr("matrix_c_source_dest_vector_read_write_dim"));
  top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
  top.setAttr("dest_data_per_write", gop.getAttr("matrix_c_dest_data_per_write"));
}

static void affixThreadwiseCopyV2Attributes(miopen::ThreadwiseCopyV2Op top, miopen::GridwiseGemmV2Op gop, OpBuilder &b) {
  top.setAttr("dim_access_order", b.getArrayAttr({
                                      b.getI32IntegerAttr(0),
                                      b.getI32IntegerAttr(1),
                                      b.getI32IntegerAttr(2),
                                      b.getI32IntegerAttr(3),
                                  }));
  top.setAttr("vector_read_write_dim",
              gop.getAttr("matrix_c_source_dest_vector_read_write_dim"));
  top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
  top.setAttr("dest_data_per_write", gop.getAttr("matrix_c_dest_data_per_write"));
}

// XXX: Figure out a way to do away with isThreadwiseLoad parameter.
static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::BlockwiseCopyOp bop,
                                          OpBuilder &b,
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
    // XXX: TBD review how vector load/store attributes are passed down.
    //top.setAttr("source_data_per_read", bop.getAttr("source_data_per_read"));
    top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
    top.setAttr("dest_data_per_write", b.getI32IntegerAttr(1));
  } else {
    top.setAttr("dim_access_order", bop.getAttr("dest_dim_access_order"));
    // XXX. Figure this out. Symmetry is somehow lost here.
    // top.setAttr("vector_read_write_dim",
    // bop.getAttr("dest_vector_write_dim"));
    top.setAttr("vector_read_write_dim", bop.getAttr("source_vector_read_dim"));
    top.setAttr("source_data_per_read", b.getI32IntegerAttr(1));
    // XXX: TBD review how vector load/store attributes are passed down.
    //top.setAttr("dest_data_per_write", bop.getAttr("dest_data_per_write"));
    top.setAttr("dest_data_per_write", b.getI32IntegerAttr(1));
  }
}

// XXX: figure out a better way to get rid of isMatrixA parameter.
static void affixThreadwiseCopyAttributes(miopen::ThreadwiseCopyOp top,
                                          miopen::BlockwiseGemmOp bop,
                                          OpBuilder &b, bool isMatrixA) {
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
    // XXX: TBD review how vector load/store attributes are passed down.
    //top.setAttr("data_per_access", bop.getAttr("m_per_thread"));
    top.setAttr("data_per_access", b.getI32IntegerAttr(1));
  } else {
    top.setAttr("n_slice_row", bop.getAttr("k_per_thread"));
    top.setAttr("n_slice_col", bop.getAttr("n_per_thread"));
    // XXX: TBD review how vector load/store attributes are passed down.
    //top.setAttr("data_per_access", bop.getAttr("n_per_thread"));
    top.setAttr("data_per_access", b.getI32IntegerAttr(1));
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

     // llvm::errs() << "a_block_space: " << a_block_space << "\n";
     // llvm::errs() << "b_block_space: " << b_block_space << "\n";
     // llvm::errs() << "double_block_space: " << double_block_space << "\n\n";
  }

  // XXX. Figure out a way to do away with isMatrixA parameter.
  void affixBlockwiseCopyAttributes(miopen::BlockwiseCopyOp bop,
                                    miopen::GridwiseGemmOp gop,
                                    OpBuilder &b, bool isMatrixA) const {
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

  void affixXdlopsGemmV2Attributes(miopen::XdlopsGemmV2Op xop,
                                   miopen::GridwiseGemmOp gop,
                                   OpBuilder &b) const {
    xop.setAttr("block_size", gop.getAttr("block_size"));
    // xdlopsV2.
    auto xdlopsV2Attr = gop.template getAttrOfType<BoolAttr>("xdlopsV2");
    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      int64_t MPerBlock =
          gop.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerBlock =
          gop.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t MPerWave =
          gop.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerWave =
          gop.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t MWaves = MPerBlock / MPerWave;
      int64_t NWaves = NPerBlock / NPerWave;

      xop.setAttr("m_per_wave", gop.getAttr("m_per_thread"));
      xop.setAttr("n_per_wave", gop.getAttr("n_per_thread"));
      xop.setAttr("m_waves", b.getI32IntegerAttr(MWaves));
      xop.setAttr("n_waves", b.getI32IntegerAttr(NWaves));

      xop.setAttr("xdlopsV2", b.getBoolAttr(true));
    }
  }

  void affixBlockwiseGemmAttributes(miopen::BlockwiseGemmOp bop,
                                    miopen::GridwiseGemmOp gop,
                                    OpBuilder &b) const {
    bop.setAttr("block_size", gop.getAttr("block_size"));

    // xdlops.
    auto xdlopsAttr = gop.template getAttrOfType<BoolAttr>("xdlops");
    if (xdlopsAttr && xdlopsAttr.getValue() == true) {
      int64_t MPerBlock =
          gop.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerBlock =
          gop.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t MPerWave =
          gop.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerWave =
          gop.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t MWaves = MPerBlock / MPerWave;
      int64_t NWaves = NPerBlock / NPerWave;

      bop.setAttr("m_per_wave", gop.getAttr("m_per_thread"));
      bop.setAttr("n_per_wave", gop.getAttr("n_per_thread"));
      bop.setAttr("m_waves", b.getI32IntegerAttr(MWaves));
      bop.setAttr("n_waves", b.getI32IntegerAttr(NWaves));

      bop.setAttr("xdlops", b.getBoolAttr(true));
    } else {
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
 
      // Attributes used in non-xdlops lowering path.
      bop.setAttr("m_per_thread", gop.getAttr("m_per_thread"));
      bop.setAttr("n_per_thread", gop.getAttr("n_per_thread"));
      bop.setAttr("k_per_thread", gop.getAttr("k_per_thread"));
      bop.setAttr("m_level0_cluster", gop.getAttr("m_level0_cluster"));
      bop.setAttr("m_level1_cluster", gop.getAttr("m_level1_cluster"));
      bop.setAttr("n_level0_cluster", gop.getAttr("n_level0_cluster"));
      bop.setAttr("n_level1_cluster", gop.getAttr("n_level1_cluster"));
    }
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

    auto elementType = op.output().getType().cast<MemRefType>().getElementType();

    // Prepare some useful constants.
    auto zeroConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);

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
    auto MPerThreadConstantOp = b.create<ConstantIndexOp>(loc, MPerThread);
    auto NPerThreadConstantOp = b.create<ConstantIndexOp>(loc, NPerThread);

    int64_t MLevel0Cluster = op.getAttr("m_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t MLevel1Cluster = op.getAttr("m_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel0Cluster = op.getAttr("n_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
    int64_t NLevel1Cluster = op.getAttr("n_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
    auto NLevel0ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel0Cluster);
    auto NLevel1ClusterConstantOp =
        b.create<ConstantIndexOp>(loc, NLevel1Cluster);

    int64_t matrix_a_source_data_per_read =
        op.getAttr("matrix_a_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op.getAttr("matrix_b_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();

    // XDLOPS.
    int64_t MPerWave = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;
    auto dataType = op.input().getType().template dyn_cast<MemRefType>().getElementType().template dyn_cast<FloatType>();

    auto MPerWaveConstantOp = b.create<ConstantIndexOp>(loc, MPerWave);
    auto NPerWaveConstantOp = b.create<ConstantIndexOp>(loc, NPerWave);
    auto MWavesConstantOp = b.create<ConstantIndexOp>(loc, MWaves);
    auto NWavesConstantOp = b.create<ConstantIndexOp>(loc, NWaves);

    int64_t WaveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, WaveSize);

    // Get current workgroup ID.
    auto bid = b.create<miopen::WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: "  << N << "\n";
    // llvm::errs() << "K: "  << K << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";
    // llvm::errs() << "MWaves = MPerBlock / MPerWave: " << MWaves << "\n";
    // llvm::errs() << "NWaves = NPerBlock / NPerWave: " << NWaves << "\n";

    auto MBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, MBlockWork);
    auto NBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, NBlockWork);
    auto block_work_id_m =
        b.create<SignedDivIOp>(loc, bid, NBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<SignedRemIOp>(loc, bid, NBlockWorkConstantOp);
    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);
    auto m_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, m_block_data_on_global, b.getIntegerType(32));
    auto n_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, n_block_data_on_global, b.getIntegerType(32));

    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "matrix_a_source_data_per_read: " << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: " << matrix_b_source_data_per_read << "\n";

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock /
        ((MPerBlock * KPerBlock / BlockSize) / matrix_a_source_data_per_read);
    int64_t GemmABlockCopyClusterLengths_GemmM =
        MPerBlock / matrix_a_source_data_per_read;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << "\n";

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

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyThreadSliceLengths_GemmK =
        KPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN =
        NPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";

    // Get current workitem ID.

    // Original logic.
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

    auto GemmABlockCopyThreadClusterId_Y = b.create<SignedRemIOp>(
        loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
    auto GemmABlockCopyThreadClusterId_X = b.create<SignedDivIOp>(
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

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockADoubleSize = ldsBlockASize * 2;
    auto ldsBlockAOffset = 0;

    auto ldsBlockAOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOffset);
    auto ldsBlockADoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockAOffset,
                                 {ldsBlockADoubleSize}, elementType);
    auto ldsBlockADoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockADoubleMemRefType, ldsGpuAllocOp,
        ldsBlockAOffsetConstantOp);

    auto ldsBlockAEvenOffset = 0;
    auto ldsBlockAEvenOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAEvenOffset);
    auto ldsBlockAEvenMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAEvenOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAEvenMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAEvenOffsetConstantOp);

    auto ldsBlockAOddOffset = ldsBlockADoubleSize / 2;
    auto ldsBlockAOddOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOddOffset);
    auto ldsBlockAOddMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAOddOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAOddMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAOddOffsetConstantOp);

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
        zeroConstantOp);
    auto lds2DMatrixAOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixAOddMemRefType,
                                    ldsBlockAOddSubviewOp, zeroConstantOp);

    // Subviews for Matrix B.
    auto ldsBlockBDoubleSize = ldsBlockBSize * 2;
    auto ldsBlockBOffset = ldsBlockADoubleSize;

    auto ldsBlockBOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOffset);
    auto ldsBlockBDoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockBOffset,
                                 {ldsBlockBDoubleSize}, elementType);
    auto ldsBlockBDoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBDoubleMemRefType, ldsGpuAllocOp,
        ldsBlockBOffsetConstantOp);

    auto ldsBlockBEvenOffset = 0;
    auto ldsBlockBEvenOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBEvenOffset);
    auto ldsBlockBEvenMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBEvenOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBEvenMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBEvenOffsetConstantOp);

    auto ldsBlockBOddOffset = ldsBlockBDoubleSize / 2;
    auto ldsBlockBOddOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOddOffset);
    auto ldsBlockBOddMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBOddOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBOddMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBOddOffsetConstantOp);

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
        zeroConstantOp);
    auto lds2DMatrixBOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixBOddMemRefType,
                                    ldsBlockBOddSubviewOp, zeroConstantOp);

    // Alloc for Matrix C on registers.
    // Compute register size from attributes.
    int64_t GemmMRepeat = 0, GemmNRepeat = 0;
    Value registerMatrixCAllocOp;
    auto xdlopsAttr = op.template getAttrOfType<BoolAttr>("xdlops");
    auto xdlopsV2Attr = op.template getAttrOfType<BoolAttr>("xdlopsV2");
    if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
        (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {
      // XDLOPS.
      int64_t WaveSize = 64;
      int64_t TotalRegSize = MPerWave * NPerWave / WaveSize;
      //llvm::errs() << "TotalRegSize: " << TotalRegSize << "\n";
      auto threadCRegisterMemRefType =
          MemRefType::get({TotalRegSize}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
      registerMatrixCAllocOp =
          b.create<miopen::GpuAllocOp>(loc, threadCRegisterMemRefType);
    } else {
      // Non-XDLOPS.

      // Original C++ logic.
      // constexpr index_t GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
      // constexpr index_t GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);
      // constexpr auto c_m0m1_n0n1_thread_mtx_desc = make_ConstantMatrixDescriptor_packed(
      //     Number<GemmMRepeat * MPerThread>{}, Number<GemmNRepeat * NPerThread>{});

      // llvm::errs() << "MPerThread: " << MPerThread << "\n";
      // llvm::errs() << "NPerThread: " << NPerThread << "\n";

      GemmMRepeat = MPerBlock / (MPerThread * MLevel0Cluster * MLevel1Cluster);
      GemmNRepeat = NPerBlock / (NPerThread * NLevel0Cluster * NLevel1Cluster);

      // llvm::errs() << "GemmMRepeat: " << GemmMRepeat << "\n";
      // llvm::errs() << "GemmNRepeat: " << GemmNRepeat << "\n";

      auto threadCRegisterMemRefType = MemRefType::get(
          {GemmMRepeat * MPerThread, GemmNRepeat * NPerThread}, elementType, {},
          gpu::GPUDialect::getPrivateAddressSpace());
      registerMatrixCAllocOp =
          b.create<miopen::GpuAllocOp>(loc, threadCRegisterMemRefType);
    }

    // Alloc for Matrix A / B on registers.
    auto threadARegisterMemRefType = MemRefType::get(
        {GemmABlockCopyThreadSliceLengths_GemmK,
         GemmABlockCopyThreadSliceLengths_GemmM},
        elementType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    auto threadAEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);
    auto threadAOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType = MemRefType::get(
        {GemmBBlockCopyThreadSliceLengths_GemmK,
         GemmBBlockCopyThreadSliceLengths_GemmN},
        elementType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    auto threadBEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);
    auto threadBOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Zero init Matrix C on registers.
    b.create<miopen::FillOp>(loc, registerMatrixCAllocOp,
                             zeroConstantFloatOp);

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // Compute source and destination coordinates for BlockwiseCopy ops.
    auto blockwiseCopyCoordType =
        MemRefType::get({2}, b.getIntegerType(32), {},
                        gpu::GPUDialect::getPrivateAddressSpace());

    // Matrix A: {0, m_block_data_on_global}, {0, 0}
    auto blockwiseCopyASrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_Y_i32, blockwiseCopyASrc,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_X_i32, blockwiseCopyASrc,
                      ValueRange{oneConstantOp});

    auto blockwiseCopyADst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_Y_i32, blockwiseCopyADst,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_X_i32, blockwiseCopyADst,
                      ValueRange{oneConstantOp});

    // Matrix B: {0, n_block_data_on_global}, {0, 0}
    auto blockwiseCopyBSrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_Y_i32, blockwiseCopyBSrc,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_X_i32, blockwiseCopyBSrc,
                      ValueRange{oneConstantOp});

    auto blockwiseCopyBDst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_Y_i32, blockwiseCopyBDst,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_X_i32, blockwiseCopyBDst,
                      ValueRange{oneConstantOp});

    Value mMyThreadOffsetA, mMyThreadOffsetB;
    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32;
    Value m_thread_data_on_global_i32, n_thread_data_on_global_i32;
    if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
        (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {
      // XDLOPS path.

      // c_thread_mtx_index_row(_i32), c_thread_mtx_index_col(_i32)
      // m_thread_data_on_global_i32, n_thread_data_on_global_i32
      // are emitted in matrix C writeout logic later in the function
      // because they are bounded by a loop.

      // Original C++ logic:
      // index_t mMyWaveOffsetA;
      // index_t mMyWaveOffsetB;
      // const index_t waveId   = get_thread_local_1d_id() / WaveSize;
      // const index_t waveId_m = waveId / GemmNWaves;
      // const index_t waveId_n = waveId % GemmNWaves;
      // mMyWaveOffsetA = waveId_m * GemmMPerWave;
      // mMyWaveOffsetB = waveId_n * GemmNPerWave;

      auto waveId = b.create<SignedDivIOp>(loc, tid, waveSizeConstantOp);
      auto waveId_m = b.create<SignedDivIOp>(loc, waveId, NWavesConstantOp);
      auto waveId_n = b.create<SignedRemIOp>(loc, waveId, NWavesConstantOp);
      mMyThreadOffsetA = b.create<MulIOp>(loc, waveId_m, MPerWaveConstantOp);
      mMyThreadOffsetB = b.create<MulIOp>(loc, waveId_n, NPerWaveConstantOp);
    } else {
      // non-XDLOPS path.

      // Compute c_thread_mtx_index for Matrix C.
      int64_t ThreadPerLevel0Cluster = MLevel0Cluster * NLevel0Cluster;
      auto ThreadPerLevel0ClusterConstantOp =
          b.create<ConstantIndexOp>(loc, ThreadPerLevel0Cluster);
      auto level1_id =
          b.create<SignedDivIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
      auto level1_m_id =
          b.create<SignedDivIOp>(loc, level1_id, NLevel1ClusterConstantOp);
      auto level1_n_id =
          b.create<SignedRemIOp>(loc, level1_id, NLevel1ClusterConstantOp);

      auto level0_id =
          b.create<SignedRemIOp>(loc, tid, ThreadPerLevel0ClusterConstantOp);
      auto level0_m_id =
          b.create<SignedDivIOp>(loc, level0_id, NLevel0ClusterConstantOp);
      auto level0_n_id =
          b.create<SignedRemIOp>(loc, level0_id, NLevel0ClusterConstantOp);

      int64_t MPerLevel0Cluster = MPerThread * MLevel0Cluster;
      int64_t NPerLevel0Cluster = NPerThread * NLevel0Cluster;
      auto MPerLevel0ClusterConstantOp =
          b.create<ConstantIndexOp>(loc, MPerLevel0Cluster);
      auto NPerLevel0ClusterConstantOp =
          b.create<ConstantIndexOp>(loc, NPerLevel0Cluster);

      // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0, c_thread_mtx_index.row} = c_thread_mtx_index_row
      c_thread_mtx_index_row = b.create<AddIOp>(
          loc,
          b.create<MulIOp>(loc, level1_m_id, MPerLevel0ClusterConstantOp),
          b.create<MulIOp>(loc, level0_m_id, MPerThreadConstantOp));
      mMyThreadOffsetA = c_thread_mtx_index_row;
      c_thread_mtx_index_row_i32 = b.create<IndexCastOp>(
          loc, c_thread_mtx_index_row, b.getIntegerType(32));

      // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0, c_thread_mtx_index.col} = c_thread_mtx_index_col
      c_thread_mtx_index_col = b.create<AddIOp>(
          loc,
          b.create<MulIOp>(loc, level1_n_id, NPerLevel0ClusterConstantOp),
          b.create<MulIOp>(loc, level0_n_id, NPerThreadConstantOp));
      mMyThreadOffsetB = c_thread_mtx_index_col;
      c_thread_mtx_index_col_i32 = b.create<IndexCastOp>(
          loc, c_thread_mtx_index_col, b.getIntegerType(32));

      m_thread_data_on_global_i32 = b.create<AddIOp>(
          loc, m_block_data_on_global_i32, c_thread_mtx_index_row_i32);
      n_thread_data_on_global_i32 = b.create<AddIOp>(
          loc, n_block_data_on_global_i32, c_thread_mtx_index_col_i32);
    }

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

    auto KPerBlockConstantI32Op =
        b.create<ConstantIntOp>(loc, KPerBlock, b.getIntegerType(32));

    int64_t loopIteration;
    // For XDLOPS path iteration is less by 1.
    if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
        (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {
      loopIteration = K / (KPerBlock * 2) - 1;
    } else {
      loopIteration = K / (KPerBlock * 2);
    }
    auto loopIterationConstantOp =
        b.create<ConstantIndexOp>(loc, loopIteration);
    auto loopOp =
        b.create<scf::ForOp>(loc, zeroConstantOp,
                             loopIterationConstantOp, oneConstantOp);

    // inside the loop.
    auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

    // LDS barrier.
    lb.create<miopen::WorkgroupBarrierOp>(loc);

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

    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      // TBD: Emit xdlops_gemm_v2.
      // TBD: assign attributes.
    } else {
      // Emit blockwise GEMM.
      auto blockwiseGemmEvenOp = lb.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
          registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
      affixBlockwiseGemmAttributes(blockwiseGemmEvenOp, op, b);
    }

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

    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      // TBD: Emit xdlops_gemm_v2.
      // TBD: assign attributes.
    } else {
      // Emit blockwise GEMM.
      auto blockwiseGemmOddOp = lb.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
          registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
      affixBlockwiseGemmAttributes(blockwiseGemmOddOp, op, b);
    }

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

    // For XDLOPS path no need to emit loop tail.
    if (xdlopsAttr && xdlopsAttr.getValue() == true) {
      // LDS barrier.
      b.create<miopen::WorkgroupBarrierOp>(loc);

      // Emit blockwise GEMM for the loop tail.
      auto blockwiseGemmTailEvenOp = b.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
          registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
      affixBlockwiseGemmAttributes(blockwiseGemmTailEvenOp, op, b);

      auto blockwiseGemmTailOddOp = b.create<miopen::BlockwiseGemmOp>(
          loc, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
          registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
      affixBlockwiseGemmAttributes(blockwiseGemmTailOddOp, op, b);
    } else if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      // TBD. emit read VaccGPR.
    } else {
      // LDS barrier.
      b.create<miopen::WorkgroupBarrierOp>(loc);

      // Emit blockwise GEMM for the loop tail.
      if (loopIteration % 2) {
        auto blockwiseGemmTailEvenOp = b.create<miopen::BlockwiseGemmOp>(
            loc, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
            registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
        affixBlockwiseGemmAttributes(blockwiseGemmTailEvenOp, op, b);
      } else {
        auto blockwiseGemmTailOddOp = b.create<miopen::BlockwiseGemmOp>(
            loc, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
            registerMatrixCAllocOp, mMyThreadOffsetA, mMyThreadOffsetB);
        affixBlockwiseGemmAttributes(blockwiseGemmTailOddOp, op, b);
      }
    }

    if ((xdlopsAttr && xdlopsAttr.getValue() == true) ||
        (xdlopsV2Attr && xdlopsV2Attr.getValue() == true)) {
      // XDLOPS-specific logic.
      XdlopsCodeSelection xcs = XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

      // Extract values from XdlopsCodeSelection.
      StringRef mfmaInstr = xcs.mfmaInstr;
      int64_t MPerXdlops = xcs.MPerXdlops;
      int64_t NPerXdlops = xcs.NPerXdlops;
      int64_t MRepeats = xcs.MRepeats;
      int64_t NRepeats = xcs.NRepeats;

      int64_t group_size = xcs.group_size;
      int64_t num_groups_blk = xcs.num_groups_blk;
      int64_t num_regs_blk = xcs.num_regs_blk;
      int64_t num_threads_blk = xcs.num_threads_blk;
      int64_t wave_size = xcs.wave_size;
      int64_t num_input_blks = xcs.num_input_blks;
      int64_t num_output_blks = xcs.num_output_blks;
      int64_t num_regs_xdlops = xcs.num_regs_xdlops;
      int64_t m = xcs.m;
      int64_t n = xcs.n;
      int64_t k = xcs.k;
      int64_t cycles = xcs.cycles;
      int64_t k_base = xcs.k_base;

      auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
      auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
      auto MRepeatsConstantOp = b.create<ConstantIndexOp>(loc, MRepeats);
      auto NRepeatsConstantOp = b.create<ConstantIndexOp>(loc, NRepeats);

      auto group_size_ConstantOp = b.create<ConstantIndexOp>(loc, group_size);
      auto wave_size_ConstantOp = b.create<ConstantIndexOp>(loc, wave_size);
      auto num_threads_blk_ConstantOp = b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto num_output_blks_ConstantOp = b.create<ConstantIndexOp>(loc, num_output_blks);
      auto m_ConstantOp = b.create<ConstantIndexOp>(loc, m);
      auto n_ConstantOp = b.create<ConstantIndexOp>(loc, n);

      // XDLOPS.

      // Original C++ logic.
      // __device__ static constexpr index_t GetNumBlksPerXdlops() {
      //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
      // }
      //
      // struct OutputLayout {
      //     __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_blk; }
      //     __device__ static constexpr index_t GetNumBlks() {
      //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
      //     }
      // };
      // using CThreadCopySliceLengths = Sequence<M0, 1, M2, 1>;
      // constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
      // constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

      int BlkSize = num_regs_blk;
      int NumBlksPerXdlops = (MPerXdlops * NPerXdlops) / (m * n);
      int NumBlks = NumBlksPerXdlops * MRepeats * NRepeats;

      // llvm::errs() << "MPerWave: " << MPerWave << "\n";
      // llvm::errs() << "NPerWave: " << NPerWave << "\n\n";

      // llvm::errs() << "MPerXlops: " << MPerXdlops << "\n";
      // llvm::errs() << "NPerXlops: " << NPerXdlops << "\n";
      // llvm::errs() << "m: " << m << "\n";
      // llvm::errs() << "n: " << n << "\n";
      // llvm::errs() << "MRepeat: " << MRepeats << "\n";
      // llvm::errs() << "NRepeat: " << NRepeats << "\n\n";

      // llvm::errs() << "BlkSize: " << BlkSize << "\n";
      // llvm::errs() << "NumBlksPerXdlops: " << NumBlksPerXdlops << "\n";
      // llvm::errs() << "NumBlks: " << NumBlks << "\n\n";

      auto BlkSizeConstantI32Op = b.create<ConstantIntOp>(loc, BlkSize, b.getIntegerType(32));
      auto NumBlksPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NumBlksPerXdlops);
      auto NumBlksConstantOp = b.create<ConstantIndexOp>(loc, NumBlks);
 
      // Threadwise copy from register (naive tensor) to global (generic tensor).
      // Original C++ logic:
      //
      // __device__ static constexpr index_t GetNumBlksPerXdlops() {
      //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
      // }
      //
      // struct OutputLayout {
      //     __device__ static constexpr index_t M1() { return mfma_type.num_groups_blk; }
      //     __device__ static constexpr index_t M0() { return mfma_type.group_size; }
      //     __device__ static constexpr index_t N1() { return mfma_type.num_input_blks; }
      //     __device__ static constexpr index_t N0() { return mfma_type.num_threads_blk; }
      //     __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_blk; }
      //     __device__ static constexpr index_t GetNumBlks() {
      //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
      //     }
      // };
      //
      // // CLayout.M1() = num_groups;
      // // CLayout.M0() = group_size;
      // // CLayout.N1() = num_blks_per_wave;
      // // CLayout.N0() = num_threads_per_blks;
      // constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
      // constexpr index_t M0   = CLayout.M1();
      // constexpr index_t M1   = CLayout.N1();
      // constexpr index_t M2   = CLayout.M0();

      int64_t M3 = num_groups_blk;
      int64_t M1 = num_input_blks;
      int64_t M2 = group_size;
      int64_t M0 = M / (M1 * M2);

      // llvm::errs() << "M0: " << M0 << "\n";
      // llvm::errs() << "M1: num_input_blks: " << M1 << "\n";
      // llvm::errs() << "M2: group_size: " << M2 << "\n";
      // llvm::errs() << "M3: num_groups_blk: " << M3 << "\n\n";

      auto M0ConstantI32Op =
          b.create<ConstantIntOp>(loc, M0, b.getIntegerType(32));
      auto M1ConstantI32Op =
          b.create<ConstantIntOp>(loc, M1, b.getIntegerType(32));
      auto M2ConstantI32Op =
          b.create<ConstantIntOp>(loc, M2, b.getIntegerType(32));
      auto M3ConstantI32Op =
          b.create<ConstantIntOp>(loc, M3, b.getIntegerType(32));
      auto NConstantI32Op =
          b.create<ConstantIntOp>(loc, N, b.getIntegerType(32));

      // constexpr auto c_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
      //     c_m_n_global_desc,
      //     make_tuple(UnMerge<Sequence<M0, M1, M2>>{}, PassThrough<N>{}),
      //     make_tuple(Sequence<0>{}, Sequence<1>{}),
      //     make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));
 
      // build affine expression:
      // (d0, d1, d2, d3) -> (d0 * M1 * M2 + d1 * M2 + d2, d3)
      auto affineMap4to2 =
          AffineMap::get(4, 0,
                         {getAffineDimExpr(0, op.getContext()) *
                              getAffineConstantExpr(M1, op.getContext()) *
                              getAffineConstantExpr(M2, op.getContext()) +
                          getAffineDimExpr(1, op.getContext()) *
                              getAffineConstantExpr(M2, op.getContext()) +
                          getAffineDimExpr(2, op.getContext()),
                          getAffineDimExpr(3, op.getContext())},
                         op.getContext());

      // compose with output tensor affine map.
      auto outputType = op.output().getType().template dyn_cast<MemRefType>();
      auto outputAffineMap2to4 = outputType.getAffineMaps()[0];
      auto affineMap4to2to4 = outputAffineMap2to4.compose(affineMap4to2);

      // emit TransformOp for output tensor.
      auto newOutputType = MemRefType::get(
          {M0, M1, M2, N}, outputType.getElementType(), {affineMap4to2to4});
      auto newOutputTransformOp =
          b.create<miopen::TransformOp>(loc, newOutputType, op.output());

      // Original C++ logic.
      // //     src descriptor
      // constexpr auto c_m0_m1_m2_n_thread_desc =
      //     make_native_tensor_descriptor_packed(Sequence<M0, 1, M2, 1>{});
      // 
      // Note: turns out this layout is not enough to cover the whole matrix C
      //       on VGPR. It only covers 1/NumBlks of it.

      // A layout of Sequence<NumBlks * M3, 1, M2, 1> would cover the whole matrix C
      // on VGPR.
      // build affine expression for Sequence<NumBlks * M3, 1, M2, 1>
      // (d0, d1, d2, d3) -> (d0 * M2 + d2)
      auto matrixCAffineMap4to1 = AffineMap::get(
          4, 0,
          {getAffineDimExpr(0, op.getContext()) * getAffineConstantExpr(M2, op.getContext()) +
           getAffineDimExpr(2, op.getContext())},
          op.getContext());

      // emit TransformOp for Matrix C on VGPR.
      auto register4DMatrixCType = MemRefType::get(
          {NumBlks * M3, 1, M2, 1}, elementType,
          {matrixCAffineMap4to1}, gpu::GPUDialect::getPrivateAddressSpace());
      auto matrixCTransformOp = b.create<miopen::TransformOp>(
          loc, register4DMatrixCType, registerMatrixCAllocOp);

      // for(index_t i = 0; i < NumBlks; ++i)
      // {
      //     // calculate origin of thread output tensor on global memory
      //     //     blockwise GEMM c matrix starting index
      //     const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);
      //     const index_t m_thread_data_on_global =
      //         m_block_data_on_global + c_thread_mtx_on_block.row;
      //     const index_t n_thread_data_on_global =
      //         n_block_data_on_global + c_thread_mtx_on_block.col;
      //     ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_m2_n_thread_desc),
      //                                           decltype(c_m0_m1_m2_n_global_desc),
      //                                           CThreadCopySliceLengths,
      //                                           arithmetic_sequence_gen<0, 4, 1>::type,
      //                                           3,
      //                                           1,
      //                                           1,
      //                                           AddressSpace::Vgpr,
      //                                           AddressSpace::Global,
      //                                           CGlobalMemoryDataOperation>(
      //         {0, 0, 0, 0},
      //         {m_thread_data_on_global / (M2 * M1),
      //          m_thread_data_on_global % (M2 * M1) / M2,
      //          m_thread_data_on_global % M2,
      //          n_thread_data_on_global})
      //         .Run(p_c_thread + i * BlkSize, p_c_global);
      // }

      // emit loop.
      auto loopOp = b.create<scf::ForOp>(loc, zeroConstantOp, NumBlksConstantOp, oneConstantOp);

      // inside the main loop.
      auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());
      auto iv = loopOp.getInductionVar();
      auto iv_i32 = lb.create<IndexCastOp>(loc, iv, lb.getIntegerType(32));

      // Original C++ logic.
      //
      // In xdlops_gemm.hpp:
      //
      // __device__ static constexpr index_t GetNumBlksPerXdlops() {
      //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
      // }
      // static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops; }
      // __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i) {
      //     const index_t xdlops_i = i / GetNumBlksPerXdlops();
      //     const index_t j        = i % GetNumBlksPerXdlops();
      //     const index_t m_i = xdlops_i / NRepeats;
      //     const index_t n_i = xdlops_i % NRepeats;
      //     const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;
      //     index_t col_blk = j % mfma_type.num_output_blks;
      //     index_t row_blk = j / mfma_type.num_output_blks;
      //     static_if<!IsABroadcast>{}([&](auto) {
      //         col_blk = j / mfma_type.num_output_blks;
      //         row_blk = j % mfma_type.num_output_blks;
      //     });
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;
      //     return MatrixIndex{row, col};
      // }
      //
      // In blockwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i) {
      //     const index_t waveId = get_thread_local_1d_id() / WaveSize;
      //     const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(i);
      //     const index_t col = (waveId % GemmNWaves) * GemmNPerWave + thread_mtx_on_blk.col;
      //     const index_t row = (waveId / GemmNWaves) * GemmMPerWave + thread_mtx_on_blk.row;
      //     return MatrixIndex{row, col};
      // }
      //
      // In gridwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);
      // const index_t m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      // compute thread_mtx_on_blk_row and thread_mtx_on_blk_col.
      auto xdlops_i = lb.create<SignedDivIOp>(loc, iv, NumBlksPerXdlopsConstantOp);
      auto j = lb.create<SignedRemIOp>(loc, iv, NumBlksPerXdlopsConstantOp);
      auto m_i = lb.create<SignedDivIOp>(loc, xdlops_i, NRepeatsConstantOp);
      auto n_i = lb.create<SignedRemIOp>(loc, xdlops_i, NRepeatsConstantOp);
      auto laneId = lb.create<SignedRemIOp>(loc, tid, wave_size_ConstantOp);
      auto blk_id = lb.create<SignedDivIOp>(loc, laneId, num_threads_blk_ConstantOp);
      auto blk_td = lb.create<SignedRemIOp>(loc, laneId, num_threads_blk_ConstantOp);
      Value col_blk, row_blk;
      if (NPerXdlops >= MPerXdlops) {
        // IsABroadcast
        col_blk = lb.create<SignedRemIOp>(loc, j, num_output_blks_ConstantOp);
        row_blk = lb.create<SignedDivIOp>(loc, j, num_output_blks_ConstantOp);
      } else {
        // !IsABroadcast
        col_blk = lb.create<SignedDivIOp>(loc, j, num_output_blks_ConstantOp);
        row_blk = lb.create<SignedRemIOp>(loc, j, num_output_blks_ConstantOp);
      } 
      // Original C++ logic.
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      auto thread_mtx_on_blk_col = lb.create<AddIOp>(loc,
        lb.create<AddIOp>(loc,                     
          lb.create<MulIOp>(loc, col_blk, n_ConstantOp),
          blk_td),
        lb.create<MulIOp>(loc, n_i, NPerXdlopsConstantOp));
      // Original C++ logic.
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;
      auto thread_mtx_on_blk_row = lb.create<AddIOp>(loc,
        lb.create<AddIOp>(loc,
          lb.create<MulIOp>(loc, row_blk, m_ConstantOp),
          lb.create<MulIOp>(loc, blk_id, group_size_ConstantOp)),
        lb.create<MulIOp>(loc, m_i, MPerXdlopsConstantOp));

      // compute c_thread_mtx_index_row, c_thread_mtx_index_col.
      // compute c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32.

      // compute waveId.
      auto waveId = lb.create<SignedDivIOp>(loc, tid, waveSizeConstantOp);
      
      // Original C++ logic.
      // const index_t col = (waveId % GemmNWaves) * GemmNPerWave + thread_mtx_on_blk.col;
      c_thread_mtx_index_col = lb.create<AddIOp>(loc,
        lb.create<MulIOp>(loc,
          lb.create<SignedRemIOp>(loc, waveId, NWavesConstantOp),
          NPerWaveConstantOp),
        thread_mtx_on_blk_col);
      c_thread_mtx_index_col_i32 = lb.create<IndexCastOp>(loc, c_thread_mtx_index_col, lb.getIntegerType(32));

      // Original C++ logic.
      // const index_t row = (waveId / GemmNWaves) * GemmMPerWave + thread_mtx_on_blk.row;
      c_thread_mtx_index_row = lb.create<AddIOp>(loc,
        lb.create<MulIOp>(loc,
          lb.create<SignedDivIOp>(loc, waveId, NWavesConstantOp),
          MPerWaveConstantOp),
        thread_mtx_on_blk_row); 
      c_thread_mtx_index_row_i32 = lb.create<IndexCastOp>(loc, c_thread_mtx_index_row, lb.getIntegerType(32));

      // Original C++ logic:
      // const index_t m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      m_thread_data_on_global_i32 = lb.create<AddIOp>(
          loc, m_block_data_on_global_i32, c_thread_mtx_index_row_i32);
      n_thread_data_on_global_i32 = lb.create<AddIOp>(
          loc, n_block_data_on_global_i32, c_thread_mtx_index_col_i32);
 
      SmallVector<Value, 8> matrixCThreadwiseCopySourceAndDestCoords;
      auto coord0 = lb.create<MulIOp>(loc, iv_i32, M2ConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(coord0);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);

      matrixCThreadwiseCopySourceAndDestCoords.push_back(lb.create<SignedDivIOp>(
          loc, m_thread_data_on_global_i32, lb.create<MulIOp>(loc, M2ConstantI32Op, M1ConstantI32Op)));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(lb.create<SignedDivIOp>(
          loc, lb.create<SignedRemIOp>(loc, m_thread_data_on_global_i32, lb.create<MulIOp>(loc, M2ConstantI32Op, M1ConstantI32Op)),
               M2ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(lb.create<SignedRemIOp>(
          loc, m_thread_data_on_global_i32, M2ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(n_thread_data_on_global_i32);

      auto threadwiseCopyCMatrixOp = lb.create<miopen::ThreadwiseCopyOp>(
          loc, matrixCTransformOp, newOutputTransformOp,
          matrixCThreadwiseCopySourceAndDestCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyCMatrixOp, op, lb);

      // affix bound attributes.
      threadwiseCopyCMatrixOp.setAttr("bound", b.getArrayAttr({
                                      b.getI32IntegerAttr(M3),
                                      b.getI32IntegerAttr(1),
                                      b.getI32IntegerAttr(M2),
                                      b.getI32IntegerAttr(1),
                                     }));
    } else {
      // non-XDLOPS path.

      // Threadwise copy from register (naive tensor) to global (generic tensor).
      // Original C++ logic:
      //
      // constexpr index_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
      // constexpr index_t M0 = M / M1;
      //
      // constexpr index_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;
      // constexpr index_t N0 = N / N1;
      //
      // // define input tensor descriptor for threadwise copy
      // //     thread input tensor, src of threadwise copy
      // constexpr auto c_m0_m1_n0_n1_thread_desc =
      // make_native_tensor_descriptor_packed(
      //     Sequence<GemmMRepeat, MPerThread, GemmNRepeat, NPerThread>{});
      //
      // constexpr auto c_m0_m1_n0_n1_global_desc = transform_tensor_descriptor(
      //     c_m_n_global_desc,
      //     make_tuple(UnMerge<Sequence<M0, M1>>{}, UnMerge<Sequence<N0, N1>>{}),
      //     make_tuple(Sequence<0>{}, Sequence<1>{}),
      //     make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}));
      //
      // // calculate origin of thread input tensor on global memory
      // //     blockwise GEMM c matrix starting index
      // const auto c_thread_mtx_on_block =
      //     blockwise_gemm.GetBeginOfThreadMatrixC(get_thread_local_1d_id());
      //
      // const index_t m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      //
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;
      //
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

      int64_t M1 = MPerThread * MLevel0Cluster * MLevel1Cluster;
      int64_t M0 = M / M1;
      int64_t N1 = NPerThread * NLevel0Cluster * NLevel1Cluster;
      int64_t N0 = N / N1;

      auto M0ConstantI32Op =
          b.create<ConstantIntOp>(loc, M0, b.getIntegerType(32));
      auto M1ConstantI32Op =
          b.create<ConstantIntOp>(loc, M1, b.getIntegerType(32));
      auto N0ConstantI32Op =
          b.create<ConstantIntOp>(loc, N0, b.getIntegerType(32));
      auto N1ConstantI32Op =
          b.create<ConstantIntOp>(loc, N1, b.getIntegerType(32));

      // build affine expression:
      // (d0, d1, d2, d3) -> (d0 * M1 + d1, d2 * N1 + d3)
      auto affineMap4to2 =
          AffineMap::get(4, 0,
                         {getAffineDimExpr(1, op.getContext()) +
                              getAffineDimExpr(0, op.getContext()) *
                                  getAffineConstantExpr(M1, op.getContext()),
                          getAffineDimExpr(3, op.getContext()) +
                              getAffineDimExpr(2, op.getContext()) *
                                  getAffineConstantExpr(N1, op.getContext())},
                         op.getContext());

      // compose with output tensor affine map.
      auto outputType = op.output().getType().template dyn_cast<MemRefType>();
      auto outputAffineMap2to4 = outputType.getAffineMaps()[0];
      auto affineMap4to2to4 = outputAffineMap2to4.compose(affineMap4to2);

      // emit TransformOp for output tensor.
      auto newOutputType = MemRefType::get(
          {M0, M1, N0, N1}, outputType.getElementType(), {affineMap4to2to4});
      auto newOutputTransformOp =
          b.create<miopen::TransformOp>(loc, newOutputType, op.output());

      // build affine expression:
      // (d0, d1, d2, d3) -> (d0 * MPerThread + d1, d2 * NPerThread + d3)
      auto matrixCAffineMap4to2 = AffineMap::get(
          4, 0,
          {getAffineDimExpr(1, op.getContext()) +
               getAffineDimExpr(0, op.getContext()) *
                   getAffineConstantExpr(MPerThread, op.getContext()),
           getAffineDimExpr(3, op.getContext()) +
               getAffineDimExpr(2, op.getContext()) *
                   getAffineConstantExpr(MPerThread, op.getContext())},
          op.getContext());

      // emit TransformOp for Matrix C on VGPR.
      auto register4DMatrixCType = MemRefType::get(
          {GemmMRepeat, MPerThread, GemmNRepeat, NPerThread}, elementType,
          {matrixCAffineMap4to2}, gpu::GPUDialect::getPrivateAddressSpace());
      auto matrixCTransformOp = b.create<miopen::TransformOp>(
          loc, register4DMatrixCType, registerMatrixCAllocOp);

      SmallVector<Value, 8> matrixCThreadwiseCopySourceAndDestCoords;
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);

      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedDivIOp>(
          loc, m_thread_data_on_global_i32, M1ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedRemIOp>(
          loc, m_thread_data_on_global_i32, M1ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedDivIOp>(
          loc, n_thread_data_on_global_i32, N1ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedRemIOp>(
          loc, n_thread_data_on_global_i32, N1ConstantI32Op));

      auto threadwiseCopyCMatrixOp = b.create<miopen::ThreadwiseCopyOp>(
          loc, matrixCTransformOp, newOutputTransformOp,
          matrixCThreadwiseCopySourceAndDestCoords);
      affixThreadwiseCopyAttributes(threadwiseCopyCMatrixOp, op, b);
    }

    op.erase();

    return success();
  }
};

//===----------------------------------------------------------------------===//
// GridwiseGemmV2 lowering.
//===----------------------------------------------------------------------===//

struct GridwiseGemmV2RewritePattern : public OpRewritePattern<miopen::GridwiseGemmV2Op> {
  using OpRewritePattern<miopen::GridwiseGemmV2Op>::OpRewritePattern;

  void computeLDSBlockSizes(miopen::GridwiseGemmV2Op op, int64_t &a_block_space, int64_t &b_block_space, int64_t &double_block_space) const {
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

     // llvm::errs() << "a_block_space: " << a_block_space << "\n";
     // llvm::errs() << "b_block_space: " << b_block_space << "\n";
     // llvm::errs() << "double_block_space: " << double_block_space << "\n\n";
  }

  // XXX. Figure out a way to do away with isMatrixA parameter.
  void affixBlockwiseCopyAttributes(miopen::BlockwiseCopyOp bop,
                                    miopen::GridwiseGemmV2Op gop,
                                    OpBuilder &b, bool isMatrixA) const {
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

  void affixXdlopsGemmV2Attributes(miopen::XdlopsGemmV2Op xop,
                                   miopen::GridwiseGemmV2Op gop,
                                   OpBuilder &b) const {
    xop.setAttr("block_size", gop.getAttr("block_size"));
    // xdlopsV2.
    auto xdlopsV2Attr = gop.template getAttrOfType<BoolAttr>("xdlopsV2");
    if (xdlopsV2Attr && xdlopsV2Attr.getValue() == true) {
      int64_t MPerBlock =
          gop.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerBlock =
          gop.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();
      int64_t MPerWave =
          gop.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerWave =
          gop.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t MWaves = MPerBlock / MPerWave;
      int64_t NWaves = NPerBlock / NPerWave;

      xop.setAttr("m_per_wave", gop.getAttr("m_per_thread"));
      xop.setAttr("n_per_wave", gop.getAttr("n_per_thread"));
      xop.setAttr("m_waves", b.getI32IntegerAttr(MWaves));
      xop.setAttr("n_waves", b.getI32IntegerAttr(NWaves));

      xop.setAttr("xdlopsV2", b.getBoolAttr(true));
    }
  }

  void affixBlockwiseGemmV2Attributes(miopen::BlockwiseGemmV2Op bop,
                                      miopen::GridwiseGemmV2Op gop,
                                      OpBuilder &b) const {
    bop.setAttr("block_size", gop.getAttr("block_size"));

    int64_t MPerBlock =
        gop.getAttr("m_per_block").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerBlock =
        gop.getAttr("n_per_block").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerWave =
        gop.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        gop.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;

    bop.setAttr("m_per_wave", gop.getAttr("m_per_thread"));
    bop.setAttr("n_per_wave", gop.getAttr("n_per_thread"));
    bop.setAttr("m_waves", b.getI32IntegerAttr(MWaves));
    bop.setAttr("n_waves", b.getI32IntegerAttr(NWaves));

    int64_t M = bop.matrixA().getType().template dyn_cast<MemRefType>().getShape()[1];
    int64_t N = bop.matrixB().getType().template dyn_cast<MemRefType>().getShape()[1];
    int64_t K = bop.matrixA().getType().template dyn_cast<MemRefType>().getShape()[0];

    bop.setAttr("m", b.getI32IntegerAttr(M));
    bop.setAttr("n", b.getI32IntegerAttr(N));
    bop.setAttr("k", b.getI32IntegerAttr(K));

    bop.setAttr("coord_transforms", b.getArrayAttr({}));
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

  LogicalResult matchAndRewrite(miopen::GridwiseGemmV2Op op, PatternRewriter &b) const override {
    auto loc = op.getLoc();

    auto elementType = op.output().getType().cast<MemRefType>().getElementType();

    // Prepare some useful constants.
    auto zeroConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 0, b.getIntegerType(32));

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);

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
    auto MPerThreadConstantOp = b.create<ConstantIndexOp>(loc, MPerThread);
    auto NPerThreadConstantOp = b.create<ConstantIndexOp>(loc, NPerThread);

    int64_t matrix_a_source_data_per_read =
        op.getAttr("matrix_a_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();
    int64_t matrix_b_source_data_per_read =
        op.getAttr("matrix_b_source_data_per_read")
            .template dyn_cast<IntegerAttr>()
            .getInt();

    // XDLOPS.
    int64_t MPerWave = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();
    int64_t MWaves = MPerBlock / MPerWave;
    int64_t NWaves = NPerBlock / NPerWave;
    auto dataType = op.input().getType().template dyn_cast<MemRefType>().getElementType().template dyn_cast<FloatType>();

    auto MPerWaveConstantOp = b.create<ConstantIndexOp>(loc, MPerWave);
    auto NPerWaveConstantOp = b.create<ConstantIndexOp>(loc, NPerWave);
    auto MWavesConstantOp = b.create<ConstantIndexOp>(loc, MWaves);
    auto NWavesConstantOp = b.create<ConstantIndexOp>(loc, NWaves);

    int64_t WaveSize = 64;
    auto waveSizeConstantOp = b.create<ConstantIndexOp>(loc, WaveSize);

    // Get current workgroup ID.
    auto bid = b.create<miopen::WorkgroupIdOp>(loc, b.getIndexType());

    int64_t MBlockWork = M / MPerBlock;
    int64_t NBlockWork = N / NPerBlock;

    // llvm::errs() << "M: " << M << "\n";
    // llvm::errs() << "N: "  << N << "\n";
    // llvm::errs() << "K: "  << K << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MBlockWork = M / MPerBlock: " << MBlockWork << "\n";
    // llvm::errs() << "NBlockWork = N / NPerBlock: " << NBlockWork << "\n";
    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n";
    // llvm::errs() << "MWaves = MPerBlock / MPerWave: " << MWaves << "\n";
    // llvm::errs() << "NWaves = NPerBlock / NPerWave: " << NWaves << "\n";

    auto MBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, MBlockWork);
    auto NBlockWorkConstantOp = b.create<ConstantIndexOp>(loc, NBlockWork);
    auto block_work_id_m =
        b.create<SignedDivIOp>(loc, bid, NBlockWorkConstantOp);
    auto block_work_id_n =
        b.create<SignedRemIOp>(loc, bid, NBlockWorkConstantOp);
    auto MPerBlockConstantOp = b.create<ConstantIndexOp>(loc, MPerBlock);
    auto NPerBlockConstantOp = b.create<ConstantIndexOp>(loc, NPerBlock);
    auto m_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_m, MPerBlockConstantOp);
    auto n_block_data_on_global =
        b.create<MulIOp>(loc, block_work_id_n, NPerBlockConstantOp);
    auto m_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, m_block_data_on_global, b.getIntegerType(32));
    auto n_block_data_on_global_i32 = b.create<IndexCastOp>(
        loc, n_block_data_on_global, b.getIntegerType(32));

    // llvm::errs() << "KPerBlock: " << KPerBlock << "\n";
    // llvm::errs() << "MPerBlock: " << MPerBlock << "\n";
    // llvm::errs() << "NPerBlock: " << NPerBlock << "\n";
    // llvm::errs() << "matrix_a_source_data_per_read: " << matrix_a_source_data_per_read << "\n";
    // llvm::errs() << "matrix_b_source_data_per_read: " << matrix_b_source_data_per_read << "\n";

    // Compute ThreadClusterLengths for Matrix A.
    int64_t GemmABlockCopyClusterLengths_GemmK =
        KPerBlock /
        ((MPerBlock * KPerBlock / BlockSize) / matrix_a_source_data_per_read);
    int64_t GemmABlockCopyClusterLengths_GemmM =
        MPerBlock / matrix_a_source_data_per_read;

    // llvm::errs() << "thread cluster lengths for Matrix A\n";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmABlockCopyClusterLengths_GemmM << "\n";

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

    // llvm::errs() << "thread cluster lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyClusterLengths_GemmN << "\n";

    // Compute ThreadSliceLengths for Matrix B.
    int64_t GemmBBlockCopyThreadSliceLengths_GemmK =
        KPerBlock / GemmBBlockCopyClusterLengths_GemmK;
    int64_t GemmBBlockCopyThreadSliceLengths_GemmN =
        NPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    // llvm::errs() << "thread slice lengths for Matrix B\n";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmK << " ";
    // llvm::errs() << GemmBBlockCopyThreadSliceLengths_GemmN << "\n";

    // Get current workitem ID.

    // Original logic.
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

    auto GemmABlockCopyThreadClusterId_Y = b.create<SignedRemIOp>(
        loc, tid, GemmABlockCopyClusterLengths_GemmKConstantOp);
    auto GemmABlockCopyThreadClusterId_X = b.create<SignedDivIOp>(
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

    // Allocate LDS.
    auto ldsMemRefType =
        MemRefType::get({ldsBlockSize}, elementType, {},
                        gpu::GPUDialect::getWorkgroupAddressSpace());
    auto ldsGpuAllocOp = b.create<miopen::GpuAllocOp>(loc, ldsMemRefType);

    // Subviews for Matrix A.
    auto ldsBlockADoubleSize = ldsBlockASize * 2;
    auto ldsBlockAOffset = 0;

    auto ldsBlockAOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOffset);
    auto ldsBlockADoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockAOffset,
                                 {ldsBlockADoubleSize}, elementType);
    auto ldsBlockADoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockADoubleMemRefType, ldsGpuAllocOp,
        ldsBlockAOffsetConstantOp);

    auto ldsBlockAEvenOffset = 0;
    auto ldsBlockAEvenOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAEvenOffset);
    auto ldsBlockAEvenMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAEvenOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAEvenMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAEvenOffsetConstantOp);

    auto ldsBlockAOddOffset = ldsBlockADoubleSize / 2;
    auto ldsBlockAOddOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockAOddOffset);
    auto ldsBlockAOddMemRefType = computeSubviewResultType(
        op, ldsBlockADoubleMemRefType, ldsBlockAOddOffset, {ldsBlockASize},
        elementType);
    auto ldsBlockAOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockAOddMemRefType, ldsBlockADoubleSubviewOp,
        ldsBlockAOddOffsetConstantOp);

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
        zeroConstantOp);
    auto lds2DMatrixAOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixAOddMemRefType,
                                    ldsBlockAOddSubviewOp, zeroConstantOp);

    // Subviews for Matrix B.
    auto ldsBlockBDoubleSize = ldsBlockBSize * 2;
    auto ldsBlockBOffset = ldsBlockADoubleSize;

    auto ldsBlockBOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOffset);
    auto ldsBlockBDoubleMemRefType =
        computeSubviewResultType(op, ldsMemRefType, ldsBlockBOffset,
                                 {ldsBlockBDoubleSize}, elementType);
    auto ldsBlockBDoubleSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBDoubleMemRefType, ldsGpuAllocOp,
        ldsBlockBOffsetConstantOp);

    auto ldsBlockBEvenOffset = 0;
    auto ldsBlockBEvenOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBEvenOffset);
    auto ldsBlockBEvenMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBEvenOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBEvenSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBEvenMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBEvenOffsetConstantOp);

    auto ldsBlockBOddOffset = ldsBlockBDoubleSize / 2;
    auto ldsBlockBOddOffsetConstantOp =
        b.create<ConstantIndexOp>(loc, ldsBlockBOddOffset);
    auto ldsBlockBOddMemRefType = computeSubviewResultType(
        op, ldsBlockBDoubleMemRefType, ldsBlockBOddOffset, {ldsBlockBSize},
        elementType);
    auto ldsBlockBOddSubviewOp = b.create<miopen::SubviewOp>(
        loc, ldsBlockBOddMemRefType, ldsBlockBDoubleSubviewOp,
        ldsBlockBOddOffsetConstantOp);

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
        zeroConstantOp);
    auto lds2DMatrixBOddSubviewOp =
        b.create<miopen::SubviewOp>(loc, lds2DMatrixBOddMemRefType,
                                    ldsBlockBOddSubviewOp, zeroConstantOp);

    // Alloc for Matrix A / B on registers.
    auto threadARegisterMemRefType = MemRefType::get(
        {GemmABlockCopyThreadSliceLengths_GemmK,
         GemmABlockCopyThreadSliceLengths_GemmM},
        elementType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    auto threadAEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);
    auto threadAOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType = MemRefType::get(
        {GemmBBlockCopyThreadSliceLengths_GemmK,
         GemmBBlockCopyThreadSliceLengths_GemmN},
        elementType, {}, gpu::GPUDialect::getPrivateAddressSpace());
    auto threadBEvenAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);
    auto threadBOddAllocOp =
        b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

    // XDLOPS code selection.
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

    int64_t group_size = xcs.group_size;
    int64_t num_groups_blk = xcs.num_groups_blk;
    int64_t num_regs_blk = xcs.num_regs_blk;
    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t num_regs_xdlops = xcs.num_regs_xdlops;
    int64_t m = xcs.m;
    int64_t n = xcs.n;
    int64_t k = xcs.k;
    int64_t cycles = xcs.cycles;
    int64_t k_base = xcs.k_base;

    // Allocate 0-initialized vectors for C.
    SmallVector<Value, 4> vectorCs;
    SmallVector<Type, 4> vectorCTypes;
    for (int64_t iter = 0; iter < vectorNumber; ++iter) {
      auto vectorC = b.create<SplatOp>(loc, zeroConstantFloatOp, vectorType);
      vectorCs.push_back(vectorC);
      vectorCTypes.push_back(vectorType);
    }

    // Blockwise copies before the loop.
    // Blockwise copy from global (generic tensor) to LDS (naive tensor).

    // Compute source and destination coordinates for BlockwiseCopy ops.
    auto blockwiseCopyCoordType =
        MemRefType::get({2}, b.getIntegerType(32), {},
                        gpu::GPUDialect::getPrivateAddressSpace());

    // Matrix A: {0, m_block_data_on_global}, {0, 0}
    auto blockwiseCopyASrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_Y_i32, blockwiseCopyASrc,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmABlockCopySourceCoord_X_i32, blockwiseCopyASrc,
                      ValueRange{oneConstantOp});

    auto blockwiseCopyADst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_Y_i32, blockwiseCopyADst,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmABlockCopyDestCoord_X_i32, blockwiseCopyADst,
                      ValueRange{oneConstantOp});

    // Matrix B: {0, n_block_data_on_global}, {0, 0}
    auto blockwiseCopyBSrc =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_Y_i32, blockwiseCopyBSrc,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmBBlockCopySourceCoord_X_i32, blockwiseCopyBSrc,
                      ValueRange{oneConstantOp});

    auto blockwiseCopyBDst =
        b.create<miopen::GpuAllocOp>(loc, blockwiseCopyCoordType);
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_Y_i32, blockwiseCopyBDst,
                      ValueRange{zeroConstantOp});
    b.create<StoreOp>(loc, GemmBBlockCopyDestCoord_X_i32, blockwiseCopyBDst,
                      ValueRange{oneConstantOp});

    Value mMyThreadOffsetA, mMyThreadOffsetB;
    Value c_thread_mtx_index_row, c_thread_mtx_index_col;
    Value c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32;
    Value m_thread_data_on_global_i32, n_thread_data_on_global_i32;

    // c_thread_mtx_index_row(_i32), c_thread_mtx_index_col(_i32)
    // m_thread_data_on_global_i32, n_thread_data_on_global_i32
    // are emitted in matrix C writeout logic later in the function
    // because they are bounded by a loop.

    // Original C++ logic:
    // index_t mMyWaveOffsetA;
    // index_t mMyWaveOffsetB;
    // const index_t waveId   = get_thread_local_1d_id() / WaveSize;
    // const index_t waveId_m = waveId / GemmNWaves;
    // const index_t waveId_n = waveId % GemmNWaves;
    // mMyWaveOffsetA = waveId_m * GemmMPerWave;
    // mMyWaveOffsetB = waveId_n * GemmNPerWave;
    auto waveId = b.create<SignedDivIOp>(loc, tid, waveSizeConstantOp);
    auto waveId_m = b.create<SignedDivIOp>(loc, waveId, NWavesConstantOp);
    auto waveId_n = b.create<SignedRemIOp>(loc, waveId, NWavesConstantOp);
    mMyThreadOffsetA = b.create<MulIOp>(loc, waveId_m, MPerWaveConstantOp);
    mMyThreadOffsetB = b.create<MulIOp>(loc, waveId_n, NPerWaveConstantOp);
    
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

    auto KPerBlockConstantI32Op =
        b.create<ConstantIntOp>(loc, KPerBlock, b.getIntegerType(32));

    int64_t loopIteration = K / (KPerBlock * 2);
    auto loopIterationConstantOp =
        b.create<ConstantIndexOp>(loc, loopIteration);
    auto mfmaLoopOp =
        b.create<scf::ForOp>(loc, zeroConstantOp,
                             loopIterationConstantOp, oneConstantOp, vectorCs);

    // inside the loop.
    auto mfmalb = OpBuilder::atBlockBegin(mfmaLoopOp.getBody());

    // LDS barrier.
    mfmalb.create<miopen::WorkgroupBarrierOp>(loc);

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    mfmalb.create<miopen::MovePosOp>(
        loc, blockwiseCopyASrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpAEven = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, op.filter(), threadAEvenAllocOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAEven, op, b,
                                 /*isMatrixA=*/true);
    mfmalb.create<miopen::MovePosOp>(
        loc, blockwiseCopyBSrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpBEven = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, op.input(), threadBEvenAllocOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBEven, op, b,
                                 /*isMatrixA=*/false);

    // Emit blockwise V2 GEMM.
    auto blockwiseGemmV2EvenOp = mfmalb.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
        mMyThreadOffsetA, mMyThreadOffsetB, mfmaLoopOp.getRegionIterArgs());
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2EvenOp, op, b);
 
    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    auto blockwiseCopyOpAOdd = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, threadAEvenAllocOp, lds2DMatrixAOddSubviewOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAOdd, op, b,
                                 /*isMatrixA=*/true);
    auto blockwiseCopyOpBOdd = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, threadBEvenAllocOp, lds2DMatrixBOddSubviewOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBOdd, op, b,
                                 /*isMatrixA=*/false);

    // LDS barrier.
    mfmalb.create<miopen::WorkgroupBarrierOp>(loc);

    // Blockwise copy from global (generic tensor) to register (naive tensor).
    mfmalb.create<miopen::MovePosOp>(
        loc, blockwiseCopyASrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpAOddSecondIteration =
        mfmalb.create<miopen::BlockwiseCopyOp>(loc, op.filter(), threadAOddAllocOp,
                                           blockwiseCopyASrc, blockwiseCopyADst,
                                           /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpAOddSecondIteration, op, b,
                                 /*isMatrixA=*/true);
    mfmalb.create<miopen::MovePosOp>(
        loc, blockwiseCopyBSrc,
        ValueRange{KPerBlockConstantI32Op, zeroConstantI32Op});
    auto blockwiseCopyOpBOddSecondIteration =
        mfmalb.create<miopen::BlockwiseCopyOp>(loc, op.input(), threadBOddAllocOp,
                                           blockwiseCopyBSrc, blockwiseCopyBDst,
                                           /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyOpBOddSecondIteration, op, b,
                                 /*isMatrixA=*/false);

    // Emit blockwise V2 GEMM.
    auto blockwiseGemmV2OddOp = mfmalb.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
        mMyThreadOffsetA, mMyThreadOffsetB, blockwiseGemmV2EvenOp.getResults());
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2OddOp, op, b);

    // Blockwise copy from register (naive tensor) to LDS (naive tensor).
    auto blockwiseCopyAEvenSecondIteration = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, threadAOddAllocOp, lds2DMatrixAEvenSubviewOp, blockwiseCopyASrc,
        blockwiseCopyADst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyAEvenSecondIteration, op, b,
                                 /*isMatrixA=*/true);
    auto blockwiseCopyBEvenSecondIteration = mfmalb.create<miopen::BlockwiseCopyOp>(
        loc, threadBOddAllocOp, lds2DMatrixBEvenSubviewOp, blockwiseCopyBSrc,
        blockwiseCopyBDst, /*buffer=*/nullptr);
    affixBlockwiseCopyAttributes(blockwiseCopyBEvenSecondIteration, op, b,
                                 /*isMatrixA=*/false);

    mfmalb.create<scf::YieldOp>(loc, blockwiseGemmV2OddOp.getResults());
    // outside the loop.

    // Emit loop tail.

    // LDS barrier.
    b.create<miopen::WorkgroupBarrierOp>(loc);

    // Emit blockwise GEMM for the loop tail.
    auto blockwiseGemmV2TailEvenOp = b.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, lds2DMatrixAEvenSubviewOp, lds2DMatrixBEvenSubviewOp,
        mMyThreadOffsetA, mMyThreadOffsetB, mfmaLoopOp.getResults());
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2TailEvenOp, op, b);

    auto blockwiseGemmV2TailOddOp = b.create<miopen::BlockwiseGemmV2Op>(
        loc, vectorCTypes, lds2DMatrixAOddSubviewOp, lds2DMatrixBOddSubviewOp,
        mMyThreadOffsetA, mMyThreadOffsetB, blockwiseGemmV2TailEvenOp.getResults());
    affixBlockwiseGemmV2Attributes(blockwiseGemmV2TailOddOp, op, b);
    
    // Matrix C write out logic.

    // Original C++ logic.
    // __device__ static constexpr index_t GetNumBlksPerXdlops() {
    //     return (MPerXdlops * NPerXdlops) / (mfma_type.m * mfma_type.n);
    // }
    //
    // struct OutputLayout {
    //     __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_blk; }
    //     __device__ static constexpr index_t GetNumBlks() {
    //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
    //     }
    // };
    // using CThreadCopySliceLengths = Sequence<M0, 1, M2, 1>;
    // constexpr index_t BlkSize = blockwise_gemm.GetBlkSize();
    // constexpr index_t NumBlks = blockwise_gemm.GetNumBlks();

    int64_t BlkSize = num_regs_blk;
    int64_t NumBlksPerXdlops = (MPerXdlops * NPerXdlops) / (m * n);
    int64_t NumBlks = NumBlksPerXdlops * MRepeats * NRepeats;

    int64_t iterationsPerVectorC = NumBlks / vectorNumber;
    int64_t vectorCoffset = vectorType.getShape()[0] / BlkSize;

    // llvm::errs() << "MPerWave: " << MPerWave << "\n";
    // llvm::errs() << "NPerWave: " << NPerWave << "\n\n";

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

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto MRepeatsConstantOp = b.create<ConstantIndexOp>(loc, MRepeats);
    auto NRepeatsConstantOp = b.create<ConstantIndexOp>(loc, NRepeats);

    auto group_size_ConstantOp = b.create<ConstantIndexOp>(loc, group_size);
    auto wave_size_ConstantOp = b.create<ConstantIndexOp>(loc, wave_size);
    auto num_threads_blk_ConstantOp = b.create<ConstantIndexOp>(loc, num_threads_blk);
    auto num_output_blks_ConstantOp = b.create<ConstantIndexOp>(loc, num_output_blks);
    auto m_ConstantOp = b.create<ConstantIndexOp>(loc, m);
    auto n_ConstantOp = b.create<ConstantIndexOp>(loc, n);
 
    auto BlkSizeConstantI32Op = b.create<ConstantIntOp>(loc, BlkSize, b.getIntegerType(32));
    auto NumBlksPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NumBlksPerXdlops);
    auto NumBlksConstantOp = b.create<ConstantIndexOp>(loc, NumBlks);

    auto iterationsPerVectorCConstantOp = b.create<ConstantIndexOp>(loc, iterationsPerVectorC);
    auto vectorCoffsetConstantOp = b.create<ConstantIndexOp>(loc, vectorCoffset);
 
    // Threadwise copy from register (naive tensor) to global (generic tensor).
    // Original C++ logic:
    //
    // struct OutputLayout {
    //     __device__ static constexpr index_t M1() { return mfma_type.num_groups_blk; }
    //     __device__ static constexpr index_t M0() { return mfma_type.group_size; }
    //     __device__ static constexpr index_t N1() { return mfma_type.num_input_blks; }
    //     __device__ static constexpr index_t N0() { return mfma_type.num_threads_blk; }
    //     __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_blk; }
    //     __device__ static constexpr index_t GetNumBlks() {
    //         return GetNumBlksPerXdlops() * MRepeats * NRepeats;
    //     }
    // };
    //
    // // CLayout.M1() = num_groups;
    // // CLayout.M0() = group_size;
    // // CLayout.N1() = num_blks_per_wave;
    // // CLayout.N0() = num_threads_per_blks;
    // constexpr auto CLayout = blockwise_gemm.GetOutputLayout();
    // constexpr index_t M0   = CLayout.M1();
    // constexpr index_t M1   = CLayout.N1();
    // constexpr index_t M2   = CLayout.M0();

    int64_t M3 = num_groups_blk;
    int64_t M1 = num_input_blks;
    int64_t M2 = group_size;
    int64_t M0 = M / (M1 * M2);

    // llvm::errs() << "M0: " << M0 << "\n";
    // llvm::errs() << "M1: num_input_blks: " << M1 << "\n";
    // llvm::errs() << "M2: group_size: " << M2 << "\n";
    // llvm::errs() << "M3: num_groups_blk: " << M3 << "\n\n";

    auto M0ConstantI32Op =
        b.create<ConstantIntOp>(loc, M0, b.getIntegerType(32));
    auto M1ConstantI32Op =
        b.create<ConstantIntOp>(loc, M1, b.getIntegerType(32));
    auto M2ConstantI32Op =
        b.create<ConstantIntOp>(loc, M2, b.getIntegerType(32));
    auto M3ConstantI32Op =
        b.create<ConstantIntOp>(loc, M3, b.getIntegerType(32));
    auto NConstantI32Op =
        b.create<ConstantIntOp>(loc, N, b.getIntegerType(32));

    // constexpr auto c_m0_m1_m2_n_global_desc = transform_tensor_descriptor(
    //     c_m_n_global_desc,
    //     make_tuple(UnMerge<Sequence<M0, M1, M2>>{}, PassThrough<N>{}),
    //     make_tuple(Sequence<0>{}, Sequence<1>{}),
    //     make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}));
 
    // build affine expression:
    // (d0, d1, d2, d3) -> (d0 * M1 * M2 + d1 * M2 + d2, d3)
    auto affineMap4to2 =
        AffineMap::get(4, 0,
                       {getAffineDimExpr(0, op.getContext()) *
                            getAffineConstantExpr(M1, op.getContext()) *
                            getAffineConstantExpr(M2, op.getContext()) +
                        getAffineDimExpr(1, op.getContext()) *
                            getAffineConstantExpr(M2, op.getContext()) +
                        getAffineDimExpr(2, op.getContext()),
                        getAffineDimExpr(3, op.getContext())},
                       op.getContext());

    // compose with output tensor affine map.
    auto outputType = op.output().getType().template dyn_cast<MemRefType>();
    auto outputAffineMap2to4 = outputType.getAffineMaps()[0];
    auto affineMap4to2to4 = outputAffineMap2to4.compose(affineMap4to2);

    // emit TransformOp for output tensor.
    auto newOutputType = MemRefType::get(
        {M0, M1, M2, N}, outputType.getElementType(), {affineMap4to2to4});
    auto newOutputTransformOp =
        b.create<miopen::TransformOp>(loc, newOutputType, op.output());

    // Original C++ logic.
    // //     src descriptor
    // constexpr auto c_m0_m1_m2_n_thread_desc =
    //     make_native_tensor_descriptor_packed(Sequence<M0, 1, M2, 1>{});

    // Build affine expression for Sequence<M0, 1, M2, 1>
    // (d0, d1, d2, d3) -> (d0 * M2 + d2)
    auto matrixCAffineMap4to1 = AffineMap::get(
        4, 0,
        {getAffineDimExpr(0, op.getContext()) * getAffineConstantExpr(M2, op.getContext()) +
         getAffineDimExpr(2, op.getContext())},
        op.getContext());

    // Original C++ logic.
    // for(index_t i = 0; i < NumBlks; ++i)
    // {
    //     // calculate origin of thread output tensor on global memory
    //     //     blockwise GEMM c matrix starting index
    //     const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);
    //     const index_t m_thread_data_on_global =
    //         m_block_data_on_global + c_thread_mtx_on_block.row;
    //     const index_t n_thread_data_on_global =
    //         n_block_data_on_global + c_thread_mtx_on_block.col;
    //     ThreadwiseGenericTensorSliceCopy_v4r2<decltype(c_m0_m1_m2_n_thread_desc),
    //                                           decltype(c_m0_m1_m2_n_global_desc),
    //                                           CThreadCopySliceLengths,
    //                                           arithmetic_sequence_gen<0, 4, 1>::type,
    //                                           3,
    //                                           1,
    //                                           1,
    //                                           AddressSpace::Vgpr,
    //                                           AddressSpace::Global,
    //                                           CGlobalMemoryDataOperation>(
    //         {0, 0, 0, 0},
    //         {m_thread_data_on_global / (M2 * M1),
    //          m_thread_data_on_global % (M2 * M1) / M2,
    //          m_thread_data_on_global % M2,
    //          n_thread_data_on_global})
    //         .Run(c_thread_vec.n + i * BlkSize, p_c_global);
    // }

    // emit unrolled loop.
    for (int64_t iter = 0; iter < NumBlks; ++iter) {
      auto iv = b.create<ConstantIndexOp>(loc, iter);
      auto iv_i32 = b.create<IndexCastOp>(loc, iv, b.getIntegerType(32));

      // Original C++ logic.
      //
      // In xdlops_gemm.hpp:
      //
      // static constexpr bool IsABroadcast() { return NPerXdlops >= MPerXdlops; }
      // __device__ static MatrixIndex GetBeginOfThreadBlk(index_t i) {
      //     const index_t xdlops_i = i / GetNumBlksPerXdlops();
      //     const index_t j        = i % GetNumBlksPerXdlops();
      //     const index_t m_i = xdlops_i / NRepeats;
      //     const index_t n_i = xdlops_i % NRepeats;
      //     const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;
      //     index_t col_blk = j % mfma_type.num_output_blks;
      //     index_t row_blk = j / mfma_type.num_output_blks;
      //     static_if<!IsABroadcast>{}([&](auto) {
      //         col_blk = j / mfma_type.num_output_blks;
      //         row_blk = j % mfma_type.num_output_blks;
      //     });
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;
      //     return MatrixIndex{row, col};
      // }
      //
      // In blockwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // __device__ static MatrixIndex GetBeginOfThreadMatrixC(index_t i) {
      //     const index_t waveId = get_thread_local_1d_id() / WaveSize;
      //     const index_t xdlops_i = i / XdlopsGemm.GetOutputLayout().GetNumBlks();
      //     const index_t j        = i % XdlopsGemm.GetOutputLayout().GetNumBlks();
      //     const index_t m = xdlops_i / NRepeats;
      //     const index_t n = xdlops_i % NRepeats;
      //     const auto thread_mtx_on_blk = XdlopsGemm.GetBeginOfThreadBlk(j);
      //     const index_t col =
      //         (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops + thread_mtx_on_blk.col;
      //     const index_t row =
      //         (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops + thread_mtx_on_blk.row;
      //     return MatrixIndex{row, col};
      // }
      //
      // In gridwise_gemm_xdlops.hpp:
      //
      // Original C++ logic:
      // const auto c_thread_mtx_on_block = blockwise_gemm.GetBeginOfThreadMatrixC(i);
      // const index_t m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      // compute thread_mtx_on_blk_row and thread_mtx_on_blk_col.
      auto xdlops_i = b.create<SignedDivIOp>(loc, iv, NumBlksPerXdlopsConstantOp);
      auto j = b.create<SignedRemIOp>(loc, iv, NumBlksPerXdlopsConstantOp);
      auto m_i = b.create<SignedDivIOp>(loc, xdlops_i, NRepeatsConstantOp);
      auto n_i = b.create<SignedRemIOp>(loc, xdlops_i, NRepeatsConstantOp);

      auto laneId = b.create<SignedRemIOp>(loc, tid, wave_size_ConstantOp);
      auto blk_id = b.create<SignedDivIOp>(loc, laneId, num_threads_blk_ConstantOp);
      auto blk_td = b.create<SignedRemIOp>(loc, laneId, num_threads_blk_ConstantOp);
      Value col_blk, row_blk;
      if (NPerXdlops >= MPerXdlops) {
        // IsABroadcast
        col_blk = b.create<SignedRemIOp>(loc, j, num_output_blks_ConstantOp);
        row_blk = b.create<SignedDivIOp>(loc, j, num_output_blks_ConstantOp);
      } else {
        // !IsABroadcast
        col_blk = b.create<SignedDivIOp>(loc, j, num_output_blks_ConstantOp);
        row_blk = b.create<SignedRemIOp>(loc, j, num_output_blks_ConstantOp);
      } 

      // Original C++ logic.
      //     index_t col = col_blk * mfma_type.n + blk_td + n_i * NPerXdlops;
      auto thread_mtx_on_blk_col = b.create<AddIOp>(loc,
        b.create<AddIOp>(loc,                     
          b.create<MulIOp>(loc, col_blk, n_ConstantOp),
          blk_td),
        b.create<MulIOp>(loc, n_i, NPerXdlopsConstantOp));
      // Original C++ logic.
      //     index_t row = row_blk * mfma_type.m + blk_id * mfma_type.group_size + m_i * MPerXdlops;
      auto thread_mtx_on_blk_row = b.create<AddIOp>(loc,
        b.create<AddIOp>(loc,
          b.create<MulIOp>(loc, row_blk, m_ConstantOp),
          b.create<MulIOp>(loc, blk_id, group_size_ConstantOp)),
        b.create<MulIOp>(loc, m_i, MPerXdlopsConstantOp));

      // compute c_thread_mtx_index_row, c_thread_mtx_index_col.
      // compute c_thread_mtx_index_row_i32, c_thread_mtx_index_col_i32.

      // Original C++ logic.
      // const index_t col = (waveId % GemmNWaves) * GemmNPerWave + n * NPerXdlops + thread_mtx_on_blk.col;
      c_thread_mtx_index_col = b.create<AddIOp>(loc,
        b.create<AddIOp>(loc,
          b.create<MulIOp>(loc,
            b.create<SignedRemIOp>(loc, waveId, NWavesConstantOp),
            NPerWaveConstantOp),
          b.create<MulIOp>(loc,
            n_ConstantOp, NPerXdlopsConstantOp)),
        thread_mtx_on_blk_col);
      c_thread_mtx_index_col_i32 = b.create<IndexCastOp>(loc, c_thread_mtx_index_col, b.getIntegerType(32));

      // Original C++ logic.
      // const index_t row = (waveId / GemmNWaves) * GemmMPerWave + m * MPerXdlops + thread_mtx_on_blk.row;
      c_thread_mtx_index_row = b.create<AddIOp>(loc,
        b.create<AddIOp>(loc,
          b.create<MulIOp>(loc,
            b.create<SignedDivIOp>(loc, waveId, NWavesConstantOp),
            MPerWaveConstantOp),
          b.create<MulIOp>(loc,
            m_ConstantOp, MPerXdlopsConstantOp)),
        thread_mtx_on_blk_row); 
      c_thread_mtx_index_row_i32 = b.create<IndexCastOp>(loc, c_thread_mtx_index_row, b.getIntegerType(32));

      // Original C++ logic:
      // const index_t m_thread_data_on_global =
      //     m_block_data_on_global + c_thread_mtx_on_block.row;
      // const index_t n_thread_data_on_global =
      //     n_block_data_on_global + c_thread_mtx_on_block.col;

      m_thread_data_on_global_i32 = b.create<AddIOp>(
          loc, m_block_data_on_global_i32, c_thread_mtx_index_row_i32);
      n_thread_data_on_global_i32 = b.create<AddIOp>(
          loc, n_block_data_on_global_i32, c_thread_mtx_index_col_i32);
 
      SmallVector<Value, 8> matrixCThreadwiseCopySourceAndDestCoords;
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);
      matrixCThreadwiseCopySourceAndDestCoords.push_back(zeroConstantI32Op);

      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedDivIOp>(
          loc, m_thread_data_on_global_i32, b.create<MulIOp>(loc, M2ConstantI32Op, M1ConstantI32Op)));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedDivIOp>(
          loc, b.create<SignedRemIOp>(loc, m_thread_data_on_global_i32, b.create<MulIOp>(loc, M2ConstantI32Op, M1ConstantI32Op)),
               M2ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(b.create<SignedRemIOp>(
          loc, m_thread_data_on_global_i32, M2ConstantI32Op));
      matrixCThreadwiseCopySourceAndDestCoords.push_back(n_thread_data_on_global_i32);

      // Select which vector C to use, and offset.
      int64_t vectorCIndex = iter / iterationsPerVectorC;
      int64_t vectorCOffset = vectorCoffset * (iter % iterationsPerVectorC);
      auto vectorCOffsetConstantOp = b.create<ConstantIntOp>(loc, vectorCOffset, b.getIntegerType(32));

      auto threadwiseCopyV2CMatrixOp = b.create<miopen::ThreadwiseCopyV2Op>(
          loc, blockwiseGemmV2TailOddOp.getResults()[vectorCIndex], newOutputTransformOp,
          vectorCOffsetConstantOp,
          matrixCThreadwiseCopySourceAndDestCoords);
      affixThreadwiseCopyV2Attributes(threadwiseCopyV2CMatrixOp, op, b);

      // affix coord_transforms attributes.
      threadwiseCopyV2CMatrixOp.setAttr("coord_transforms",
                                    b.getArrayAttr({
                                      b.getDictionaryAttr({
                                        b.getNamedAttr("operand", b.getI32IntegerAttr(0)),
                                        b.getNamedAttr("transforms", b.getAffineMapArrayAttr(matrixCAffineMap4to1))
                                      })
                                    }));
 
      // affix bound attributes.
      threadwiseCopyV2CMatrixOp.setAttr("bound",
                                    b.getArrayAttr({
                                     b.getI32IntegerAttr(M3),
                                     b.getI32IntegerAttr(1),
                                     b.getI32IntegerAttr(M2),
                                     b.getI32IntegerAttr(1),
                                    }));
    }

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
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);

    auto blockAType = op.matrixA().getType().cast<MemRefType>();
    auto blockBType = op.matrixA().getType().cast<MemRefType>();

    auto elementType = op.matrixC().getType().cast<MemRefType>().getElementType();

    // Obtain critical matrix dimensions.
    int64_t K = blockAType.getShape()[0];
    int64_t M = blockAType.getShape()[1];
    int64_t N = blockBType.getShape()[1];

    // xdlops.
    auto xdlopsAttr = op.template getAttrOfType<BoolAttr>("xdlops");
    if (xdlopsAttr && xdlopsAttr.getValue() == true) {
      // Original C++ logic:
      // static constexpr auto XdlopsGemm =
      //     XdlopsGemm_t<Float, MPerXdlops, NPerXdlops, GemmDataPerReadA, GemmDataPerReadB>{};
      // XdlopsGemm.template Run<M, N, K>(&p_a_block[mMyWaveOffsetA],
      //                                  &p_b_block[mMyWaveOffsetB],
      //                                  p_c_thread);

      auto xdlopsGemm = b.create<miopen::XdlopsGemmOp>(loc, op.matrixA(), op.matrixB(), op.matrixC(), op.threadOffsetA(), op.threadOffsetB());
      xdlopsGemm.setAttr("m", b.getI32IntegerAttr(M));
      xdlopsGemm.setAttr("n", b.getI32IntegerAttr(N));
      xdlopsGemm.setAttr("k", b.getI32IntegerAttr(K));
      xdlopsGemm.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemm.setAttr("n_per_wave", op.getAttr("n_per_wave"));
    } else {
      // Non-xdlops path.
 
      // Obtain critical attributes.
      int64_t KPerThread = op.getAttr("k_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t MPerThread =
          op.matrixC().getType().template dyn_cast<MemRefType>().getShape()[0];
      int64_t NPerThread =
          op.matrixC().getType().template dyn_cast<MemRefType>().getShape()[1];
      int64_t MPerThreadSubC = op.getAttr("m_per_thread").template dyn_cast<IntegerAttr>().getInt();
      int64_t NPerThreadSubC = op.getAttr("n_per_thread").template dyn_cast<IntegerAttr>().getInt();

      // llvm::errs() << "MPerThread: " << MPerThread << "\n";
      // llvm::errs() << "MPerThreadSubC: " << MPerThreadSubC << "\n";
      // llvm::errs() << "NPerThread: " << NPerThread << "\n";
      // llvm::errs() << "NPerThreadSubC: " << NPerThreadSubC << "\n";

      auto MPerThreadSubCConstantI32Op =
          b.create<ConstantIntOp>(loc, MPerThreadSubC, b.getIntegerType(32));
      auto NPerThreadSubCConstantI32Op =
          b.create<ConstantIntOp>(loc, NPerThreadSubC, b.getIntegerType(32));

      int64_t MLevel0Cluster = op.getAttr("m_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
      int64_t MLevel1Cluster = op.getAttr("m_level1_cluster").template dyn_cast<IntegerAttr>().getInt();
      int64_t NLevel0Cluster = op.getAttr("n_level0_cluster").template dyn_cast<IntegerAttr>().getInt();
      int64_t NLevel1Cluster = op.getAttr("n_level1_cluster").template dyn_cast<IntegerAttr>().getInt();

      int64_t MPerLevel1Cluster = MPerThreadSubC * MLevel0Cluster * MLevel1Cluster;
      int64_t NPerLevel1Cluster = NPerThreadSubC * NLevel0Cluster * NLevel1Cluster;
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
          MemRefType::get({KPerThread, MPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
      auto threadAAllocOp =
          b.create<miopen::GpuAllocOp>(loc, threadARegisterMemRefType);

      auto threadBRegisterMemRefType =
          MemRefType::get({KPerThread, NPerThread}, elementType, {},
                          gpu::GPUDialect::getPrivateAddressSpace());
      auto threadBAllocOp =
          b.create<miopen::GpuAllocOp>(loc, threadBRegisterMemRefType);

      // Main loop.
      auto loopIteration = K / KPerThread;
      auto loopIterationConstantOp =
          b.create<ConstantIndexOp>(loc, loopIteration);
      auto loopOp =
          b.create<scf::ForOp>(loc, zeroConstantOp,
                               loopIterationConstantOp, oneConstantOp);

      // inside the main loop.
      auto lb = OpBuilder::atBlockTerminator(loopOp.getBody());

      auto iv = loopOp.getInductionVar();
      auto iv_i32 = lb.create<IndexCastOp>(loc, iv, lb.getIntegerType(32));

      // read matrix A loop.
      auto loopReadMatrixAIteration = MRepeat;
      auto loopReadMatrixAIterationConstantOp =
          lb.create<ConstantIndexOp>(loc, loopReadMatrixAIteration);
      auto loopReadMatrixAOp = lb.create<scf::ForOp>(
          loc, zeroConstantOp, loopReadMatrixAIterationConstantOp,
          oneConstantOp);

      // inside read matrix A loop.
      auto lab = OpBuilder::atBlockTerminator(loopReadMatrixAOp.getBody());

      auto iva = loopReadMatrixAOp.getInductionVar();
      auto iva_i32 = lab.create<IndexCastOp>(loc, iva, lab.getIntegerType(32));

      // Threadwise copy from LDS (naive tensor) to register (generic tensor).

      // Set copy sorce and dest coordinate acoording to original C++ logic:
      SmallVector<Value, 4> matrixAThreadwiseCopySourceAndDestCoords;
      // a_thread_copy.Run(
      //   p_a_block + a_block_mtx.CalculateOffset(k_begin, m_repeat *  MPerLevel1Cluster) + mMyThreadOffsetA),
      // mMyThreadOffsetA = BlockMatrixA::GetOffsetFromMultiIndex{0, c_thread_mtx_index.row} = c_thread_mtx_index_row
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
      auto loopReadMatrixBIterationConstantOp =
          lb.create<ConstantIndexOp>(loc, loopReadMatrixBIteration);
      auto loopReadMatrixBOp = lb.create<scf::ForOp>(
          loc, zeroConstantOp, loopReadMatrixBIterationConstantOp,
          oneConstantOp);

      // inside read matrix B loop.
      auto lbb = OpBuilder::atBlockTerminator(loopReadMatrixBOp.getBody());

      auto ivb = loopReadMatrixBOp.getInductionVar();
      auto ivb_i32 = lbb.create<IndexCastOp>(loc, ivb, lbb.getIntegerType(32));

      // Threadwise copy from LDS (naive tensor) to register (generic tensor).

      // Set copy sorce and dest coordinate acoording to original C++ logic:
      SmallVector<Value, 4> matrixBThreadwiseCopySourceAndDestCoords;
      // b_thread_copy.Run(
      //   p_b_block + b_block_mtx.CalculateOffset(k_begin, n_repeat * NPerLevel1Cluster) + mMyThreadOffsetB),
      // mMyThreadOffsetB = BlockMatrixB::GetOffsetFromMultiIndex{0, c_thread_mtx_index.col} = c_thread_mtx_index_col
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
    }

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
    //     non-XDLOPS: switch between 2x2 and naive pipeline.
    //     XDLOPS: always use naive pipeline.
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
    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);

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
      //                       DstMatrix::RowStride() % DataPerAccess == 0,
      //                       "wrong! wrong alignment");
      //         static_assert(NSliceCol % DataPerAccess == 0,
      //                       "wrong! should be NSliceCol % DataPerAccess ==
      //                       0");
      //     }
      //
      //     template <typename Data>
      //     __device__ static void Run(const Data* p_src, Data* p_dst)
      //     {
      //         using vector_t = typename vector_type<Data, DataPerAccess>::MemoryType;
      //
      //         for(index_t i = 0; i < NSliceRow; ++i)
      //         {
      //             for(index_t j = 0; j < NSliceCol; j += DataPerAccess)
      //             {
      //                 const index_t src_index = SrcMatrix::CalculateOffset(i, j);
      //                 const index_t dst_index = DstMatrix::CalculateOffset(i, j);
      //
      //                 *reinterpret_cast<vector_t*>(&p_dst[dst_index]) =
      //                     *reinterpret_cast<const vector_t*>(&p_src[src_index]);
      //             }
      //         }
      //     }
      // };
      auto NSliceRowConstantOp = b.create<ConstantIndexOp>(loc, NSliceRow);
      auto NSliceColConstantOp = b.create<ConstantIndexOp>(loc, NSliceCol);
      auto DataPerAccessConstantOp =
          b.create<ConstantIndexOp>(loc, DataPerAccess);

      // outer loop.
      auto outerLoopOp =
          b.create<scf::ForOp>(loc, zeroConstantOp,
                               NSliceRowConstantOp, oneConstantOp);

      // inside the outer loop.
      auto lob = OpBuilder::atBlockTerminator(outerLoopOp.getBody());
      auto ivo = outerLoopOp.getInductionVar();
      auto ivo_i32 = lob.create<IndexCastOp>(loc, ivo, b.getIntegerType(32));

      // inner loop
      auto innerLoopOp = lob.create<scf::ForOp>(loc, zeroConstantOp,
                                                NSliceColConstantOp,
                                                DataPerAccessConstantOp);

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

      Value vectorValue;
      Value scalarValue;
      // Load from source.
      if (DataPerAccess > 1) {
        // Issue vector load.
        auto vectorType =
            VectorType::get(DataPerAccess, sourceType.getElementType());
        auto srcExpr =
            getAffineDimExpr(sourceType.getRank() - 1, op.getContext());
        auto srcProjection = AffineMap::get(sourceType.getRank(), 0, srcExpr);
        vectorValue = lib.create<vector::TransferReadOp>(
            loc, vectorType, op.source(), srcLowerIndices, srcProjection);
      } else {
        // Issue scalar load.
        scalarValue = lib.create<LoadOp>(loc, sourceType.getElementType(), op.source(), srcLowerIndices);
      }

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
      if (DataPerAccess > 1) {
        auto dstExpr = getAffineDimExpr(destType.getRank() - 1, op.getContext());
        auto dstProjection = AffineMap::get(destType.getRank(), 0, dstExpr);
        lib.create<vector::TransferWriteOp>(loc, vectorValue, op.dest(),
                                            destLowerIndices, dstProjection);
      } else {
        // Issue scalar store.
        lib.create<StoreOp>(loc, scalarValue, op.dest(), destLowerIndices);
      }

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
          // Use domain attribute from source memref.
          for (auto attr : coordTransformsAttr) {
            auto dictAttr = attr.template cast<DictionaryAttr>();
            auto operandIndex =
                dictAttr.get("operand").template cast<IntegerAttr>().getInt();
            if (operandIndex == 0) {
              // bound attribute take precendence over domain attribute.
              if (op.getAttr("bound")) {
                auto boundAttr =
                    op.getAttr("bound").template cast<ArrayAttr>();
                for (unsigned i = 0; i < boundAttr.size(); ++i)
                  sliceLengths.push_back(
                      boundAttr[i].template cast<IntegerAttr>().getInt());
              } else {
                auto domainAttr =
                    dictAttr.get("domain").template cast<ArrayAttr>();
                for (unsigned i = 0; i < domainAttr.size(); ++i)
                  sliceLengths.push_back(
                      domainAttr[i].template cast<IntegerAttr>().getInt());
              }
            }
          }
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
            loc, zeroConstantOp, loopBounds[dim], oneConstantOp);
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
      Value vectorValue;
      Value scalarValue;
      if (srcDataPerRead > 1) {
        // Issue vector load.
        auto sourceVectorType =
            VectorType::get(srcDataPerRead, sourceType.getElementType());
        auto srcExpr =
            getAffineDimExpr(sourceType.getRank() - 1, op.getContext());
        auto srcProjection = AffineMap::get(sourceType.getRank(), 0, srcExpr);
        vectorValue = innerLoopBuilder.create<vector::TransferReadOp>(
            loc, sourceVectorType, op.source(), srcLowerIndices, srcProjection);
      } else {
        // Issue scalar load.
        scalarValue = innerLoopBuilder.create<LoadOp>(loc, sourceType.getElementType(), op.source(), srcLowerIndices);
      }

      // Compute high-level coordinate for dest memref.
      // dst_index = (iv_0, iv_1, ...) + destCoord
      SmallVector<Value, 8> destUpperIndices;
      for (unsigned iter = 0; iter < loopIV_i32s.size(); ++iter)
        destUpperIndices.push_back(innerLoopBuilder.create<IndexCastOp>(
            loc,
            innerLoopBuilder.create<AddIOp>(
                loc,
                loopIV_i32s[dimAccessOrder[iter]
                                .template cast<IntegerAttr>()
                                .getInt()],
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
      if (destDataPerWrite > 1) {
        // Issue vector store.
        auto dstExpr = getAffineDimExpr(destType.getRank() - 1, op.getContext());
        auto dstProjection = AffineMap::get(destType.getRank(), 0, dstExpr);
        innerLoopBuilder.create<vector::TransferWriteOp>(
            loc, vectorValue, op.dest(), destLowerIndices, dstProjection);
      } else {
        // Issue scalar store.
        innerLoopBuilder.create<StoreOp>(loc, scalarValue, op.dest(), destLowerIndices);
      }
    }

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

    auto zeroConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(0.0f), b.getF32Type());
    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());
    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);
    auto zeroConstantI32Op =
        b.create<ConstantIntOp>(loc, 1, b.getIntegerType(32));
    auto oneConstantI32Op =
        b.create<ConstantIntOp>(loc, 1, b.getIntegerType(32));

    auto sourceType = op.source().getType().cast<VectorType>();
    auto destType = op.dest().getType().cast<MemRefType>();

    // Get source offset, and dest coordinates.
    //
    // 1. For memrefs with no externally defined affine maps in coord_transforms
    //    attribute, or embedded affine maps. Use its rank.
    // 2. For memrefs with externally defined maps, use its input rank.
    // 3. For memrefs with embedded maps, use its input rank.
    auto sourceAndDestCoord = op.sourceAndDestCoord();
    auto destTypeAffineMaps = destType.getAffineMaps();
    auto coordTransformsAttr = op.getAttr("coord_transforms");

    unsigned sourceCoordLength = sourceType.getRank();
    unsigned destCoordLength = destType.getRank();

    bool sourceEmbeddedTransform = false;
    bool destEmbeddedTransform = false;
    bool sourceExternalTransform = false;
    bool destExternalTransform = false;
    AffineMap sourceTransform;
    AffineMap destTransform;

    if (destTypeAffineMaps.size()) {
      // Use the first affine map in the attribute array.
      destCoordLength = destTypeAffineMaps[0].getNumInputs();
      destEmbeddedTransform = true;
      destTransform = destTypeAffineMaps[0];
    }
    if (coordTransformsAttr) {
      for (auto attr : coordTransformsAttr.template cast<ArrayAttr>()) {
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

    // Refer to ThreadwiseGenericTensorSliceCopy_v4r2::Run() for the original
    // C++ implementation.

    // llvm::errs() << "\nthreadwise_copy_v2 op:\n";
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

    // Figure out slice lengths.
    SmallVector<int64_t, 2> sliceLengths;

    if (sourceExternalTransform || sourceEmbeddedTransform) {
      // Use bound or domain attribute from source vector.
      for (auto attr : coordTransformsAttr.template cast<ArrayAttr>()) {
        auto dictAttr = attr.template cast<DictionaryAttr>();
        auto operandIndex =
            dictAttr.get("operand").template cast<IntegerAttr>().getInt();
        if (operandIndex == 0) {
          // bound attribute take precendence over domain attribute.
          if (op.getAttr("bound")) {
            auto boundAttr =
                op.getAttr("bound").template cast<ArrayAttr>();
            for (unsigned i = 0; i < boundAttr.size(); ++i)
              sliceLengths.push_back(
                  boundAttr[i].template cast<IntegerAttr>().getInt());
          } else {
            auto domainAttr =
                dictAttr.get("domain").template cast<ArrayAttr>();
            for (unsigned i = 0; i < domainAttr.size(); ++i)
              sliceLengths.push_back(
                  domainAttr[i].template cast<IntegerAttr>().getInt());
          }
        }
      }
    } else {
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
          loc, zeroConstantOp, loopBounds[dim], oneConstantOp);
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

    // Add sourceOffset to derive the position in the vector.
    auto srcPosition = innerLoopBuilder.create<AddIOp>(loc,
                          innerLoopBuilder.create<IndexCastOp>(loc, srcLowerIndices[0], b.getIntegerType(32)),
                          op.sourceOffset());

    // Load from source.
    // TBD. Issue vector load.
    // Value vectorValue;
    Value scalarValue;
    if (srcDataPerRead > 1) {
      // TBD. Issue vector load.
      // auto sourceVectorType =
      //     VectorType::get(srcDataPerRead, sourceType.getElementType());
      // auto srcExpr =
      //     getAffineDimExpr(sourceType.getRank() - 1, op.getContext());
      // auto srcProjection = AffineMap::get(sourceType.getRank(), 0, srcExpr);
      // vectorValue = innerLoopBuilder.create<vector::TransferReadOp>(
      //     loc, sourceVectorType, op.source(), srcLowerIndices, srcProjection);
    } else {
      // Issue scalar load.
      scalarValue = innerLoopBuilder.create<vector::ExtractElementOp>(loc, sourceType.getElementType(), op.source(), srcPosition);
    }
    
    // Compute high-level coordinate for dest memref.
    // dst_index = (iv_0, iv_1, ...) + destCoord
    SmallVector<Value, 8> destUpperIndices;
    for (unsigned iter = 0; iter < loopIV_i32s.size(); ++iter)
      destUpperIndices.push_back(innerLoopBuilder.create<IndexCastOp>(
          loc,
          innerLoopBuilder.create<AddIOp>(
              loc,
              loopIV_i32s[dimAccessOrder[iter]
                              .template cast<IntegerAttr>()
                              .getInt()],
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
    if (destDataPerWrite > 1) {
      // TBD. Issue vector store.
      // auto dstExpr = getAffineDimExpr(destType.getRank() - 1, op.getContext());
      // auto dstProjection = AffineMap::get(destType.getRank(), 0, dstExpr);
      // innerLoopBuilder.create<vector::TransferWriteOp>(
      //     loc, vectorValue, op.dest(), destLowerIndices, dstProjection);
    } else {
      // Issue scalar store.
      innerLoopBuilder.create<StoreOp>(loc, scalarValue, op.dest(), destLowerIndices);
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
      if (!coordTransformAttrs) {
        user->setAttr("coord_transforms",
                      b.getArrayAttr({
                        b.getDictionaryAttr({
                          b.getNamedAttr("operand", b.getI32IntegerAttr(userOperandIndex)),
                          b.getNamedAttr("transforms", b.getAffineMapArrayAttr(outputType.getAffineMaps()))
                        })
                      }));
      } else {
        // XXX. Only do this for miopen.xdlops_gemm operation.
        //      and miopen.xdlops_gemm_v2 operation.
        // miopen.threadwise_copy will NOT be affected.
        if ((user->getName().getStringRef() == miopen::XdlopsGemmOp::getOperationName()) ||
            (user->getName().getStringRef() == miopen::XdlopsGemmV2Op::getOperationName())) {

          // create a deep-copy of existing attributes, and amend the new one.
          // need to figure out if there's a better way than this.
          auto arrayAttr = coordTransformAttrs.cast<ArrayAttr>();
          llvm::SmallVector<Attribute, 2> augmentedArrayAttr;

          //llvm::errs() << "\nexisting transforms:\n";
          //coordTransformAttrs.dump();
          //llvm::errs() << "\ntransform to be added:\n";
          //llvm::errs() << "operand: " << userOperandIndex << "\n";
          //if (outputType.getAffineMaps().size() > 0) {
          //  llvm::errs() << "transforms: " << outputType.getAffineMaps()[0] << "\n";
          //}

          bool augmented = false;
          for (unsigned idx = 0; idx < arrayAttr.size(); ++idx) {
            auto dictAttr = arrayAttr.getValue()[idx].cast<DictionaryAttr>();
            auto operandIndex =
                dictAttr.get("operand").cast<IntegerAttr>().getInt();

            if (operandIndex != userOperandIndex) {
              augmentedArrayAttr.push_back(dictAttr);
            } else {
              //auto existingTransforms =
              //    dictAttr.get("transforms").cast<ArrayAttr>();
              llvm::SmallVector<Attribute, 4> augmentedTransforms;
              //augmentedTransforms.append(existingTransforms.begin(),
              //                           existingTransforms.end());
              if (outputType.getAffineMaps().size() > 0)
                augmentedTransforms.push_back(
                    AffineMapAttr::get(outputType.getAffineMaps()[0]));

              augmentedArrayAttr.push_back(b.getDictionaryAttr(
                  {b.getNamedAttr("operand",
                                  b.getI32IntegerAttr(userOperandIndex)),
                   b.getNamedAttr("transforms",
                                  b.getArrayAttr(augmentedTransforms))}));
              augmented = true;
            }
          }
          if (!augmented)
            augmentedArrayAttr.push_back(b.getDictionaryAttr(
                {b.getNamedAttr("operand",
                                b.getI32IntegerAttr(userOperandIndex)),
                 b.getNamedAttr("transforms", b.getAffineMapArrayAttr(
                                                  outputType.getAffineMaps()))}));

          //llvm::errs() << "\naugmented transforms:\n";
          //b.getArrayAttr(augmentedArrayAttr).dump();
          user->setAttr("coord_transforms", b.getArrayAttr(augmentedArrayAttr));
        }
      }
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
    auto outputShape = outputType.getShape();

    // determine output shape and track it in an attribute.
    llvm::SmallVector<Attribute, 4> shapeAttrVec;
    for (unsigned i = 0; i < outputShape.size(); ++i) {
      shapeAttrVec.push_back(b.getI32IntegerAttr(outputShape[i]));
    }

    // auto attr = b.getNamedAttr("domain",
    //               b.getArrayAttr(shapeAttrVec));
    // llvm::errs() << "\n\ndomain attr:\n";
    // attr.second.dump();
    // llvm::errs() << "\n";
    // llvm::errs() << "\n========\nTransformOp:\n";
    // op.dump();

    // Pass the output affine map to users of this op.
    if (outputType.getAffineMaps().size() > 0)
      for (auto user : op.output().getUsers()) {
        // llvm::errs() << "\n========\nTransformOp user:\n";
        // user->dump();

        // determine user domain

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
                   b.getNamedAttr(
                       "transforms",
                       b.getAffineMapArrayAttr(outputType.getAffineMaps())),
                   b.getNamedAttr("domain", b.getArrayAttr(shapeAttrVec))})}));
        else {
          // create a deep-copy of existing attributes, and amend the new one.
          // need to figure out if there's a better way than this.
          auto arrayAttr = coordTransformAttrs.cast<ArrayAttr>();
          llvm::SmallVector<Attribute, 2> augmentedArrayAttr;

          bool augmented = false;
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

              auto existingDomain = dictAttr.get("domain").cast<ArrayAttr>();

              augmentedArrayAttr.push_back(b.getDictionaryAttr(
                  {b.getNamedAttr("operand",
                                  b.getI32IntegerAttr(userOperandIndex)),
                   b.getNamedAttr("transforms",
                                  b.getArrayAttr(augmentedTransforms)),
                   b.getNamedAttr("domain", existingDomain)}));
              augmented = true;
            }
          }
          if (!augmented)
            augmentedArrayAttr.push_back(b.getDictionaryAttr(
                {b.getNamedAttr("operand",
                                b.getI32IntegerAttr(userOperandIndex)),
                 b.getNamedAttr("transforms", b.getAffineMapArrayAttr(
                                                  outputType.getAffineMaps())),
                 b.getNamedAttr("domain", b.getArrayAttr(shapeAttrVec))}));
          user->setAttr("coord_transforms", b.getArrayAttr(augmentedArrayAttr));
        }

        // llvm::errs() << "\n========\nTransformOp updated user:\n";
        // user->dump();
        // llvm::errs() << "\n";
      }

    // Pass the input to uses of this op.
    op.replaceAllUsesWith(op.input());

    op.erase();
    return success();
  }
};


//===----------------------------------------------------------------------===//
// XdlopsGemm lowering.
//===----------------------------------------------------------------------===//

struct XdlopsGemmRewritePattern
    : public OpRewritePattern<miopen::XdlopsGemmOp> {
  using OpRewritePattern<miopen::XdlopsGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(miopen::XdlopsGemmOp op,
                                PatternRewriter &b) const override {
    auto loc = op.getLoc();

    // Obtain critical information.
    int64_t M = op.getAttr("m").template dyn_cast<IntegerAttr>().getInt();
    int64_t N = op.getAttr("n").template dyn_cast<IntegerAttr>().getInt();
    int64_t K = op.getAttr("k").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerWave = op.getAttr("m_per_wave").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave = op.getAttr("n_per_wave").template dyn_cast<IntegerAttr>().getInt();
    auto dataType = op.matrixA()
                        .getType()
                        .template dyn_cast<MemRefType>()
                        .getElementType()
                        .template dyn_cast<FloatType>();

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

    XdlopsCodeSelection xcs = XdlopsCodeSelection::get(dataType, MPerWave, NPerWave, b);

    // Extract values from XdlopsCodeSelection.
    StringRef mfmaInstr = xcs.mfmaInstr;
    int64_t MPerXdlops = xcs.MPerXdlops;
    int64_t NPerXdlops = xcs.NPerXdlops;
    int64_t MRepeats = xcs.MRepeats;
    int64_t NRepeats = xcs.NRepeats;

    int64_t group_size = xcs.group_size;
    int64_t num_groups_blk = xcs.num_groups_blk;
    int64_t num_regs_blk = xcs.num_regs_blk;
    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t num_regs_xdlops = xcs.num_regs_xdlops;
    int64_t m = xcs.m;
    int64_t n = xcs.n;
    int64_t k = xcs.k;
    int64_t cycles = xcs.cycles;
    int64_t k_base = xcs.k_base;

    bool IsABroadcast = (NPerXdlops >= MPerXdlops);
    bool IsKReduction = (num_output_blks == 1) && (num_input_blks > 1);

    int64_t RegSizePerXdlops = MPerXdlops * NPerXdlops / wave_size;
    auto RegSizePerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, RegSizePerXdlops);

    // Original C++ logic.
    // const index_t laneId = get_thread_local_1d_id() % mfma_type.wave_size;
    // FloatA a[K * MRepeats];
    // FloatB b[K * NRepeats];
    // constexpr index_t nxdlops = sizeof(FloatA) / (sizeof(data_type) * mfma_type.k_base);

    auto tid = b.create<miopen::WorkitemIdOp>(loc, b.getIndexType());
    auto laneId = b.create<SignedRemIOp>(
        loc, tid, b.create<ConstantIndexOp>(loc, wave_size));
    auto arrayAType =
        MemRefType::get({K * MRepeats}, dataType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto arrayA = b.create<miopen::GpuAllocOp>(loc, arrayAType);
    auto arrayBType =
        MemRefType::get({K * NRepeats}, dataType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto arrayB = b.create<miopen::GpuAllocOp>(loc, arrayBType);
    auto NXDlopsConstantOp = b.create<ConstantIndexOp>(
        loc, dataType.getWidth() / (dataType.getWidth() * k_base));

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);

    //llvm::errs() << "mfmaInstr: " << mfmaInstr << "\n";
    //llvm::errs() << "MPerXdlops: " << MPerXdlops << "\n";
    //llvm::errs() << "NPerXdlops: " << NPerXdlops << "\n";
    //llvm::errs() << "MRepeats: " << MRepeats << "\n";
    //llvm::errs() << "NRepeats: " << NRepeats << "\n";
    //llvm::errs() << "IsABroadcast: " << IsABroadcast << "\n";
    //llvm::errs() << "IsKReduction: " << IsKReduction << "\n";

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);
    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());

    auto MRepeatsConstantOp = b.create<ConstantIndexOp>(loc, MRepeats);
    auto NRepeatsConstantOp = b.create<ConstantIndexOp>(loc, NRepeats);

    if (!IsKReduction) {
      // Original C++ logic.
      // static_if<!IsKReduction>{}([&](auto) {
      //     for(index_t m_i = 0; m_i < MRepeats; ++m_i)
      //         for(index_t k_i      = 0; k_i < K; ++k_i)
      //             a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by threadOffsetA.

      auto outerLoopM = b.create<scf::ForOp>(loc, zeroConstantOp, MRepeatsConstantOp, oneConstantOp);
      auto olmb = OpBuilder::atBlockTerminator(outerLoopM.getBody());
      auto olmiv = outerLoopM.getInductionVar();
      auto innerLoopMK = olmb.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, oneConstantOp);
      auto ilmkb = OpBuilder::atBlockTerminator(innerLoopMK.getBody());
      auto ilmkiv = innerLoopMK.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //             a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by threadOffsetA.
      auto sourceOffsetA = ilmkb.create<AddIOp>(
          loc, op.threadOffsetA(),
          ilmkb.create<AddIOp>(
              loc,
              ilmkb.create<AddIOp>(
                  loc, ilmkb.create<MulIOp>(loc, ilmkiv, MConstantOp),
                  laneId),
              ilmkb.create<MulIOp>(loc, MPerXdlopsConstantOp, olmiv)));
      auto destOffsetA = ilmkb.create<AddIOp>(
          loc, ilmkiv, ilmkb.create<MulIOp>(loc, olmiv, KConstantOp));

      auto valueA = ilmkb.create<LoadOp>(loc, dataType, op.matrixA(),
                                         ValueRange{sourceOffsetA});
      ilmkb.create<StoreOp>(loc, valueA, arrayA, ValueRange{destOffsetA});

      // Original C++ logic.
      //     for(index_t n_i = 0; n_i < NRepeats; ++n_i)
      //         for(index_t k_i      = 0; k_i < K; ++k_i)
      //             b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by threadOffsetB.

      auto outerLoopN = b.create<scf::ForOp>(loc, zeroConstantOp, NRepeatsConstantOp, oneConstantOp);
      auto olnb = OpBuilder::atBlockTerminator(outerLoopN.getBody());
      auto olniv = outerLoopN.getInductionVar();
      auto innerLoopNK = olnb.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, oneConstantOp);
      auto ilnkb = OpBuilder::atBlockTerminator(innerLoopNK.getBody());
      auto ilnkiv = innerLoopNK.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //             b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by threadOffsetB.

      auto sourceOffsetB = ilnkb.create<AddIOp>(
          loc, op.threadOffsetB(),
          ilnkb.create<AddIOp>(
              loc,
              ilnkb.create<AddIOp>(
                  loc, ilnkb.create<MulIOp>(loc, ilnkiv, NConstantOp),
                  laneId),
              ilnkb.create<MulIOp>(loc, NPerXdlopsConstantOp, olniv)));
      auto destOffsetB = ilnkb.create<AddIOp>(
          loc, ilnkiv, ilnkb.create<MulIOp>(loc, olniv, KConstantOp));

      auto valueB = ilnkb.create<LoadOp>(loc, dataType, op.matrixB(),
                                         ValueRange{sourceOffsetB});
      ilnkb.create<StoreOp>(loc, valueB, arrayB, ValueRange{destOffsetB});

      // Original C++ logic.
      //     // get pointer of registers
      //     auto pa = reinterpret_cast<const data_type*>(&a);
      //     auto pb = reinterpret_cast<const data_type*>(&b);
      //     for(index_t m_i = 0; m_i < MRepeats; ++m_i) {
      //         for(index_t n_i = 0; n_i < NRepeats; ++n_i) {
      //             for(index_t k_i = 0; k_i < K; ++k_i) {
      //                 for(index_t i = 0; i < nxdlops; ++i)

      auto loopM =
          b.create<scf::ForOp>(loc, zeroConstantOp,
                               MRepeatsConstantOp, oneConstantOp);
      auto lmb = OpBuilder::atBlockTerminator(loopM.getBody());
      auto lmiv = loopM.getInductionVar();
      auto loopN =
          lmb.create<scf::ForOp>(loc, zeroConstantOp,
                                 NRepeatsConstantOp, oneConstantOp);
      auto lnb = OpBuilder::atBlockTerminator(loopN.getBody());
      auto lniv = loopN.getInductionVar();
      auto loopK = lnb.create<scf::ForOp>(loc, zeroConstantOp,
                                          KConstantOp, oneConstantOp);
      auto lkb = OpBuilder::atBlockTerminator(loopK.getBody());
      auto lkiv = loopK.getInductionVar();
      auto loopI = lkb.create<scf::ForOp>(
          loc, zeroConstantOp, NXDlopsConstantOp, oneConstantOp);
      auto lib = OpBuilder::atBlockTerminator(loopI.getBody());
      auto liiv = loopI.getInductionVar();

      // Original C++ logic.
      //                     mfma_type.template run<MPerXdlops, NPerXdlops>(
      //                         &pa[(k_i * nxdlops + i) * mfma_type.k_base +
      //                             m_i * K * nxdlops * mfma_type.k_base],
      //                         &pb[(k_i * nxdlops + i) * mfma_type.k_base +
      //                             n_i * K * nxdlops * mfma_type.k_base],
      //                         p_c_thread + (NRepeats * m_i + n_i) *
      //                         GetRegSizePerXdlops());
      auto addressA = lib.create<AddIOp>(loc,
        lib.create<MulIOp>(loc,
          lib.create<AddIOp>(loc,
            lib.create<MulIOp>(loc, lkiv, NXDlopsConstantOp),
            liiv),
          KBaseConstantOp),
        lib.create<MulIOp>(loc,
          lmiv,
          lib.create<MulIOp>(loc,
            KConstantOp,
            lib.create<MulIOp>(loc,
              NXDlopsConstantOp, KBaseConstantOp))));
      auto addressB = lib.create<AddIOp>(loc,
        lib.create<MulIOp>(loc,
          lib.create<AddIOp>(loc,
            lib.create<MulIOp>(loc, lkiv, NXDlopsConstantOp),
            liiv),
          KBaseConstantOp),
        lib.create<MulIOp>(loc,
          lniv,
          lib.create<MulIOp>(loc,
            KConstantOp,
            lib.create<MulIOp>(loc,
              NXDlopsConstantOp, KBaseConstantOp))));
      // TBD: use vector.type_cast for FP16/BF16 types.
      auto argA =
          lib.create<LoadOp>(loc, dataType, arrayA, ValueRange{addressA});
      auto argB =
          lib.create<LoadOp>(loc, dataType, arrayB, ValueRange{addressB});

      auto addressC = lib.create<MulIOp>(
          loc,
          lib.create<AddIOp>(
              loc, lib.create<MulIOp>(loc, NRepeatsConstantOp, lmiv),
              lniv),
          RegSizePerXdlopsConstantOp);

      // auto mfma = lib.create<miopen::MFMAOp>(loc, argA, argB, op.matrixC(), addressC);
      // mfma.setAttr("m_per_wave", lib.getI32IntegerAttr(MPerWave));
      // mfma.setAttr("n_per_wave", lib.getI32IntegerAttr(NPerWave));

      // XXX. directly emit 32 load, add by 1, store.
      auto thirtyTwoConstantOp = lib.create<ConstantIndexOp>(loc, 32);
      auto loopHack = lib.create<scf::ForOp>(loc, zeroConstantOp, thirtyTwoConstantOp, oneConstantOp);
      auto lhb = OpBuilder::atBlockTerminator(loopHack.getBody());
      auto lhiv = loopHack.getInductionVar();
 
      auto addressFinal = lhb.create<AddIOp>(loc, addressC, lhiv);
      auto value = lhb.create<LoadOp>(loc, op.matrixC(), ValueRange{addressFinal});
      auto addByOne = lhb.create<AddFOp>(loc, value, oneConstantFloatOp);
      lhb.create<StoreOp>(loc, addByOne, op.matrixC(), ValueRange{addressFinal});

      auto addressFinal2 = lhb.create<AddIOp>(loc, addressFinal, thirtyTwoConstantOp);
      auto value2 = lhb.create<LoadOp>(loc, op.matrixC(), ValueRange{addressFinal2});
      auto addByOne2 = lhb.create<AddFOp>(loc, value2, oneConstantFloatOp);
      lhb.create<StoreOp>(loc, addByOne2, op.matrixC(), ValueRange{addressFinal2});

    } else {
      // Original C++ logic.
      // }).Else([&](auto) {
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;

      auto NumThreadsBlkConstantOp = b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto blk_id = b.create<SignedDivIOp>(loc, laneId, NumThreadsBlkConstantOp);
      auto blk_td = b.create<SignedRemIOp>(loc, laneId, NumThreadsBlkConstantOp);

      // Original C++ logic.
      //     // load into registers
      //     for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks) {
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      //     }
      // p_a_wave need to be offseted by threadOffsetA.
      // p_b_wave need to be offseted by threadOffsetB.

      auto NumInputBlksConstantOp = b.create<ConstantIndexOp>(loc, num_input_blks);
      auto loopKLoad = b.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, NumInputBlksConstantOp);
      auto lklb = OpBuilder::atBlockTerminator(loopKLoad.getBody());
      auto lkliv = loopKLoad.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      // p_a_wave need to be offseted by threadOffsetA.
      // p_b_wave need to be offseted by threadOffsetB.
      auto sourceOffsetA = lklb.create<AddIOp>(
          loc, op.threadOffsetA(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(loc, lklb.create<AddIOp>(loc, lkliv, blk_id),
                                  MConstantOp),
              blk_td));

      auto valueA = lklb.create<LoadOp>(loc, dataType, op.matrixA(),
                                        ValueRange{sourceOffsetA});
      lklb.create<StoreOp>(loc, valueA, arrayA, ValueRange{lkliv});

      auto sourceOffsetB = lklb.create<AddIOp>(
          loc, op.threadOffsetB(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(loc, lklb.create<AddIOp>(loc, lkliv, blk_id),
                                  NConstantOp),
              blk_td));

      auto valueB = lklb.create<LoadOp>(loc, dataType, op.matrixB(),
                                        ValueRange{sourceOffsetB});
      lklb.create<StoreOp>(loc, valueB, arrayB, ValueRange{lkliv});

      //     // get pointer of registers
      //     auto pa = reinterpret_cast<const data_type*>(&a);
      //     auto pb = reinterpret_cast<const data_type*>(&b);
      //     for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks) {
      //         for(index_t i = 0; i < nxdlops; ++i)

      auto loopKMFMA = b.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, NumInputBlksConstantOp);
      auto lkmb = OpBuilder::atBlockTerminator(loopKMFMA.getBody());
      auto lkmiv = loopKMFMA.getInductionVar();
      auto loopI = lkmb.create<scf::ForOp>(loc, zeroConstantOp, NXDlopsConstantOp, oneConstantOp);
      auto lib = OpBuilder::atBlockTerminator(loopI.getBody());
      auto liiv = loopI.getInductionVar();

      //             mfma_type.template run<MPerXdlops, NPerXdlops>(
      //                 &pa[(k_i * nxdlops + i) * mfma_type.k_base],
      //                 &pb[(k_i * nxdlops + i) * mfma_type.k_base],
      //                 p_c_thread);
      //     }
      // });

      auto addressAB = lib.create<MulIOp>(
          loc,
          lib.create<AddIOp>(
              loc, lib.create<MulIOp>(loc, lkmiv, NXDlopsConstantOp),
              liiv),
          KBaseConstantOp);

      // TBD: use vector.type_cast for FP16/BF16 types.
      auto argA =
          lib.create<LoadOp>(loc, dataType, arrayA, ValueRange{addressAB});
      auto argB =
          lib.create<LoadOp>(loc, dataType, arrayB, ValueRange{addressAB});

      auto mfma = lib.create<miopen::MFMAOp>(loc, argA, argB, op.matrixC(), zeroConstantOp);
      mfma.setAttr("m_per_wave", lib.getI32IntegerAttr(MPerWave));
      mfma.setAttr("n_per_wave", lib.getI32IntegerAttr(NPerWave));
    }

    op.erase();
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
    int64_t M = op.getAttr("m").template dyn_cast<IntegerAttr>().getInt();
    int64_t N = op.getAttr("n").template dyn_cast<IntegerAttr>().getInt();
    int64_t K = op.getAttr("k").template dyn_cast<IntegerAttr>().getInt();
    int64_t MPerWave = op.getAttr("m_per_wave").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave = op.getAttr("n_per_wave").template dyn_cast<IntegerAttr>().getInt();

    auto dataType = op.matrixA().getType().template dyn_cast<MemRefType>().getElementType().template dyn_cast<FloatType>();

    auto MConstantOp = b.create<ConstantIndexOp>(loc, M);
    auto NConstantOp = b.create<ConstantIndexOp>(loc, N);
    auto KConstantOp = b.create<ConstantIndexOp>(loc, K);

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

    int64_t group_size = xcs.group_size;
    int64_t num_groups_blk = xcs.num_groups_blk;
    int64_t num_regs_blk = xcs.num_regs_blk;
    int64_t num_threads_blk = xcs.num_threads_blk;
    int64_t wave_size = xcs.wave_size;
    int64_t num_input_blks = xcs.num_input_blks;
    int64_t num_output_blks = xcs.num_output_blks;
    int64_t num_regs_xdlops = xcs.num_regs_xdlops;
    int64_t m = xcs.m;
    int64_t n = xcs.n;
    int64_t k = xcs.k;
    int64_t cycles = xcs.cycles;
    int64_t k_base = xcs.k_base;

    bool IsABroadcast = (NPerXdlops >= MPerXdlops);
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
    auto laneId = b.create<SignedRemIOp>(
        loc, tid, b.create<ConstantIndexOp>(loc, wave_size));

    // TBD. FloatA / FloatB could be vectorized via KPack. Ignore this for now.
    auto arrayAType =
        MemRefType::get({K * MRepeats}, dataType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto arrayA = b.create<miopen::GpuAllocOp>(loc, arrayAType);
    auto arrayBType =
        MemRefType::get({K * NRepeats}, dataType, {},
                        gpu::GPUDialect::getPrivateAddressSpace());
    auto arrayB = b.create<miopen::GpuAllocOp>(loc, arrayBType);

    // TBD. FloatA / FloatB could be vectorized via KPack tuning parameter. Ignore this for now.
    // use arrayA as pa for now.
    // use arrayB as pb for now.

    // TBD. FloatA / FloatB could be vectorized via KPack tuning parameter. Ignore this for now.
    // This must be fixed when we test fp16 / bf16 data types.
    int64_t KRepeats = (dataType.getWidth() / 8) / (dataType.getWidth() / 8 * k_base);
    int64_t AStride = K * KRepeats;
    int64_t BStride = K * KRepeats;

    auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
    auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
    auto KBaseConstantOp = b.create<ConstantIndexOp>(loc, k_base);

    auto zeroConstantOp = b.create<ConstantIndexOp>(loc, 0);
    auto oneConstantOp = b.create<ConstantIndexOp>(loc, 1);
    auto MRepeatsConstantOp = b.create<ConstantIndexOp>(loc, MRepeats);
    auto NRepeatsConstantOp = b.create<ConstantIndexOp>(loc, NRepeats);
    auto KRepeatsConstantOp = b.create<ConstantIndexOp>(loc, KRepeats);

    auto oneConstantFloatOp =
        b.create<ConstantFloatOp>(loc, APFloat(1.0f), b.getF32Type());

    if (!IsKReduction) {
      // Original C++ logic.
      // static_if<!IsKReduction>{}([&](auto) {
      //     for(index_t m_i = 0; m_i < MRepeats; ++m_i)
      //         for(index_t k_i      = 0; k_i < K; ++k_i)
      //             a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by threadOffsetA.

      auto outerLoopM = b.create<scf::ForOp>(loc, zeroConstantOp, MRepeatsConstantOp, oneConstantOp);
      auto olmb = OpBuilder::atBlockTerminator(outerLoopM.getBody());
      auto olmiv = outerLoopM.getInductionVar();
      auto innerLoopMK = olmb.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, oneConstantOp);
      auto ilmkb = OpBuilder::atBlockTerminator(innerLoopMK.getBody());
      auto ilmkiv = innerLoopMK.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //             a[k_i + m_i * K] = p_a_wave[k_i * M + laneId + MPerXdlops * m_i];
      // p_a_wave need to be offseted by threadOffsetA.
      auto sourceOffsetA = ilmkb.create<AddIOp>(
          loc, op.threadOffsetA(),
          ilmkb.create<AddIOp>(
              loc,
              ilmkb.create<AddIOp>(
                  loc, ilmkb.create<MulIOp>(loc, ilmkiv, MConstantOp),
                  laneId),
              ilmkb.create<MulIOp>(loc, MPerXdlopsConstantOp, olmiv)));
      auto destOffsetA = ilmkb.create<AddIOp>(
          loc, ilmkiv, ilmkb.create<MulIOp>(loc, olmiv, KConstantOp));

      auto valueA = ilmkb.create<LoadOp>(loc, dataType, op.matrixA(),
                                         ValueRange{sourceOffsetA});
      ilmkb.create<StoreOp>(loc, valueA, arrayA, ValueRange{destOffsetA});

      // Original C++ logic.
      //     for(index_t n_i = 0; n_i < NRepeats; ++n_i)
      //         for(index_t k_i      = 0; k_i < K; ++k_i)
      //             b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by threadOffsetB.

      auto outerLoopN = b.create<scf::ForOp>(loc, zeroConstantOp, NRepeatsConstantOp, oneConstantOp);
      auto olnb = OpBuilder::atBlockTerminator(outerLoopN.getBody());
      auto olniv = outerLoopN.getInductionVar();
      auto innerLoopNK = olnb.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, oneConstantOp);
      auto ilnkb = OpBuilder::atBlockTerminator(innerLoopNK.getBody());
      auto ilnkiv = innerLoopNK.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //             b[k_i + n_i * K] = p_b_wave[k_i * N + laneId + NPerXdlops * n_i];
      // p_b_wave need to be offseted by threadOffsetB.

      auto sourceOffsetB = ilnkb.create<AddIOp>(
          loc, op.threadOffsetB(),
          ilnkb.create<AddIOp>(
              loc,
              ilnkb.create<AddIOp>(
                  loc, ilnkb.create<MulIOp>(loc, ilnkiv, NConstantOp),
                  laneId),
              ilnkb.create<MulIOp>(loc, NPerXdlopsConstantOp, olniv)));
      auto destOffsetB = ilnkb.create<AddIOp>(
          loc, ilnkiv, ilnkb.create<MulIOp>(loc, olniv, KConstantOp));

      auto valueB = ilnkb.create<LoadOp>(loc, dataType, op.matrixB(),
                                         ValueRange{sourceOffsetB});
      ilnkb.create<StoreOp>(loc, valueB, arrayB, ValueRange{destOffsetB});

      // Original C++ logic.
      // for(index_t k_i = 0; k_i < K * KRepeats; ++k_i)
      // {
      //     p_c_thread = mfma_type.template run<MPerXdlops * MRepeats,
      //                                         NPerXdlops * NRepeats,
      //                                         AStride,
      //                                         BStride>(
      //         &pa[k_i * mfma_type.k_base], &pb[k_i * mfma_type.k_base], p_c_thread);
      // }

      auto loopIterationConstantOp = b.create<ConstantIndexOp>(loc, K * KRepeats);
      auto loopK = b.create<scf::ForOp>(loc, zeroConstantOp, loopIterationConstantOp, oneConstantOp, op.vectorCs());
      auto loopKb = OpBuilder::atBlockBegin(loopK.getBody());
      auto loopKiv = loopK.getInductionVar();

      auto offset = loopKb.create<MulIOp>(loc, loopKiv, KBaseConstantOp);
      auto argA = loopKb.create<LoadOp>(loc, dataType, arrayA, ValueRange{offset});
      auto argB = loopKb.create<LoadOp>(loc, dataType, arrayB, ValueRange{offset});

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = loopK.getRegionIterArgs()[i];
        // TBD: use constant 1.0f for A and B for now.
        auto mfma = loopKb.create<miopen::MFMAV2Op>(loc, vectorType, oneConstantFloatOp, oneConstantFloatOp, vectorC);

        // TBD: need to consider the case to use argA[AStride] and argB[BStride]
        //auto mfma = loopKb.create<miopen::MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma.setAttr("instr", loopKb.getStringAttr(mfmaInstr));
        mfma.setAttr("imm", loopKb.getArrayAttr({
                              loopKb.getI32IntegerAttr(imms[i][0]),
                              loopKb.getI32IntegerAttr(imms[i][1]),
                              loopKb.getI32IntegerAttr(imms[i][2])
                            }));
        mfmas.push_back(mfma);
      }
      loopKb.create<scf::YieldOp>(loc, mfmas);
      op.replaceAllUsesWith(loopK.results());
      op.erase();
    } else {
      // Original C++ logic.
      //     const index_t blk_id = laneId / mfma_type.num_threads_blk;
      //     const index_t blk_td = laneId % mfma_type.num_threads_blk;

      auto NumThreadsBlkConstantOp = b.create<ConstantIndexOp>(loc, num_threads_blk);
      auto blk_id = b.create<SignedDivIOp>(loc, laneId, NumThreadsBlkConstantOp);
      auto blk_td = b.create<SignedRemIOp>(loc, laneId, NumThreadsBlkConstantOp);

      // Original C++ logic.
      //     // load into registers
      //     for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks) {
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      //     }
      // p_a_wave need to be offseted by threadOffsetA.
      // p_b_wave need to be offseted by threadOffsetB.

      auto NumInputBlksConstantOp = b.create<ConstantIndexOp>(loc, num_input_blks);
      auto loopKLoad = b.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, NumInputBlksConstantOp);
      auto lklb = OpBuilder::atBlockTerminator(loopKLoad.getBody());
      auto lkliv = loopKLoad.getInductionVar();

      // TBD. Check if we need to apply coord_transform as well.
      //         a[k_i] = p_a_wave[(k_i + blk_id) * M + blk_td];
      //         b[k_i] = p_b_wave[(k_i + blk_id) * N + blk_td];
      // p_a_wave need to be offseted by threadOffsetA.
      // p_b_wave need to be offseted by threadOffsetB.
      auto sourceOffsetA = lklb.create<AddIOp>(
          loc, op.threadOffsetA(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(loc, lklb.create<AddIOp>(loc, lkliv, blk_id),
                                  MConstantOp),
              blk_td));

      auto valueA = lklb.create<LoadOp>(loc, dataType, op.matrixA(),
                                        ValueRange{sourceOffsetA});
      lklb.create<StoreOp>(loc, valueA, arrayA, ValueRange{lkliv});

      auto sourceOffsetB = lklb.create<AddIOp>(
          loc, op.threadOffsetB(),
          lklb.create<AddIOp>(
              loc,
              lklb.create<MulIOp>(loc, lklb.create<AddIOp>(loc, lkliv, blk_id),
                                  NConstantOp),
              blk_td));

      auto valueB = lklb.create<LoadOp>(loc, dataType, op.matrixB(),
                                        ValueRange{sourceOffsetB});
      lklb.create<StoreOp>(loc, valueB, arrayB, ValueRange{lkliv});

      // Original C++ logic.
      // for(index_t k_i = 0; k_i < K; k_i += mfma_type.num_input_blks)
      // {
      //     for(index_t i = 0; i < KRepeats; ++i)
      //         p_c_thread = mfma_type.template run<MPerXdlops, NPerXdlops, AStride, BStride>(
      //             &pa[(k_i * KRepeats + i) * mfma_type.k_base],
      //             &pb[(k_i * KRepeats + i) * mfma_type.k_base],
      //             p_c_thread);
      // }

      auto outerLoop = b.create<scf::ForOp>(loc, zeroConstantOp, KConstantOp, NumInputBlksConstantOp, op.vectorCs());
      auto outerLoopb = OpBuilder::atBlockBegin(outerLoop.getBody());
      auto outerLoopiv = outerLoop.getInductionVar();

      auto innerLoop = outerLoopb.create<scf::ForOp>(loc, zeroConstantOp, KRepeatsConstantOp, oneConstantOp, outerLoop.getRegionIterArgs());
      auto innerLoopb = OpBuilder::atBlockBegin(innerLoop.getBody());
      auto innerLoopiv = innerLoop.getInductionVar();

      auto offset = innerLoopb.create<MulIOp>(loc, innerLoopb.create<AddIOp>(loc, innerLoopb.create<MulIOp>(loc, outerLoopiv, KRepeatsConstantOp), innerLoopiv), KBaseConstantOp);
      auto argA = innerLoopb.create<LoadOp>(loc, dataType, arrayA, ValueRange{offset});
      auto argB = innerLoopb.create<LoadOp>(loc, dataType, arrayB, ValueRange{offset});

      SmallVector<Value, 4> mfmas;
      for (int64_t i = 0; i < vectorNumber; ++i) {
        auto vectorC = innerLoop.getRegionIterArgs()[i];
        // TBD: use constant 1.0f for A and B for now.
        auto mfma = innerLoopb.create<miopen::MFMAV2Op>(loc, vectorType, oneConstantFloatOp, oneConstantFloatOp, vectorC);

        // TBD: need to consider the case to use argA[AStride] and argB[BStride]
        //auto mfma = loopKb.create<miopen::MFMAV2Op>(loc, vectorType, argA, argB, vectorC);

        mfma.setAttr("instr", innerLoopb.getStringAttr(mfmaInstr));
        mfma.setAttr("imm", innerLoopb.getArrayAttr({
                              innerLoopb.getI32IntegerAttr(imms[i][0]),
                              innerLoopb.getI32IntegerAttr(imms[i][1]),
                              innerLoopb.getI32IntegerAttr(imms[i][2])
                            }));
        mfmas.push_back(mfma);
      }
      innerLoopb.create<scf::YieldOp>(loc, mfmas);

      outerLoopb.create<scf::YieldOp>(loc, innerLoop.results());

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
        op.getAttr("m_per_wave").template dyn_cast<IntegerAttr>().getInt();
    int64_t NPerWave =
        op.getAttr("n_per_wave").template dyn_cast<IntegerAttr>().getInt();

    auto dataType = op.matrixA()
                        .getType()
                        .template dyn_cast<MemRefType>()
                        .getElementType()
                        .template dyn_cast<FloatType>();

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
          loc, resultTypes, op.matrixA(), op.matrixB(), op.threadOffsetA(),
          op.threadOffsetB(), op.vectorCs());

      xdlopsGemmV2Op.setAttr("m", op.getAttr("m"));
      xdlopsGemmV2Op.setAttr("n", op.getAttr("n"));
      xdlopsGemmV2Op.setAttr("k", op.getAttr("k"));
      xdlopsGemmV2Op.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemmV2Op.setAttr("n_per_wave", op.getAttr("n_per_wave"));
      xdlopsGemmV2Op.setAttr("coord_transforms",
                             op.getAttr("coord_transforms"));

      op.replaceAllUsesWith(xdlopsGemmV2Op.vectorDs());
      op.erase();
    } else if (MRepeats == 2 && NRepeats == 1) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block, p_c_thread.s.x.l);
      // p_c_thread.s.y.l = XdlopsGemm.template Run<M, N, K>(p_a_block + MPerXdlops, p_b_block, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.threadOffsetA(),
          op.threadOffsetB(), ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

      xdlopsGemmV2Op0.setAttr("m", op.getAttr("m"));
      xdlopsGemmV2Op0.setAttr("n", op.getAttr("n"));
      xdlopsGemmV2Op0.setAttr("k", op.getAttr("k"));
      xdlopsGemmV2Op0.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemmV2Op0.setAttr("n_per_wave", op.getAttr("n_per_wave"));
      xdlopsGemmV2Op0.setAttr("coord_transforms",
                              op.getAttr("coord_transforms"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto MPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, MPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes1, op.matrixA(), op.matrixB(),
          b.create<AddIOp>(loc, op.threadOffsetA(), MPerXdlopsConstantOp),
          op.threadOffsetB(), ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1.setAttr("m", op.getAttr("m"));
      xdlopsGemmV2Op1.setAttr("n", op.getAttr("n"));
      xdlopsGemmV2Op1.setAttr("k", op.getAttr("k"));
      xdlopsGemmV2Op1.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemmV2Op1.setAttr("n_per_wave", op.getAttr("n_per_wave"));
      xdlopsGemmV2Op1.setAttr("coord_transforms",
                              op.getAttr("coord_transforms"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
    } else if (MRepeats == 1 && NRepeats == 2) {
      // Original C++ logic.
      // p_c_thread.s.x.l = XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block, p_c_thread.s.x.l);
      // p_c_thread.s.y.l = XdlopsGemm.template Run<M, N, K>(p_a_block, p_b_block + NPerXdlops, p_c_thread.s.y.l);

      SmallVector<Type, 2> resultTypes0;
      resultTypes0.push_back(op.vectorDs()[0].getType());
      resultTypes0.push_back(op.vectorDs()[1].getType());

      auto xdlopsGemmV2Op0 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes0, op.matrixA(), op.matrixB(), op.threadOffsetA(),
          op.threadOffsetB(), ValueRange{op.vectorCs()[0], op.vectorCs()[1]});

      xdlopsGemmV2Op0.setAttr("m", op.getAttr("m"));
      xdlopsGemmV2Op0.setAttr("n", op.getAttr("n"));
      xdlopsGemmV2Op0.setAttr("k", op.getAttr("k"));
      xdlopsGemmV2Op0.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemmV2Op0.setAttr("n_per_wave", op.getAttr("n_per_wave"));
      xdlopsGemmV2Op0.setAttr("coord_transforms",
                              op.getAttr("coord_transforms"));

      SmallVector<Type, 2> resultTypes1;
      resultTypes1.push_back(op.vectorDs()[2].getType());
      resultTypes1.push_back(op.vectorDs()[3].getType());

      auto NPerXdlopsConstantOp = b.create<ConstantIndexOp>(loc, NPerXdlops);
      auto xdlopsGemmV2Op1 = b.create<miopen::XdlopsGemmV2Op>(
          loc, resultTypes1, op.matrixA(), op.matrixB(), op.threadOffsetA(),
          b.create<AddIOp>(loc, op.threadOffsetB(), NPerXdlopsConstantOp),
          ValueRange{op.vectorCs()[2], op.vectorCs()[3]});

      xdlopsGemmV2Op1.setAttr("m", op.getAttr("m"));
      xdlopsGemmV2Op1.setAttr("n", op.getAttr("n"));
      xdlopsGemmV2Op1.setAttr("k", op.getAttr("k"));
      xdlopsGemmV2Op1.setAttr("m_per_wave", op.getAttr("m_per_wave"));
      xdlopsGemmV2Op1.setAttr("n_per_wave", op.getAttr("n_per_wave"));
      xdlopsGemmV2Op1.setAttr("coord_transforms",
                              op.getAttr("coord_transforms"));

      op.replaceAllUsesWith(ValueRange{
          xdlopsGemmV2Op0.vectorDs()[0], xdlopsGemmV2Op0.vectorDs()[1],
          xdlopsGemmV2Op1.vectorDs()[0], xdlopsGemmV2Op1.vectorDs()[1]});
      op.erase();
    }

    return success();
  }
};

