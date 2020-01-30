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
// This pass converts miopen.conv2d into miopen.transform and
// miopen.gridwise_gemm.
//
//===----------------------------------------------------------------------===//

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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;

struct Conv2DOpRewritePattern : public OpRewritePattern<miopen::Conv2DOp> {
  using OpRewritePattern<miopen::Conv2DOp>::OpRewritePattern;

  PatternMatchResult
  matchAndRewrite(miopen::Conv2DOp op, PatternRewriter &b) const override {
    auto filterLayoutAttr = op.getAttrOfType<ArrayAttr>("filter_layout");
    auto inputLayoutAttr = op.getAttrOfType<ArrayAttr>("input_layout");
    auto outputLayoutAttr = op.getAttrOfType<ArrayAttr>("output_layout");

    // TBD: handle dilations, strides, padding.

    // Transform filter tensor.
    auto filterType = op.filter().getType().dyn_cast<MemRefType>();
    auto filterShape = filterType.getShape();
    auto filterElementType = filterType.getElementType();
    
    llvm::SmallVector<int64_t, 2> transformedFilterShape;
    transformedFilterShape.set_size(filterShape.size() - 2);
    // TBD: compute transformed filter shape dimensions.
    std::fill(transformedFilterShape.begin(), transformedFilterShape.end(), -1);
    auto transformedFilterMemRefType = MemRefType::get(transformedFilterShape, filterElementType);

    llvm::SmallVector<NamedAttribute, 3> transformedFilterAttrs;

    // set layout attribute.
    // Weight tensor transformation:
    // - Part 1: Merge non-K dimensions to dimension 0, name it as gemmK.
    // - Part 2: PassThrough K dimension to dimension 1, name it as gemmM.
    {
      llvm::SmallVector<IntegerAttr, 3> nonKDims;
      IntegerAttr kDim;
      llvm::SmallVector<StringAttr, 3> nonKDimNames;
      StringAttr kDimName;
      for (unsigned i = 0; i < filterLayoutAttr.size(); ++i) {
        if (auto strAttr = filterLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
          if (strAttr.getValue() == "k") {
            kDim = b.getI32IntegerAttr(i);
            kDimName = strAttr;
          } else {
            nonKDims.push_back(b.getI32IntegerAttr(i));
            nonKDimNames.push_back(strAttr);
          }
        }
      } 

      // Part 1: Merge part.
      llvm::SmallVector<NamedAttribute, 5> transformedFilterLayoutPart1Specs;
      transformedFilterLayoutPart1Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(0)})));
      transformedFilterLayoutPart1Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmK")})));
      transformedFilterLayoutPart1Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Merge")));
      transformedFilterLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions",
                                                  b.getArrayAttr(ArrayRef<Attribute>(nonKDims.begin(), nonKDims.end()))));
      transformedFilterLayoutPart1Specs.push_back(b.getNamedAttr("source_names",
                                                  b.getArrayAttr(ArrayRef<Attribute>(nonKDimNames.begin(), nonKDimNames.end()))));

      // Part 2: Passthrough part.
      llvm::SmallVector<NamedAttribute, 5> transformedFilterLayoutPart2Specs;
      transformedFilterLayoutPart2Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(1)})));
      transformedFilterLayoutPart2Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmM")})));
      transformedFilterLayoutPart2Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      transformedFilterLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions",
                                                  b.getArrayAttr({kDim})));
      transformedFilterLayoutPart2Specs.push_back(b.getNamedAttr("source_names",
                                                  b.getArrayAttr({kDimName})));

      auto transformedFilterLayoutAttr = b.getNamedAttr("layout",
                                                               b.getArrayAttr({
                                                                   b.getDictionaryAttr(transformedFilterLayoutPart1Specs),
                                                                   b.getDictionaryAttr(transformedFilterLayoutPart2Specs)
                                                               }));
      transformedFilterAttrs.push_back(transformedFilterLayoutAttr);
    }

    // set source_layout attribute.
    auto filterSrcLayoutAttr = b.getNamedAttr("source_layout", filterLayoutAttr);
    transformedFilterAttrs.push_back(filterSrcLayoutAttr);
    // set output_layout attribute.
    auto filterOutputLayoutAttr = b.getNamedAttr("output_layout",
                                                        b.getArrayAttr({
                                                            b.getStringAttr("gemmK"),
                                                            b.getStringAttr("gemmM")
                                                        }));
    transformedFilterAttrs.push_back(filterOutputLayoutAttr);
    // set gridwise_gemm_argument_pos attribute.
    auto filterGridwiseGemmArgPosAttr = b.getNamedAttr("gridwise_gemm_argument_position", b.getI32IntegerAttr(0));
    transformedFilterAttrs.push_back(filterGridwiseGemmArgPosAttr);
    auto gemmA = b.create<miopen::TransformOp>(op.getLoc(), transformedFilterMemRefType, op.filter(), transformedFilterAttrs);


    // Transform input tensor.
    // Input tensor step 1: padded input.
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
    auto inputShape = inputType.getShape();
    auto inputElementType = inputType.getElementType();

    // TBD: compute padded input shape dimensions.

    llvm::SmallVector<NamedAttribute, 3> paddedInputAttrs;

    // reorderedPaddedInputDimNames would be used by the next stage.
    llvm::SmallVector<StringAttr, 4> reorderedPaddedInputDimNames;

    // set layout attribute.
    // Padded input tensor transformation:
    // - Part 1: PassThrough ni dimension to its original dimension, name it as ni.
    // - Part 2: PassThrough ci dimension to its original dimension, name it as ci.
    // - Part 3: Pad hi/wi dimensions to their original dimensions, name it as hipad/wipad.
    {
      IntegerAttr nDim, cDim;
      StringAttr nDimName, cDimName;
      llvm::SmallVector<IntegerAttr, 2> hwDims;
      llvm::SmallVector<StringAttr, 2> hwDimNames;
      for (unsigned i = 0; i < inputLayoutAttr.size(); ++i) {
        if (auto strAttr = inputLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
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
        hwPaddedDimNames.push_back(b.getStringAttr((strAttr.getValue() + "pad").str()));
      }

      for (unsigned i = 0, j = 0; i < inputLayoutAttr.size(); ++i) {
        if (APInt(32, i) == nDim.getValue()) {
          reorderedPaddedInputDimNames.push_back(nDimName);
        } else if (APInt(32, i) == cDim.getValue()) {
          reorderedPaddedInputDimNames.push_back(cDimName);
        } else {
          reorderedPaddedInputDimNames.push_back(hwPaddedDimNames[j++]);
        }
      }
  
      // Part 1: Passthrough for ni dimension.
      llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart1Specs;
      paddedInputLayoutPart1Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({nDim})));
      paddedInputLayoutPart1Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({nDimName})));
      paddedInputLayoutPart1Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      paddedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({nDim})));
      paddedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({nDimName})));

      // Part 2: Passthrough for ci dimension.
      llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart2Specs;
      paddedInputLayoutPart2Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({cDim})));
      paddedInputLayoutPart2Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({cDimName})));
      paddedInputLayoutPart2Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      paddedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({cDim})));
      paddedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({cDimName})));

      // Part 3: Pad for h/w dimensions.
      llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart3Specs;
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr(ArrayRef<Attribute>(hwDims.begin(), hwDims.end()))));
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("names", b.getArrayAttr(ArrayRef<Attribute>(hwPaddedDimNames.begin(), hwPaddedDimNames.end()))));
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Pad")));
      // TBD: padding parmeters.
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("parameters",
                                                  b.getArrayAttr({
                                                      b.getI32IntegerAttr(0),
                                                      b.getI32IntegerAttr(0)
                                                  })));
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr(ArrayRef<Attribute>(hwDims.begin(), hwDims.end()))));
      paddedInputLayoutPart3Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr(ArrayRef<Attribute>(hwDimNames.begin(), hwDimNames.end()))));

      auto paddedInputLayoutAttr = b.getNamedAttr("layout",
                                                               b.getArrayAttr({
                                                                   b.getDictionaryAttr(paddedInputLayoutPart1Specs),
                                                                   b.getDictionaryAttr(paddedInputLayoutPart2Specs),
                                                                   b.getDictionaryAttr(paddedInputLayoutPart3Specs)
                                                               }));
      paddedInputAttrs.push_back(paddedInputLayoutAttr);

      // set source_layout attribute.
      auto inputSrcLayoutAttr = b.getNamedAttr("source_layout", inputLayoutAttr);
      paddedInputAttrs.push_back(inputSrcLayoutAttr);
      // set output_layout attribute.
      auto paddedInputOutputLayoutAttr = b.getNamedAttr("output_layout", b.getArrayAttr(ArrayRef<Attribute>(reorderedPaddedInputDimNames.begin(), reorderedPaddedInputDimNames.end())));
      paddedInputAttrs.push_back(paddedInputOutputLayoutAttr);
    }
    auto paddedInput = b.create<miopen::TransformOp>(op.getLoc(), inputType, op.input(), paddedInputAttrs);

    // Input tensor step 2 : embedded input.
    llvm::SmallVector<int64_t, 6> embeddedInputShape;
    embeddedInputShape.set_size(inputShape.size() + 2);
    // TBD: compute embedded input shape dimensions.
    std::fill(embeddedInputShape.begin(), embeddedInputShape.end(), -1);
    auto embeddedInputMemRefType = MemRefType::get(embeddedInputShape, inputElementType);

    llvm::SmallVector<NamedAttribute, 3> embeddedInputAttrs;

    // reorderedEmbeddedInputDimNames would be used by the next stage.
    llvm::SmallVector<StringAttr, 6> reorderedEmbeddedInputDimNames;

    // Embedded input tensor transformation:
    // - Part 1: PassThrough ni dimension to its original dimension, name it as ni.
    // - Part 2: PassThrough ci dimension to its original dimension, name it as ci.
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
        } else if (strAttr.getValue() == "ci") {
          cDim = b.getI32IntegerAttr(i);
          cDimName = strAttr;

          reorderedCDim = b.getI32IntegerAttr(dimCtr++);

          reorderedEmbeddedInputDimNames.push_back(strAttr);
        } else if (strAttr.getValue() == "hipad") {
          hDim = b.getI32IntegerAttr(i);
          hDimName = strAttr;

          reorderedYHoDim.push_back(b.getI32IntegerAttr(dimCtr++));
          reorderedYHoDim.push_back(b.getI32IntegerAttr(dimCtr++));

          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("y"));
          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("ho"));
        } else if (strAttr.getValue() == "wipad") {
          wDim = b.getI32IntegerAttr(i);
          wDimName = strAttr;

          reorderedXWoDim.push_back(b.getI32IntegerAttr(dimCtr++));
          reorderedXWoDim.push_back(b.getI32IntegerAttr(dimCtr++));

          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("x"));
          reorderedEmbeddedInputDimNames.push_back(b.getStringAttr("wo"));
        }
      }

      // Part 1: Passthrough for ni dimension.
      llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart1Specs;
      embeddedInputLayoutPart1Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({reorderedNDim})));
      embeddedInputLayoutPart1Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({nDimName})));
      embeddedInputLayoutPart1Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      embeddedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({nDim})));
      embeddedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({nDimName})));

      // Part 2: Passthrough for ci dimension.
      llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart2Specs;
      embeddedInputLayoutPart2Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({reorderedCDim})));
      embeddedInputLayoutPart2Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({cDimName})));
      embeddedInputLayoutPart2Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      embeddedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({cDim})));
      embeddedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({cDimName})));

      // Part 3: Embed for y, ho dimensions.
      llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart3Specs;
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr(ArrayRef<Attribute>(reorderedYHoDim.begin(), reorderedYHoDim.end()))));
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("names",
                                                  b.getArrayAttr({
                                                      b.getStringAttr("y"),
                                                      b.getStringAttr("ho")
                                                  })));
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Embed")));
      // TBD: padding parmeters.
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("parameters",
                                                  b.getArrayAttr({
                                                      b.getI32IntegerAttr(2),
                                                      b.getI32IntegerAttr(1),
                                                      b.getI32IntegerAttr(1),
                                                      b.getI32IntegerAttr(0)
                                                  })));
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({hDim})));
      embeddedInputLayoutPart3Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({hDimName})));

      // Part 4: Embed for x, wo dimensions.
      llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart4Specs;
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr(ArrayRef<Attribute>(reorderedXWoDim.begin(), reorderedXWoDim.end()))));
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("names",
                                                  b.getArrayAttr({
                                                      b.getStringAttr("x"),
                                                      b.getStringAttr("wo")
                                                  })));
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Embed")));
      // TBD: embed parmeters.
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("parameters",
                                                  b.getArrayAttr({
                                                      b.getI32IntegerAttr(2),
                                                      b.getI32IntegerAttr(1),
                                                      b.getI32IntegerAttr(1),
                                                      b.getI32IntegerAttr(0)
                                                  })));
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({wDim})));
      embeddedInputLayoutPart4Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({wDimName})));

      auto embeddedInputLayoutAttr = b.getNamedAttr("layout",
                                                               b.getArrayAttr({
                                                                   b.getDictionaryAttr(embeddedInputLayoutPart1Specs),
                                                                   b.getDictionaryAttr(embeddedInputLayoutPart2Specs),
                                                                   b.getDictionaryAttr(embeddedInputLayoutPart3Specs),
                                                                   b.getDictionaryAttr(embeddedInputLayoutPart4Specs)
                                                               }));
      embeddedInputAttrs.push_back(embeddedInputLayoutAttr);

      // set intermediate_layout attribute.
      auto embeddedInputImmLayoutAttr = b.getNamedAttr("intermediate_layout", b.getArrayAttr(ArrayRef<Attribute>(reorderedPaddedInputDimNames.begin(), reorderedPaddedInputDimNames.end())));
      embeddedInputAttrs.push_back(embeddedInputImmLayoutAttr);
      // set output_layout attribute.
      auto embeddedInputOutputLayoutAttr = b.getNamedAttr("output_layout", b.getArrayAttr(ArrayRef<Attribute>(reorderedEmbeddedInputDimNames.begin(), reorderedEmbeddedInputDimNames.end())));
      embeddedInputAttrs.push_back(embeddedInputOutputLayoutAttr);
    }
    auto embeddedInput = b.create<miopen::TransformOp>(op.getLoc(), embeddedInputMemRefType, ArrayRef<Value>(paddedInput), embeddedInputAttrs);

    // Input tensor step 3: transformed input.
    llvm::SmallVector<int64_t, 2> transformedInputShape;
    transformedInputShape.set_size(inputShape.size() - 2);
    // TBD: compute transformed input shape dimensions.
    std::fill(transformedInputShape.begin(), transformedInputShape.end(), -1);
    auto transformedInputMemRefType = MemRefType::get(transformedInputShape, inputElementType);

    llvm::SmallVector<NamedAttribute, 3> transformedInputAttrs;

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

      // Part 1: Merge ci, y, x dimensions.
      llvm::SmallVector<NamedAttribute, 5> transformedInputLayoutPart1Specs;
      transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(0)})));
      transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmK")})));
      transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Merge")));

      // XXX: use better way to match output tensor layout for c/y/x dimension.
      if (cDim.getInt() < yDim.getInt()) {
        transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({cDim, yDim, xDim})));
        transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({cDimName, yDimName, xDimName})));
      } else {
        transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({yDim, xDim, cDim})));
        transformedInputLayoutPart1Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({yDimName, xDimName, cDimName})));
      }

      // Part 2: Merge ni, ho, wo dimensions.
      llvm::SmallVector<NamedAttribute, 5> transformedInputLayoutPart2Specs;
      transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(1)})));
      transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmN")})));
      transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Merge")));

      // XXX: use better way to match output tensor layout for n/h/w dimension.
      if (nDim.getInt() < hDim.getInt()) {
        transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({nDim, hDim, wDim})));
        transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({nDimName, hDimName, wDimName})));
      } else {
        transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions", b.getArrayAttr({hDim, wDim, nDim})));
        transformedInputLayoutPart2Specs.push_back(b.getNamedAttr("source_names", b.getArrayAttr({hDimName, wDimName, nDimName})));
      }

      auto transformedInputLayoutAttr = b.getNamedAttr("layout",
                                                               b.getArrayAttr({
                                                                   b.getDictionaryAttr(transformedInputLayoutPart1Specs),
                                                                   b.getDictionaryAttr(transformedInputLayoutPart2Specs)
                                                               }));
      transformedInputAttrs.push_back(transformedInputLayoutAttr);
    }
    // set intermediate_layout attribute.
    auto transformedInputImmLayoutAttr = b.getNamedAttr("intermediate_layout", b.getArrayAttr(ArrayRef<Attribute>(reorderedEmbeddedInputDimNames.begin(), reorderedEmbeddedInputDimNames.end())));
    transformedInputAttrs.push_back(transformedInputImmLayoutAttr);
    // set output_layout attribute.
    auto transformedInputOutputLayoutAttr = b.getNamedAttr("output_layout",
                                                        b.getArrayAttr({
                                                            b.getStringAttr("gemmK"),
                                                            b.getStringAttr("gemmN")
                                                        }));
    transformedInputAttrs.push_back(transformedInputOutputLayoutAttr);

    // set gridwise_gemm_argument_pos attribute.
    auto inputGridwiseGemmArgPosAttr = b.getNamedAttr("gridwise_gemm_argument_position", 
                                                             b.getI32IntegerAttr(1));
    transformedInputAttrs.push_back(inputGridwiseGemmArgPosAttr);
    auto gemmB = b.create<miopen::TransformOp>(op.getLoc(), transformedInputMemRefType, ArrayRef<Value>(embeddedInput), transformedInputAttrs);


    // Transform output tensor.
    auto outputType = op.output().getType().dyn_cast<MemRefType>();
    auto outputShape = outputType.getShape();
    auto outputElementType = outputType.getElementType();

    llvm::SmallVector<int64_t, 2> transformedOutputShape;
    transformedOutputShape.set_size(outputShape.size() - 2);
    // TBD: compute transformed output shape dimensions.
    std::fill(transformedOutputShape.begin(), transformedOutputShape.end(), -1);
    auto transformedOutputMemRefType = MemRefType::get(transformedOutputShape, outputElementType);

    llvm::SmallVector<NamedAttribute, 3> transformedOutputAttrs;

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
        if (auto strAttr = outputLayoutAttr.getValue()[i].dyn_cast<StringAttr>()) {
          if (strAttr.getValue() == "ko") {
            kDim = b.getI32IntegerAttr(i);
            kDimName = strAttr;
          } else {
            nonKDims.push_back(b.getI32IntegerAttr(i));
            nonKDimNames.push_back(strAttr);
          }
        }
      }
 
      // Part 1: Passthrough.
      llvm::SmallVector<NamedAttribute, 5> transformedOutputLayoutPart1Specs;
      transformedOutputLayoutPart1Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(0)})));
      transformedOutputLayoutPart1Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmM")})));
      transformedOutputLayoutPart1Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("PassThrough")));
      transformedOutputLayoutPart1Specs.push_back(b.getNamedAttr("source_dimensions",
                                                  b.getArrayAttr({kDim})));
      transformedOutputLayoutPart1Specs.push_back(b.getNamedAttr("source_names",
                                                  b.getArrayAttr({kDimName})));

      // Part 2: Merge.
      llvm::SmallVector<NamedAttribute, 5> transformedOutputLayoutPart2Specs;
      transformedOutputLayoutPart2Specs.push_back(b.getNamedAttr("dimensions", b.getArrayAttr({b.getI32IntegerAttr(1)})));
      transformedOutputLayoutPart2Specs.push_back(b.getNamedAttr("names", b.getArrayAttr({b.getStringAttr("gemmN")})));
      transformedOutputLayoutPart2Specs.push_back(b.getNamedAttr("transformation", b.getStringAttr("Merge")));
      transformedOutputLayoutPart2Specs.push_back(b.getNamedAttr("source_dimensions",
                                                  b.getArrayAttr(ArrayRef<Attribute>(nonKDims.begin(), nonKDims.end()))));
      transformedOutputLayoutPart2Specs.push_back(b.getNamedAttr("source_names",
                                                  b.getArrayAttr(ArrayRef<Attribute>(nonKDimNames.begin(), nonKDimNames.end()))));

      auto transformedOutputLayoutAttr = b.getNamedAttr("layout",
                                                               b.getArrayAttr({
                                                                   b.getDictionaryAttr(transformedOutputLayoutPart1Specs),
                                                                   b.getDictionaryAttr(transformedOutputLayoutPart2Specs)
                                                               }));
      transformedOutputAttrs.push_back(transformedOutputLayoutAttr);
    }

    // set source_layout attribute.
    auto outputSrcLayoutAttr = b.getNamedAttr("source_layout", outputLayoutAttr);
    transformedOutputAttrs.push_back(outputSrcLayoutAttr);
    // set output_layout attribute.
    auto transformedOutputOutputLayoutAttr = b.getNamedAttr("output_layout",
                                                        b.getArrayAttr({
                                                            b.getStringAttr("gemmM"),
                                                            b.getStringAttr("gemmN")
                                                        }));
    transformedOutputAttrs.push_back(transformedOutputOutputLayoutAttr);

    // set gridwise_gemm_argument_pos attribute.
    auto outputGridwiseGemmArgPosAttr = b.getNamedAttr("gridwise_gemm_argument_position", b.getI32IntegerAttr(2));
    transformedOutputAttrs.push_back(outputGridwiseGemmArgPosAttr);
    auto gemmC = b.create<miopen::TransformOp>(op.getLoc(), transformedOutputMemRefType, op.output(), transformedOutputAttrs);

    // Emit miopen.gridwise_gemm op.
    b.create<miopen::GridwiseGemmOp>(op.getLoc(), gemmA, gemmB, gemmC);

    // Finally, erase the original Conv2D op.
    op.erase();

    return matchSuccess();
  }
};

namespace {
struct LowerMIOpenOpsPass : public ModulePass<LowerMIOpenOpsPass> {
  void runOnModule() override;
};
} // end anonymous namespace

void LowerMIOpenOpsPass::runOnModule() {
  OwningRewritePatternList patterns;
  patterns.insert<Conv2DOpRewritePattern>(&getContext());
  applyPatternsGreedily(getModule(), patterns);
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::miopen::createLowerMIOpenOpsPass() {
  return std::make_unique<LowerMIOpenOpsPass>();
}

static PassRegistration<LowerMIOpenOpsPass>
    lowerMIOpenOpsPass("miopen-lowering",
                       "Lower MIOpen conv2d into transform and gridwise_gemm.");
