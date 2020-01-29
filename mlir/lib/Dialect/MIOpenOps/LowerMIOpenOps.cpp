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
  matchAndRewrite(miopen::Conv2DOp op, PatternRewriter &rewriter) const override {
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
            kDim = IntegerAttr::get(IntegerType::get(32, op.getContext()), i);
            kDimName = StringAttr::get(strAttr.getValue(), op.getContext());
          } else {
            nonKDims.push_back(IntegerAttr::get(IntegerType::get(32, op.getContext()), i));
            nonKDimNames.push_back(StringAttr::get(strAttr.getValue(), op.getContext()));
          }
        }
      } 

      // Part 1: Merge part.
      llvm::SmallVector<NamedAttribute, 5> transformedFilterLayoutPart1Specs;
      transformedFilterLayoutPart1Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)}, op.getContext())));
      transformedFilterLayoutPart1Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmK", op.getContext())}, op.getContext())));
      transformedFilterLayoutPart1Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Merge", op.getContext())));
      transformedFilterLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                  ArrayAttr::get(ArrayRef<Attribute>(nonKDims.begin(), nonKDims.end()), op.getContext())));
      transformedFilterLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_names",
                                                  ArrayAttr::get(ArrayRef<Attribute>(nonKDimNames.begin(), nonKDimNames.end()), op.getContext())));

      // Part 2: Passthrough part.
      llvm::SmallVector<NamedAttribute, 5> transformedFilterLayoutPart2Specs;
      transformedFilterLayoutPart2Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 1)}, op.getContext())));
      transformedFilterLayoutPart2Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmM", op.getContext())}, op.getContext())));
      transformedFilterLayoutPart2Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
      transformedFilterLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                  ArrayAttr::get({kDim}, op.getContext())));
      transformedFilterLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_names",
                                                  ArrayAttr::get({kDimName}, op.getContext())));

      auto transformedFilterLayoutAttr = rewriter.getNamedAttr("layout",
                                                               ArrayAttr::get({
                                                                   DictionaryAttr::get(transformedFilterLayoutPart1Specs, op.getContext()),
                                                                   DictionaryAttr::get(transformedFilterLayoutPart2Specs, op.getContext())
                                                               }, op.getContext()));
      transformedFilterAttrs.push_back(transformedFilterLayoutAttr);
    }

    // set source_layout attribute.
    auto filterSrcLayoutAttr = rewriter.getNamedAttr("source_layout", filterLayoutAttr);
    transformedFilterAttrs.push_back(filterSrcLayoutAttr);
    // set output_layout attribute.
    auto filterOutputLayoutAttr = rewriter.getNamedAttr("output_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("gemmK", op.getContext()),
                                                            StringAttr::get("gemmM", op.getContext())
                                                        }, op.getContext()));
    transformedFilterAttrs.push_back(filterOutputLayoutAttr);
    // set gridwise_gemm_argument_pos attribute.
    auto filterGridwiseGemmArgPosAttr = rewriter.getNamedAttr("gridwise_gemm_argument_position", 
                                                              IntegerAttr::get(IntegerType::get(32, op.getContext()), 0));
    transformedFilterAttrs.push_back(filterGridwiseGemmArgPosAttr);
    auto gemmA = rewriter.create<miopen::TransformOp>(op.getLoc(), transformedFilterMemRefType, op.filter(), transformedFilterAttrs);


    // Transform input tensor.
    // Input tensor step 1: padded input.
    auto inputType = op.input().getType().dyn_cast<MemRefType>();
    auto inputShape = inputType.getShape();
    auto inputElementType = inputType.getElementType();

    // TBD: compute padded input shape dimensions.

    llvm::SmallVector<NamedAttribute, 3> paddedInputAttrs;

    // TBD: set layout attribute.
    // TBD: part 1: Passthrough.
    llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart1Specs;
    paddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)}, op.getContext())));
    paddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("ni", op.getContext())}, op.getContext())));
    paddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
    paddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0),
                                                }, op.getContext())));
    paddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ni", op.getContext())
                                                }, op.getContext())));

    // TBD: part 2: Passthrough.
    llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart2Specs;
    paddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 1)}, op.getContext())));
    paddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("ci", op.getContext())}, op.getContext())));
    paddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
    paddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                }, op.getContext())));
    paddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ci", op.getContext())
                                                }, op.getContext())));

    // TBD: part 3: Pad.
    llvm::SmallVector<NamedAttribute, 5> paddedInputLayoutPart3Specs;
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 3)
                                                }, op.getContext())));
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("names",
                                                ArrayAttr::get({
                                                    StringAttr::get("hipad", op.getContext()),
                                                    StringAttr::get("wipad", op.getContext()),
                                                }, op.getContext())));
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Pad", op.getContext())));
    // TBD: padding parmeters.
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("parameters",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)
                                                }, op.getContext())));
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 3)
                                                }, op.getContext())));
    paddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("hi", op.getContext()),
                                                    StringAttr::get("wi", op.getContext())
                                                }, op.getContext())));

    auto paddedInputLayoutAttr = rewriter.getNamedAttr("layout",
                                                             ArrayAttr::get({
                                                                 DictionaryAttr::get(paddedInputLayoutPart1Specs, op.getContext()),
                                                                 DictionaryAttr::get(paddedInputLayoutPart2Specs, op.getContext()),
                                                                 DictionaryAttr::get(paddedInputLayoutPart3Specs, op.getContext())
                                                             }, op.getContext()));
    paddedInputAttrs.push_back(paddedInputLayoutAttr);

    // set source_layout attribute.
    auto inputSrcLayoutAttr = rewriter.getNamedAttr("source_layout", inputLayoutAttr);
    paddedInputAttrs.push_back(inputSrcLayoutAttr);
    // TBD: set output_layout attribute.
    auto paddedInputOutputLayoutAttr = rewriter.getNamedAttr("output_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("ni", op.getContext()),
                                                            StringAttr::get("ci", op.getContext()),
                                                            StringAttr::get("hipad", op.getContext()),
                                                            StringAttr::get("wipad", op.getContext())
                                                        }, op.getContext()));
    paddedInputAttrs.push_back(paddedInputOutputLayoutAttr);
    auto paddedInput = rewriter.create<miopen::TransformOp>(op.getLoc(), inputType, op.input(), paddedInputAttrs);

    // Input tensor step 2 : embedded input.
    llvm::SmallVector<int64_t, 6> embeddedInputShape;
    embeddedInputShape.set_size(inputShape.size() + 2);
    // TBD: compute embedded input shape dimensions.
    std::fill(embeddedInputShape.begin(), embeddedInputShape.end(), -1);
    auto embeddedInputMemRefType = MemRefType::get(embeddedInputShape, inputElementType);

    llvm::SmallVector<NamedAttribute, 3> embeddedInputAttrs;

    // TBD: set layout attribute.
    // TBD: part 1: Passthrough.
    llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart1Specs;
    embeddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)}, op.getContext())));
    embeddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("ni", op.getContext())}, op.getContext())));
    embeddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
    embeddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0),
                                                }, op.getContext())));
    embeddedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ni", op.getContext())
                                                }, op.getContext())));

    // TBD: part 2: Passthrough.
    llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart2Specs;
    embeddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 1)}, op.getContext())));
    embeddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("ci", op.getContext())}, op.getContext())));
    embeddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
    embeddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                }, op.getContext())));
    embeddedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ci", op.getContext())
                                                }, op.getContext())));
    // TBD: part 3: Embed.
    llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart3Specs;
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 3)
                                                }, op.getContext())));
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("names",
                                                ArrayAttr::get({
                                                    StringAttr::get("y", op.getContext()),
                                                    StringAttr::get("ho", op.getContext()),
                                                }, op.getContext())));
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Embed", op.getContext())));
    // TBD: padding parmeters.
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("parameters",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)
                                                }, op.getContext())));
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2)
                                                }, op.getContext())));
    embeddedInputLayoutPart3Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("hipad", op.getContext()),
                                                }, op.getContext())));

    // TBD: part 4: Embed.
    llvm::SmallVector<NamedAttribute, 5> embeddedInputLayoutPart4Specs;
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 4),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 5)
                                                }, op.getContext())));
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("names",
                                                ArrayAttr::get({
                                                    StringAttr::get("x", op.getContext()),
                                                    StringAttr::get("wo", op.getContext()),
                                                }, op.getContext())));
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Embed", op.getContext())));
    // TBD: embed parmeters.
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("parameters",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)
                                                }, op.getContext())));
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 3)
                                                }, op.getContext())));
    embeddedInputLayoutPart4Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("wipad", op.getContext())
                                                }, op.getContext())));

    auto embeddedInputLayoutAttr = rewriter.getNamedAttr("layout",
                                                             ArrayAttr::get({
                                                                 DictionaryAttr::get(embeddedInputLayoutPart1Specs, op.getContext()),
                                                                 DictionaryAttr::get(embeddedInputLayoutPart2Specs, op.getContext()),
                                                                 DictionaryAttr::get(embeddedInputLayoutPart3Specs, op.getContext()),
                                                                 DictionaryAttr::get(embeddedInputLayoutPart4Specs, op.getContext())
                                                             }, op.getContext()));
    embeddedInputAttrs.push_back(embeddedInputLayoutAttr);


    // TBD: set intermediate_layout attribute.
    auto embeddedInputImmLayoutAttr = rewriter.getNamedAttr("intermediate_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("ni", op.getContext()),
                                                            StringAttr::get("ci", op.getContext()),
                                                            StringAttr::get("hipad", op.getContext()),
                                                            StringAttr::get("wipad", op.getContext())
                                                        }, op.getContext()));
    embeddedInputAttrs.push_back(embeddedInputImmLayoutAttr);
    // TBD: set output_layout attribute.
    auto embeddedInputOutputLayoutAttr = rewriter.getNamedAttr("output_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("ni", op.getContext()),
                                                            StringAttr::get("ci", op.getContext()),
                                                            StringAttr::get("y", op.getContext()),
                                                            StringAttr::get("ho", op.getContext()),
                                                            StringAttr::get("x", op.getContext()),
                                                            StringAttr::get("wo", op.getContext())
                                                        }, op.getContext()));
    embeddedInputAttrs.push_back(embeddedInputOutputLayoutAttr);
    auto embeddedInput = rewriter.create<miopen::TransformOp>(op.getLoc(), embeddedInputMemRefType, ArrayRef<Value>(paddedInput), embeddedInputAttrs);

    // Input tensor step 3: transformed input.
    llvm::SmallVector<int64_t, 2> transformedInputShape;
    transformedInputShape.set_size(inputShape.size() - 2);
    // TBD: compute transformed input shape dimensions.
    std::fill(transformedInputShape.begin(), transformedInputShape.end(), -1);
    auto transformedInputMemRefType = MemRefType::get(transformedInputShape, inputElementType);

    llvm::SmallVector<NamedAttribute, 3> transformedInputAttrs;

    // TBD: set layout attribute.
    // TBD: Part 1: Merge.
    llvm::SmallVector<NamedAttribute, 5> transformedInputLayoutPart1Specs;
    transformedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)}, op.getContext())));
    transformedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmK", op.getContext())}, op.getContext())));
    transformedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Merge", op.getContext())));
    transformedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 1),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 2),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 4)
                                                }, op.getContext())));
    transformedInputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ci", op.getContext()),
                                                    StringAttr::get("y", op.getContext()),
                                                    StringAttr::get("x", op.getContext())
                                                }, op.getContext())));

    // TBD: Part 2: Merge.
    llvm::SmallVector<NamedAttribute, 5> transformedInputLayoutPart2Specs;
    transformedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 1)}, op.getContext())));
    transformedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmN", op.getContext())}, op.getContext())));
    transformedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Merge", op.getContext())));
    transformedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                ArrayAttr::get({
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 0),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 3),
                                                    IntegerAttr::get(IntegerType::get(32, op.getContext()), 5)
                                                }, op.getContext())));
    transformedInputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_names",
                                                ArrayAttr::get({
                                                    StringAttr::get("ni", op.getContext()),
                                                    StringAttr::get("ho", op.getContext()),
                                                    StringAttr::get("wo", op.getContext())
                                                }, op.getContext())));

    auto transformedInputLayoutAttr = rewriter.getNamedAttr("layout",
                                                             ArrayAttr::get({
                                                                 DictionaryAttr::get(transformedInputLayoutPart1Specs, op.getContext()),
                                                                 DictionaryAttr::get(transformedInputLayoutPart2Specs, op.getContext())
                                                             }, op.getContext()));
    transformedInputAttrs.push_back(transformedInputLayoutAttr);

    // TBD: set intermediate_layout attribute.
    auto transformedInputImmLayoutAttr = rewriter.getNamedAttr("intermediate_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("ni", op.getContext()),
                                                            StringAttr::get("ci", op.getContext()),
                                                            StringAttr::get("y", op.getContext()),
                                                            StringAttr::get("ho", op.getContext()),
                                                            StringAttr::get("x", op.getContext()),
                                                            StringAttr::get("wo", op.getContext())
                                                        }, op.getContext()));
    transformedInputAttrs.push_back(transformedInputImmLayoutAttr);
    // set output_layout attribute.
    auto transformedInputOutputLayoutAttr = rewriter.getNamedAttr("output_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("gemmK", op.getContext()),
                                                            StringAttr::get("gemmN", op.getContext()),
                                                        }, op.getContext()));
    transformedInputAttrs.push_back(transformedInputOutputLayoutAttr);

    // set gridwise_gemm_argument_pos attribute.
    auto inputGridwiseGemmArgPosAttr = rewriter.getNamedAttr("gridwise_gemm_argument_position", 
                                                             IntegerAttr::get(IntegerType::get(32, op.getContext()), 1));
    transformedInputAttrs.push_back(inputGridwiseGemmArgPosAttr);
    auto gemmB = rewriter.create<miopen::TransformOp>(op.getLoc(), transformedInputMemRefType, ArrayRef<Value>(embeddedInput), transformedInputAttrs);


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
    // Weight tensor transformation:
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
            kDim = IntegerAttr::get(IntegerType::get(32, op.getContext()), i);
            kDimName = StringAttr::get(strAttr.getValue(), op.getContext());
          } else {
            nonKDims.push_back(IntegerAttr::get(IntegerType::get(32, op.getContext()), i));
            nonKDimNames.push_back(StringAttr::get(strAttr.getValue(), op.getContext()));
          }
        }
      }
 
      // Part 1: Passthrough.
      llvm::SmallVector<NamedAttribute, 5> transformedOutputLayoutPart1Specs;
      transformedOutputLayoutPart1Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 0)}, op.getContext())));
      transformedOutputLayoutPart1Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmM", op.getContext())}, op.getContext())));
      transformedOutputLayoutPart1Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("PassThrough", op.getContext())));
      transformedOutputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                  ArrayAttr::get({kDim}, op.getContext())));
      transformedOutputLayoutPart1Specs.push_back(rewriter.getNamedAttr("source_names",
                                                  ArrayAttr::get({kDimName}, op.getContext())));

      // Part 2: Merge.
      llvm::SmallVector<NamedAttribute, 5> transformedOutputLayoutPart2Specs;
      transformedOutputLayoutPart2Specs.push_back(rewriter.getNamedAttr("dimensions", ArrayAttr::get({IntegerAttr::get(IntegerType::get(32, op.getContext()), 1)}, op.getContext())));
      transformedOutputLayoutPart2Specs.push_back(rewriter.getNamedAttr("names", ArrayAttr::get({StringAttr::get("gemmN", op.getContext())}, op.getContext())));
      transformedOutputLayoutPart2Specs.push_back(rewriter.getNamedAttr("transformation", StringAttr::get("Merge", op.getContext())));
      transformedOutputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_dimensions",
                                                  ArrayAttr::get(ArrayRef<Attribute>(nonKDims.begin(), nonKDims.end()), op.getContext())));
      transformedOutputLayoutPart2Specs.push_back(rewriter.getNamedAttr("source_names",
                                                  ArrayAttr::get(ArrayRef<Attribute>(nonKDimNames.begin(), nonKDimNames.end()), op.getContext())));

      auto transformedOutputLayoutAttr = rewriter.getNamedAttr("layout",
                                                               ArrayAttr::get({
                                                                   DictionaryAttr::get(transformedOutputLayoutPart1Specs, op.getContext()),
                                                                   DictionaryAttr::get(transformedOutputLayoutPart2Specs, op.getContext())
                                                               }, op.getContext()));
      transformedOutputAttrs.push_back(transformedOutputLayoutAttr);
    }

    // set source_layout attribute.
    auto outputSrcLayoutAttr = rewriter.getNamedAttr("source_layout", outputLayoutAttr);
    transformedOutputAttrs.push_back(outputSrcLayoutAttr);
    // set output_layout attribute.
    auto transformedOutputOutputLayoutAttr = rewriter.getNamedAttr("output_layout",
                                                        ArrayAttr::get({
                                                            StringAttr::get("gemmM", op.getContext()),
                                                            StringAttr::get("gemmN", op.getContext()),
                                                        }, op.getContext()));
    transformedOutputAttrs.push_back(transformedOutputOutputLayoutAttr);

    // set gridwise_gemm_argument_pos attribute.
    auto outputGridwiseGemmArgPosAttr = rewriter.getNamedAttr("gridwise_gemm_argument_position", 
                                                             IntegerAttr::get(IntegerType::get(32, op.getContext()), 2));
    transformedOutputAttrs.push_back(outputGridwiseGemmArgPosAttr);
    auto gemmC = rewriter.create<miopen::TransformOp>(op.getLoc(), transformedOutputMemRefType, op.output(), transformedOutputAttrs);

    // Emit miopen.gridwise_gemm op.
    rewriter.create<miopen::GridwiseGemmOp>(op.getLoc(), gemmA, gemmB, gemmC);

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
