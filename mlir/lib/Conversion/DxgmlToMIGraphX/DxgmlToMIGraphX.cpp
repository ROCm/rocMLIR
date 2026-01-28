//===- DxgmlToMIGraphX.cpp - DXGML to MIGraphX conversion ----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion from DXGML dialect to MIGraphX dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/DxgmlToMIGraphX/DxgmlToMIGraphX.h"

#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

/// Convert DXGML types to MIGraphX types.
class DxgmlToMIGraphXTypeConverter : public TypeConverter {
public:
  DxgmlToMIGraphXTypeConverter() {
    // Convert DXGML tensor to MIGraphX shaped type
    addConversion([](dxgml::TensorType type) -> Type {
      // Get shape and element type
      auto shape = type.getShape();
      Type elemType = type.getElementType();
      
      // Convert element type
      Type migraphxElemType = convertElementType(elemType);
      if (!migraphxElemType)
        return nullptr;
      
      // Calculate standard row-major strides
      SmallVector<int64_t> strides;
      int64_t stride = 1;
      for (int i = shape.size() - 1; i >= 0; --i) {
        strides.insert(strides.begin(), stride);
        stride *= shape[i];
      }
      
      // Create MIGraphX shaped type
      return migraphx::MIXRShapedType::get(shape, strides, migraphxElemType);
    });
    
    // Convert DXGML scalar types to MLIR builtin types
    addConversion([](dxgml::Float16Type type) { return Float16Type::get(type.getContext()); });
    addConversion([](dxgml::Float32Type type) { return Float32Type::get(type.getContext()); });
    addConversion([](dxgml::Float64Type type) { return Float64Type::get(type.getContext()); });
    addConversion([](dxgml::BFloat16Type type) { return BFloat16Type::get(type.getContext()); });
    
    addConversion([](dxgml::Int8Type type) { return IntegerType::get(type.getContext(), 8, IntegerType::Signed); });
    addConversion([](dxgml::Int16Type type) { return IntegerType::get(type.getContext(), 16, IntegerType::Signed); });
    addConversion([](dxgml::Int32Type type) { return IntegerType::get(type.getContext(), 32, IntegerType::Signed); });
    addConversion([](dxgml::Int64Type type) { return IntegerType::get(type.getContext(), 64, IntegerType::Signed); });
    
    // Default: pass through unchanged
    addConversion([](Type type) { return type; });
  }

private:
  static Type convertElementType(Type type) {
    if (auto ft = dyn_cast<dxgml::Float16Type>(type))
      return Float16Type::get(type.getContext());
    if (auto ft = dyn_cast<dxgml::Float32Type>(type))
      return Float32Type::get(type.getContext());
    if (auto ft = dyn_cast<dxgml::BFloat16Type>(type))
      return BFloat16Type::get(type.getContext());
    if (auto it = dyn_cast<dxgml::Int8Type>(type))
      return IntegerType::get(type.getContext(), 8, IntegerType::Signed);
    // Add more conversions as needed
    return type;
  }
};

//===----------------------------------------------------------------------===//
// Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert dxgml_op.convolution to migraphx.convolution.
struct ConvertConvolutionOp : public OpConversionPattern<dxgml_op::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml_op::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Extract attributes
    auto groupCount = op.getGroupCount().getInt();
    
    // Extract strides, dilations, padding from DenseIntegerElements
    auto stridesAttr = op.getStrides();
    auto dilationsAttr = op.getDilations();
    auto startPaddingAttr = op.getStartPadding();
    auto endPaddingAttr = op.getEndPadding();
    
    // Combine start and end padding: [start[0], start[1], end[0], end[1]]
    SmallVector<int64_t> padding;
    for (int64_t p : startPaddingAttr.getValue())
      padding.push_back(p);
    for (int64_t p : endPaddingAttr.getValue())
      padding.push_back(p);
    
    // Create MIGraphX convolution
    // TODO: Convert attributes to MIGraphX format
    // rewriter.replaceOpWithNewOp<migraphx::ConvolutionOp>(
    //     op, adaptor.getInput(), adaptor.getFilter(),
    //     padding, strides, dilations, groupCount);
    
    return failure(); // TODO: Implement
  }
};

/// Convert dxgml_op.relu to migraphx.relu.
struct ConvertReluOp : public OpConversionPattern<dxgml_op::ReluOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml_op::ReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Simply replace with MIGraphX relu
    rewriter.replaceOpWithNewOp<migraphx::ReluOp>(
        op, adaptor.getInput());
    
    return success();
  }
};

/// Convert dxgml_op.add to migraphx.add.
struct ConvertAddOp : public OpConversionPattern<dxgml_op::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml_op::AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    rewriter.replaceOpWithNewOp<migraphx::AddOp>(
        op, adaptor.getLhs(), adaptor.getRhs());
    
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertDxgmlToMIGraphXPass
    : public PassWrapper<ConvertDxgmlToMIGraphXPass, OperationPass<ModuleOp>> {
  
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertDxgmlToMIGraphXPass)
  
  StringRef getArgument() const final { return "convert-dxgml-to-migraphx"; }
  StringRef getDescription() const final {
    return "Convert DXGML dialect to MIGraphX dialect";
  }
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect>();
  }
  
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();
    
    // Set up type converter
    DxgmlToMIGraphXTypeConverter typeConverter;
    
    // Set up conversion target
    ConversionTarget target(*context);
    target.addIllegalDialect<dxgml::DxgmlDialect, dxgml_op::DxgmlOpDialect>();
    target.addLegalDialect<migraphx::MIGraphXDialect>();
    target.addLegalDialect<func::FuncDialect>();
    
    // Set up conversion patterns
    RewritePatternSet patterns(context);
    populateDxgmlToMIGraphXConversionPatterns(typeConverter, patterns);
    
    // Apply conversion
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void mlir::populateDxgmlToMIGraphXConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  
  patterns.add<
      ConvertConvolutionOp,
      ConvertReluOp,
      ConvertAddOp
      // TODO: Add more operation conversions
  >(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertDxgmlToMIGraphXPass() {
  return std::make_unique<ConvertDxgmlToMIGraphXPass>();
}
