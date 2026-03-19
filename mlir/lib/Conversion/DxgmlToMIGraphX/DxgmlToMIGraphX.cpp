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
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

static FailureOr<SmallVector<int64_t>>
extractI64ValuesFromConstantAttr(dxgml::ConstantAttr attr, Operation *op,
                                 StringRef name) {
  SmallVector<int64_t> values;
  values.reserve(attr.getValue().size());

  for (Attribute element : attr.getValue()) {
    auto intElement = dyn_cast<IntegerAttr>(element);
    if (!intElement) {
      op->emitOpError() << "attribute '" << name
                        << "' expects #DxGML.ConstantValue with integer "
                           "elements";
      return failure();
    }
    values.push_back(intElement.getInt());
  }

  if (values.empty()) {
    op->emitOpError() << "attribute '" << name
                      << "' expects at least one integer element";
    return failure();
  }

  return values;
}

static FailureOr<int64_t>
extractSingleI64FromConstantAttr(dxgml::ConstantAttr attr, Operation *op,
                                 StringRef name) {
  FailureOr<SmallVector<int64_t>> values =
      extractI64ValuesFromConstantAttr(attr, op, name);
  if (failed(values))
    return failure();

  if (values->size() != 1) {
    op->emitOpError() << "attribute '" << name
                      << "' expects exactly one integer element";
    return failure();
  }

  return values->front();
}

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
struct ConvertConvolutionOp : public OpConversionPattern<dxgml::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Extract attributes from #dxgml.constant_value.
    FailureOr<int64_t> groupCount =
        extractSingleI64FromConstantAttr(op.getGroupCount(), op, "group_count");
    if (failed(groupCount))
      return failure();

    FailureOr<SmallVector<int64_t>> strides =
        extractI64ValuesFromConstantAttr(op.getStrides(), op, "strides");
    if (failed(strides))
      return failure();

    FailureOr<SmallVector<int64_t>> dilations =
        extractI64ValuesFromConstantAttr(op.getDilations(), op, "dilations");
    if (failed(dilations))
      return failure();

    FailureOr<SmallVector<int64_t>> startPadding =
        extractI64ValuesFromConstantAttr(op.getStartPadding(), op,
                                         "start_padding");
    if (failed(startPadding))
      return failure();

    FailureOr<SmallVector<int64_t>> endPadding =
        extractI64ValuesFromConstantAttr(op.getEndPadding(), op,
                                         "end_padding");
    if (failed(endPadding))
      return failure();
    
    // Combine start and end padding: [start[0], start[1], end[0], end[1]]
    SmallVector<int64_t> padding;
    padding.reserve(startPadding->size() + endPadding->size());
    padding.append(startPadding->begin(), startPadding->end());
    padding.append(endPadding->begin(), endPadding->end());

    int64_t groupCountValue = *groupCount;
    (void)groupCountValue;
    (void)strides;
    (void)dilations;
    (void)padding;
    
    // Create MIGraphX convolution
    // TODO: Convert attributes to MIGraphX format
    // rewriter.replaceOpWithNewOp<migraphx::ConvolutionOp>(
    //     op, adaptor.getInput(), adaptor.getFilter(),
    //     padding, strides, dilations, groupCount);
    
    return failure(); // TODO: Implement
  }
};

/// Convert dxgml_op.relu to migraphx.relu.
struct ConvertReluOp : public OpConversionPattern<dxgml::ReluOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml::ReluOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    
    // Simply replace with MIGraphX relu
    rewriter.replaceOpWithNewOp<migraphx::ReluOp>(
        op, adaptor.getInput());
    
    return success();
  }
};

/// Convert dxgml_op.add to migraphx.add.
struct ConvertAddOp : public OpConversionPattern<dxgml::AddOp> {
  using OpConversionPattern::OpConversionPattern;
  
  LogicalResult matchAndRewrite(
      dxgml::AddOp op, OpAdaptor adaptor,
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
    target.addIllegalDialect<dxgml::DxgmlDialect>();
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
