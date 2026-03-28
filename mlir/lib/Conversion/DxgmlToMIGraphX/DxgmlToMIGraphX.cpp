//===- DxgmlToMIGraphX.cpp - DXGML to MIGraphX conversion ----------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion from DXGML dialect to MIGraphX dialect.
// The lowering strategy is:
//   dxgml.module / dxgml.entry_point  ->  func.func
//   dxgml_op.*                         ->  migraphx.*  (where a direct mapping exists)
//   dxgml_op.*                         ->  left as-is  (complex ops without a direct map)
// After this pass the module contains standard func.func ops with migraphx.*
// ops in their bodies, which can then be lowered via the existing MIGraphX
// pipeline (MIGraphXToTosa -> TosaToRock -> GPU/ROCDL/binary).
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/DxgmlToMIGraphX/DxgmlToMIGraphX.h"

#include "mlir/Dialect/Dxgml/IR/Dxgml.h"
#include "mlir/Dialect/Dxgml/DxgmlOp/IR/DxgmlOp.h"
#include "mlir/Dialect/MIGraphX/IR/MIGraphX.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Type Conversion Helpers
//===----------------------------------------------------------------------===//

/// Convert a DXML element type (dxgml::Float16Type etc.) to a standard MLIR
/// type.
static Type convertDxgmlElementType(Type type) {
  MLIRContext *ctx = type.getContext();
  if (isa<dxgml::Float16Type>(type))   return Float16Type::get(ctx);
  if (isa<dxgml::Float32Type>(type))   return Float32Type::get(ctx);
  if (isa<dxgml::Float64Type>(type))   return Float64Type::get(ctx);
  if (isa<dxgml::BFloat16Type>(type))  return BFloat16Type::get(ctx);
  if (isa<dxgml::Int8Type>(type))
    return IntegerType::get(ctx, 8,  IntegerType::Signed);
  if (isa<dxgml::UInt8Type>(type))
    return IntegerType::get(ctx, 8,  IntegerType::Unsigned);
  if (isa<dxgml::Int4Type>(type))
    return IntegerType::get(ctx, 4,  IntegerType::Signed);
  if (isa<dxgml::UInt4Type>(type))
    return IntegerType::get(ctx, 4,  IntegerType::Unsigned);
  if (isa<dxgml::Int16Type>(type))
    return IntegerType::get(ctx, 16, IntegerType::Signed);
  if (isa<dxgml::UInt16Type>(type))
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  if (isa<dxgml::Int32Type>(type))
    return IntegerType::get(ctx, 32, IntegerType::Signed);
  if (isa<dxgml::Int64Type>(type))
    return IntegerType::get(ctx, 64, IntegerType::Signed);
  return type; // pass through unknown types unchanged
}

/// Convert a dxgml::DxgmlTensorType to migraphx::MIXRShapedType with
/// standard row-major strides.
static migraphx::MIXRShapedType
convertTensorType(dxgml::DxgmlTensorType type) {
  auto shape = type.getShape();
  Type elemType = convertDxgmlElementType(type.getElementType());

  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int i = (int)shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] != ShapedType::kDynamic)
      stride *= shape[i];
  }
  return migraphx::MIXRShapedType::get(shape, strides, elemType);
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class DxgmlToMIGraphXTypeConverter : public TypeConverter {
public:
  DxgmlToMIGraphXTypeConverter() {
    // Conversions are tried in LIFO order; add the catch-all FIRST so it
    // is tried LAST (letting the specific dxgml converters take priority).
    addConversion([](Type type) -> Type { return type; });

    // Source materialization: dxgml.tensor -> migraphx.shaped
    // (needed when a converted op receives a dxgml value from an unconverted op)
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) return nullptr;
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // Target materialization: migraphx.shaped -> dxgml.tensor
    // (needed when an unconverted op receives a migraphx value from a converted op)
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1) return nullptr;
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    });

    // dxgml scalar element types -> builtin MLIR types
    addConversion([](dxgml::Int64Type t) -> Type {
      return IntegerType::get(t.getContext(), 64, IntegerType::Signed);
    });
    addConversion([](dxgml::Int32Type t) -> Type {
      return IntegerType::get(t.getContext(), 32, IntegerType::Signed);
    });
    addConversion([](dxgml::Int16Type t) -> Type {
      return IntegerType::get(t.getContext(), 16, IntegerType::Signed);
    });
    addConversion([](dxgml::Int8Type  t) -> Type {
      return IntegerType::get(t.getContext(), 8,  IntegerType::Signed);
    });
    addConversion([](dxgml::UInt8Type t) -> Type {
      return IntegerType::get(t.getContext(), 8,  IntegerType::Unsigned);
    });
    addConversion([](dxgml::Int4Type t) -> Type {
      return IntegerType::get(t.getContext(), 4, IntegerType::Signed);
    });
    addConversion([](dxgml::UInt4Type t) -> Type {
      return IntegerType::get(t.getContext(), 4, IntegerType::Unsigned);
    });
    addConversion([](dxgml::Int16Type t) -> Type {
      return IntegerType::get(t.getContext(), 16, IntegerType::Signed);
    });
    addConversion([](dxgml::UInt16Type t) -> Type {
      return IntegerType::get(t.getContext(), 16, IntegerType::Unsigned);
    });
    addConversion([](dxgml::BFloat16Type t) -> Type { return BFloat16Type::get(t.getContext()); });
    addConversion([](dxgml::Float64Type  t) -> Type { return Float64Type::get(t.getContext()); });
    addConversion([](dxgml::Float32Type  t) -> Type { return Float32Type::get(t.getContext()); });
    addConversion([](dxgml::Float16Type  t) -> Type { return Float16Type::get(t.getContext()); });
    // dxgml.tensor<...> -> migraphx shaped type (highest priority — tried first)
    addConversion([](dxgml::DxgmlTensorType type) -> Type {
      return convertTensorType(type);
    });

  }
};

//===----------------------------------------------------------------------===//
// Utility: extract int64 values from an optional DenseIntegerElementsAttr
//===----------------------------------------------------------------------===//
static SmallVector<int64_t>
extractI64Array(std::optional<dxgml::DenseIntegerElementsAttr> optAttr,
                ArrayRef<int64_t> defaults) {
  if (!optAttr)
    return SmallVector<int64_t>(defaults.begin(), defaults.end());
  return SmallVector<int64_t>(optAttr->getValue().begin(),
                              optAttr->getValue().end());
}

//===----------------------------------------------------------------------===//
// Op Conversions: dxgml_op.* -> migraphx.*
//===----------------------------------------------------------------------===//

/// Re-wrap func.return so that its operand types match the converted types.
struct ConvertFuncReturnOp : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant / Null
//===----------------------------------------------------------------------===//

/// dxgml_op.constant -> migraphx.literal (when the value is a dense elements
/// attr) or left in place by marking it legal in the target.
/// For constant_resource references (external weights), we emit a literal with
/// a zero-splat placeholder — downstream passes must resolve these externally.
/// In practice for parse/validate flows this is sufficient; full compilation
/// needs weight loading which is outside this pass's scope.
struct ConvertConstantOp : public OpConversionPattern<dxgml_op::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::ConstantOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getResult().getType());
    if (!outType) return failure();

    auto shaped = dyn_cast<migraphx::MIXRShapedType>(outType);
    if (!shaped) return failure();

    // If the value is already a dense elements attribute, use it directly.
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      // Re-encode the dense attr with the converted element type if needed.
      auto tensorTy = RankedTensorType::get(shaped.getShape(),
                                            shaped.getElementType());
      ElementsAttr newDense = denseAttr.isSplat()
          ? DenseElementsAttr::get(tensorTy, denseAttr.getSplatValue<Attribute>())
          : denseAttr;
      rewriter.replaceOpWithNewOp<migraphx::LiteralOp>(op, outType, newDense);
      return success();
    }

    // For constant_resource attrs (external weight references): emit a
    // zero-splat literal as a placeholder. The actual data would be loaded
    // by a later weight-loading pass or runtime.
    auto tensorTy = RankedTensorType::get(shaped.getShape(),
                                          shaped.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(tensorTy);
    if (!zeroAttr) return failure();
    rewriter.replaceOpWithNewOp<migraphx::LiteralOp>(op, outType,
                                                      cast<ElementsAttr>(zeroAttr));
    return success();
  }
};

/// dxgml_op.null_ptr — represents an absent optional tensor.
/// Map to migraphx.literal of an empty tensor (shape [0]) as a sentinel.
/// This allows optional-input migraphx ops to receive a typed null value.
/// When the result type is !dxgml.null (non-tensor), we cannot convert to
/// migraphx — return failure so the op stays legal (pass-through).
struct ConvertNullPtrOp : public OpConversionPattern<dxgml_op::NullPtrOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::NullPtrOp op, OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getResult().getType());
    if (!outType) return failure();

    // If the result is not a migraphx shaped type (e.g. !dxgml.null), we
    // cannot produce a migraphx literal — leave the op unchanged.
    auto shaped = dyn_cast<migraphx::MIXRShapedType>(outType);
    if (!shaped) return failure();
    auto zeroAttr = rewriter.getZeroAttr(
        RankedTensorType::get(shaped.getShape(), shaped.getElementType()));
    if (!zeroAttr) return failure();
    rewriter.replaceOpWithNewOp<migraphx::LiteralOp>(op, outType,
                                                      cast<ElementsAttr>(zeroAttr));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Shape Operations
//===----------------------------------------------------------------------===//

/// dxgml_op.reshape -> migraphx.reshape
struct ConvertReshapeOp : public OpConversionPattern<dxgml_op::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getResult().getType());
    if (!outType) return failure();
    auto shaped = dyn_cast<migraphx::MIXRShapedType>(outType);
    if (!shaped) return failure();

    // migraphx.reshape requires a static dims attr from the output shape.
    SmallVector<int64_t> dims(shaped.getShape().begin(), shaped.getShape().end());
    rewriter.replaceOpWithNewOp<migraphx::ReshapeOp>(
        op, outType, adaptor.getInputTensor(),
        rewriter.getI64ArrayAttr(dims));
    return success();
  }
};

/// dxgml_op.transpose -> migraphx.transpose
struct ConvertTransposeOp : public OpConversionPattern<dxgml_op::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    // Extract permutation (default: reverse all dims).
    SmallVector<int64_t> perm;
    if (auto permAttr = op.getPermutation()) {
      perm.assign(permAttr->getValue().begin(), permAttr->getValue().end());
    } else {
      auto inShaped = dyn_cast<ShapedType>(adaptor.getInput().getType());
      if (!inShaped) return failure();
      int64_t rank = inShaped.getRank();
      for (int64_t i = rank - 1; i >= 0; --i) perm.push_back(i);
    }
    rewriter.replaceOpWithNewOp<migraphx::TransposeOp>(
        op, outType, adaptor.getInput(),
        rewriter.getI64ArrayAttr(perm));
    return success();
  }
};

/// dxgml_op.slice -> migraphx.slice
struct ConvertSliceOp : public OpConversionPattern<dxgml_op::SliceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    auto inShaped = dyn_cast<ShapedType>(adaptor.getInput().getType());
    if (!inShaped) return failure();
    int64_t rank = inShaped.getRank();

    // Build axes (default all dims), starts, ends.
    SmallVector<int64_t> axes;
    if (auto axesAttr = op.getAxes()) {
      axes.assign(axesAttr->getValue().begin(), axesAttr->getValue().end());
    } else {
      for (int64_t i = 0; i < rank; ++i) axes.push_back(i);
    }

    auto starts = SmallVector<int64_t>(op.getStarts().getValue().begin(),
                                       op.getStarts().getValue().end());
    auto ends   = SmallVector<int64_t>(op.getEnds().getValue().begin(),
                                       op.getEnds().getValue().end());

    rewriter.replaceOpWithNewOp<migraphx::SliceOp>(
        op, outType, adaptor.getInput(),
        rewriter.getI64ArrayAttr(axes),
        rewriter.getI64ArrayAttr(ends),
        rewriter.getI64ArrayAttr(starts));
    return success();
  }
};

/// dxgml_op.concat -> migraphx... no direct concat in MIGraphX.
/// Lower by chaining reshape+multibroadcast is complex; for now, fold to
/// a passthrough by marking this as handled via a simple identity if single
/// input, otherwise emit migraphx.reshape on the first input as an approximation.
/// For full correctness, this would need a custom lowering.
/// We use the migraphx reshape of output shape as a placeholder.
struct ConvertConcatOp : public OpConversionPattern<dxgml_op::ConcatOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::ConcatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    auto shaped = dyn_cast<migraphx::MIXRShapedType>(outType);
    if (!shaped) return failure();

    if (adaptor.getInputs().empty()) return failure();

    // Single input: reshape to output shape.
    if (adaptor.getInputs().size() == 1) {
      SmallVector<int64_t> dims(shaped.getShape().begin(), shaped.getShape().end());
      rewriter.replaceOpWithNewOp<migraphx::ReshapeOp>(
          op, outType, adaptor.getInputs()[0],
          rewriter.getI64ArrayAttr(dims));
      return success();
    }

    // Multi-input concat: not directly supported. Return failure so it stays
    // as an unhandled op (partial conversion allows this).
    return failure();
  }
};

/// dxgml_op.gather -> migraphx.gather (if the axis attr maps)
struct ConvertGatherOp : public OpConversionPattern<dxgml_op::GatherOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::GatherOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    // migraphx has no direct gather — not yet converted.
    return failure();
  }
};

/// dxgml_op.split -> not directly in MIGraphX; skip for now.

/// dxgml_op.depth_to_space -> migraphx.reshape + migraphx.transpose + migraphx.reshape
///
/// DepthToSpace (pixel shuffle) rearranges data from the depth (channel) dimension
/// into spatial dimensions. For NCHW input [N, C, H, W] with block_size bs:
///   Output: [N, C/(bs*bs), H*bs, W*bs]
///
/// CRD (column_row_depth) order:
///   1. reshape [N, C, H, W] → [N, C_out, bs, bs, H, W]
///   2. transpose perm [0, 1, 4, 2, 5, 3] → [N, C_out, H, bs, W, bs]
///   3. reshape → [N, C_out, H*bs, W*bs]
///
/// DCR (depth_column_row) order:
///   1. reshape [N, C, H, W] → [N, bs, bs, C_out, H, W]
///   2. transpose perm [0, 3, 4, 1, 5, 2] → [N, C_out, H, bs, W, bs]
///   3. reshape → [N, C_out, H*bs, W*bs]
struct ConvertDepthToSpaceOp
    : public OpConversionPattern<dxgml_op::DepthToSpaceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::DepthToSpaceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    auto outShaped = dyn_cast<migraphx::MIXRShapedType>(outType);
    if (!outShaped) return failure();

    Value input = adaptor.getInput();
    auto inShaped = dyn_cast<migraphx::MIXRShapedType>(input.getType());
    if (!inShaped || inShaped.getRank() != 4) return failure();

    ArrayRef<int64_t> inShape = inShaped.getShape();
    int64_t N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
    int64_t bs = cast<dxgml::IntegerAttr>(op.getBlockSize()).getInt();
    if (bs <= 0) return failure();
    int64_t C_out = C / (bs * bs);

    Location loc = op.getLoc();
    Type elemType = inShaped.getElementType();

    // Compute standard row-major strides for a given shape.
    auto makeStrides = [](ArrayRef<int64_t> shape) -> SmallVector<int64_t> {
      SmallVector<int64_t> strides(shape.size());
      int64_t s = 1;
      for (int i = (int)shape.size() - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
      }
      return strides;
    };

    bool isCRD = (op.getDepthSpaceOrder().getValue() ==
                  dxgml_op::DepthSpaceOrder::column_row_depth);

    // Step 1: Reshape input to 6D expanded form.
    SmallVector<int64_t> expandShape;
    SmallVector<int64_t> transpPerm;
    if (isCRD) {
      expandShape = {N, C_out, bs, bs, H, W};
      transpPerm  = {0, 1, 4, 2, 5, 3};
    } else {
      expandShape = {N, bs, bs, C_out, H, W};
      transpPerm  = {0, 3, 4, 1, 5, 2};
    }

    auto expandType = migraphx::MIXRShapedType::get(
        expandShape, makeStrides(expandShape), elemType);
    Value expanded = migraphx::ReshapeOp::create(
        rewriter, loc, expandType, input,
        rewriter.getI64ArrayAttr(expandShape)).getOutput();

    // Step 2: Transpose to [N, C_out, H, bs, W, bs].
    SmallVector<int64_t> transpShape(6);
    for (int i = 0; i < 6; ++i)
      transpShape[i] = expandShape[transpPerm[i]];
    auto transpType = migraphx::MIXRShapedType::get(
        transpShape, makeStrides(transpShape), elemType);
    Value transposed = migraphx::TransposeOp::create(
        rewriter, loc, transpType, expanded,
        rewriter.getI64ArrayAttr(transpPerm)).getOutput();

    // Step 3: Reshape to final output [N, C_out, H*bs, W*bs].
    rewriter.replaceOpWithNewOp<migraphx::ReshapeOp>(
        op, outType, transposed,
        rewriter.getI64ArrayAttr(outShaped.getShape()));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quantization
//===----------------------------------------------------------------------===//

/// dxgml_op.dequantize_linear -> migraphx.dequantizelinear
struct ConvertDequantizeLinearOp
    : public OpConversionPattern<dxgml_op::DequantizeLinearOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::DequantizeLinearOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    // migraphx.dequantizelinear(input, scale, optional zero_point)
    Value zeroPoint = adaptor.getZeroPoint();
    if (zeroPoint && isa<dxgml::NullType>(zeroPoint.getType()))
      zeroPoint = Value{};
    rewriter.replaceOpWithNewOp<migraphx::DeQuantizeLinearOp>(
        op, outType, adaptor.getInput(), adaptor.getScale(), zeroPoint);
    return success();
  }
};

/// dxgml_op.quantize_linear -> migraphx.quantizelinear
struct ConvertQuantizeLinearOp
    : public OpConversionPattern<dxgml_op::QuantizeLinearOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::QuantizeLinearOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    Value zeroPoint = adaptor.getZeroPoint();
    if (zeroPoint && isa<dxgml::NullType>(zeroPoint.getType()))
      zeroPoint = Value{};
    rewriter.replaceOpWithNewOp<migraphx::QuantizeLinearOp>(
        op, outType, adaptor.getInput(), adaptor.getScale(), zeroPoint);
    return success();
  }
};

/// dxgml_op.cast -> migraphx.convert
struct ConvertCastOp : public OpConversionPattern<dxgml_op::CastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    rewriter.replaceOpWithNewOp<migraphx::ConvertOp>(
        op, outType, adaptor.getInput());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pooling / Normalization
//===----------------------------------------------------------------------===//

/// dxgml_op.softmax -> migraphx.softmax
struct ConvertSoftmaxOp : public OpConversionPattern<dxgml_op::SoftmaxOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::SoftmaxOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();
    // Default axis = -1 (last dim). migraphx.softmax takes an i64 axis attr.
    int64_t axis = -1;
    if (auto axisAttr = op.getAxis()) {
      if (auto intAttr = dyn_cast<dxgml::IntegerAttr>(*axisAttr))
        axis = intAttr.getInt();
    }
    // Normalize negative axis.
    if (auto shaped = dyn_cast<ShapedType>(adaptor.getInput().getType()))
      if (axis < 0) axis += shaped.getRank();

    rewriter.replaceOpWithNewOp<migraphx::SoftmaxOp>(
        op, outType, adaptor.getInput(),
        rewriter.getI64IntegerAttr(axis));
    return success();
  }
};

/// dxgml_op.reduce -> migraphx.reduce_sum / reduce_max / reduce_mean
struct ConvertReduceOp : public OpConversionPattern<dxgml_op::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::ReduceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    // Get the reduction axes (default: all dims if not set).
    SmallVector<int64_t> axes;
    if (auto axesAttr = op.getAxes()) {
      axes.assign(axesAttr->getValue().begin(), axesAttr->getValue().end());
    } else {
      auto inShaped = dyn_cast<ShapedType>(adaptor.getInput().getType());
      if (!inShaped) return failure();
      for (int64_t i = 0; i < inShaped.getRank(); ++i) axes.push_back(i);
    }

    // Map reduce function enum to the appropriate MIGraphX op.
    auto fn = op.getReductionFunction().getValue();
    if (fn == dxgml_op::ReduceFunction::sum) {
      rewriter.replaceOpWithNewOp<migraphx::ReduceSumOp>(
          op, outType, adaptor.getInput(),
          rewriter.getI64ArrayAttr(axes));
    } else if (fn == dxgml_op::ReduceFunction::max) {
      rewriter.replaceOpWithNewOp<migraphx::ReduceMaxOp>(
          op, outType, adaptor.getInput(),
          rewriter.getI64ArrayAttr(axes));
    } else if (fn == dxgml_op::ReduceFunction::average) {
      rewriter.replaceOpWithNewOp<migraphx::ReduceMeanOp>(
          op, outType, adaptor.getInput(),
          rewriter.getI64ArrayAttr(axes));
    } else {
      return failure(); // Other reductions not yet mapped.
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise Conversions
//===----------------------------------------------------------------------===//

/// dxgml_op.convolution -> migraphx.convolution (+ optional fused-bias add)
struct ConvertConvolutionOp
    : public OpConversionPattern<dxgml_op::ConvolutionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      dxgml_op::ConvolutionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    SmallVector<int64_t> strides   = extractI64Array(op.getStrides(),      {1, 1});
    SmallVector<int64_t> dilations = extractI64Array(op.getDilations(),    {1, 1});
    SmallVector<int64_t> startPad  = extractI64Array(op.getStartPadding(), {0, 0});
    SmallVector<int64_t> endPad    = extractI64Array(op.getEndPadding(),   {0, 0});
    int64_t groupCount = op.getGroupCount() ? op.getGroupCount()->getInt() : 1;

    SmallVector<int64_t> padding;
    for (int64_t p : startPad) padding.push_back(p);
    for (int64_t p : endPad)   padding.push_back(p);

    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    Value convOut =
        migraphx::ConvolutionOp::create(
            rewriter, op.getLoc(), outType,
            adaptor.getInput(), adaptor.getFilter(),
            rewriter.getI64ArrayAttr(padding),
            rewriter.getI64ArrayAttr(strides),
            rewriter.getI64ArrayAttr(dilations),
            rewriter.getI64IntegerAttr(groupCount),
            /*padding_mode=*/nullptr,
            /*perf_config=*/nullptr)
            .getOutput();

    Value result = convOut;
    // Only fuse bias if it is a real tensor (not a null_ptr / !dxgml.null).
    Value bias = adaptor.getBias();
    bool hasBias = bias && !isa<dxgml::NullType>(bias.getType());
    if (hasBias) {
      auto outShaped = cast<migraphx::MIXRShapedType>(outType);
      auto biasShaped = dyn_cast<migraphx::MIXRShapedType>(bias.getType());
      // Bias is typically 1D [C_out]; broadcast it to the output shape [N,C,H,W]
      // using broadcast strides (0 for all but the channel dim) so that TOSA
      // lowering sees shapes that are compatible for element-wise add.
      if (biasShaped && biasShaped.getRank() == 1 && outShaped.getRank() > 1) {
        ArrayRef<int64_t> outShape = outShaped.getShape();
        int64_t outRank = outShaped.getRank();
        // Channel axis is dim 1 (NCHW). Assign stride 1 there, 0 elsewhere.
        SmallVector<int64_t> bcastStrides(outRank, 0);
        bcastStrides[1] = 1;
        auto bcastType = migraphx::MIXRShapedType::get(
            outShape, bcastStrides, outShaped.getElementType());
        bias = migraphx::MultiBroadcastOp::create(
                   rewriter, op.getLoc(), bcastType, bias,
                   rewriter.getI64ArrayAttr(outShape))
                   .getOutput();
      }
      result = migraphx::AddOp::create(rewriter, op.getLoc(), outType,
                                       convOut, bias)
                   .getOutput();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// dxgml_op.gemm -> migraphx.dot (with optional transpose & bias add)
struct ConvertGemmOp : public OpConversionPattern<dxgml_op::GemmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      dxgml_op::GemmOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getResult().getType());
    if (!outType) return failure();

    Value inA = adaptor.getATensor();
    Value inB = adaptor.getBTensor();

    auto maybeTranspose = [&](Value v,
                              std::optional<dxgml_op::MatrixTransform> trans) -> Value {
      if (!trans || *trans == dxgml_op::MatrixTransform::none)
        return v;
      auto shaped = dyn_cast<ShapedType>(v.getType());
      if (!shaped) return v;
      int64_t rank = shaped.getRank();
      SmallVector<int64_t> perm;
      for (int64_t i = 0; i < rank - 2; ++i) perm.push_back(i);
      perm.push_back(rank - 1);
      perm.push_back(rank - 2);

      SmallVector<int64_t> tShape(shaped.getShape());
      std::swap(tShape[rank - 2], tShape[rank - 1]);
      SmallVector<int64_t> tStrides(rank);
      int64_t s = 1;
      for (int i = rank - 1; i >= 0; --i) {
        tStrides[i] = s;
        if (tShape[i] != ShapedType::kDynamic) s *= tShape[i];
      }
      auto transTy = migraphx::MIXRShapedType::get(tShape, tStrides,
                                                    shaped.getElementType());
      return migraphx::TransposeOp::create(rewriter, op.getLoc(), transTy, v,
                                           rewriter.getI64ArrayAttr(perm))
          .getOutput();
    };

    inA = maybeTranspose(inA, op.getTransA());
    inB = maybeTranspose(inB, op.getTransB());

    Value dotResult =
        migraphx::DotOp::create(rewriter, op.getLoc(), outType, inA, inB)
            .getOutput();

    Value bias = adaptor.getCTensor();
    bool hasBias = bias && !isa<dxgml::NullType>(bias.getType());
    if (hasBias) {
      dotResult =
          migraphx::AddOp::create(rewriter, op.getLoc(), outType, dotResult,
                                  bias)
              .getOutput();
    }

    rewriter.replaceOp(op, dotResult);
    return success();
  }
};

/// dxgml_op.batch_normalization -> migraphx.batch_norm_inference
///
/// Lowers to the same MIGraphX op that migraphx.batch_norm_inference produces,
/// following the same lowering strategy as the MIGraphX pipeline.  Ops without
/// a TOSA lowering (batch_norm, pooling) are kept as migraphx.* ops and the
/// rocmlir-driver dxgml host-pipeline skips addHighLevelPipeline when such ops
/// are present, passing the module directly to the kernel pipeline (identical
/// to how native MIGraphX models work with --kernel-pipeline=gpu).
struct ConvertBatchNormalizationOp
    : public OpConversionPattern<dxgml_op::BatchNormalizationOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      dxgml_op::BatchNormalizationOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    // Extract epsilon (default 1e-5 to match MIGraphX default).
    float epsilon = 1e-5f;
    if (auto epsAttr = op.getEpsilon())
      epsilon = static_cast<float>(epsAttr->getValue().convertToDouble());

    // MIGraphX batch_norm_inference: spatial mode (bn_mode=1), momentum unused.
    rewriter.replaceOpWithNewOp<migraphx::BatchNormOp>(
        op, outType,
        adaptor.getInput(),
        adaptor.getScale(),
        adaptor.getBias(),
        adaptor.getMean(),
        adaptor.getVariance(),
        rewriter.getF32FloatAttr(epsilon),
        /*momentum=*/rewriter.getF32FloatAttr(0.9f),
        /*bn_mode=*/rewriter.getI64IntegerAttr(1));
    return success();
  }
};

/// dxgml_op.max_pooling -> migraphx.pooling (mode="max")
struct ConvertMaxPoolingOp
    : public OpConversionPattern<dxgml_op::MaxPoolingOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      dxgml_op::MaxPoolingOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    const TypeConverter *tc = getTypeConverter();
    // MaxPoolingOp has an optional output_indices; we only lower the pooling
    // result (index output is not needed for inference).
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    SmallVector<int64_t> strides    = extractI64Array(op.getStrides(),      {1, 1});
    SmallVector<int64_t> windowSize = extractI64Array(op.getWindowSize(),   {1, 1});
    SmallVector<int64_t> startPad   = extractI64Array(op.getStartPadding(), {0, 0});
    SmallVector<int64_t> endPad     = extractI64Array(op.getEndPadding(),   {0, 0});

    SmallVector<int64_t> padding;
    for (int64_t p : startPad) padding.push_back(p);
    for (int64_t p : endPad)   padding.push_back(p);

    rewriter.replaceOpWithNewOp<migraphx::PoolingOp>(
        op, outType,
        adaptor.getInput(),
        rewriter.getStringAttr("max"),
        rewriter.getI64ArrayAttr(padding),
        rewriter.getI64ArrayAttr(strides),
        rewriter.getI64ArrayAttr(windowSize),
        /*ceil_mode=*/rewriter.getI64IntegerAttr(0));
    return success();
  }
};

/// dxgml_op.average_pooling -> migraphx.pooling (mode="average")
struct ConvertAveragePoolingOp
    : public OpConversionPattern<dxgml_op::AveragePoolingOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      dxgml_op::AveragePoolingOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    const TypeConverter *tc = getTypeConverter();
    Type outType = tc->convertType(op.getOutput().getType());
    if (!outType) return failure();

    SmallVector<int64_t> strides    = extractI64Array(op.getStrides(),      {1, 1});
    SmallVector<int64_t> windowSize = extractI64Array(op.getWindowSize(),   {1, 1});
    SmallVector<int64_t> startPad   = extractI64Array(op.getStartPadding(), {0, 0});
    SmallVector<int64_t> endPad     = extractI64Array(op.getEndPadding(),   {0, 0});

    SmallVector<int64_t> padding;
    for (int64_t p : startPad) padding.push_back(p);
    for (int64_t p : endPad)   padding.push_back(p);

    rewriter.replaceOpWithNewOp<migraphx::PoolingOp>(
        op, outType,
        adaptor.getInput(),
        rewriter.getStringAttr("average"),
        rewriter.getI64ArrayAttr(padding),
        rewriter.getI64ArrayAttr(strides),
        rewriter.getI64ArrayAttr(windowSize),
        /*ceil_mode=*/rewriter.getI64IntegerAttr(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unary element-wise ops
//===----------------------------------------------------------------------===//

/// Helper macro to define a simple unary pattern DxgmlSrc -> MigraphxDst.
#define DEFINE_UNARY_CONV(PatternName, DxgmlSrcOp, MigraphxDstOp)           \
struct PatternName : public OpConversionPattern<DxgmlSrcOp> {               \
  using OpConversionPattern::OpConversionPattern;                            \
  LogicalResult matchAndRewrite(DxgmlSrcOp op, OpAdaptor adaptor,           \
                                ConversionPatternRewriter &rewriter)         \
      const override {                                                        \
    Type outType = getTypeConverter()->convertType(op.getOutput().getType());\
    if (!outType) return failure();                                           \
    rewriter.replaceOpWithNewOp<MigraphxDstOp>(op, outType,                  \
                                               adaptor.getInput());          \
    return success();                                                         \
  }                                                                           \
};

DEFINE_UNARY_CONV(ConvertReluOp,      dxgml_op::ReluOp,      migraphx::ReluOp)
DEFINE_UNARY_CONV(ConvertSigmoidOp,   dxgml_op::SigmoidOp,   migraphx::SigmoidOp)
DEFINE_UNARY_CONV(ConvertTanhOp,      dxgml_op::TanhOp,       migraphx::TanhOp)
DEFINE_UNARY_CONV(ConvertErfOp,       dxgml_op::ErfOp,        migraphx::ErfOp)
DEFINE_UNARY_CONV(ConvertExpOp,       dxgml_op::ExpOp,        migraphx::ExpOp)
DEFINE_UNARY_CONV(ConvertLogOp,       dxgml_op::LogOp,        migraphx::LogOp)
DEFINE_UNARY_CONV(ConvertSqrtOp,      dxgml_op::SqrtOp,       migraphx::SqrtOp)
DEFINE_UNARY_CONV(ConvertRsqrtOp,     dxgml_op::ReciprocalOp, migraphx::RsqrtOp)
DEFINE_UNARY_CONV(ConvertNegateOp,    dxgml_op::NegateOp,     migraphx::NegOp)
DEFINE_UNARY_CONV(ConvertAbsOp,       dxgml_op::AbsOp,        migraphx::AbsOp)
DEFINE_UNARY_CONV(ConvertCeilOp,      dxgml_op::CeilOp,       migraphx::CeilOp)
DEFINE_UNARY_CONV(ConvertFloorOp,     dxgml_op::FloorOp,      migraphx::FloorOp)

//===----------------------------------------------------------------------===//
// Binary element-wise ops
//===----------------------------------------------------------------------===//

/// Helper macro to define a simple binary pattern DxgmlSrc -> MigraphxDst.
#define DEFINE_BINARY_CONV(PatternName, DxgmlSrcOp, MigraphxDstOp)          \
struct PatternName : public OpConversionPattern<DxgmlSrcOp> {               \
  using OpConversionPattern::OpConversionPattern;                            \
  LogicalResult matchAndRewrite(DxgmlSrcOp op, OpAdaptor adaptor,           \
                                ConversionPatternRewriter &rewriter)         \
      const override {                                                        \
    Type outType = getTypeConverter()->convertType(op.getOutput().getType());\
    if (!outType) return failure();                                           \
    rewriter.replaceOpWithNewOp<MigraphxDstOp>(op, outType,                  \
        adaptor.getLhs(), adaptor.getRhs());                                  \
    return success();                                                         \
  }                                                                           \
};

DEFINE_BINARY_CONV(ConvertAddOp,      dxgml_op::AddOp,      migraphx::AddOp)
DEFINE_BINARY_CONV(ConvertSubtractOp, dxgml_op::SubtractOp, migraphx::SubOp)
DEFINE_BINARY_CONV(ConvertMultiplyOp, dxgml_op::MultiplyOp, migraphx::MulOp)
DEFINE_BINARY_CONV(ConvertDivideOp,   dxgml_op::DivideOp,   migraphx::DivOp)
DEFINE_BINARY_CONV(ConvertMaxBinaryOp,dxgml_op::MaxOp,       migraphx::MulOp) // placeholder

/// dxgml_op.pow -> migraphx.pow, casting the exponent to match the base type
/// if needed (e.g. int32 exponent with float16 base).
struct ConvertPowOp : public OpConversionPattern<dxgml_op::PowOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(dxgml_op::PowOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type outType = getTypeConverter()->convertType(op.getOutput().getType());
    if (!outType) return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // If lhs and rhs element types differ, cast rhs to match lhs's element type.
    auto lhsShaped = dyn_cast<migraphx::MIXRShapedType>(lhs.getType());
    auto rhsShaped = dyn_cast<migraphx::MIXRShapedType>(rhs.getType());
    if (lhsShaped && rhsShaped &&
        lhsShaped.getElementType() != rhsShaped.getElementType()) {
      auto newRhsTy = migraphx::MIXRShapedType::get(
          rhsShaped.getShape(), rhsShaped.getStrides(),
          lhsShaped.getElementType());
      rhs = migraphx::ConvertOp::create(rewriter, op.getLoc(), newRhsTy, rhs)
                .getOutput();
    }

    rewriter.replaceOpWithNewOp<migraphx::PowOp>(op, outType, lhs, rhs);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct ConvertDxgmlToMIGraphXPass
    : public PassWrapper<ConvertDxgmlToMIGraphXPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertDxgmlToMIGraphXPass)

  StringRef getArgument() const final { return "convert-dxgml-to-migraphx"; }
  StringRef getDescription() const final {
    return "Convert DXGML dialect (dxgml.* / dxgml_op.*) to MIGraphX dialect "
           "and func.func, enabling the standard MIGraphX -> Rock -> GPU pipeline";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<migraphx::MIGraphXDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp module = getOperation();

    DxgmlToMIGraphXTypeConverter typeConverter;

    // -----------------------------------------------------------------------
    // Phase 1: Lift dxgml.entry_point → func.func (with converted arg types),
    // erase dxgml.module.  dxgml.return → func.return.
    // The body still contains dxgml_op.* at this point.
    // -----------------------------------------------------------------------
    {
      OpBuilder builder(module.getBodyRegion());

      SmallVector<dxgml::ModuleOp> dxgmlModules;
      module->walk([&](dxgml::ModuleOp mod) { dxgmlModules.push_back(mod); });

      for (auto dxgmlMod : dxgmlModules) {
        builder.setInsertionPoint(dxgmlMod);

        SmallVector<dxgml::EntryPointOp> entries;
        dxgmlMod->walk([&](dxgml::EntryPointOp ep) { entries.push_back(ep); });

        for (auto ep : entries) {
          auto origFuncType = cast<FunctionType>(ep.getFunctionType());

          SmallVector<Type> convArgs, convResults;
          for (Type t : origFuncType.getInputs()) {
            Type conv = typeConverter.convertType(t);
            if (!conv) { signalPassFailure(); return; }
            convArgs.push_back(conv);
          }
          if (failed(typeConverter.convertTypes(origFuncType.getResults(),
                                               convResults))) {
            signalPassFailure(); return;
          }

          auto newFuncType = FunctionType::get(context, convArgs, convResults);
          auto funcOp = builder.create<func::FuncOp>(ep.getLoc(),
                                                     ep.getSymName(), newFuncType);

          funcOp.getBody().takeBody(ep.getBody());

          if (!funcOp.getBody().empty()) {
            Block &entryBlock = funcOp.getBody().front();
            for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i)
              entryBlock.getArgument(i).setType(convArgs[i]);
          }

          funcOp->walk([&](dxgml::ReturnOp ret) {
            OpBuilder retBuilder(ret);
            retBuilder.create<func::ReturnOp>(ret.getLoc(), ret.getOperands());
            ret.erase();
          });

          ep.erase();
        }

        dxgmlMod.erase();
      }
    }

    // -----------------------------------------------------------------------
    // Phase 2: Convert known dxgml_op.* → migraphx.* inside func.func bodies.
    // Unknown/unsupported ops are left as-is (partial conversion).
    // -----------------------------------------------------------------------
    {
      ConversionTarget target(*context);

      // By default everything is legal.  We selectively mark dxgml_op ops as
      // illegal only for the ones that have concrete patterns.  Unknown or
      // unmapped ops stay legal and pass through unchanged (partial conversion).
      target.addLegalDialect<dxgml_op::DxgmlOpDialect>();

      // Mark the ops we actively convert as illegal (patterns must handle them).
      // ConstantOp: illegal only when its result type has a supported element type.
      target.addDynamicallyLegalOp<dxgml_op::ConstantOp>([](dxgml_op::ConstantOp op) {
        auto tensorTy = dyn_cast<dxgml::DxgmlTensorType>(op.getResult().getType());
        if (!tensorTy) return true;
        Type elem = tensorTy.getElementType();
        return !isa<dxgml::Float16Type, dxgml::Float32Type, dxgml::Float64Type,
                    dxgml::BFloat16Type, dxgml::Int8Type, dxgml::UInt8Type,
                    dxgml::Int16Type, dxgml::UInt16Type,
                    dxgml::Int32Type, dxgml::Int64Type>(elem);
      });
      // NullPtrOp: illegal only when result is a dxgml.tensor (convertible).
      target.addDynamicallyLegalOp<dxgml_op::NullPtrOp>([](dxgml_op::NullPtrOp op) {
        return !isa<dxgml::DxgmlTensorType>(op.getResult().getType());
      });

      // Helper to check if all result types have supported element types.
      auto hasSupportedElemTypes = [](Operation *op) -> bool {
        for (Type resTy : op->getResultTypes()) {
          auto tensorTy = dyn_cast<dxgml::DxgmlTensorType>(resTy);
          if (!tensorTy) return false;
          Type elem = tensorTy.getElementType();
          if (!isa<dxgml::Float16Type, dxgml::Float32Type, dxgml::Float64Type,
                   dxgml::BFloat16Type, dxgml::Int8Type,
                   dxgml::Int16Type, dxgml::Int32Type, dxgml::Int64Type>(elem))
            return false;
        }
        return !op->getResultTypes().empty();
      };

      ConversionTarget::DynamicLegalityCallbackFn dynamicIllegal =
          [hasSupportedElemTypes](Operation *op) -> std::optional<bool> {
        return !hasSupportedElemTypes(op); // illegal when element types are supported
      };

      target.addDynamicallyLegalOp<
          dxgml_op::ReshapeOp, dxgml_op::TransposeOp, dxgml_op::SliceOp,
          dxgml_op::DepthToSpaceOp,
          dxgml_op::DequantizeLinearOp, dxgml_op::QuantizeLinearOp, dxgml_op::CastOp,
          dxgml_op::ConvolutionOp, dxgml_op::GemmOp,
          dxgml_op::BatchNormalizationOp, dxgml_op::MaxPoolingOp, dxgml_op::AveragePoolingOp,
          dxgml_op::SoftmaxOp, dxgml_op::ReduceOp,
          dxgml_op::ReluOp, dxgml_op::SigmoidOp, dxgml_op::TanhOp, dxgml_op::ErfOp,
          dxgml_op::ExpOp, dxgml_op::LogOp, dxgml_op::SqrtOp, dxgml_op::ReciprocalOp,
          dxgml_op::NegateOp, dxgml_op::AbsOp, dxgml_op::CeilOp, dxgml_op::FloorOp,
          dxgml_op::AddOp, dxgml_op::SubtractOp, dxgml_op::MultiplyOp,
          dxgml_op::DivideOp, dxgml_op::PowOp
      >(dynamicIllegal);

      // func.return is legal when all its operands have migraphx types.
      // Otherwise mark it illegal so the ConvertFuncReturnOp pattern converts it.
      target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
        return llvm::none_of(op.getOperandTypes(), [](Type t) {
          return isa<dxgml::DxgmlTensorType>(t);
        });
      });

      // func.func, module, and UnrealizedConversionCastOp are legal.
      // UnrealizedConversionCastOp is legal to allow type bridging at boundaries
      // between converted migraphx ops and unconverted dxgml_op pass-through ops.
      target.addLegalOp<ModuleOp, func::FuncOp, UnrealizedConversionCastOp>();
      target.addLegalDialect<migraphx::MIGraphXDialect>();

      RewritePatternSet patterns(context);
      patterns.add<
          ConvertFuncReturnOp,
          // Constants / null
          ConvertConstantOp,
          ConvertNullPtrOp,
          // Shape ops
          ConvertReshapeOp,
          ConvertTransposeOp,
          ConvertSliceOp,
          ConvertConcatOp,
          ConvertGatherOp,
          ConvertDepthToSpaceOp,
          // Quantization
          ConvertDequantizeLinearOp,
          ConvertQuantizeLinearOp,
          ConvertCastOp,
          // Compute ops
          ConvertConvolutionOp,
          ConvertGemmOp,
          ConvertBatchNormalizationOp,
          ConvertMaxPoolingOp,
          ConvertAveragePoolingOp,
          ConvertSoftmaxOp,
          ConvertReduceOp,
          // Unary
          ConvertReluOp,
          ConvertSigmoidOp,
          ConvertTanhOp,
          ConvertErfOp,
          ConvertExpOp,
          ConvertLogOp,
          ConvertSqrtOp,
          ConvertRsqrtOp,
          ConvertNegateOp,
          ConvertAbsOp,
          ConvertCeilOp,
          ConvertFloorOp,
          // Binary
          ConvertAddOp,
          ConvertSubtractOp,
          ConvertMultiplyOp,
          ConvertDivideOp,
          ConvertPowOp,
          ConvertMaxBinaryOp
      >(typeConverter, context);

      // Use partial conversion: ops without a pattern are left as-is.
      if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
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
      ConvertConstantOp,
      ConvertNullPtrOp,
      ConvertReshapeOp,
      ConvertTransposeOp,
      ConvertSliceOp,
      ConvertConcatOp,
      ConvertDepthToSpaceOp,
      ConvertDequantizeLinearOp,
      ConvertQuantizeLinearOp,
      ConvertCastOp,
      ConvertConvolutionOp,
      ConvertGemmOp,
      ConvertBatchNormalizationOp,
      ConvertMaxPoolingOp,
      ConvertAveragePoolingOp,
      ConvertSoftmaxOp,
      ConvertReduceOp,
      ConvertReluOp,
      ConvertSigmoidOp,
      ConvertTanhOp,
      ConvertErfOp,
      ConvertExpOp,
      ConvertLogOp,
      ConvertSqrtOp,
      ConvertRsqrtOp,
      ConvertNegateOp,
      ConvertAbsOp,
      ConvertCeilOp,
      ConvertFloorOp,
      ConvertAddOp,
      ConvertSubtractOp,
      ConvertMultiplyOp,
      ConvertDivideOp,
      ConvertPowOp
  >(typeConverter, patterns.getContext());
}

std::unique_ptr<Pass> mlir::createConvertDxgmlToMIGraphXPass() {
  return std::make_unique<ConvertDxgmlToMIGraphXPass>();
}
