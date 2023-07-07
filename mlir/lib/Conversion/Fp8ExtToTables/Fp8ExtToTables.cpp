//===- Fp8ExtToTables.cpp - arith.extf on fp8 by table lookup -----------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2023 Advanced Micro Devices Inc.
//===----------------------------------------------------------------------===//
//
// Declares the passes for remapping `arith.extf` on fp8 types to a table lookup
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Fp8ExtToTables/Fp8ExtToTables.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_FP8EXTTOTABLESPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;

namespace {
struct Fp8ExtToTablesPass final
    : public impl::Fp8ExtToTablesPassBase<Fp8ExtToTablesPass> {
  using impl::Fp8ExtToTablesPassBase<
      Fp8ExtToTablesPass>::Fp8ExtToTablesPassBase;

  void runOnOperation() override;
};

struct Fp8ExtToTableLookupPattern final : public OpConversionPattern<ExtFOp> {
  using OpConversionPattern<ExtFOp>::OpConversionPattern;

  LogicalResult match(ExtFOp op) const override;
  void rewrite(ExtFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override;
};
} // namespace

static bool isFp8(Type t) {
  return t.isFloat8E5M2() || t.isFloat8E4M3FN() || t.isFloat8E5M2FNUZ() ||
         t.isFloat8E4M3FNUZ();
}

static LogicalResult canRewriteToTable(ExtFOp op) {
  Type inType = op.getIn().getType();
  Type inElemType = getElementTypeOrSelf(inType);
  if (!isFp8(inElemType))
    return failure();
  if (isa<FloatType>(inType))
    return success();
  if (auto vecType = dyn_cast<VectorType>(inType))
    return success(vecType.hasStaticShape());
  return failure();
}

LogicalResult Fp8ExtToTableLookupPattern::match(ExtFOp op) const {
  return canRewriteToTable(op);
}

static Value getFloatValueTableFor(Type elementType, Operation *op,
                                   ConversionPatternRewriter &rewriter) {
  assert(isFp8(elementType) &&
         "tables can only be generated for scalar float types");
  auto type = cast<FloatType>(elementType);
  Operation *module = SymbolTable::getNearestSymbolTable(op);
  auto globalType = MemRefType::get(256, rewriter.getF32Type());
  SmallString<32> extTableName;
  // Name collisions are unlikely to be an issue as
  // - in an XMIR context, this'll be placed within individual copies of the
  // code,
  //   , which tend to add suffixes to function names etc.
  // - In the MIGraphX context, fp8 isn't supported (so even if they feed us
  //   arbitrarily evil function names, we won't hit this case).
  // - In our testing context, we control the top-level module names and won't
  //    pick one like this.
  llvm::raw_svector_ostream extTableNameGen(extTableName);
  extTableNameGen << "__rocmlir_extf_tbl_" << type;
  auto table = dyn_cast_if_present<memref::GlobalOp>(
      SymbolTable::lookupSymbolIn(module, extTableName));
  if (table) {
    return rewriter.createOrFold<memref::GetGlobalOp>(op->getLoc(), globalType,
                                                      extTableName);
  }
  SmallVector<float, 0> tableElems;
  tableElems.reserve(256);
  const auto &sem = type.getFloatSemantics();
  for (uint32_t i = 0; i < 256; ++i) {
    APFloat entry(sem, APInt(8, i));
    tableElems.push_back(entry.convertToFloat());
  }
  ElementsAttr tableElemsAttr = DenseElementsAttr::get<float>(
      RankedTensorType::get(256, rewriter.getF32Type()), tableElems);
  OpBuilder nowhereBuilder(module->getContext(), &rewriter);
  table = nowhereBuilder.create<memref::GlobalOp>(
      op->getLoc(), extTableName,
      /*sym_visibility=*/rewriter.getStringAttr("private"),
      /*type=*/globalType,
      /*initial_value=*/tableElemsAttr,
      /*constant=*/true,
      /*alignment=*/nullptr);
  SymbolTable(module).insert(table);
  return rewriter.createOrFold<memref::GetGlobalOp>(op->getLoc(), globalType,
                                                    extTableName);
}

void Fp8ExtToTableLookupPattern::rewrite(
    ExtFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Type inType = op.getIn().getType();
  Type outType = op.getResult().getType();
  Type outElemType = getElementTypeOrSelf(outType);
  Type elemType = getElementTypeOrSelf(inType);
  Type f32 = rewriter.getF32Type();

  Value table = getFloatValueTableFor(elemType, op, rewriter);
  auto oneToFloat = [&](Value fp8) -> Value {
    Value bitcast =
        rewriter.create<arith::BitcastOp>(loc, rewriter.getI8Type(), fp8);
    // Don't sign-extend the byte when index casting.
    Value i32 =
        rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(), bitcast);
    Value index =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), i32);
    Value extended = rewriter.create<memref::LoadOp>(loc, table, index);
    return extended;
  };

  auto floatsToResult = [&](Value floats) -> Value {
    if (outElemType.isF32())
      return floats;
    if (outElemType.getIntOrFloatBitWidth() < 32)
      return rewriter.create<arith::TruncFOp>(loc, outType, floats);
    if (outElemType.getIntOrFloatBitWidth() > 32)
      return rewriter.create<arith::ExtFOp>(loc, outType, floats);
    llvm_unreachable("f32 is the only 32-bit float type");
  };
  auto inVecType = dyn_cast<VectorType>(inType);
  if (!inVecType) {
    Value ret = floatsToResult(oneToFloat(adaptor.getIn()));
    return rewriter.replaceOp(op, ret);
  }
  VectorType floatVecType = inVecType.clone(f32);
  Value floats = rewriter.createOrFold<vector::SplatOp>(
      loc,
      rewriter.createOrFold<arith::ConstantOp>(loc, f32,
                                               rewriter.getF32FloatAttr(0.0f)),
      floatVecType);
  SmallVector<int64_t> strides = computeStrides(inVecType.getShape());
  for (int64_t i = 0, e = inVecType.getNumElements(); i < e; ++i) {
    SmallVector<int64_t> idx = delinearize(i, strides);
    Value scalar =
        rewriter.create<vector::ExtractOp>(loc, adaptor.getIn(), idx);
    Value extended = oneToFloat(scalar);
    floats = rewriter.create<vector::InsertOp>(loc, extended, floats, idx);
  }
  Value ret = floatsToResult(floats);
  return rewriter.replaceOp(op, ret);
}

void mlir::addFp8ExtToTablesPatterns(RewritePatternSet &patterns) {
  patterns.add<Fp8ExtToTableLookupPattern>(patterns.getContext());
}

void Fp8ExtToTablesPass::runOnOperation() {
  Operation *op = getOperation();
  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    emitError(op->getLoc(), "fp8-ext-to-tables requires a module-like (symbol "
                            "table having) root operation");
    return signalPassFailure();
  }

  MLIRContext *ctx = &getContext();
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect,
                         vector::VectorDialect>();
  target.addDynamicallyLegalOp<arith::ExtFOp>(
      [](ExtFOp op) { return failed(canRewriteToTable(op)); });
  RewritePatternSet rewrites(ctx);
  addFp8ExtToTablesPatterns(rewrites);
  if (failed(applyPartialConversion(op, target, std::move(rewrites))))
    return signalPassFailure();
}
