//===- EmulateFp8ExtTrunc.cpp - arith.extf on fp8 by table lookup -------===//
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

#include "mlir/Conversion/EmulateFp8ExtTrunc/EmulateFp8ExtTrunc.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
#define GEN_PASS_DEF_EMULATEFP8EXTTRUNCPASS
#include "mlir/Conversion/RocMLIRPasses.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;

namespace {
struct EmulateFp8ExtTruncPass final
    : public impl::EmulateFp8ExtTruncPassBase<EmulateFp8ExtTruncPass> {
  using impl::EmulateFp8ExtTruncPassBase<
      EmulateFp8ExtTruncPass>::EmulateFp8ExtTruncPassBase;

  void runOnOperation() override;
};

struct Fp8ExtToTableLookupPattern final : public OpConversionPattern<ExtFOp> {
  using OpConversionPattern<ExtFOp>::OpConversionPattern;

  LogicalResult match(ExtFOp op) const override;
  void rewrite(ExtFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override;
};

struct Fp8TruncToCallPattern final : public OpConversionPattern<TruncFOp> {
  FlatSymbolRefAttr f8E4M3FNUZFunc;
  FlatSymbolRefAttr f8E5M2FNUZFunc;

  // The functions are optional - if they aren't provided for a type (the null
  // attribute is sent in) the pattern will not apply.
  Fp8TruncToCallPattern(MLIRContext *ctx, FlatSymbolRefAttr f8E4M3FNUZFunc,
                        FlatSymbolRefAttr f8E5M2FNUZFunc)
      : OpConversionPattern<TruncFOp>::OpConversionPattern(ctx),
        f8E4M3FNUZFunc(f8E4M3FNUZFunc), f8E5M2FNUZFunc(f8E5M2FNUZFunc) {}

  LogicalResult match(TruncFOp op) const override;
  void rewrite(TruncFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override;
};
} // namespace

static bool isFp8(Type t) {
  return t.isFloat8E5M2() || t.isFloat8E4M3FN() || t.isFloat8E5M2FNUZ() ||
         t.isFloat8E4M3FNUZ();
}

static LogicalResult canBeConverted(Type t) {
  if (!isFp8(getElementTypeOrSelf(t)))
    return failure();
  if (auto vecType = dyn_cast<VectorType>(t))
    return success(vecType.hasStaticShape());
  return success();
}

LogicalResult Fp8ExtToTableLookupPattern::match(ExtFOp op) const {
  return canBeConverted(op.getIn().getType());
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
  OpBuilder nowhereBuilder(module->getContext(), rewriter.getListener());
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
    Value bitcast = rewriter.create<BitcastOp>(loc, rewriter.getI8Type(), fp8);
    // Don't sign-extend the byte when index casting.
    Value i32 = rewriter.create<ExtUIOp>(loc, rewriter.getI32Type(), bitcast);
    Value index =
        rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(), i32);
    Value extended = rewriter.create<memref::LoadOp>(loc, table, index);
    return extended;
  };

  auto floatsToResult = [&](Value floats) -> Value {
    if (outElemType.isF32())
      return floats;
    if (outElemType.getIntOrFloatBitWidth() < 32)
      return rewriter.create<TruncFOp>(loc, outType, floats);
    if (outElemType.getIntOrFloatBitWidth() > 32)
      return rewriter.create<ExtFOp>(loc, outType, floats);
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
      rewriter.createOrFold<ConstantOp>(loc, f32,
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

/// Creates a function that trunctates input floats to the 8-bit `ooutTYpe`,
/// where `outType` is one of the NANOO float types (f8E4M3FNUZ or f8E5M2FNUZ),
/// and inserts it into `module`, returning a reference to the inserted
/// function.
///
/// This truncation saturates: values too large in absolute value to be
/// represented by the maximum value of `outType` are clamped into `outType`'s
/// range instead of being rounded to NaN.
///
/// Based off
/// https://github.com/ROCm/AMDMIGraphX/blob/a41cd5c0b493bbb7d21078f1a842675ff824d2b7/src/include/migraphx/float8_impl.hpp#L37
/// but trimmed (Clip = true, NegativeZeroNan = true) and run through
/// clang -O3 on godbolt to get LLVM IR that I could mechanically recreate.
/// See mlir/docs/fnuz-float-software-truncation-sources/
/// for the inputs to and outputs of this process.
static FlatSymbolRefAttr makeFp8TruncFunction(Location loc, FloatType outType,
                                              Operation *module) {
  ImplicitLocOpBuilder b(loc, loc.getContext());
  SymbolTable symtab(module);

  SmallString<32> funcName;
  (Twine("_rocmlir_trunc_f32_to_") +
   (TypeSwitch<FloatType, StringRef>(outType)
        .Case<Float8E4M3FNUZType>(
            [](auto ignored) -> StringRef { return "f8E4M3FNUZ"; })
        .Case<Float8E5M2FNUZType>(
            [](auto ignored) -> StringRef { return "f8E5M2FNUZ"; })
        .Default([](auto ignored) -> StringRef { return "unknownERROR"; })))
      .toVector(funcName);
  auto func = func::FuncOp::create(
      loc, funcName, b.getFunctionType({b.getF32Type()}, {outType}));
  StringAttr realFuncName = symtab.insert(func);
  auto symbolRef = FlatSymbolRefAttr::get(realFuncName);
  symtab.setSymbolVisibility(func, SymbolTable::Visibility::Private);

  Block *entry = func.addEntryBlock();
  b.setInsertionPointToStart(entry);
  Value in = entry->getArgument(0);

  Type i32 = b.getI32Type();
  Type i8 = b.getI8Type();
  auto i32Const = [&](uint32_t value) -> Value {
    return b.createOrFold<ConstantOp>(i32, b.getI32IntegerAttr(value));
  };
  // Created here so we can branch to it, will be inserted last
  Block *ret = new Block();
  ret->addArgument(outType, loc);

  Value bits = b.create<BitcastOp>(i32, in);
  const llvm::fltSemantics &outSem = outType.getFloatSemantics();

  Value and1 = b.create<AndIOp>(bits, i32Const((1u << 23u) - 1));
  Value shr = b.create<ShRUIOp>(bits, i32Const(23));
  Value and2 = b.create<AndIOp>(shr, i32Const(0xff));
  Value ir1 = b.create<ShRUIOp>(bits, i32Const(24));
  Value shl = b.create<AndIOp>(ir1, i32Const(128));
  Value infNanConst = i32Const(0x7f800000);
  Value and4 = b.create<AndIOp>(bits, infNanConst);
  Value cmp = b.create<CmpIOp>(CmpIPredicate::eq, and4, infNanConst);

  Block *notInfNan = func.addBlock();
  Value outNan = b.create<ConstantFloatOp>(APFloat::getQNaN(outSem), outType);
  b.create<cf::CondBranchOp>(cmp, ret, ValueRange{outNan}, notInfNan,
                             ValueRange{});
  b.setInsertionPointToStart(notInfNan);

  // A deviation from the MIGraphX: denormals are zero here
  Value cmp5 = b.create<CmpIOp>(CmpIPredicate::eq, and2, i32Const(0));
  Value outZero = b.create<ConstantFloatOp>(APFloat::getZero(outSem), outType);
  Block *notZero = func.addBlock();
  b.create<cf::CondBranchOp>(cmp5, ret, ValueRange{outZero}, notZero,
                             ValueRange{});
  b.setInsertionPointToStart(notZero);

  // For some reason, this is off by one
  uint32_t mBits = outType.getFPMantissaWidth() - 1;
  uint32_t eBits = 7 - mBits;
  Value sub = b.create<AddIOp>(and2, i32Const(-127));
  Value reducedConst1 = i32Const(127 - ((1 << (eBits - 1)) - 2));
  Value cmp8 = b.create<CmpIOp>(CmpIPredicate::ult, and2, reducedConst1);
  Value reducedConst2 = i32Const(127 - ((1 << (eBits - 1)) - 1));
  Value sub10 = b.create<SubIOp>(reducedConst2, and2);
  Value exponentDiff0 = b.create<SelectOp>(cmp8, sub10, i32Const(0));

  Value add12 = b.create<OrIOp>(and1, i32Const(1 << 23));
  Value ir2 = b.create<MinUIOp>(exponentDiff0, i32Const(15 - eBits));
  Value notmaskConst = i32Const(~((1 << (16 + eBits)) - 1));
  Value notmask = b.create<ShLIOp>(notmaskConst, ir2);
  Value sub16 = b.create<XOrIOp>(notmask, i32Const(-1));
  Value and17 = b.create<AndIOp>(add12, sub16);
  Value sub21 = b.create<AddIOp>(exponentDiff0, i32Const(15 + eBits));
  Value sroaSpeculataed140 = b.create<MinUIOp>(sub21, i32Const(31));
  Value shl23 = b.create<ShLIOp>(i32Const(1), sroaSpeculataed140);
  Value cmp24 = b.create<CmpIOp>(CmpIPredicate::eq, and17, shl23);
  Value cmp25 =
      b.create<CmpIOp>(CmpIPredicate::sgt, exponentDiff0, i32Const(0));
  Value sroaSpeculated = b.create<MinUIOp>(exponentDiff0, i32Const(31));
  Value shr30 = b.create<SelectOp>(cmp25, sroaSpeculated, i32Const(0));
  Value mantissa0 = b.create<ShRUIOp>(add12, shr30);

  Value add40 = b.create<AddIOp>(sub, exponentDiff0);
  Value and38 = b.create<ShRUIOp>(mantissa0, i32Const(23));
  Value ir3 = b.create<OrIOp>(and38, i32Const(-2));
  Value add41 = b.create<AddIOp>(add40, ir3);
  Value sub43 = b.create<AddIOp>(add41, i32Const((1 << (eBits - 1)) + 1));
  Value and44 = b.create<ShRUIOp>(mantissa0, i32Const(16 + eBits));
  Value ir4 = b.create<AndIOp>(and44, i32Const(1));
  Value resolvedConst3 = i32Const((1 << (16 + eBits)) - 1);
  Value sext = b.create<AddIOp>(ir4, resolvedConst3);
  Value cond51 = b.create<SelectOp>(cmp24, sext, i32Const(0));
  Value cond54 = b.create<AddIOp>(cond51, mantissa0);
  Value and55 = b.create<AndIOp>(cond54, resolvedConst3);
  Value add56 = b.create<AddIOp>(and55, mantissa0);
  Value cmp57 = b.create<CmpIOp>(CmpIPredicate::ne, sub43, i32Const(0));
  Value and58 = b.create<AndIOp>(add56, i32Const(1 << 23));
  Value tobool59Not = b.create<CmpIOp>(CmpIPredicate::eq, and58, i32Const(0));
  Value trueConst = b.create<ConstantIntOp>(true, 1);
  Value brCond133 = b.create<SelectOp>(cmp57, trueConst, tobool59Not);

  Block *ifElse61 = func.addBlock();
  Block *ifThen70 = func.addBlock();
  Block *ifEnd71 = func.addBlock();
  Block *ifEnd76 = func.addBlock();
  ifEnd76->addArguments({i32, i32}, {loc, loc});
  Value oneConst = i32Const(1);
  b.create<cf::CondBranchOp>(brCond133, ifElse61, ValueRange{}, ifEnd76,
                             ValueRange{oneConst, add56});

  b.setInsertionPointToStart(ifElse61);
  Value tobool63Not =
      b.create<CmpIOp>(CmpIPredicate::ugt, add56, i32Const((1 << 24) - 1));
  Value incConst = i32Const((1 << (eBits - 1)) + 2);
  Value inc = b.create<AddIOp>(add41, incConst);
  Value f8Exponent0 = b.create<SelectOp>(tobool63Not, inc, sub43);
  Value cmp69 = b.create<CmpIOp>(CmpIPredicate::sgt, f8Exponent0,
                                 i32Const((1 << eBits) - 1));
  b.create<cf::CondBranchOp>(cmp69, ifThen70, ValueRange{}, ifEnd71,
                             ValueRange{});

  b.setInsertionPointToStart(ifThen70);
  Value ir5 = b.create<TruncIOp>(i8, ir1);
  Value conv =
      b.create<OrIOp>(ir5, b.create<ConstantIntOp>(127, b.getI8Type()));
  Value convOut = b.create<BitcastOp>(outType, conv);
  b.create<cf::BranchOp>(ret, convOut);

  b.setInsertionPointToStart(ifEnd71);
  Value shr65 = b.create<ExtUIOp>(i32, tobool63Not);
  Value mantissa1 = b.create<ShRUIOp>(add56, shr65);
  Value cmp72 = b.create<CmpIOp>(CmpIPredicate::eq, f8Exponent0, i32Const(0));
  Value cmp74 = b.create<CmpIOp>(CmpIPredicate::ult, mantissa1,
                                 i32Const(1 << (16 + eBits)));
  Value falseConst = b.create<ConstantIntOp>(false, 1);
  Value brCond = b.create<SelectOp>(cmp72, cmp74, falseConst);
  b.create<cf::CondBranchOp>(brCond, ret, ValueRange{outZero}, ifEnd76,
                             ValueRange{f8Exponent0, mantissa1});

  b.setInsertionPointToStart(ifEnd76);
  Value f8Exponent015 = ifEnd76->getArgument(0);
  Value shr681In = ifEnd76->getArgument(1);
  Value shr681 = b.create<ShRUIOp>(shr681In, i32Const(16 + eBits));
  Value and77 = b.create<AndIOp>(shr681, i32Const((1 << mBits) - 1));
  Value shl79 = b.create<ShLIOp>(f8Exponent015, i32Const(mBits));
  Value irOr = b.create<OrIOp>(shl79, shl);
  Value or80 = b.create<OrIOp>(irOr, and77);
  Value conv81 = b.create<TruncIOp>(i8, or80);
  Value conv81Out = b.create<BitcastOp>(outType, conv81);
  b.create<cf::BranchOp>(ret, ValueRange{conv81Out});

  func.push_back(ret);
  b.setInsertionPointToStart(ret);
  Value retVal = ret->getArgument(0);
  b.create<func::ReturnOp>(retVal);

  return symbolRef;
}

LogicalResult Fp8TruncToCallPattern::match(TruncFOp op) const {
  if (failed(canBeConverted(op.getResult().getType())))
    return failure();
  Type resType = getElementTypeOrSelf(op.getOut().getType());
  if (resType.isFloat8E4M3FN() && !f8E4M3FNUZFunc)
    return failure();
  if (resType.isFloat8E5M2FNUZ() && !f8E5M2FNUZFunc)
    return failure();
  return success();
}

static Type cloneOrReplace(Type t, Type newElementType) {
  if (auto shaped = dyn_cast<ShapedType>(t))
    return shaped.clone(newElementType);
  return newElementType;
}

void Fp8TruncToCallPattern::rewrite(TruncFOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  Value rawIn = adaptor.getIn();
  Type rawInType = rawIn.getType();
  Type rawInElemType = getElementTypeOrSelf(rawInType);
  Type outType = op.getOut().getType();
  FloatType outElemType = cast<FloatType>(getElementTypeOrSelf(outType));

  FlatSymbolRefAttr func = TypeSwitch<Type, FlatSymbolRefAttr>(outElemType)
                               .Case<Float8E4M3FNUZType>(
                                   [&](auto ignored) { return f8E4M3FNUZFunc; })
                               .Case<Float8E5M2FNUZType>(
                                   [&](auto ignored) { return f8E5M2FNUZFunc; })
                               .Default([](auto ignored) { return nullptr; });

  auto oneToOut = [&](Value f32) -> Value {
    auto call = rewriter.create<func::CallOp>(loc, func, outElemType, f32);
    return call.getResult(0);
  };

  Type inType = cloneOrReplace(rawInType, rewriter.getF32Type());
  Value in = rawIn;
  if (rawInElemType.getIntOrFloatBitWidth() < 32)
    in = rewriter.create<arith::ExtFOp>(loc, inType, rawIn);
  else if (rawInElemType.getIntOrFloatBitWidth() > 32)
    in = rewriter.create<arith::TruncFOp>(loc, inType, rawIn);

  auto inVecType = dyn_cast<VectorType>(inType);
  if (!inVecType)
    return rewriter.replaceOp(op, oneToOut(in));

  VectorType retVecType = inVecType.clone(outElemType);
  Value rets = rewriter.createOrFold<vector::SplatOp>(
      loc,
      rewriter.createOrFold<ConstantFloatOp>(
          loc, APFloat::getZero(outElemType.getFloatSemantics()), outElemType),
      retVecType);
  SmallVector<int64_t> strides = computeStrides(inVecType.getShape());
  for (int64_t i = 0, e = inVecType.getNumElements(); i < e; ++i) {
    SmallVector<int64_t> idx = delinearize(i, strides);
    Value scalar =
        rewriter.create<vector::ExtractOp>(loc, adaptor.getIn(), idx);
    Value truncated = oneToOut(scalar);
    rets = rewriter.create<vector::InsertOp>(loc, truncated, rets, idx);
  }
  return rewriter.replaceOp(op, rets);
}

void mlir::addEmulateFp8ExtTruncPatterns(
    RewritePatternSet &patterns, FlatSymbolRefAttr f8E4M3FNUZTruncFunc,
    FlatSymbolRefAttr f8E5M2FNUZTruncFunc) {
  patterns.add<Fp8ExtToTableLookupPattern>(patterns.getContext());
  patterns.add<Fp8TruncToCallPattern>(patterns.getContext(),
                                      f8E4M3FNUZTruncFunc, f8E5M2FNUZTruncFunc);
}

void EmulateFp8ExtTruncPass::runOnOperation() {
  Operation *op = getOperation();
  if (!op->hasTrait<OpTrait::SymbolTable>()) {
    emitError(op->getLoc(),
              "emulate-fp8-ext-trunc requires a module-like (symbol "
              "table having) root operation");
    return signalPassFailure();
  }

  MLIRContext *ctx = &getContext();
  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, vector::VectorDialect>();
  target.addDynamicallyLegalOp<arith::ExtFOp>(
      [](ExtFOp op) { return failed(canBeConverted(op.getIn().getType())); });
  target.addDynamicallyLegalOp<arith::TruncFOp>([](TruncFOp op) {
    return failed(canBeConverted(op.getOut().getType()));
  });

  FlatSymbolRefAttr f8E4M3FNUZTruncFunc = nullptr;
  FlatSymbolRefAttr f8E5M2FNUZTruncFunc = nullptr;
  SmallVector<Location> f8E4M3FNUZLocs, f8E5M2FNUZLocs;
  op->walk([&](TruncFOp op) {
    Type outElemType = getElementTypeOrSelf(op.getOut().getType());
    if (outElemType.isFloat8E4M3FNUZ())
      f8E4M3FNUZLocs.push_back(op->getLoc());
    else if (outElemType.isFloat8E5M2FNUZ())
      f8E5M2FNUZLocs.push_back(op->getLoc());
  });

  if (!f8E4M3FNUZLocs.empty()) {
    f8E4M3FNUZTruncFunc = makeFp8TruncFunction(
        FusedLoc::get(ctx, f8E4M3FNUZLocs), Float8E4M3FNUZType::get(ctx), op);
  }
  if (!f8E5M2FNUZLocs.empty()) {
    f8E5M2FNUZTruncFunc = makeFp8TruncFunction(
        FusedLoc::get(ctx, f8E5M2FNUZLocs), Float8E5M2FNUZType::get(ctx), op);
  }

  RewritePatternSet rewrites(ctx);
  addEmulateFp8ExtTruncPatterns(rewrites, f8E4M3FNUZTruncFunc,
                                f8E5M2FNUZTruncFunc);
  if (failed(applyPartialConversion(op, target, std::move(rewrites))))
    return signalPassFailure();
}
