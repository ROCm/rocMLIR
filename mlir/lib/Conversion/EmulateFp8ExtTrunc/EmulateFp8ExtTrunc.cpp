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

  bool hasF8ConversionInstrs = false;
  bool hasOcpF8ConversionInstrs = false;
  Fp8ExtToTableLookupPattern(MLIRContext *ctx, bool hasF8ConversionInstrs,
                             bool hasOcpF8ConversionInstrs)
      : OpConversionPattern(ctx), hasF8ConversionInstrs(hasF8ConversionInstrs),
        hasOcpF8ConversionInstrs(hasOcpF8ConversionInstrs) {}
  LogicalResult match(ExtFOp op) const;
  void rewrite(ExtFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const;
  LogicalResult
  matchAndRewrite(ExtFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = match(op);
    if (result.succeeded()) {
      rewrite(op, adaptor, rewriter);
    }
    return result;
  }
};

struct Fp8TruncToCallPattern final : public OpConversionPattern<TruncFOp> {
  using OpConversionPattern::OpConversionPattern;

  FlatSymbolRefAttr f8E4M3FNUZFunc;
  FlatSymbolRefAttr f8E5M2FNUZFunc;
  FlatSymbolRefAttr f8E4M3FNFunc; // OCP
  FlatSymbolRefAttr f8E5M2Func;   // OCP
  bool hasF8ConversionInstrs = false;
  bool hasOcpF8ConversionInstrs = false;

  // The functions are optional - if they aren't provided for a type (the null
  // attribute is sent in) the pattern will not apply.
  Fp8TruncToCallPattern(MLIRContext *ctx, FlatSymbolRefAttr f8E4M3FNUZFunc,
                        FlatSymbolRefAttr f8E5M2FNUZFunc,
                        FlatSymbolRefAttr f8E4M3FNFunc,
                        FlatSymbolRefAttr f8E5M2Func,
                        bool hasF8ConversionInstrs,
                        bool hasOcpF8ConversionInstrs)
      : OpConversionPattern(ctx), f8E4M3FNUZFunc(f8E4M3FNUZFunc),
        f8E5M2FNUZFunc(f8E5M2FNUZFunc), f8E4M3FNFunc(f8E4M3FNFunc),
        f8E5M2Func(f8E5M2Func), hasF8ConversionInstrs(hasF8ConversionInstrs),
        hasOcpF8ConversionInstrs(hasOcpF8ConversionInstrs) {}

  LogicalResult match(TruncFOp op) const;
  void rewrite(TruncFOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const;
  LogicalResult
  matchAndRewrite(TruncFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto result = match(op);
    if (result.succeeded()) {
      rewrite(op, adaptor, rewriter);
    }
    return result;
  }
};

} // namespace

static bool isFp8(Type t) {
  return isa<FloatType>(t) && t.getIntOrFloatBitWidth() == 8 &&
         !isa<Float8E8M0FNUType>(t);
}

static bool isNanooF8(Type t) {
  return isa<Float8E5M2FNUZType, Float8E4M3FNUZType>(t);
}

static bool isOcpF8(Type t) { return isa<Float8E5M2Type, Float8E4M3FNType>(t); }

static LogicalResult canBeConverted(Type t, bool hasF8ConversionInstrs,
                                    bool hasOcpF8ConversionInstrs) {
  Type elemType = getElementTypeOrSelf(t);
  if (!isFp8(elemType))
    return failure();
  if (hasF8ConversionInstrs && isNanooF8(elemType)) {
    return failure();
  }
  if (hasOcpF8ConversionInstrs && isOcpF8(elemType)) {
    return failure();
  }
  if (auto vecType = dyn_cast<VectorType>(t))
    return success(vecType.hasStaticShape());
  return success();
}

LogicalResult Fp8ExtToTableLookupPattern::match(ExtFOp op) const {
  return canBeConverted(op.getIn().getType(), hasF8ConversionInstrs,
                        hasOcpF8ConversionInstrs);
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
    float x = entry.convertToFloat();
    uint32_t u = llvm::bit_cast<uint32_t>(x);
    // Hack:  Navi4 uses 0x7f800001 for all three NaN, and that's not
    // what APFloat will do.
    if (isa<Float8E5M2Type>(type)) {
      if (i == 0x7d || i == 0x7e || i == 0x7f)
        u = 0x7f800001;
      if (i == 0xfd || i == 0xfe || i == 0xff)
        u = 0xff800001;
      x = llvm::bit_cast<float>(u);
    }
    tableElems.push_back(x);
  }
  ElementsAttr tableElemsAttr = DenseElementsAttr::get<float>(
      RankedTensorType::get(256, rewriter.getF32Type()), tableElems);
  OpBuilder nowhereBuilder(module->getContext(), rewriter.getListener());
  table = memref::GlobalOp::create(
      nowhereBuilder, op->getLoc(), extTableName,
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
    Value bitcast = BitcastOp::create(rewriter, loc, rewriter.getI8Type(), fp8);
    // Don't sign-extend the byte when index casting.
    Value i32 = ExtUIOp::create(rewriter, loc, rewriter.getI32Type(), bitcast);
    Value index =
        IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), i32);
    Value extended = memref::LoadOp::create(rewriter, loc, table, index);
    return extended;
  };

  auto floatsToResult = [&](Value floats) -> Value {
    if (outElemType.isF32())
      return floats;
    if (outElemType.getIntOrFloatBitWidth() < 32)
      return TruncFOp::create(rewriter, loc, outType, floats);
    if (outElemType.getIntOrFloatBitWidth() > 32)
      return ExtFOp::create(rewriter, loc, outType, floats);
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
        vector::ExtractOp::create(rewriter, loc, adaptor.getIn(), idx);
    Value extended = oneToFloat(scalar);
    floats = vector::InsertOp::create(rewriter, loc, extended, floats, idx);
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

  Value bits = BitcastOp::create(b, i32, in);
  const llvm::fltSemantics &outSem = outType.getFloatSemantics();

  Value and1 = AndIOp::create(b, bits, i32Const((1u << 23u) - 1));
  Value shr = ShRUIOp::create(b, bits, i32Const(23));
  Value and2 = AndIOp::create(b, shr, i32Const(0xff));
  Value ir1 = ShRUIOp::create(b, bits, i32Const(24));
  Value shl = AndIOp::create(b, ir1, i32Const(128));
  Value infNanConst = i32Const(0x7f800000);
  Value and4 = AndIOp::create(b, bits, infNanConst);
  Value cmp = CmpIOp::create(b, CmpIPredicate::eq, and4, infNanConst);

  Block *notInfNan = func.addBlock();
  Value outNan = ConstantFloatOp::create(b, outType, APFloat::getQNaN(outSem));
  cf::CondBranchOp::create(b, cmp, ret, ValueRange{outNan}, notInfNan,
                           ValueRange{});
  b.setInsertionPointToStart(notInfNan);

  // A deviation from the MIGraphX: denormals are zero here
  Value cmp5 = CmpIOp::create(b, CmpIPredicate::eq, and2, i32Const(0));
  Value outZero = ConstantFloatOp::create(b, outType, APFloat::getZero(outSem));
  Block *notZero = func.addBlock();
  cf::CondBranchOp::create(b, cmp5, ret, ValueRange{outZero}, notZero,
                           ValueRange{});
  b.setInsertionPointToStart(notZero);

  // For some reason, this is off by one
  uint32_t mBits = outType.getFPMantissaWidth() - 1;
  uint32_t eBits = 7 - mBits;
  Value sub = AddIOp::create(b, and2, i32Const(-127));
  Value reducedConst1 = i32Const(127 - ((1 << (eBits - 1)) - 2));
  Value cmp8 = CmpIOp::create(b, CmpIPredicate::ult, and2, reducedConst1);
  Value reducedConst2 = i32Const(127 - ((1 << (eBits - 1)) - 1));
  Value sub10 = SubIOp::create(b, reducedConst2, and2);
  Value exponentDiff0 = SelectOp::create(b, cmp8, sub10, i32Const(0));

  Value add12 = OrIOp::create(b, and1, i32Const(1 << 23));
  Value ir2 = MinUIOp::create(b, exponentDiff0, i32Const(15 - eBits));
  Value notmaskConst = i32Const(~((1 << (16 + eBits)) - 1));
  Value notmask = ShLIOp::create(b, notmaskConst, ir2);
  Value sub16 = XOrIOp::create(b, notmask, i32Const(-1));
  Value and17 = AndIOp::create(b, add12, sub16);
  Value sub21 = AddIOp::create(b, exponentDiff0, i32Const(15 + eBits));
  Value sroaSpeculataed140 = MinUIOp::create(b, sub21, i32Const(31));
  Value shl23 = ShLIOp::create(b, i32Const(1), sroaSpeculataed140);
  Value cmp24 = CmpIOp::create(b, CmpIPredicate::eq, and17, shl23);
  Value cmp25 =
      CmpIOp::create(b, CmpIPredicate::sgt, exponentDiff0, i32Const(0));
  Value sroaSpeculated = MinUIOp::create(b, exponentDiff0, i32Const(31));
  Value shr30 = SelectOp::create(b, cmp25, sroaSpeculated, i32Const(0));
  Value mantissa0 = ShRUIOp::create(b, add12, shr30);

  Value add40 = AddIOp::create(b, sub, exponentDiff0);
  Value and38 = ShRUIOp::create(b, mantissa0, i32Const(23));
  Value ir3 = OrIOp::create(b, and38, i32Const(-2));
  Value add41 = AddIOp::create(b, add40, ir3);
  Value sub43 = AddIOp::create(b, add41, i32Const((1 << (eBits - 1)) + 1));
  Value and44 = ShRUIOp::create(b, mantissa0, i32Const(16 + eBits));
  Value ir4 = AndIOp::create(b, and44, i32Const(1));
  Value resolvedConst3 = i32Const((1 << (16 + eBits)) - 1);
  Value sext = AddIOp::create(b, ir4, resolvedConst3);
  Value cond51 = SelectOp::create(b, cmp24, sext, i32Const(0));
  Value cond54 = AddIOp::create(b, cond51, mantissa0);
  Value and55 = AndIOp::create(b, cond54, resolvedConst3);
  Value add56 = AddIOp::create(b, and55, mantissa0);
  Value cmp57 = CmpIOp::create(b, CmpIPredicate::ne, sub43, i32Const(0));
  Value and58 = AndIOp::create(b, add56, i32Const(1 << 23));
  Value tobool59Not = CmpIOp::create(b, CmpIPredicate::eq, and58, i32Const(0));
  Value trueConst = ConstantIntOp::create(b, b.getI1Type(), true);
  Value brCond133 = SelectOp::create(b, cmp57, trueConst, tobool59Not);

  Block *ifElse61 = func.addBlock();
  Block *ifThen70 = func.addBlock();
  Block *ifEnd71 = func.addBlock();
  Block *ifEnd76 = func.addBlock();
  ifEnd76->addArguments({i32, i32}, {loc, loc});
  Value oneConst = i32Const(1);
  cf::CondBranchOp::create(b, brCond133, ifElse61, ValueRange{}, ifEnd76,
                           ValueRange{oneConst, add56});

  b.setInsertionPointToStart(ifElse61);
  Value tobool63Not =
      CmpIOp::create(b, CmpIPredicate::ugt, add56, i32Const((1 << 24) - 1));
  Value incConst = i32Const((1 << (eBits - 1)) + 2);
  Value inc = AddIOp::create(b, add41, incConst);
  Value f8Exponent0 = SelectOp::create(b, tobool63Not, inc, sub43);
  Value cmp69 = CmpIOp::create(b, CmpIPredicate::sgt, f8Exponent0,
                               i32Const((1 << eBits) - 1));
  cf::CondBranchOp::create(b, cmp69, ifThen70, ValueRange{}, ifEnd71,
                           ValueRange{});

  b.setInsertionPointToStart(ifThen70);
  Value ir5 = TruncIOp::create(b, i8, ir1);
  Value c127 = ConstantIntOp::create(b, i8, 127);
  Value conv = OrIOp::create(b, ir5, c127);
  Value convOut = BitcastOp::create(b, outType, conv);
  cf::BranchOp::create(b, ret, convOut);

  b.setInsertionPointToStart(ifEnd71);
  Value shr65 = ExtUIOp::create(b, i32, tobool63Not);
  Value mantissa1 = ShRUIOp::create(b, add56, shr65);
  Value cmp72 = CmpIOp::create(b, CmpIPredicate::eq, f8Exponent0, i32Const(0));
  Value cmp74 = CmpIOp::create(b, CmpIPredicate::ult, mantissa1,
                               i32Const(1 << (16 + eBits)));
  Value falseConst = ConstantIntOp::create(b, b.getI1Type(), false);
  Value brCond = SelectOp::create(b, cmp72, cmp74, falseConst);
  cf::CondBranchOp::create(b, brCond, ret, ValueRange{outZero}, ifEnd76,
                           ValueRange{f8Exponent0, mantissa1});

  b.setInsertionPointToStart(ifEnd76);
  Value f8Exponent015 = ifEnd76->getArgument(0);
  Value shr681In = ifEnd76->getArgument(1);
  Value shr681 = ShRUIOp::create(b, shr681In, i32Const(16 + eBits));
  Value and77 = AndIOp::create(b, shr681, i32Const((1 << mBits) - 1));
  Value shl79 = ShLIOp::create(b, f8Exponent015, i32Const(mBits));
  Value irOr = OrIOp::create(b, shl79, shl);
  Value or80 = OrIOp::create(b, irOr, and77);
  Value conv81 = TruncIOp::create(b, i8, or80);
  Value conv81Out = BitcastOp::create(b, outType, conv81);
  cf::BranchOp::create(b, ret, ValueRange{conv81Out});

  func.push_back(ret);
  b.setInsertionPointToStart(ret);
  Value retVal = ret->getArgument(0);
  func::ReturnOp::create(b, retVal);

  return symbolRef;
}

// Float8E5M2 and Float8E4M3FN
static FlatSymbolRefAttr
makeOCPFp8TruncFunction(Location loc, FloatType outType, Operation *module) {
  ImplicitLocOpBuilder b(loc, loc.getContext());
  SymbolTable symtab(module);

  SmallString<32> funcName;
  (Twine("_rocmlir_trunc_f32_to_") +
   (TypeSwitch<FloatType, StringRef>(outType)
        .Case<Float8E4M3FNType>(
            [](auto ignored) -> StringRef { return "f8E4M3FN"; })
        .Case<Float8E5M2Type>(
            [](auto ignored) -> StringRef { return "f8E5M2"; })
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
  Type i1 = b.getI1Type();
  auto i32Const = [&](uint32_t value) -> Value {
    return b.createOrFold<ConstantOp>(i32, b.getI32IntegerAttr(value));
  };
  auto i8Const = [&](uint32_t value) -> Value {
    return b.createOrFold<ConstantOp>(i8, b.getI8IntegerAttr(value));
  };
  auto i1Const = [&](bool value) -> Value {
    return b.createOrFold<ConstantOp>(i1, b.getBoolAttr(value));
  };

  // Mantissa width includes hidden bit so subtract.
  uint32_t mBits = outType.getFPMantissaWidth() - 1;
  uint32_t eBits = outType.getWidth() - 1 - mBits;
  Value mWidth = i32Const(mBits);
  Value eWidth = i32Const(eBits);

  // Created here so we can branch to it, will be inserted last
  Block *ret = new Block();
  ret->addArgument(i8, loc);

  Value bits = BitcastOp::create(b, i32, in);
  Value and22 = AndIOp::create(b, bits, i32Const((1u << 23u) - 1));
  Value shr23 = ShRUIOp::create(b, bits, i32Const(23));
  Value and24 = AndIOp::create(b, shr23, i32Const(0xff));
  Value shr25 = ShRUIOp::create(b, bits, i32Const(24));
  Value and26 = AndIOp::create(b, shr25, i32Const(128));
  Value shl27 = ShLIOp::create(b, i32Const(1), eWidth);
  Value add28 = AddIOp::create(b, shl27, i32Const(-1));
  Value shl29 = ShLIOp::create(b, add28, mWidth);
  Value add30 = AddIOp::create(b, shl29, and26);
  Value shl31 = ShLIOp::create(b, i32Const(-1), mWidth);
  Value xor32 = XOrIOp::create(b, shl31, i32Const(-1));
  Value add33 = AddIOp::create(b, add30, xor32);
  Value cmp34 = CmpIOp::create(b, CmpIPredicate::eq, mWidth, i32Const(2));
  Value infNanConst = i32Const(0x7f800000);
  Value and35 = AndIOp::create(b, bits, infNanConst);
  Value cmp36 = CmpIOp::create(b, CmpIPredicate::eq, and35, infNanConst);

  Block *bb1 = func.addBlock();
  Block *bb2 = func.addBlock();
  Block *bb3 = func.addBlock();
  Block *bb4 = func.addBlock();
  cf::CondBranchOp::create(b, cmp36, bb1, bb4);

  b.setInsertionPointToStart(bb1);
  Value cmp37 = CmpIOp::create(b, CmpIPredicate::eq, mWidth, i32Const(3));
  cf::CondBranchOp::create(b, cmp37, bb2, bb3);

  b.setInsertionPointToStart(bb2);
  Value trunc38 = TruncIOp::create(b, i8, add33);
  cf::BranchOp::create(b, ret, trunc38);

  // This block is a later edit, thus the numbers don't fit with bb6, etc.
  b.setInsertionPointToStart(bb3);
  Value cmp39 = CmpIOp::create(b, CmpIPredicate::eq, and22, i32Const(0));
  Value select40 = SelectOp::create(b, cmp39, add30, add33);
  Value trunc41 = TruncIOp::create(b, i8, select40);
  cf::BranchOp::create(b, ret, trunc41);

  Block *bb5 = func.addBlock();
  Block *bb6 = func.addBlock();
  b.setInsertionPointToStart(bb4);
  SmallVector<int32_t> caseLabels({0, std::numeric_limits<int32_t>::max()});
  SmallVector<Block *> caseSuccessors({ret, bb5});
  SmallVector<ValueRange> caseOperands({ValueRange{i8Const(0)}, ValueRange{}});
  cf::SwitchOp::create(b, loc, bits, bb6, ValueRange{}, caseLabels,
                       caseSuccessors, caseOperands);

  b.setInsertionPointToStart(bb5);
  cf::BranchOp::create(b, ret, i8Const(-128));

  Block *bb7 = func.addBlock();
  Block *bb8 = func.addBlock();
  b.setInsertionPointToStart(bb6);
  Value add41 = AddIOp::create(b, eWidth, i32Const(-1));
  Value shl42 = ShLIOp::create(b, i32Const(-1), add41);
  Value cmp43 = CmpIOp::create(b, CmpIPredicate::eq, and24, i32Const(0));
  Value cmp44 = CmpIOp::create(b, CmpIPredicate::ne, and22, i32Const(0));
  Value and45 = AndIOp::create(b, cmp44, cmp43);
  cf::CondBranchOp::create(b, and45, bb7, bb8);

  Block *bb9 = func.addBlock();
  bb9->addArgument(i32, loc);
  bb9->addArgument(i32, loc);
  bb9->addArgument(i32, loc);
  b.setInsertionPointToStart(bb7);
  Value add46 = AddIOp::create(b, shl42, i32Const(128));
  cf::BranchOp::create(b, bb9, ValueRange{i32Const(-126), add46, and22});

  b.setInsertionPointToStart(bb8);
  Value add47 = AddIOp::create(b, shl42, i32Const(2));
  Value add48 = AddIOp::create(b, and24, i32Const(-127));
  Value cmp49 = CmpIOp::create(b, CmpIPredicate::sgt, add48, add47);
  Value sub50 = SubIOp::create(b, add47, add48);
  Value or51 = OrIOp::create(b, and22, i32Const(0x800000));
  Value select52 = SelectOp::create(b, cmp49, i32Const(0), sub50);
  cf::BranchOp::create(b, bb9, ValueRange{add48, select52, or51});

  Block *bb10 = func.addBlock();
  Block *bb11 = func.addBlock();
  Block *bb12 = func.addBlock();
  bb12->addArgument(i32, loc);
  b.setInsertionPointToStart(bb9);
  Value bb9arg0 = bb9->getArgument(0);
  Value bb9arg1 = bb9->getArgument(1);
  Value bb9arg2 = bb9->getArgument(2);
  Value sub56 = SubIOp::create(b, i32Const(23), mWidth);
  Value add57 = AddIOp::create(b, bb9arg1, sub56);
  Value min58 = MinUIOp::create(b, add57, i32Const(31));
  Value shl59 = ShLIOp::create(b, i32Const(-1), min58);
  Value xor60 = XOrIOp::create(b, shl59, i32Const(-1));
  Value and61 = AndIOp::create(b, bb9arg2, xor60);
  Value add62 = AddIOp::create(b, add57, i32Const(-1));
  Value min63 = MinUIOp::create(b, add62, i32Const(31));
  Value shl64 = ShLIOp::create(b, i32Const(1), min63);
  Value cmp65 = CmpIOp::create(b, CmpIPredicate::eq, and61, shl64);
  Value cmp66 = CmpIOp::create(b, CmpIPredicate::sgt, bb9arg1, i32Const(0));
  cf::CondBranchOp::create(b, cmp66, bb10, bb11);

  b.setInsertionPointToStart(bb10);
  Value min67 = MinUIOp::create(b, bb9arg1, i32Const(31));
  Value shr68 = ShRUIOp::create(b, bb9arg2, min67);
  cf::BranchOp::create(b, bb12, ValueRange{shr68});

  b.setInsertionPointToStart(bb11);
  Value cmp69 = CmpIOp::create(b, CmpIPredicate::eq, bb9arg1, i32Const(-1));
  Value zext70 = ExtUIOp::create(b, i32, cmp69);
  Value shl71 = ShLIOp::create(b, bb9arg2, zext70);
  cf::BranchOp::create(b, bb12, ValueRange{shl71});

  Block *bb13 = func.addBlock();
  Block *bb14 = func.addBlock();
  Block *bb15 = func.addBlock();
  bb15->addArgument(i32, loc);
  bb15->addArgument(i32, loc);
  b.setInsertionPointToStart(bb12);
  Value bb12arg0 = bb12->getArgument(0);
  Value shr73 = ShRUIOp::create(b, bb12arg0, i32Const(23));
  Value or74 = OrIOp::create(b, shr73, i32Const(-2));
  Value sub75 = SubIOp::create(b, bb9arg0, shl42);
  Value add76 = AddIOp::create(b, sub75, bb9arg1);
  Value add77 = AddIOp::create(b, add76, or74);
  Value shl78 = ShLIOp::create(b, i32Const(1), sub56);
  Value add79 = AddIOp::create(b, shl78, i32Const(-1));
  Value and80 = AndIOp::create(b, bb12arg0, shl78);
  Value cmp81 = CmpIOp::create(b, CmpIPredicate::eq, and80, i32Const(0));
  Value select82 = SelectOp::create(b, cmp65, cmp81, i1Const(false));
  Value sext83 = ExtSIOp::create(b, i32, select82);
  Value add84 = AddIOp::create(b, bb12arg0, sext83);
  Value stoch = i1Const(false); // Defaulted arguments.
  Value rng = i32Const(0);
  Value select85 = SelectOp::create(b, stoch, rng, add84);
  Value and86 = AndIOp::create(b, select85, add79);
  Value add87 = AddIOp::create(b, and86, bb12arg0);
  Value cmp88 = CmpIOp::create(b, CmpIPredicate::ne, add77, i32Const(0));
  Value and89 = AndIOp::create(b, add87, i32Const(0x800000));
  Value cmp90 = CmpIOp::create(b, CmpIPredicate::eq, and89, i32Const(0));
  Value select91 = SelectOp::create(b, cmp88, i1Const(true), cmp90);
  cf::CondBranchOp::create(b, select91, bb13, bb15,
                           ValueRange{i32Const(1), add87});

  b.setInsertionPointToStart(bb13);
  Value and92 = AndIOp::create(b, add87, i32Const(0x1000000));
  Value cmp93 = CmpIOp::create(b, CmpIPredicate::eq, and92, i32Const(0));
  cf::CondBranchOp::create(b, cmp93, bb15, ValueRange{add77, add87}, bb14,
                           ValueRange{});

  b.setInsertionPointToStart(bb14);
  Value shr94 = ShRUIOp::create(b, add87, i32Const(1));
  Value add95 = AddIOp::create(b, add77, i32Const(1));
  cf::BranchOp::create(b, bb15, ValueRange{add95, shr94});

  Block *bb16 = func.addBlock();
  Block *bb17 = func.addBlock();
  b.setInsertionPointToStart(bb15);
  Value bb15arg0 = bb15->getArgument(0);
  Value bb15arg1 = bb15->getArgument(1);
  Value shr98 = ShRUIOp::create(b, bb15arg1, sub56);
  Value cmp99 = CmpIOp::create(b, CmpIPredicate::eq, mWidth, i32Const(3));
  Value select100 = SelectOp::create(b, cmp99, i32Const(-1), i32Const(-2));
  Value add101 = AddIOp::create(b, select100, shl27);
  Value cmp102 = CmpIOp::create(b, CmpIPredicate::sgt, bb15arg0, add101);
  cf::CondBranchOp::create(b, cmp102, bb16, bb17);

  Block *bb19 = func.addBlock();
  bb19->addArgument(i32, loc);
  b.setInsertionPointToStart(bb16);
  Value select103 = SelectOp::create(b, cmp34, add30, add33);
  cf::BranchOp::create(b, bb19, ValueRange{select103});

  Block *bb18 = func.addBlock();
  b.setInsertionPointToStart(bb17);
  Value cmp104 = CmpIOp::create(b, CmpIPredicate::eq, bb15arg0, i32Const(0));
  Value cmp105 = CmpIOp::create(b, CmpIPredicate::eq, shr98, i32Const(0));
  Value select106 = SelectOp::create(b, cmp104, cmp105, i1Const(false));
  cf::CondBranchOp::create(b, select106, bb19, ValueRange{and26}, bb18,
                           ValueRange{});

  b.setInsertionPointToStart(bb18);
  Value and107 = AndIOp::create(b, shr98, xor32);
  Value shl108 = ShLIOp::create(b, bb15arg0, mWidth);
  Value or109 = OrIOp::create(b, shl108, and107);
  Value or110 = OrIOp::create(b, or109, and26);
  cf::BranchOp::create(b, bb19, ValueRange{or110});

  b.setInsertionPointToStart(bb19);
  Value bb19arg0 = bb19->getArgument(0);
  Value trunc112 = TruncIOp::create(b, i8, bb19arg0);
  cf::BranchOp::create(b, ret, ValueRange{trunc112});

  func.push_back(ret);
  b.setInsertionPointToStart(ret);
  Value retVal = ret->getArgument(0);
  Value retOut = BitcastOp::create(b, outType, retVal);
  func::ReturnOp::create(b, retOut);

  return symbolRef;
}

LogicalResult Fp8TruncToCallPattern::match(TruncFOp op) const {
  if (failed(canBeConverted(op.getResult().getType(), hasF8ConversionInstrs,
                            hasOcpF8ConversionInstrs)))
    return failure();
  Type resType = getElementTypeOrSelf(op.getOut().getType());
  if (isa<Float8E4M3FNUZType>(resType) && !f8E4M3FNUZFunc)
    return failure();
  if (isa<Float8E5M2FNUZType>(resType) && !f8E5M2FNUZFunc)
    return failure();
  if (isa<Float8E4M3FNType>(resType) && !f8E4M3FNFunc)
    return failure();
  if (isa<Float8E5M2Type>(resType) && !f8E5M2Func)
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

  FlatSymbolRefAttr func =
      TypeSwitch<Type, FlatSymbolRefAttr>(outElemType)
          .Case<Float8E4M3FNUZType>(
              [&](auto ignored) { return f8E4M3FNUZFunc; })
          .Case<Float8E5M2FNUZType>(
              [&](auto ignored) { return f8E5M2FNUZFunc; })
          .Case<Float8E4M3FNType>([&](auto ignored) { return f8E4M3FNFunc; })
          .Case<Float8E5M2Type>([&](auto ignored) { return f8E5M2Func; })
          .Default([](auto ignored) { return nullptr; });

  auto oneToOut = [&](Value f32) -> Value {
    auto call = func::CallOp::create(rewriter, loc, func, outElemType, f32);
    return call.getResult(0);
  };

  Type inType = cloneOrReplace(rawInType, rewriter.getF32Type());
  Value in = rawIn;
  if (rawInElemType.getIntOrFloatBitWidth() < 32)
    in = arith::ExtFOp::create(rewriter, loc, inType, rawIn);
  else if (rawInElemType.getIntOrFloatBitWidth() > 32)
    in = arith::TruncFOp::create(rewriter, loc, inType, rawIn);

  auto inVecType = dyn_cast<VectorType>(inType);
  if (!inVecType)
    return rewriter.replaceOp(op, oneToOut(in));

  VectorType retVecType = inVecType.clone(outElemType);
  Value rets = rewriter.createOrFold<vector::SplatOp>(
      loc,
      rewriter.createOrFold<ConstantFloatOp>(
          loc, outElemType, APFloat::getZero(outElemType.getFloatSemantics())),
      retVecType);
  SmallVector<int64_t> strides = computeStrides(inVecType.getShape());
  for (int64_t i = 0, e = inVecType.getNumElements(); i < e; ++i) {
    SmallVector<int64_t> idx = delinearize(i, strides);
    Value scalar =
        vector::ExtractOp::create(rewriter, loc, adaptor.getIn(), idx);
    Value truncated = oneToOut(scalar);
    rets = vector::InsertOp::create(rewriter, loc, truncated, rets, idx);
  }
  return rewriter.replaceOp(op, rets);
}

void mlir::addEmulateFp8ExtTruncPatterns(RewritePatternSet &patterns,
                                         FlatSymbolRefAttr f8E4M3FNUZTruncFunc,
                                         FlatSymbolRefAttr f8E5M2FNUZTruncFunc,
                                         FlatSymbolRefAttr f8E4M3FNTruncFunc,
                                         FlatSymbolRefAttr f8E5M2TruncFunc,
                                         bool hasF8ConversionInstrs,
                                         bool hasOcpF8ConversionInstrs) {
  patterns.add<Fp8ExtToTableLookupPattern>(
      patterns.getContext(), hasF8ConversionInstrs, hasOcpF8ConversionInstrs);
  patterns.add<Fp8TruncToCallPattern>(
      patterns.getContext(), f8E4M3FNUZTruncFunc, f8E5M2FNUZTruncFunc,
      f8E4M3FNTruncFunc, f8E5M2TruncFunc, hasF8ConversionInstrs,
      hasOcpF8ConversionInstrs);
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
  target.addDynamicallyLegalOp<arith::ExtFOp>([this](ExtFOp op) {
    return failed(canBeConverted(op.getIn().getType(), hasFp8ConversionInstrs,
                                 hasOcpFp8ConversionInstrs));
  });
  target.addDynamicallyLegalOp<arith::TruncFOp>([this](TruncFOp op) {
    return failed(canBeConverted(op.getOut().getType(), hasFp8ConversionInstrs,
                                 hasOcpFp8ConversionInstrs));
  });

  FlatSymbolRefAttr f8E4M3FNUZTruncFunc = nullptr;
  FlatSymbolRefAttr f8E5M2FNUZTruncFunc = nullptr;
  FlatSymbolRefAttr f8E4M3FNTruncFunc = nullptr;
  FlatSymbolRefAttr f8E5M2TruncFunc = nullptr;
  SmallVector<Location> f8E4M3FNUZLocs, f8E5M2FNUZLocs, f8E4M3FNLocs,
      f8E5M2Locs;
  op->walk([&](TruncFOp op) {
    Type outElemType = getElementTypeOrSelf(op.getOut().getType());
    if (!hasFp8ConversionInstrs) {
      if (isa<Float8E4M3FNUZType>(outElemType))
        f8E4M3FNUZLocs.push_back(op->getLoc());
      else if (isa<Float8E5M2FNUZType>(outElemType))
        f8E5M2FNUZLocs.push_back(op->getLoc());
    }
    if (!hasOcpFp8ConversionInstrs) {
      if (isa<Float8E4M3FNType>(outElemType))
        f8E4M3FNLocs.push_back(op->getLoc());
      else if (isa<Float8E5M2Type>(outElemType))
        f8E5M2Locs.push_back(op->getLoc());
    }
  });

  if (!f8E4M3FNUZLocs.empty()) {
    f8E4M3FNUZTruncFunc = makeFp8TruncFunction(
        FusedLoc::get(ctx, f8E4M3FNUZLocs), Float8E4M3FNUZType::get(ctx), op);
  }
  if (!f8E5M2FNUZLocs.empty()) {
    f8E5M2FNUZTruncFunc = makeFp8TruncFunction(
        FusedLoc::get(ctx, f8E5M2FNUZLocs), Float8E5M2FNUZType::get(ctx), op);
  }
  if (!f8E4M3FNLocs.empty()) {
    f8E4M3FNTruncFunc = makeOCPFp8TruncFunction(
        FusedLoc::get(ctx, f8E4M3FNLocs), Float8E4M3FNType::get(ctx), op);
  }
  if (!f8E5M2Locs.empty()) {
    f8E5M2TruncFunc = makeOCPFp8TruncFunction(FusedLoc::get(ctx, f8E5M2Locs),
                                              Float8E5M2Type::get(ctx), op);
  }

  RewritePatternSet rewrites(ctx);
  addEmulateFp8ExtTruncPatterns(
      rewrites, f8E4M3FNUZTruncFunc, f8E5M2FNUZTruncFunc, f8E4M3FNTruncFunc,
      f8E5M2TruncFunc, hasFp8ConversionInstrs, hasOcpFp8ConversionInstrs);
  if (failed(applyPartialConversion(op, target, std::move(rewrites))))
    return signalPassFailure();
}
