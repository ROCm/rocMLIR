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
  FlatSymbolRefAttr f8E4M3FNFunc; // OCP
  FlatSymbolRefAttr f8E5M2Func;   // OCP

  // The functions are optional - if they aren't provided for a type (the null
  // attribute is sent in) the pattern will not apply.
  Fp8TruncToCallPattern(MLIRContext *ctx, FlatSymbolRefAttr f8E4M3FNUZFunc,
                        FlatSymbolRefAttr f8E5M2FNUZFunc,
                        FlatSymbolRefAttr f8E4M3FNFunc,
                        FlatSymbolRefAttr f8E5M2Func)
      : OpConversionPattern<TruncFOp>::OpConversionPattern(ctx),
        f8E4M3FNUZFunc(f8E4M3FNUZFunc), f8E5M2FNUZFunc(f8E5M2FNUZFunc),
        f8E4M3FNFunc(f8E4M3FNFunc), f8E5M2Func(f8E5M2Func) {}

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
    float x = entry.convertToFloat();
    uint32_t u = llvm::bit_cast<uint32_t>(x);
    // Hack:  Navi4 uses 0x7f800001 for all three NaN, and that's not
    // what APFloat will do.
    if (type.isFloat8E5M2()) {
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

  Value bits = b.create<BitcastOp>(i32, in);
  Value and22 = b.create<AndIOp>(bits, i32Const((1u << 23u) - 1));
  Value shr23 = b.create<ShRUIOp>(bits, i32Const(23));
  Value and24 = b.create<AndIOp>(shr23, i32Const(0xff));
  Value shr25 = b.create<ShRUIOp>(bits, i32Const(24));
  Value and26 = b.create<AndIOp>(shr25, i32Const(128));
  Value shl27 = b.create<ShLIOp>(i32Const(1), eWidth);
  Value add28 = b.create<AddIOp>(shl27, i32Const(-1));
  Value shl29 = b.create<ShLIOp>(add28, mWidth);
  Value add30 = b.create<AddIOp>(shl29, and26);
  Value shl31 = b.create<ShLIOp>(i32Const(-1), mWidth);
  Value xor32 = b.create<XOrIOp>(shl31, i32Const(-1));
  Value add33 = b.create<AddIOp>(add30, xor32);
  Value cmp34 = b.create<CmpIOp>(CmpIPredicate::eq, mWidth, i32Const(2));
  Value infNanConst = i32Const(0x7f800000);
  Value and35 = b.create<AndIOp>(bits, infNanConst);
  Value cmp36 = b.create<CmpIOp>(CmpIPredicate::eq, and35, infNanConst);

  Block *bb1 = func.addBlock();
  Block *bb2 = func.addBlock();
  Block *bb3 = func.addBlock();
  Block *bb4 = func.addBlock();
  b.create<cf::CondBranchOp>(cmp36, bb1, bb4);

  b.setInsertionPointToStart(bb1);
  Value cmp37 = b.create<CmpIOp>(CmpIPredicate::eq, mWidth, i32Const(3));
  b.create<cf::CondBranchOp>(cmp37, bb2, bb3);

  b.setInsertionPointToStart(bb2);
  Value trunc38 = b.create<TruncIOp>(i8, add33);
  b.create<cf::BranchOp>(ret, trunc38);

  // This block is a later edit, thus the numbers don't fit with bb6, etc.
  b.setInsertionPointToStart(bb3);
  Value cmp39 = b.create<CmpIOp>(CmpIPredicate::eq, and22, i32Const(0));
  Value select40 = b.create<SelectOp>(cmp39, add30, add33);
  Value trunc41 = b.create<TruncIOp>(i8, select40);
  b.create<cf::BranchOp>(ret, trunc41);

  Block *bb5 = func.addBlock();
  Block *bb6 = func.addBlock();
  b.setInsertionPointToStart(bb4);
  SmallVector<int32_t> caseLabels({0, static_cast<int>(2147483648)});
  SmallVector<Block *> caseSuccessors({ret, bb5});
  SmallVector<ValueRange> caseOperands({ValueRange{i8Const(0)}, ValueRange{}});
  b.create<cf::SwitchOp>(loc, bits, bb6, ValueRange{}, caseLabels,
                         caseSuccessors, caseOperands);

  b.setInsertionPointToStart(bb5);
  b.create<cf::BranchOp>(ret, i8Const(-128));

  Block *bb7 = func.addBlock();
  Block *bb8 = func.addBlock();
  b.setInsertionPointToStart(bb6);
  Value add41 = b.create<AddIOp>(eWidth, i32Const(-1));
  Value shl42 = b.create<ShLIOp>(i32Const(-1), add41);
  Value cmp43 = b.create<CmpIOp>(CmpIPredicate::eq, and24, i32Const(0));
  Value cmp44 = b.create<CmpIOp>(CmpIPredicate::ne, and22, i32Const(0));
  Value and45 = b.create<AndIOp>(cmp44, cmp43);
  b.create<cf::CondBranchOp>(and45, bb7, bb8);

  Block *bb9 = func.addBlock();
  bb9->addArgument(i32, loc);
  bb9->addArgument(i32, loc);
  bb9->addArgument(i32, loc);
  b.setInsertionPointToStart(bb7);
  Value add46 = b.create<AddIOp>(shl42, i32Const(128));
  b.create<cf::BranchOp>(bb9, ValueRange{i32Const(-126), add46, and22});

  b.setInsertionPointToStart(bb8);
  Value add47 = b.create<AddIOp>(shl42, i32Const(2));
  Value add48 = b.create<AddIOp>(and24, i32Const(-127));
  Value cmp49 = b.create<CmpIOp>(CmpIPredicate::sgt, add48, add47);
  Value sub50 = b.create<SubIOp>(add47, add48);
  Value or51 = b.create<OrIOp>(and22, i32Const(0x800000));
  Value select52 = b.create<SelectOp>(cmp49, i32Const(0), sub50);
  b.create<cf::BranchOp>(bb9, ValueRange{add48, select52, or51});

  Block *bb10 = func.addBlock();
  Block *bb11 = func.addBlock();
  Block *bb12 = func.addBlock();
  bb12->addArgument(i32, loc);
  b.setInsertionPointToStart(bb9);
  Value bb9arg0 = bb9->getArgument(0);
  Value bb9arg1 = bb9->getArgument(1);
  Value bb9arg2 = bb9->getArgument(2);
  Value sub56 = b.create<SubIOp>(i32Const(23), mWidth);
  Value add57 = b.create<AddIOp>(bb9arg1, sub56);
  Value min58 = b.create<MinUIOp>(add57, i32Const(31));
  Value shl59 = b.create<ShLIOp>(i32Const(-1), min58);
  Value xor60 = b.create<XOrIOp>(shl59, i32Const(-1));
  Value and61 = b.create<AndIOp>(bb9arg2, xor60);
  Value add62 = b.create<AddIOp>(add57, i32Const(-1));
  Value min63 = b.create<MinUIOp>(add62, i32Const(31));
  Value shl64 = b.create<ShLIOp>(i32Const(1), min63);
  Value cmp65 = b.create<CmpIOp>(CmpIPredicate::eq, and61, shl64);
  Value cmp66 = b.create<CmpIOp>(CmpIPredicate::sgt, bb9arg1, i32Const(0));
  b.create<cf::CondBranchOp>(cmp66, bb10, bb11);

  b.setInsertionPointToStart(bb10);
  Value min67 = b.create<MinUIOp>(bb9arg1, i32Const(31));
  Value shr68 = b.create<ShRUIOp>(bb9arg2, min67);
  b.create<cf::BranchOp>(bb12, ValueRange{shr68});

  b.setInsertionPointToStart(bb11);
  Value cmp69 = b.create<CmpIOp>(CmpIPredicate::eq, bb9arg1, i32Const(-1));
  Value zext70 = b.create<ExtUIOp>(i32, cmp69);
  Value shl71 = b.create<ShLIOp>(bb9arg2, zext70);
  b.create<cf::BranchOp>(bb12, ValueRange{shl71});

  Block *bb13 = func.addBlock();
  Block *bb14 = func.addBlock();
  Block *bb15 = func.addBlock();
  bb15->addArgument(i32, loc);
  bb15->addArgument(i32, loc);
  b.setInsertionPointToStart(bb12);
  Value bb12arg0 = bb12->getArgument(0);
  Value shr73 = b.create<ShRUIOp>(bb12arg0, i32Const(23));
  Value or74 = b.create<OrIOp>(shr73, i32Const(-2));
  Value sub75 = b.create<SubIOp>(bb9arg0, shl42);
  Value add76 = b.create<AddIOp>(sub75, bb9arg1);
  Value add77 = b.create<AddIOp>(add76, or74);
  Value shl78 = b.create<ShLIOp>(i32Const(1), sub56);
  Value add79 = b.create<AddIOp>(shl78, i32Const(-1));
  Value and80 = b.create<AndIOp>(bb12arg0, shl78);
  Value cmp81 = b.create<CmpIOp>(CmpIPredicate::eq, and80, i32Const(0));
  Value select82 = b.create<SelectOp>(cmp65, cmp81, i1Const(false));
  Value sext83 = b.create<ExtSIOp>(i32, select82);
  Value add84 = b.create<AddIOp>(bb12arg0, sext83);
  Value stoch = i1Const(false); // Defaulted arguments.
  Value rng = i32Const(0);
  Value select85 = b.create<SelectOp>(stoch, rng, add84);
  Value and86 = b.create<AndIOp>(select85, add79);
  Value add87 = b.create<AddIOp>(and86, bb12arg0);
  Value cmp88 = b.create<CmpIOp>(CmpIPredicate::ne, add77, i32Const(0));
  Value and89 = b.create<AndIOp>(add87, i32Const(0x800000));
  Value cmp90 = b.create<CmpIOp>(CmpIPredicate::eq, and89, i32Const(0));
  Value select91 = b.create<SelectOp>(cmp88, i1Const(true), cmp90);
  b.create<cf::CondBranchOp>(select91, bb13, bb15,
                             ValueRange{i32Const(1), add87});

  b.setInsertionPointToStart(bb13);
  Value and92 = b.create<AndIOp>(add87, i32Const(0x1000000));
  Value cmp93 = b.create<CmpIOp>(CmpIPredicate::eq, and92, i32Const(0));
  b.create<cf::CondBranchOp>(cmp93, bb15, ValueRange{add77, add87}, bb14,
                             ValueRange{});

  b.setInsertionPointToStart(bb14);
  Value shr94 = b.create<ShRUIOp>(add87, i32Const(1));
  Value add95 = b.create<AddIOp>(add77, i32Const(1));
  b.create<cf::BranchOp>(bb15, ValueRange{add95, shr94});

  Block *bb16 = func.addBlock();
  Block *bb17 = func.addBlock();
  b.setInsertionPointToStart(bb15);
  Value bb15arg0 = bb15->getArgument(0);
  Value bb15arg1 = bb15->getArgument(1);
  Value shr98 = b.create<ShRUIOp>(bb15arg1, sub56);
  Value cmp99 = b.create<CmpIOp>(CmpIPredicate::eq, mWidth, i32Const(3));
  Value select100 = b.create<SelectOp>(cmp99, i32Const(-1), i32Const(-2));
  Value add101 = b.create<AddIOp>(select100, shl27);
  Value cmp102 = b.create<CmpIOp>(CmpIPredicate::sgt, bb15arg0, add101);
  b.create<cf::CondBranchOp>(cmp102, bb16, bb17);

  Block *bb19 = func.addBlock();
  bb19->addArgument(i32, loc);
  b.setInsertionPointToStart(bb16);
  Value select103 = b.create<SelectOp>(cmp34, add30, add33);
  b.create<cf::BranchOp>(bb19, ValueRange{select103});

  Block *bb18 = func.addBlock();
  b.setInsertionPointToStart(bb17);
  Value cmp104 = b.create<CmpIOp>(CmpIPredicate::eq, bb15arg0, i32Const(0));
  Value cmp105 = b.create<CmpIOp>(CmpIPredicate::eq, shr98, i32Const(0));
  Value select106 = b.create<SelectOp>(cmp104, cmp105, i1Const(false));
  b.create<cf::CondBranchOp>(select106, bb19, ValueRange{and26}, bb18,
                             ValueRange{});

  b.setInsertionPointToStart(bb18);
  Value and107 = b.create<AndIOp>(shr98, xor32);
  Value shl108 = b.create<ShLIOp>(bb15arg0, mWidth);
  Value or109 = b.create<OrIOp>(shl108, and107);
  Value or110 = b.create<OrIOp>(or109, and26);
  b.create<cf::BranchOp>(bb19, ValueRange{or110});

  b.setInsertionPointToStart(bb19);
  Value bb19arg0 = bb19->getArgument(0);
  Value trunc112 = b.create<TruncIOp>(i8, bb19arg0);
  b.create<cf::BranchOp>(ret, ValueRange{trunc112});

  func.push_back(ret);
  b.setInsertionPointToStart(ret);
  Value retVal = ret->getArgument(0);
  Value retOut = b.create<BitcastOp>(outType, retVal);
  b.create<func::ReturnOp>(retOut);

  return symbolRef;
}

LogicalResult Fp8TruncToCallPattern::match(TruncFOp op) const {
  if (failed(canBeConverted(op.getResult().getType())))
    return failure();
  Type resType = getElementTypeOrSelf(op.getOut().getType());
  if (resType.isFloat8E4M3FNUZ() && !f8E4M3FNUZFunc)
    return failure();
  if (resType.isFloat8E5M2FNUZ() && !f8E5M2FNUZFunc)
    return failure();
  if (resType.isFloat8E4M3FN() && !f8E4M3FNFunc)
    return failure();
  if (resType.isFloat8E5M2() && !f8E5M2Func)
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

void mlir::addEmulateFp8ExtTruncPatterns(RewritePatternSet &patterns,
                                         FlatSymbolRefAttr f8E4M3FNUZTruncFunc,
                                         FlatSymbolRefAttr f8E5M2FNUZTruncFunc,
                                         FlatSymbolRefAttr f8E4M3FNTruncFunc,
                                         FlatSymbolRefAttr f8E5M2TruncFunc) {
  patterns.add<Fp8ExtToTableLookupPattern>(patterns.getContext());
  patterns.add<Fp8TruncToCallPattern>(patterns.getContext(),
                                      f8E4M3FNUZTruncFunc, f8E5M2FNUZTruncFunc,
                                      f8E4M3FNTruncFunc, f8E5M2TruncFunc);
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
  FlatSymbolRefAttr f8E4M3FNTruncFunc = nullptr;
  FlatSymbolRefAttr f8E5M2TruncFunc = nullptr;
  SmallVector<Location> f8E4M3FNUZLocs, f8E5M2FNUZLocs, f8E4M3FNLocs,
      f8E5M2Locs;
  op->walk([&](TruncFOp op) {
    Type outElemType = getElementTypeOrSelf(op.getOut().getType());
    if (outElemType.isFloat8E4M3FNUZ())
      f8E4M3FNUZLocs.push_back(op->getLoc());
    else if (outElemType.isFloat8E5M2FNUZ())
      f8E5M2FNUZLocs.push_back(op->getLoc());
    else if (outElemType.isFloat8E4M3FN())
      f8E4M3FNLocs.push_back(op->getLoc());
    else if (outElemType.isFloat8E5M2())
      f8E5M2Locs.push_back(op->getLoc());
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
  addEmulateFp8ExtTruncPatterns(rewrites, f8E4M3FNUZTruncFunc,
                                f8E5M2FNUZTruncFunc, f8E4M3FNTruncFunc,
                                f8E5M2TruncFunc);
  if (failed(applyPartialConversion(op, target, std::move(rewrites))))
    return signalPassFailure();
}
