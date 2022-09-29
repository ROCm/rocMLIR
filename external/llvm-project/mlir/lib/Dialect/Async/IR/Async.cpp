//===- Async.cpp - MLIR Async Operations ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Async/IR/Async.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::async;

#include "mlir/Dialect/Async/IR/AsyncOpsDialect.cpp.inc"

constexpr StringRef AsyncDialect::kAllowedToBlockAttrName;

void AsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Async/IR/AsyncOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // Get the underlying value types from async values returned from the
  // parent `async.execute` operation.
  auto executeOp = (*this)->getParentOfType<ExecuteOp>();
  auto types = llvm::map_range(executeOp.results(), [](const OpResult &result) {
    return result.getType().cast<ValueType>().getValueType();
  });

  if (getOperandTypes() != types)
    return emitOpError("operand types do not match the types returned from "
                       "the parent ExecuteOp");

  return success();
}

MutableOperandRange
YieldOp::getMutableSuccessorOperands(Optional<unsigned> index) {
  return operandsMutable();
}

//===----------------------------------------------------------------------===//
/// LaunchOp
//===----------------------------------------------------------------------===//

void LaunchOp::build(OpBuilder &builder, OperationState &result,
                     func::FuncOp func, ValueRange dependencies,
                     ValueRange operands) {
  // set callee
  result.addAttribute(calleeAttrName(result.name), SymbolRefAttr::get(func));

  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes =
      builder.getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(operand_segment_sizesAttrName(result.name),
                      operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `async.value`.
  result.addTypes({TokenType::get(result.getContext())});
  for (Type type : func.getResultTypes())
    result.addTypes(type);
}

/// Return the callee of this operation.
CallInterfaceCallable LaunchOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>(calleeAttrName());
}

/// Return the operands passed to the callee.
Operation::operand_range LaunchOp::getCallOperands() { return operands(); }

/// Return the callee results.
Operation::result_range LaunchOp::getCallResults() {
  return {++result_begin(), result_end()};
}

/// Return the callee result types.
Operation::result_type_range LaunchOp::getCallResultTypes() {
  return results();
}

/// Recompute the operand_segment_sizes attribute.
void LaunchOp::updateSegmentSizes(MLIRContext *ctx) {
  auto tokenTy = TokenType::get(ctx);
  int32_t numDependencies = 0;
  int32_t numOperands = 0;
  for (const auto &oper : getOperands()) {
    if (oper.getType() == tokenTy) {
      // All tokens should come first.
      assert(numOperands == 0);
      numDependencies++;
    } else
      numOperands++;
  }

  auto operandSegmentSizes =
      DenseI32ArrayAttr::get(ctx, {numDependencies, numOperands});
  (*this)->setAttr(operand_segment_sizesAttrName(), operandSegmentSizes);

  assert(!(*this)->hasAttr("result_segment_sizes"));
}

void LaunchOp::print(OpAsmPrinter &p) {
  // func ref
  p << " " << (*this)->getAttr(calleeAttrName());

  // [%tokens,...]
  if (!dependencies().empty())
    p << " [" << dependencies() << "]";

  // (%value, ...)
  p << " (" << operands() << ")";

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {operand_segment_sizesAttrName(), calleeAttrName()});

  // : (%value.type, ...)
  p << " : (";
  llvm::interleaveComma(operands(), p,
                        [&](Value operand) mutable { p << operand.getType(); });
  p << ")";

  // -> (return.type, ...)
  p.printArrowTypeList(llvm::drop_begin(getResultTypes()));
}

ParseResult LaunchOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();
  auto tokenTy = TokenType::get(ctx);

  FlatSymbolRefAttr calleeAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  SMLoc operandsOperandsLoc;
  ArrayRef<Type> operandsTypes;
  SmallVector<Type, 4> allResultTypes(1, tokenTy);

  if (parser.parseCustomAttributeWithFallback(
          calleeAttr, parser.getBuilder().getType<::mlir::NoneType>(),
          calleeAttrName(result.name), result.attributes)) {
    return ::mlir::failure();
  }

  // Parse dependency tokens.
  int32_t numDependencies = 0;
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> tokenArgs;
    if (parser.parseOperandList(tokenArgs) ||
        parser.resolveOperands(tokenArgs, tokenTy, result.operands) ||
        parser.parseRSquare())
      return failure();

    numDependencies = tokenArgs.size();
  }

  if (parser.parseLParen())
    return failure();
  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return failure();
  if (parser.parseRParen())
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  FunctionType operands__allResult_functionType;
  if (parser.parseType(operands__allResult_functionType))
    return failure();
  operandsTypes = operands__allResult_functionType.getInputs();
  auto resultTypes = operands__allResult_functionType.getResults();
  allResultTypes.append(resultTypes.begin(), resultTypes.end());
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(operandsOperands, operandsTypes,
                             operandsOperandsLoc, result.operands))
    return failure();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numOperands = result.operands.size() - numDependencies;
  auto operandSegmentSizes =
      DenseI32ArrayAttr::get(ctx, {numDependencies, numOperands});
  result.addAttribute(operand_segment_sizesAttrName(result.name),
                      operandSegmentSizes);

  return success();
}

LogicalResult LaunchOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callable = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!callable)
    return emitOpError("requires a 'callee' symbol reference attribute");

  func::FuncOp func =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, callable);
  if (!func)
    return emitOpError() << "'" << callable.getValue()
                         << "' does not reference a valid function";
  if (!func->hasAttr("kernel"))
    return emitOpError("requires a 'kernel' func reference");

  auto funcResultTypes = func.getResultTypes();
  // The result types should be a leading async.token and matching return types
  // of the kernel func.
  auto resultTypes = getResultTypes();
  if (resultTypes.size() != (funcResultTypes.size() + 1))
    return emitOpError(
        "requires matching result types with a leading async.token");

  auto resultItr = ++resultTypes.begin();
  for (auto resType : funcResultTypes) {
    if (*resultItr++ != resType)
      return emitOpError("requires matching result types with func");
  }

  // Match operand types
  auto funcArgumentTypes = func.getArgumentTypes();
  if (funcArgumentTypes.size() != operands().size())
    return emitOpError("incorrect number of operands for callee");

  for (auto tuple : llvm::zip(operands(), funcArgumentTypes)) {
    if (std::get<0>(tuple).getType() != std::get<1>(tuple))
      return emitOpError("requires matching operand types");
  }

  return success();
}

LogicalResult LaunchOp::verify() {
  MLIRContext *ctx = getContext();
  auto tokenTy = TokenType::get(ctx);

  // The dependencies must be async.tokens
  for (auto dep : dependencies()) {
    if (dep.getType() != tokenTy)
      return emitOpError("requires all dependencies to be async.token");
  }

  return success();
}

//===----------------------------------------------------------------------===//
/// ExecuteOp
//===----------------------------------------------------------------------===//

constexpr char kOperandSegmentSizesAttr[] = "operand_segment_sizes";

OperandRange ExecuteOp::getSuccessorEntryOperands(Optional<unsigned> index) {
  assert(index && *index == 0 && "invalid region index");
  return operands();
}

bool ExecuteOp::areTypesCompatible(Type lhs, Type rhs) {
  const auto getValueOrTokenType = [](Type type) {
    if (auto value = type.dyn_cast<ValueType>())
      return value.getValueType();
    return type;
  };
  return getValueOrTokenType(lhs) == getValueOrTokenType(rhs);
}

void ExecuteOp::getSuccessorRegions(Optional<unsigned> index,
                                    ArrayRef<Attribute>,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `body` region branch back to the parent operation.
  if (index) {
    assert(*index == 0 && "invalid region index");
    regions.push_back(RegionSuccessor(results()));
    return;
  }

  // Otherwise the successor is the body region.
  regions.push_back(RegionSuccessor(&body(), body().getArguments()));
}

void ExecuteOp::build(OpBuilder &builder, OperationState &result,
                      TypeRange resultTypes, ValueRange dependencies,
                      ValueRange operands, BodyBuilderFn bodyBuilder) {

  result.addOperands(dependencies);
  result.addOperands(operands);

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  int32_t numDependencies = dependencies.size();
  int32_t numOperands = operands.size();
  auto operandSegmentSizes =
      builder.getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // First result is always a token, and then `resultTypes` wrapped into
  // `async.value`.
  result.addTypes({TokenType::get(result.getContext())});
  for (Type type : resultTypes)
    result.addTypes(ValueType::get(type));

  // Add a body region with block arguments as unwrapped async value operands.
  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  for (Value operand : operands) {
    auto valueType = operand.getType().dyn_cast<ValueType>();
    bodyBlock.addArgument(valueType ? valueType.getValueType()
                                    : operand.getType(),
                          operand.getLoc());
  }

  // Create the default terminator if the builder is not provided and if the
  // expected result is empty. Otherwise, leave this to the caller
  // because we don't know which values to return from the execute op.
  if (resultTypes.empty() && !bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    builder.create<async::YieldOp>(result.location, ValueRange());
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock.getArguments());
  }
}

void ExecuteOp::print(OpAsmPrinter &p) {
  // [%tokens,...]
  if (!dependencies().empty())
    p << " [" << dependencies() << "]";

  // (%value as %unwrapped: !async.value<!arg.type>, ...)
  if (!operands().empty()) {
    p << " (";
    Block *entry = body().empty() ? nullptr : &body().front();
    llvm::interleaveComma(operands(), p, [&, n = 0](Value operand) mutable {
      Value argument = entry ? entry->getArgument(n++) : Value();
      p << operand << " as " << argument << ": " << operand.getType();
    });
    p << ")";
  }

  // -> (!async.value<!return.type>, ...)
  p.printOptionalArrowTypeList(llvm::drop_begin(getResultTypes()));
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {kOperandSegmentSizesAttr});
  p << ' ';
  p.printRegion(body(), /*printEntryBlockArgs=*/false);
}

ParseResult ExecuteOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();

  // Sizes of parsed variadic operands, will be updated below after parsing.
  int32_t numDependencies = 0;

  auto tokenTy = TokenType::get(ctx);

  // Parse dependency tokens.
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> tokenArgs;
    if (parser.parseOperandList(tokenArgs) ||
        parser.resolveOperands(tokenArgs, tokenTy, result.operands) ||
        parser.parseRSquare())
      return failure();

    numDependencies = tokenArgs.size();
  }

  // Parse async value operands (%value as %unwrapped : !async.value<!type>).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> valueArgs;
  SmallVector<OpAsmParser::Argument, 4> unwrappedArgs;
  SmallVector<Type, 4> valueTypes;

  // Parse a single instance of `%value as %unwrapped : !async.value<!type>`.
  auto parseAsyncValueArg = [&]() -> ParseResult {
    if (parser.parseOperand(valueArgs.emplace_back()) ||
        parser.parseKeyword("as") ||
        parser.parseArgument(unwrappedArgs.emplace_back()) ||
        parser.parseColonType(valueTypes.emplace_back()))
      return failure();

    auto valueTy = valueTypes.back().dyn_cast<ValueType>();
    unwrappedArgs.back().type = valueTy ? valueTy.getValueType() : Type();
    return success();
  };

  auto argsLoc = parser.getCurrentLocation();
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::OptionalParen,
                                     parseAsyncValueArg) ||
      parser.resolveOperands(valueArgs, valueTypes, argsLoc, result.operands))
    return failure();

  int32_t numOperands = valueArgs.size();

  // Add derived `operand_segment_sizes` attribute based on parsed operands.
  auto operandSegmentSizes =
      parser.getBuilder().getDenseI32ArrayAttr({numDependencies, numOperands});
  result.addAttribute(kOperandSegmentSizesAttr, operandSegmentSizes);

  // Parse the types of results returned from the async execute op.
  SmallVector<Type, 4> resultTypes;
  NamedAttrList attrs;
  if (parser.parseOptionalArrowTypeList(resultTypes) ||
      // Async execute first result is always a completion token.
      parser.addTypeToList(tokenTy, result.types) ||
      parser.addTypesToList(resultTypes, result.types) ||
      // Parse operation attributes.
      parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();

  result.addAttributes(attrs);

  // Parse asynchronous region.
  Region *body = result.addRegion();
  return parser.parseRegion(*body, /*arguments=*/unwrappedArgs);
}

LogicalResult ExecuteOp::verifyRegions() {
  // Unwrap async.execute value operands types.
  auto unwrappedTypes = llvm::map_range(operands(), [](Value operand) {
    return operand.getType().cast<ValueType>().getValueType();
  });

  // Verify that unwrapped argument types matches the body region arguments.
  if (body().getArgumentTypes() != unwrappedTypes)
    return emitOpError("async body region argument types do not match the "
                       "execute operation arguments types");

  return success();
}

//===----------------------------------------------------------------------===//
/// CreateGroupOp
//===----------------------------------------------------------------------===//

LogicalResult CreateGroupOp::canonicalize(CreateGroupOp op,
                                          PatternRewriter &rewriter) {
  // Find all `await_all` users of the group.
  llvm::SmallVector<AwaitAllOp> awaitAllUsers;

  auto isAwaitAll = [&](Operation *op) -> bool {
    if (AwaitAllOp awaitAll = dyn_cast<AwaitAllOp>(op)) {
      awaitAllUsers.push_back(awaitAll);
      return true;
    }
    return false;
  };

  // Check if all users of the group are `await_all` operations.
  if (!llvm::all_of(op->getUsers(), isAwaitAll))
    return failure();

  // If group is only awaited without adding anything to it, we can safely erase
  // the create operation and all users.
  for (AwaitAllOp awaitAll : awaitAllUsers)
    rewriter.eraseOp(awaitAll);
  rewriter.eraseOp(op);

  return success();
}

//===----------------------------------------------------------------------===//
/// AwaitOp
//===----------------------------------------------------------------------===//

void AwaitOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    ArrayRef<NamedAttribute> attrs) {
  result.addOperands({operand});
  result.attributes.append(attrs.begin(), attrs.end());

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = operand.getType().dyn_cast<ValueType>())
    result.addTypes(valueType.getValueType());
}

static ParseResult parseAwaitResultType(OpAsmParser &parser, Type &operandType,
                                        Type &resultType) {
  if (parser.parseType(operandType))
    return failure();

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = operandType.dyn_cast<ValueType>())
    resultType = valueType.getValueType();

  return success();
}

static void printAwaitResultType(OpAsmPrinter &p, Operation *op,
                                 Type operandType, Type resultType) {
  p << operandType;
}

LogicalResult AwaitOp::verify() {
  Type argType = operand().getType();

  // Awaiting on a token does not have any results.
  if (argType.isa<TokenType>() && !getResultTypes().empty())
    return emitOpError("awaiting on a token must have empty result");

  // Awaiting on a value unwraps the async value type.
  if (auto value = argType.dyn_cast<ValueType>()) {
    if (*getResultType() != value.getValueType())
      return emitOpError() << "result type " << *getResultType()
                           << " does not match async value type "
                           << value.getValueType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Async/IR/AsyncOpsTypes.cpp.inc"

void ValueType::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getValueType());
  printer << '>';
}

Type ValueType::parse(mlir::AsmParser &parser) {
  Type ty;
  if (parser.parseLess() || parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "failed to parse async value type");
    return Type();
  }
  return ValueType::get(ty);
}
