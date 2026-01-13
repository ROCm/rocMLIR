//===- TosaToRock.cpp - Lowering Tosa to Rock Dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These rewriters lower from the Tosa to the Rock dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TosaToRock/TosaToRock.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockTosaCustomOps.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Tuning/ConvContext.h"
#include "mlir/Dialect/Rock/Tuning/RockTuning.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/tosaUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>
#include <utility>

#define DEBUG_TYPE "convert-tosa-to-rock"

using namespace mlir;

namespace {
// Note:  we want something a bit more general than SmallString<8> for
// the layout string, but it has to allow for inserting a character into
// the string for the caller to see.
static Value expandTensor(PatternRewriter &rw, Operation *op, Value operand,
                          SmallString<8> &layout, StringRef lowerName,
                          int64_t g, uint32_t idx = 4) {
  auto loc = op->getLoc();
  auto oprType = cast<ShapedType>(operand.getType());
  if (!oprType.hasStaticShape()) {
    (void)rw.notifyMatchFailure(
        op, "tosa to rock conversion expects statically shaped tensors");
    return Value();
  }
  ArrayRef<int64_t> shape = oprType.getShape();

  SmallVector<uint32_t, 8> endDims;
  SmallVector<uint32_t, 8> startDims;
  SmallVector<StringRef, 8> startNames;

  // find the lower dimension that encodes the g dimension
  std::optional<uint32_t> groupFoldedDim = std::nullopt;

  for (uint32_t i = 0, e = shape.size(); i < e; ++i) {
    startNames.push_back(layout.substr(i, 1));
    if (layout[i] == lowerName[0]) {
      groupFoldedDim = i;
    } else {
      startDims.push_back(i);
      endDims.push_back(groupFoldedDim.has_value() ? i + 1 : i);
    }
  }

  if (!groupFoldedDim.has_value()) {
    (void)rw.notifyMatchFailure(op, "tosa conv has an invalid layout");
    return Value();
  }

  uint32_t lowerDim = groupFoldedDim.value();
  // insert 'g' dimension into layout
  rock::BottomUpTMBuilder transform(rw, ArrayRef<StringRef>(startNames), shape,
                                    loc);
  transform.passThrough(endDims, startDims);
  transform.unmerge({"g", lowerName}, {lowerDim, lowerDim + 1}, lowerName,
                    {{g, shape[lowerDim] / g}});
  layout = Twine(layout.substr(0, lowerDim) + "g" +
                 layout.substr(lowerDim, layout.size() - lowerDim))
               .str();

  return rock::TransformOp::create(rw, loc, operand, transform.get());
}

static rock::GemmFeatures getGemmFeaturesFromOp(Operation *op, Type inputType) {
  // Start by getting the arch from the Tosa op
  StringAttr arch = StringAttr::get(op->getContext(), "");
  FailureOr<StringAttr> maybeArch = rock::getArch(op);
  if (succeeded(maybeArch)) {
    arch = maybeArch.value();
  }

  // Now we can lookup the default features from the arch
  rock::AmdArchInfo archInfo = rock::lookupArchInfo(arch);
  rock::GemmFeatures features = archInfo.getDefaultFeatures(inputType);

  return features;
}

struct ConvFields {
  SmallString<8> filterLayout;
  SmallString<8> inputLayout;
  SmallString<8> outputLayout;
  Value inputExp;
  Value filterExp;
  Value outputExp;
  ArrayAttr pad;
  ArrayAttr stride;
  ArrayAttr dilation;
  rock::GemmFeaturesAttr features;
  StringAttr perfConfig;
};

static ConvFields commonConv(PatternRewriter &rw, Operation *op, Value input,
                             Value filter, Value output, DenseI64ArrayAttr pad,
                             DenseI64ArrayAttr stride,
                             DenseI64ArrayAttr dilation, int64_t group) {
  ConvFields res;

  res.filterLayout = "kyxc";
  if (auto attr = op->getAttrOfType<StringAttr>("filter_layout"))
    res.filterLayout = attr.getValue();
  else if (cast<ShapedType>(filter.getType()).getRank() > 4)
    res.filterLayout = "k012c";

  res.inputLayout = "nhwc";
  if (auto attr = op->getAttrOfType<StringAttr>("input_layout"))
    res.inputLayout = attr.getValue();
  else if (cast<ShapedType>(input.getType()).getRank() > 4)
    res.inputLayout = "n012c";
  if (output) {
    res.outputLayout = "nhwk";
    if (auto attr = op->getAttrOfType<StringAttr>("output_layout"))
      res.outputLayout = attr.getValue();
    else if (cast<ShapedType>(output.getType()).getRank() > 4)
      res.outputLayout = "n012k";
  }

  // expand tensors from rank 4 (NHWC) to rank 5 (NHWCG)
  // and add 'g into the layout
  res.inputExp = expandTensor(rw, op, input, res.inputLayout, "c", group);
  res.filterExp = expandTensor(rw, op, filter, res.filterLayout, "k", group);
  if (output)
    res.outputExp = expandTensor(rw, op, output, res.outputLayout, "k", group);

  res.pad = rw.getIndexArrayAttr(pad);
  res.stride = rw.getIndexArrayAttr(stride);
  res.dilation = rw.getIndexArrayAttr(dilation);
  res.perfConfig = op->getAttrOfType<StringAttr>("perf_config");

  return res;
}

static void addConvAttributes(PatternRewriter &rw, Operation *cop,
                              const ConvFields &convFields) {
  // specify layout attributes
  SmallVector<StringAttr, 5> filterLayoutSpec;
  SmallVector<StringAttr, 5> inputLayoutSpec;
  SmallVector<StringAttr, 5> outputLayoutSpec;
  for (size_t i = 0; i < convFields.filterLayout.size(); ++i) {
    filterLayoutSpec.push_back(
        rw.getStringAttr(convFields.filterLayout.substr(i, 1)));
    inputLayoutSpec.push_back(
        rw.getStringAttr(convFields.inputLayout.substr(i, 1) + "i"));
    if (convFields.outputExp)
      outputLayoutSpec.push_back(
          rw.getStringAttr(convFields.outputLayout.substr(i, 1) + "o"));
  }

  // arch-specific attributes
  // TODO: remove these
  if (auto attr = convFields.perfConfig)
    cop->setAttr("perf_config", attr);

  // convolution config attributes
  cop->setAttr("filter_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(filterLayoutSpec.begin(),
                                                   filterLayoutSpec.end())));
  cop->setAttr("input_layout",
               rw.getArrayAttr(ArrayRef<Attribute>(inputLayoutSpec.begin(),
                                                   inputLayoutSpec.end())));
  if (convFields.outputExp)
    cop->setAttr("output_layout",
                 rw.getArrayAttr(ArrayRef<Attribute>(outputLayoutSpec.begin(),
                                                     outputLayoutSpec.end())));
}

static FailureOr<rock::RockConvInterface>
makeRockConv(ConversionPatternRewriter &rw, Operation *op, Value input,
             Value filter, Value output, DenseI64ArrayAttr pad,
             DenseI64ArrayAttr stride, DenseI64ArrayAttr dilation,
             int64_t group, int64_t kernelID,
             std::optional<std::string> convBackwardKind) {
  Location loc = op->getLoc();
  ConvFields convFields =
      commonConv(rw, op, input, filter, output, pad, stride, dilation, group);

  Operation *cop = nullptr;
  if (convBackwardKind.has_value() &&
      convBackwardKind.value() == ROCK_CUSTOMOP_CONV_BWD_DATA) {
    cop = rock::ConvBwdDataOp::create(
        rw, loc, convFields.outputExp.getType(), convFields.filterExp,
        convFields.outputExp, convFields.inputExp,
        /*features=*/nullptr,
        /*blockSize=*/nullptr,
        /*gridSize=*/nullptr, rw.getIndexArrayAttr(pad),
        rw.getIndexArrayAttr(stride), rw.getIndexArrayAttr(dilation),
        /*params=*/nullptr, rw.getIndexAttr(kernelID),
        /*usesV4R1=*/rw.getBoolAttr(false));
  } else {
    // Handle forwards convolution
    assert((!convBackwardKind.has_value() ||
            convBackwardKind.value() != ROCK_CUSTOMOP_CONV_BWD_WEIGHT) &&
           "bwd_weight currently not implemented");
    cop = rock::ConvOp::create(
        rw, loc, convFields.outputExp.getType(), convFields.filterExp,
        convFields.inputExp, convFields.outputExp, /*features=*/nullptr,
        /*blockSize=*/nullptr, /*gridSize=*/nullptr, convFields.pad,
        convFields.stride, convFields.dilation, /*params=*/nullptr);
  }

  addConvAttributes(rw, cop, convFields);

  return cast<rock::RockConvInterface>(cop);
}

static Value traceToRes(Value tensor, DenseMap<Value, Value> &cache,
                        Value expectedTensor) {
  if (cache.contains(tensor))
    return cache.at(tensor);

  Value res = nullptr;
  if (tensor.getDefiningOp()) {
    if (expectedTensor == tensor) {
      res = tensor;
    } else if (auto view = tensor.getDefiningOp<ViewLikeOpInterface>()) {
      res = traceToRes(view.getViewSource(), cache, expectedTensor);
    } else if (auto expand = tensor.getDefiningOp<tensor::ExpandShapeOp>()) {
      res = traceToRes(expand.getSrc(), cache, expectedTensor);
    } else if (auto collapse =
                   tensor.getDefiningOp<tensor::CollapseShapeOp>()) {
      res = traceToRes(collapse.getSrc(), cache, expectedTensor);
    } else if (auto untransform =
                   tensor.getDefiningOp<rock::TensorUntransformCastOp>()) {
      res =
          traceToRes(untransform.getTransformedResult(), cache, expectedTensor);
    } else if (auto tosaOp = tensor.getDefiningOp<tosa::TosaOp>()) {
      for (auto operand : tosaOp->getOperands()) {
        if (llvm::isa<TensorType>(operand.getType())) {
          res = traceToRes(operand, cache, expectedTensor);
          if (res)
            break;
        }
      }
    }
  }

  cache.insert({tensor, res});
  return res;
}

static SetVector<int64_t> traceToRes(Value expectedTensor, func::FuncOp func) {
  llvm::DenseMap<Value, Value> cache;

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp returnOp) { returns.push_back(returnOp); });
  assert(returns.size() == 1 && "Number of returns is not one");
  func::ReturnOp returnOp = returns[0];

  SetVector<int64_t> resIndices;
  for (auto [i, res] : llvm::enumerate(returnOp->getOperands())) {
    Value out = traceToRes(res, cache, expectedTensor);
    if (out == expectedTensor)
      resIndices.insert(i);
  }
  return resIndices;
}

template <typename OpT>
static LogicalResult setSplitKAttrs(OpT op, rock::GemmFeatures features,
                                    PatternRewriter &rw) {
  auto perfConfig = op->template getAttrOfType<StringAttr>("perf_config");
  if (perfConfig && rock::isSplitKRequested(features, perfConfig)) {
    func::FuncOp func = op->template getParentOfType<func::FuncOp>();
    SetVector<int64_t> resIndices = traceToRes(op->getResult(0), func);
    if (resIndices.empty())
      return op.emitOpError(
          "can't trace the operation output to a kernel result");

    func::ReturnOp returnOp;
    func.walk([&](func::ReturnOp op) { returnOp = op; });
    for (int64_t resNumber : resIndices) {
      Type elementType =
          cast<ShapedType>(returnOp->getOperand(resNumber).getType())
              .getElementType();
      if (!isa<Float32Type, Float16Type, BFloat16Type>(elementType)) {
        return rw.notifyMatchFailure(
            op, "We only support F32, F16 and BF16 split-k, yet.");
      }
      Attribute outputInitVal = rw.getFloatAttr(elementType, 0.0);
      func.setResultAttr(resNumber, rock::PrefillAttr::getMnemonic(),
                         outputInitVal);
      func.setResultAttr(resNumber, "read_access", rw.getUnitAttr());
      // The original function also need the read access attr for the output.
      if (func->hasAttr("original_func")) {
        if (ModuleOp rootMod = func->getParentOfType<ModuleOp>()
                                   ->getParentOfType<ModuleOp>()) {
          SymbolTable symTable(rootMod);
          SymbolRefAttr originalFuncAttr =
              func->getAttrOfType<SymbolRefAttr>("original_func");
          if (func::FuncOp originalFunc = dyn_cast<func::FuncOp>(
                  symTable.lookupSymbolIn(rootMod, originalFuncAttr))) {
            originalFunc.setResultAttr(resNumber, "read_access",
                                       rw.getUnitAttr());
          }
        }
      }
    }
  }
  return success();
}

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// Elementwise trait is strict about matching dimensions, but
// broadcastable ops are also element-wise, and we know that an
// additional set of ops are also element-wise.
static bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
         // clang-format off
    isa<tosa::CastOp,
        tosa::ClampOp,
        tosa::ErfOp,
        tosa::SigmoidOp,
        tosa::TanhOp,
        tosa::AbsOp,
        tosa::CeilOp,
        tosa::ClzOp,
        tosa::ExpOp,
        tosa::FloorOp,
        tosa::LogOp,
        tosa::LogicalNotOp,
        tosa::NegateOp,
        tosa::ReciprocalOp,
        tosa::RsqrtOp,
        tosa::SelectOp,
        tosa::EqualOp,
        tosa::GreaterOp,
        tosa::GreaterEqualOp,
        tosa::MulOp
       >(op);
  // clang-format on
}

static Value addBlockArgument(OpBuilder &b, Value val, Block *block,
                              Location loc) {
  RankedTensorType valType = cast<RankedTensorType>(val.getType());
  val = block->addArgument(
      MemRefType::get(valType.getShape(), valType.getElementType()), loc);
  val = rock::getAsTensor(b, loc, val);
  return val;
}

static Operation *getConvOp(Operation *op) {
  if (isa<tensor::ExpandShapeOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }
  if (!op)
    return nullptr;

  if (isa<tensor::CollapseShapeOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
  }
  if (!op)
    return nullptr;

  while (isa<tosa::TransposeOp>(op)) {
    op = op->getOperand(0).getDefiningOp();
    if (!op)
      return nullptr;
  }
  return ((isa_and_nonnull<tosa::Conv2DOp>(op)) ||
          (isa_and_nonnull<tosa::TransposeConv2DOp>(op)))
             ? op
             : nullptr;
}

/*
GEMM+GEMM based ops can have elementwise region between first gemm and second
gemm. This helps with matching such GEMM+GEMM ops and also constructing the
elementwise region afterwards.
*/
template <typename OpT>
struct ElementwiseRegionFinder {
  /*
  This is simple DFS traversal to find out if it can hit gemm/conv op from the
  input. It keeps track of visited nodes to avoid cycles. It caches visited ops
  in topological order for rewrite. It also caches constant values and block
  argument candidates which will be used during rewrite.
  */
  void visit(Value input) {
    if (visitedSet.contains(input))
      return;
    visitedSet.insert(input);
    OpT fusionOp = input.getDefiningOp<OpT>();
    Operation *op = input.getDefiningOp();

    // We cannot handle bwd_data/weight conv ops + gemm yet, so bail early
    if (std::is_same_v<OpT, tosa::TransposeConv2DOp> && op)
      return;

    // we need to traverse tranposes if it's conv2d
    if (std::is_same_v<OpT, tosa::Conv2DOp> && op) {
      Operation *convOp = getConvOp(op);
      if (convOp)
        fusionOp = cast<OpT>(convOp);
    }
    if (fusionOp) {
      firstGemmBasedOp = fusionOp;
      firstGemmBasedVal = input;
      // cache blockArgCandidates for rewrite
      blockArgCandidates.push_back(input);
      return;
    }
    if (op && dyn_cast<tosa::ConstOp>(op)) {
      constantVals.push_back(input);
      return;
    }
    // Right now, this is a bit restricted that we only allow reshape-like
    // ops between in the elementwise tree that get fused to the fusion point.
    // TODO: however, the latest code gridwise-gemm-to-blockwise should tackle
    // more cases. The absolute restriction is gemm0Output to Linalg block
    // should contain invertible transforms, but that's future work.
    if (!op || (!isElementwiseOp(op) &&
                !isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))) {
      // cache blockArgCandidates for rewrite
      blockArgCandidates.push_back(input);
      return;
    }
    for (Value operand : op->getOperands()) {
      // do a DFS on each operand
      visit(operand);
    }
    // keep topological order for rewrite
    visitedOps.push_back(op);
  }

  FailureOr<OpT> getFirstGemmBasedOp() const {
    if (!firstGemmBasedOp)
      return failure();
    return firstGemmBasedOp;
  }

  SmallVector<Value> getElementwiseArgs() const {
    // ElementwiseArgs doesn't contain output from the first gemm explictly.
    // Therefore remove it.
    SmallVector<Value> elementwiseArgs = blockArgCandidates;
    uint64_t firstGemmBlockIndex = getFirstGemmBlockIndex();
    elementwiseArgs.erase(elementwiseArgs.begin() + firstGemmBlockIndex);
    return elementwiseArgs;
  }

  int64_t getFirstGemmBlockIndex() const {
    return std::find_if(blockArgCandidates.begin(), blockArgCandidates.end(),
                        [this](Value v) { return v == firstGemmBasedVal; }) -
           blockArgCandidates.begin();
  }

  void rewrite(Value input, OpBuilder &regionBuilder, Block *block,
               Location loc) const {
    PatternRewriter::InsertionGuard guard(regionBuilder);
    regionBuilder.setInsertionPointToEnd(block);
    IRMapping mapper;
    for (Value v : constantVals) {
      auto *newConstOp = regionBuilder.clone(*v.getDefiningOp());
      mapper.map(v, newConstOp->getResult(0));
    }
    for (Value v : blockArgCandidates) {
      auto newBlockArg = addBlockArgument(regionBuilder, v, block, loc);
      mapper.map(v, newBlockArg);
    }
    // make sure firstGemmBasedVal is passed as blockArgument for it is always
    // present
    Value lastRes = mapper.lookup(firstGemmBasedVal);
    for (Operation *op : visitedOps) {
      auto *newOp = regionBuilder.clone(*op, mapper);
      lastRes = newOp->getResult(0);
      mapper.map(lastRes, newOp->getResult(0));
    }
    RankedTensorType resTensorType = cast<RankedTensorType>(lastRes.getType());
    MemRefType resMemRefType = MemRefType::get(resTensorType.getShape(),
                                               resTensorType.getElementType());
    Value resMemref = bufferization::ToBufferOp::create(
        regionBuilder, loc,
        cast<mlir::bufferization::BufferLikeType>(resMemRefType), lastRes);
    Value outMemref = block->addArgument(resMemRefType, loc);
    memref::CopyOp::create(regionBuilder, loc, resMemref, outMemref);
    rock::YieldOp::create(regionBuilder, loc);
  }

private:
  OpT firstGemmBasedOp = nullptr;
  Value firstGemmBasedVal = nullptr;
  DenseSet<Value> visitedSet;
  SmallVector<Value> blockArgCandidates;
  SmallVector<Value> constantVals;
  SmallVector<Operation *> visitedOps;
};

static void addZeroInitPrefillAttribute(tosa::CustomOp op,
                                        Operation *rockConv) {
  // First check if the TransposeConv2D op is going to require having it's
  // output zeroinitialized, i.e., not every element of the output buffer is
  // going to be written to
  rock::ConvolutionContext ctx = rock::populateConvContext(rockConv);
  auto strideDims = ctx.getStrideVal();
  auto dilationDims = ctx.getDilationVal();
  auto filterDims = ctx.getConvDims().fil;
  auto numKernels =
      rock::backwardDataKernelIds(strideDims, dilationDims, filterDims,
                                  /*usesV4R1=*/true);

  // If there is no zeroinit kernel needed, then there is nothing more we need
  // to do here.
  if (rock::isEveryElementWrittenBwdData(strideDims, dilationDims, filterDims))
    return;

  // Now we need to determine where to add the prefill attributes. Trace through
  // the output of the TransposeConv2D op to find where the result is used.
  Value output = op.getResult(0);
  func::FuncOp func = op->getParentOfType<func::FuncOp>();
  if (!func)
    return;

  SetVector<int64_t> resIndices = traceToRes(output, func);
  // If the output cannot be traced to a result index, then we have a case that
  // we cannot yet handle
  if (resIndices.empty())
    assert(false &&
           "Output of TransposeConv2D op cannot be traced to result index");

  OpBuilder builder(op.getContext());
  for (int64_t resNumber : resIndices) {
    Type funcResType = func.getFunctionType().getResult(resNumber);
    auto shapedResType = cast<ShapedType>(funcResType);
    Type elementType = shapedResType.getElementType();

    Attribute outputInitVal;
    if (isa<FloatType>(elementType)) {
      outputInitVal = builder.getFloatAttr(elementType, 0.0);
    } else if (isa<IntegerType>(elementType)) {
      outputInitVal = builder.getIntegerAttr(elementType, 0);
    } else {
      // We only expect integer and float types for now
      assert(false && "Unsupported element type for prefill attribute");
    }

    func.setResultAttr(resNumber, rock::PrefillAttr::getMnemonic(),
                       outputInitVal);
  }
}

static FailureOr<tosa::AddOp>
replaceCstZeroWithAddNBcast(MLIRContext *context, ConversionPatternRewriter &rw,
                            Location loc, Type resTy, Value bias, Value input,
                            Value result) {
  // non-zero bias, replace with tosa.add w/ broadcast
  auto biasType = cast<ShapedType>(bias.getType());
  if (!biasType.hasStaticShape())
    return failure();

  int64_t nDims = cast<ShapedType>(input.getType()).getRank();
  SmallVector<int64_t> biasShape;
  for (int i = 0; i < nDims - 1; i++)
    biasShape.push_back(1);
  biasShape.push_back(biasType.getShape()[0]);
  auto newType = RankedTensorType::get(biasShape, biasType.getElementType());

  // [[0, 1, 2, 3]]
  ReassociationExprs exprs;
  for (int i = 0; i < nDims; i++)
    exprs.push_back(getAffineDimExpr(i, context));
  SmallVector<ReassociationExprs, 1> reassociations;
  reassociations.push_back(exprs);

  auto biasExpand =
      tensor::ExpandShapeOp::create(rw, loc, newType, bias, reassociations);

  return tosa::AddOp::create(rw, loc, resTy, ValueRange{result, biasExpand});
}

template <typename OpT>
class ForwardConvConverter final : public OpConversionPattern<OpT> {
public:
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpT op, typename OpT::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto *context = op->getContext();
    auto input = operands[0];
    auto filter = operands[1];
    auto bias = operands[2];
    auto outputType = cast<RankedTensorType>(op.getType());

    rock::GemmFeatures features = getGemmFeaturesFromOp(op, input.getType());

    if (failed(setSplitKAttrs(op, features, rw)))
      return failure();

    Value output =
        bufferization::AllocTensorOp::create(rw, loc, outputType, ValueRange{});

    auto groupAttr = op->template getAttrOfType<IntegerAttr>("group");
    auto padAttr = op->template getAttrOfType<DenseI64ArrayAttr>("pad");
    auto dilationAttr =
        op->template getAttrOfType<DenseI64ArrayAttr>("dilation");

    // Verify all required attributes are present
    int64_t group = 1;
    if (groupAttr)
      group = groupAttr.getInt();

    if (!padAttr)
      return op->emitError(
          "Expected 'pad' attribute to be present on the operation");

    if (!dilationAttr)
      return op->emitError(
          "Expected 'dilation' attribute to be present on the operation");

    FailureOr<rock::RockConvInterface> rockConv =
        makeRockConv(rw, op, input, filter, output, padAttr, op.getStrideAttr(),
                     dilationAttr, group, /*kernelID=*/0, "");

    if (failed(rockConv))
      return failure();

    Value result;
    Operation *rockConvOp = rockConv->getOperation();
    result = rock::TensorUntransformCastOp::create(
        rw, loc, outputType, rockConvOp->getResult(0), rockConv->getOutput());

    // test for zero bias, and ignore
    if (!mlir::rock::isConstantZero(op.getOperand(2))) {
      // non-zero bias, replace with tosa.add w/ broadcast
      FailureOr<tosa::AddOp> maybeResult = replaceCstZeroWithAddNBcast(
          context, rw, loc, op.getType(), bias, input, result);

      if (succeeded(maybeResult))
        result = maybeResult.value();
      else
        return failure();
    }
    rw.replaceOp(op, result);
    return success();
  }
};

class BackwardConvConverter final : public OpConversionPattern<tosa::CustomOp> {
public:
  using OpConversionPattern<tosa::CustomOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::CustomOp op,
                                tosa::CustomOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    // Make sure its a valid CustomOp representing a convolution.
    if (op.getDomainName() != ROCK_CUSTOMOP_DOMAIN_NAME)
      return op->emitError("domain isn't rock");
    if (op.getOperatorName() != ROCK_CUSTOMOP_CONV_BWD_DATA &&
        op.getOperatorName() != ROCK_CUSTOMOP_CONV_BWD_WEIGHT)
      return op->emitError("has an invalid operator_name");
    if (op.getNumOperands() < 5)
      return op->emitError("must have 5 or more operands");
    if (op.getNumResults() != 1)
      return op->emitError("must have 1 result");

    // Verify all required attributes are present. "group" is optional.
    for (std::string attrName : {"pad", "stride", "dilation"}) {
      if (!op->hasAttr(attrName))
        return op->emitError("expected '" + attrName +
                             "' attribute to be present on the op");
    }

    auto operands = adaptor.getOperands();
    auto loc = op->getLoc();
    auto *context = op->getContext();
    auto input = operands[0];
    auto filter = operands[1];
    auto bias = operands[2];
    RankedTensorType outputType = cast<RankedTensorType>(op.getType(0));

    rock::GemmFeatures features = getGemmFeaturesFromOp(op, input.getType());

    if (failed(setSplitKAttrs(op, features, rw)))
      return failure();

    Value output =
        bufferization::AllocTensorOp::create(rw, loc, outputType, ValueRange{});

    auto groupAttr = op->getAttrOfType<IntegerAttr>("group");
    auto padAttr = op->getAttrOfType<DenseI64ArrayAttr>("pad");
    auto strideAttr = op->getAttrOfType<DenseI64ArrayAttr>("stride");
    auto dilationAttr = op->getAttrOfType<DenseI64ArrayAttr>("dilation");

    int64_t group = 1;
    if (groupAttr)
      group = groupAttr.getInt();

    // If we are trying to convert bwd_weight, fail as it's currently not
    // supported.
    if (op.getOperatorName() == ROCK_CUSTOMOP_CONV_BWD_WEIGHT) {
      return op->emitError(
          "TosaToRock lowering support for bwd_weight not supported");
    }

    FailureOr<rock::RockConvInterface> rockConv = makeRockConv(
        rw, op, input, filter, output, padAttr, strideAttr, dilationAttr, group,
        /*kernelID=*/0, ROCK_CUSTOMOP_CONV_BWD_DATA);

    addZeroInitPrefillAttribute(op, rockConv->getOperation());

    if (failed(rockConv))
      return failure();

    Value result = output;

    // test for zero bias, and ignore
    if (!mlir::rock::isConstantZero(op.getOperand(2))) {
      // non-zero bias, replace with tosa.add w/ broadcast
      FailureOr<tosa::AddOp> maybeResult = replaceCstZeroWithAddNBcast(
          context, rw, loc, op.getType(0), bias, input, result);

      if (succeeded(maybeResult))
        result = maybeResult.value();
      else
        return failure();
    }
    rw.replaceOp(op, result);
    return success();
  }
};

static Value insertBroadcast(Value inp, ArrayRef<int64_t> outShape,
                             Location loc, OpBuilder &b) {
  ArrayRef<int64_t> inpShape = cast<ShapedType>(inp.getType()).getShape();
  bool broadcastDone = false;
  rock::BottomUpTMBuilder broadcastDims(b, inpShape, loc);
  for (unsigned int i = 0; i < outShape.size(); i++) {
    if (inpShape[i] == 1 && outShape[i] != 1) {
      broadcastDims.broadcast({i}, {outShape[i]});
      broadcastDone = true;
    } else {
      broadcastDims.passThrough({i}, {i});
    }
  }
  if (!broadcastDone) {
    return inp;
  }
  return rock::TransformOp::create(b, loc, inp, broadcastDims.get());
}

static FailureOr<Value> mulBroadcast(Value val, bool skipCollapseExpand = true);

static FailureOr<Value> getValueSkipping(Value val,
                                         const DenseSet<StringRef> &opsToSkip) {
  while (val.getDefiningOp() &&
         opsToSkip.contains(val.getDefiningOp()->getName().getStringRef())) {
    if (val.getDefiningOp<tosa::MulOp>()) {
      auto maybeBroadcast = mulBroadcast(val);
      if (failed(maybeBroadcast))
        return failure();
      val = maybeBroadcast.value();
    } else
      val = val.getDefiningOp()->getOperand(0);
  }
  return val;
}

template <typename TosaOp>
static FailureOr<TosaOp>
getDefiningOpSkipping(Value val, const DenseSet<StringRef> &opsToSkip) {
  auto maybeResult = getValueSkipping(val, opsToSkip);
  if (failed(maybeResult))
    return failure();

  TosaOp result = maybeResult.value().getDefiningOp<TosaOp>();
  if (!result)
    return failure();
  return result;
}

static FailureOr<Value> mulBroadcast(Value val, bool skipCollapseExpand) {
  DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                tensor::ExpandShapeOp::getOperationName()};
  if (!skipCollapseExpand)
    opsToSkip.clear();

  auto maybeMul = getDefiningOpSkipping<tosa::MulOp>(val, opsToSkip);
  if (succeeded(maybeMul)) {
    auto mul = maybeMul.value();
    // this is a broadcast multiplication, one of the arguments is the actual
    // value, the other is a constant one
    Value nonOne;
    auto maybeTosaConstIn1 =
        getDefiningOpSkipping<tosa::ConstOp>(mul.getInput1(), opsToSkip);
    auto maybeArithConstIn1 =
        getDefiningOpSkipping<arith::ConstantOp>(mul.getInput1(), opsToSkip);
    if (succeeded(maybeTosaConstIn1)) {
      if (mlir::rock::isConstantOne(maybeTosaConstIn1.value().getResult()))
        nonOne = mul.getInput2();
    } else if (succeeded(maybeArithConstIn1)) {
      if (mlir::rock::isConstantOne(maybeArithConstIn1.value().getResult()))
        nonOne = mul.getInput2();
    }

    auto maybeTosaConstIn2 =
        getDefiningOpSkipping<tosa::ConstOp>(mul.getInput2(), opsToSkip);
    auto maybeArithConstIn2 =
        getDefiningOpSkipping<arith::ConstantOp>(mul.getInput2(), opsToSkip);
    if (succeeded(maybeTosaConstIn2)) {
      if (mlir::rock::isConstantOne(maybeTosaConstIn2.value().getResult()))
        nonOne = mul.getInput1();
    } else if (succeeded(maybeArithConstIn2)) {
      if (mlir::rock::isConstantOne(maybeArithConstIn2.value().getResult()))
        nonOne = mul.getInput1();
    }
    if (nonOne)
      return nonOne;
  }
  return failure();
}

class MatMulConverter final : public OpConversionPattern<tosa::MatMulOp> {
public:
  using OpConversionPattern<tosa::MatMulOp>::OpConversionPattern;

  UnitAttr getTranspose(tosa::MatMulOp op, StringRef name) const {
    if (auto attr = op->getAttrOfType<BoolAttr>(name)) {
      if (attr.getValue())
        return UnitAttr::get(op->getContext());
    }
    return nullptr;
  }

  std::tuple<int64_t, int64_t> getLastDims(UnitAttr transposed,
                                           RankedTensorType type) const {
    ArrayRef<int64_t> shape = type.getShape();
    int64_t rank = type.getRank();
    if (transposed) {
      return {shape[rank - 1], shape[rank - 2]};
    }
    return {shape[rank - 2], shape[rank - 1]};
  }

  void setLastDims(UnitAttr transposed, SmallVectorImpl<int64_t> &shape,
                   std::pair<int64_t, int64_t> lastDims) const {
    size_t rank = shape.size();
    if (transposed) {
      shape[rank - 1] = lastDims.first;
      shape[rank - 2] = lastDims.second;
    } else {
      shape[rank - 2] = lastDims.first;
      shape[rank - 1] = lastDims.second;
    }
  }

  // Helper to extract scale and matrix from a mul operation
  FailureOr<std::pair<Value, Value>>
  tryExtractScaleAndMatrix(Value mulInput1, Value mulInput2) const {
    Value scale = nullptr;
    Value matrix = nullptr;

    // Check if input1 is a cast from Float4E2M1FN
    if (tosa::CastOp castOp = mulInput1.getDefiningOp<tosa::CastOp>()) {
      Value castInput = castOp.getInput();
      if (isa<Float4E2M1FNType>(
              cast<ShapedType>(castInput.getType()).getElementType())) {
        matrix = castInput;
        scale = mulInput2;
      }
    }
    // Check if input2 is a cast from Float4E2M1FN
    else if (tosa::CastOp castOp = mulInput2.getDefiningOp<tosa::CastOp>()) {
      Value castInput = castOp.getInput();
      if (isa<Float4E2M1FNType>(
              cast<ShapedType>(castInput.getType()).getElementType())) {
        matrix = castInput;
        scale = mulInput1;
      }
    }

    // Unwrap cast on scale if present
    if (scale && scale.getDefiningOp<tosa::CastOp>()) {
      scale = scale.getDefiningOp<tosa::CastOp>().getInput();
    }
    if (scale) {
      RankedTensorType scaleType = cast<RankedTensorType>(scale.getType());
      if (!isa<Float8E8M0FNUType>(scaleType.getElementType()) &&
          !isa<Float32Type>(scaleType.getElementType())) {
        return failure();
      }
    }

    if (!scale || !matrix) {
      return failure();
    }

    return std::make_pair(scale, matrix);
  }

  // Helper to reshape matrix and scale to match target shape
  Value reshapeIfNeeded(Value val, ArrayRef<int64_t> targetShape, Location loc,
                        ConversionPatternRewriter &rw) const {
    auto valType = cast<RankedTensorType>(val.getType());
    if (valType.getShape() == targetShape) {
      return val;
    }

    RankedTensorType newType =
        RankedTensorType::get(targetShape, valType.getElementType());
    auto normalizedShapeValue = tosa::getTosaConstShape(rw, loc, targetShape);
    return tosa::ReshapeOp::create(rw, loc, newType, val, normalizedShapeValue);
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                tosa::MatMulOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Location loc = op->getLoc();
    auto outputType = cast<RankedTensorType>(op.getType());
    auto matA = op.getA();
    auto matB = op.getB();
    Value matABeforeCast = nullptr;
    Value matBBeforeCast = nullptr;
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName()};

    // Try to extract scale and matrix for input A
    Value scaleA = nullptr;
    FailureOr<tosa::MulOp> maybeMulA =
        getDefiningOpSkipping<tosa::MulOp>(matA, opsToSkip);
    if (succeeded(maybeMulA)) {
      tosa::MulOp mulOpA = maybeMulA.value();
      FailureOr<std::pair<Value, Value>> maybeScaleMatrixA =
          tryExtractScaleAndMatrix(mulOpA.getInput1(), mulOpA.getInput2());
      if (succeeded(maybeScaleMatrixA)) {
        auto [extractedScaleA, extractedMatrixA] = maybeScaleMatrixA.value();
        scaleA = extractedScaleA;
        matABeforeCast = extractedMatrixA;
      }
    }

    // Try to extract scale and matrix for input B
    Value scaleB = nullptr;
    FailureOr<tosa::MulOp> maybeMulB =
        getDefiningOpSkipping<tosa::MulOp>(matB, opsToSkip);
    if (succeeded(maybeMulB)) {
      tosa::MulOp mulOpB = maybeMulB.value();
      FailureOr<std::pair<Value, Value>> maybeScaleMatrixB =
          tryExtractScaleAndMatrix(mulOpB.getInput1(), mulOpB.getInput2());
      if (succeeded(maybeScaleMatrixB)) {
        auto [extractedScaleB, extractedMatrixB] = maybeScaleMatrixB.value();
        scaleB = extractedScaleB;
        matBBeforeCast = extractedMatrixB;
      }
    }

    // rock.gemm requires both scaleA and scaleB to be provided, or neither
    // If only one scale is present, fall back to normal matmul
    bool hasScaleA = (scaleA != nullptr);
    bool hasScaleB = (scaleB != nullptr);
    if (hasScaleA != hasScaleB) {
      return op.emitError("Only one scale is present. For scaled GEMM, both "
                          "scaleA and scaleB must be provided.");
    }

    // Reshape matrices and scales to match the expected shapes if needed
    if (matABeforeCast && scaleA) {
      ArrayRef<int64_t> targetShape =
          cast<ShapedType>(matA.getType()).getShape();
      matABeforeCast = reshapeIfNeeded(matABeforeCast, targetShape, loc, rw);
      scaleA = reshapeIfNeeded(scaleA, targetShape, loc, rw);
      matA = cast<TypedValue<TensorType>>(matABeforeCast);
    }

    if (matBBeforeCast && scaleB) {
      ArrayRef<int64_t> targetShape =
          cast<ShapedType>(matB.getType()).getShape();
      matBBeforeCast = reshapeIfNeeded(matBBeforeCast, targetShape, loc, rw);
      scaleB = reshapeIfNeeded(scaleB, targetShape, loc, rw);
      matB = cast<TypedValue<TensorType>>(matBBeforeCast);
    }
    Value output =
        bufferization::AllocTensorOp::create(rw, loc, outputType, ValueRange{});

    rock::GemmFeatures features = getGemmFeaturesFromOp(op, matA.getType());

    if (failed(setSplitKAttrs(op, features, rw)))
      return failure();

    UnitAttr transposeA = getTranspose(op, "transpose_a"),
             transposeB = getTranspose(op, "transpose_b"),
             transposeC = getTranspose(op, "transpose_c");

    auto [mDim, nDim] = getLastDims(transposeC, outputType);

    int64_t kDimOfA;
    std::tie(std::ignore, kDimOfA) =
        getLastDims(transposeA, cast<RankedTensorType>(matA.getType()));
    int64_t kDimOfB;
    std::tie(kDimOfB, std::ignore) =
        getLastDims(transposeB, cast<RankedTensorType>(matB.getType()));
    int kDim = (kDimOfA > kDimOfB) ? kDimOfA : kDimOfB;

    SmallVector<int64_t, 3> aShape =
        llvm::to_vector<3>(cast<RankedTensorType>(matA.getType()).getShape());
    setLastDims(transposeA, aShape, {mDim, kDim});
    Value brA = insertBroadcast(matA, aShape, loc, rw);
    Value brAScale = nullptr;
    if (scaleA) {
      SmallVector<int64_t, 3> aScaleShape = llvm::to_vector<3>(
          cast<RankedTensorType>(scaleA.getType()).getShape());
      // TODO: Handle transpose of scaleA, currently TransposeRewritePattern
      // will not be able to match scaled_gemms. Update logic when we have
      // scaled_gemm support in TOSA
      setLastDims(nullptr, aScaleShape, {mDim, kDim});
      brAScale = insertBroadcast(scaleA, aScaleShape, loc, rw);
    }

    SmallVector<int64_t, 3> bShape = llvm::to_vector<3>(
        cast<RankedTensorType>(op.getB().getType()).getShape());
    setLastDims(transposeB, bShape, {kDim, nDim});
    Value brB = insertBroadcast(matB, bShape, loc, rw);

    Value brBScale = nullptr;
    if (scaleB) {
      SmallVector<int64_t, 3> bScaleShape = llvm::to_vector<3>(
          cast<RankedTensorType>(scaleB.getType()).getShape());
      // TODO: Handle transpose of scaleB, currently TransposeRewritePattern
      // will not be able to match scaled_gemms. Update logic when we have
      // scaled_gemm support in TOSA
      setLastDims(nullptr, bScaleShape, {kDim, nDim});
      brBScale = insertBroadcast(scaleB, bScaleShape, loc, rw);
    }
    auto rockGemm = rock::GemmOp::create(
        rw, loc, outputType, brA, brB, output, brAScale, brBScale, transposeA,
        transposeB, transposeC, nullptr, nullptr,
        /*features=*/nullptr,
        rw.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        /*blockSize=*/nullptr, /*gridSize=*/nullptr,
        /*params=*/nullptr);

    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      rockGemm->setAttr("perf_config", attr);

    rw.replaceOp(op, rockGemm.getResult());

    return success();
  }
};

static void permuteLayout(Operation *op, const char *attrKey,
                          const char *layoutDefault,
                          const ArrayRef<int32_t> permDims,
                          bool isInput = false) {
  StringRef currentLayout(layoutDefault);
  if (auto attr = op->getAttrOfType<StringAttr>(attrKey))
    currentLayout = attr.getValue();
  SmallString<4> layout(currentLayout);
  if (isInput) {
    for (int i = 0, e = permDims.size(); i < e; ++i)
      layout[permDims[i]] = currentLayout[i];
  } else {
    for (int i = 0, e = permDims.size(); i < e; ++i)
      layout[i] = currentLayout[permDims[i]];
  }
  op->setAttr(attrKey, StringAttr::get(op->getContext(), layout));
}

struct TransposeRewritePattern : public OpRewritePattern<tosa::TransposeOp> {
  using OpRewritePattern<tosa::TransposeOp>::OpRewritePattern;

  void setTranspose(Operation *op, StringRef name, bool isNonTrivial) const {
    bool currentValue = false;
    if (auto attr = op->getAttrOfType<BoolAttr>(name))
      currentValue = attr.getValue();
    bool newValue = currentValue ^ isNonTrivial;
    op->setAttr(name, BoolAttr::get(op->getContext(), newValue));
  }

  LogicalResult checkInputHasUses(PatternRewriter &rewriter,
                                  tosa::TransposeOp top, Value tInput) const {
    // if the input has uses (apart from this one), we can't do this
    if (!tInput.hasOneUse()) {
      return rewriter.notifyMatchFailure(
          top, "abandoning attempt to fuse transpose "
               "because the operation has other uses");
    }
    return success();
  }

  LogicalResult checkMatMulTransposeValid(tosa::MatMulOp matmulOp,
                                          const ArrayRef<int32_t> dims) const {
    // batch dimension is expected to be 3rd from the last.
    if (dims.size() >= 3 && dims[dims.size() - 3] != (int32_t)dims.size() - 3) {
      return matmulOp.emitWarning(
          "Transposing the batch dimension out of place lowers performance");
    }
    return success();
  }

  bool isMatMulNonTrivial(const ArrayRef<int32_t> dims) const {
    int32_t lastDim = dims.size() - 1;
    int32_t prevLastDim = dims.size() - 2;
    return (dims[prevLastDim] == lastDim && dims[lastDim] == prevLastDim);
  }

  // This function traverses the uses of tOutput and then modifies
  // the uses to indicate the input are transposed and replaces them
  // with tInput. If there are collapse shapes encountered, the collapse
  // is applied on the tInput.
  LogicalResult mergeTransposeWithGemmLikeOp(PatternRewriter &rewriter,
                                             Value tOutput,
                                             const ArrayRef<int32_t> dims,
                                             Value tInput) const {
    auto handleConv = [&](auto convOp) -> LogicalResult {
      if (convOp.getInput() == tOutput) {
        permuteLayout(convOp.getOperation(), "input_layout", "nhwc", dims,
                      true);
        convOp.getInputMutable().assign(tInput);
      } else if (convOp.getWeight() == tOutput) {
        permuteLayout(convOp.getOperation(), "filter_layout", "kyxc", dims,
                      true);
        convOp.getWeightMutable().assign(tInput);
      } else {
        return convOp.emitWarning("transpose found leading to a "
                                  "conv2D/transposeConv2D input other than "
                                  "data or weight");
      }
      return success();
    };

    for (auto &use : llvm::make_early_inc_range(tOutput.getUses())) {
      if (auto op = dyn_cast<tensor::CollapseShapeOp>(use.getOwner())) {
        SmallVector<ReassociationIndices, 4> reassocIndices =
            op.getReassociationIndices();
        // This is to capture new reassociations above the transpose
        llvm::SmallDenseMap<int32_t, ReassociationIndices> newReassocIdxMap;
        ArrayRef<int64_t> inShape = op.getSrcType().getShape();

        // This loops maps reassociated dims back to pre transposed dims.
        SmallVector<int32_t, 4> newDims;

        llvm::SmallDenseSet<int64_t> preTpUnitDims;
        for (ReassociationIndices indices : reassocIndices) {
          ReassociationIndices newReassocIdx;
          size_t numNonUnitDimsMerged = 0;
          for (size_t i = 0, e = indices.size(); i < e; ++i) {
            if (inShape[indices[i]] == 1) {
              preTpUnitDims.insert(dims[indices[i]]);
            } else {
              numNonUnitDimsMerged += 1;
            }
            newReassocIdx.push_back(dims[indices[i]]);
          }
          if (numNonUnitDimsMerged > 1) {
            // Per MIGraphX bug #2692, this transpsoe/collaspe swap logic
            // will be incorrect in cases like the following
            //   %0 = expand_shape [[0], [1, 2], [3]] %arg0 : tensor<7x6x5xT>
            //   to tensor<7x3x2x5xT> %1 = transpose %0, [0, 2, 1, 3] :
            //   tensor<7x2x3x5xT> %2 = collapse_shape [[0], [1, 2], [2]] %1 :
            //   tensor<7x2x3x5xT> to tensor<7x6x5xT>
            // by way of creating a trivial expand/collapse pair that isn't
            // correct.
            //
            // Therefore, as a sledgehammer fix, don't handle any cases where
            // non-trivial collapses are performed.
            return rewriter.notifyMatchFailure(
                op, "abandoning attempt to interchange transpose and "
                    "non-trivial collapse");
          }
          if (newReassocIdx.size() > 1) {
            llvm::sort(newReassocIdx);
            // Remove unit dims from larger end of reassociation indices
            // but we need at least one for the reassociation
            while (newReassocIdx.size() > 1 &&
                   preTpUnitDims.contains(newReassocIdx.back())) {
              newReassocIdx.pop_back();
            }
            for (size_t i = 1; i < newReassocIdx.size(); i++) {
              if (newReassocIdx[i] - newReassocIdx[i - 1] != 1) {
                return rewriter.notifyMatchFailure(
                    op, "CollapseShape op following transpose collapses "
                        "non-contiguous pre-transpose dims.");
              }
            }
          }
          newDims.push_back(newReassocIdx[0]);
          // minIdx is the representative of a group that is
          // being collapsed. For e.g. for a collapse of [3,4,5] is assigned
          // with 3 as the representative. I also note that we only allow
          // collapsing of contiguous pre-transpose dims.
          newReassocIdxMap[newReassocIdx[0]] = newReassocIdx;
        }

        // Assign the ordering index of reassociated dims as the dim index
        SmallVector<int32_t, 4> newDimsSorted = newDims;
        llvm::sort(newDimsSorted);
        SmallVector<ReassociationIndices, 4> newReassocIndicesSorted;
        DenseMap<int32_t, int32_t> dimMap;
        // The vector of newDims (may) contain a discontinous
        // a range of representative minIdxs. Here we make
        // it contiguous by assigning order idx.
        for (size_t i = 0; i < newDimsSorted.size(); i++) {
          dimMap[newDimsSorted[i]] = i;
          newReassocIndicesSorted.push_back(newReassocIdxMap[newDimsSorted[i]]);
        }
        // HOTFIX: glue trailing unit dimensions onto collapses that need
        // them. This is because a case like
        // %t = transpose %aRaw [0, 1, 3, 2] : tensor<1x1xKxM> ->
        // tensor<1x1xMxK> %a = collapse_shape [[0, 1], [2], [3]]
        //    : tensor<1x1xMxK> -> tensor<1xMxK>
        // will, with the above unit-dimension-removal logic, lead to the
        // invalid reassociation [[0], [2], [3]], causing a crash.
        // See MIGraphX bug #2365.
        // The entire logic here should be reviewed, or at least made less
        // complex if possible, but ... release-critical bug, what can we do?
        for (size_t i = 0, e = newReassocIndicesSorted.size() - 1; i < e; ++i) {
          ReassociationIndices &theseIndices = newReassocIndicesSorted[i];
          const ReassociationIndices &nextIndices =
              newReassocIndicesSorted[i + 1];
          while (theseIndices.back() + 1 < nextIndices[0]) {
            theseIndices.push_back(theseIndices.back() + 1);
          }
        }
        // do the same for the last set of indices too
        // where it does not match upto the rank of the input.
        ReassociationIndices &lastIndices = newReassocIndicesSorted.back();
        while (lastIndices.back() + 1 < (int64_t)inShape.size()) {
          lastIndices.push_back(lastIndices.back() + 1);
        }

        for (size_t i = 0; i < newDims.size(); i++) {
          newDims[i] = dimMap[newDims[i]];
        }

        tensor::CollapseShapeOp newCollapseShapeOp =
            tensor::CollapseShapeOp::create(rewriter, op.getLoc(), tInput,
                                            newReassocIndicesSorted);

        if (mergeTransposeWithGemmLikeOp(rewriter, op.getResult(), newDims,
                                         newCollapseShapeOp.getResult())
                .failed()) {
          rewriter.eraseOp(newCollapseShapeOp);
          return failure();
        }
        if (op->use_empty())
          rewriter.eraseOp(op);
      } else if (auto op = dyn_cast<tensor::ExpandShapeOp>(use.getOwner())) {
        return rewriter.notifyMatchFailure(
            op, "We dont support expand shapes yet.");
      } else if (auto transposeConv2D =
                     dyn_cast<tosa::TransposeConv2DOp>(use.getOwner())) {
        return handleConv(transposeConv2D);
      } else if (auto conv2D = dyn_cast<tosa::Conv2DOp>(use.getOwner())) {
        return handleConv(conv2D);
      } else if (auto matMulOp = dyn_cast<tosa::MatMulOp>(use.getOwner())) {
        if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
          return failure();
        }
        bool mmNonTrivial = isMatMulNonTrivial(dims);
        if (matMulOp.getA() == tOutput) {
          setTranspose(matMulOp, "transpose_a", mmNonTrivial);
          matMulOp.getAMutable().assign(tInput);
        } else if (matMulOp.getB() == tOutput) {
          setTranspose(matMulOp, "transpose_b", mmNonTrivial);
          matMulOp.getBMutable().assign(tInput);
        } else {
          return matMulOp.emitWarning(
              "transpose found leading to a matmul input other than A or B");
        }
      } else {
        return failure();
      }
    }
    return success();
  }

  // Fold transpose ops and convert convolution into changed layout.
  // case #0 : fold TP(NCHW2NHWC)+tosa.conv.NHWC+TP(NHWC2NCHW) back to
  //           rock.conv.NCHW
  // Pattern match start from the output transpose
  LogicalResult matchAndRewrite(tosa::TransposeOp top,
                                PatternRewriter &b) const final {
    const auto dims = top.getPerms();

    Value tInput = top.getInput1();
    Value tOutput = top.getResult();
    auto definingOp = tInput.getDefiningOp();
    if (definingOp && (isa<tosa::Conv2DOp>(definingOp) ||
                       isa<tosa::TransposeConv2DOp>(definingOp))) {
      auto transposeConv2D = dyn_cast<tosa::TransposeConv2DOp>(definingOp);
      auto conv2D = dyn_cast<tosa::Conv2DOp>(definingOp);
      auto convOp = (transposeConv2D ? transposeConv2D : conv2D);
      if (checkInputHasUses(b, top, tInput).failed()) {
        return failure();
      }
      // conv output is transpose
      permuteLayout(convOp, "output_layout", "nhwk", dims);
      convOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(convOp);
    } else if (tosa::MatMulOp matMulOp =
                   tInput.getDefiningOp<tosa::MatMulOp>()) {

      if (checkInputHasUses(b, top, tInput).failed()) {
        return failure();
      }
      if (checkMatMulTransposeValid(matMulOp, dims).failed()) {
        return failure();
      }
      setTranspose(matMulOp, "transpose_c", isMatMulNonTrivial(dims));
      matMulOp->getResult(0).setType(tOutput.getType());
      top->replaceAllUsesWith(matMulOp);
    } else {
      if (mergeTransposeWithGemmLikeOp(b, tOutput, dims, tInput).failed()) {
        return failure();
      }
    }

    if (top.use_empty())
      b.eraseOp(top);
    return success();
  }
};

// In Tosa canonicalize, a transpose of NCHW to NHWC where H==W==1 will
// convert to a reshape because it does not change memory layout. Then in
// TosaToTensor conversion, the reshape is replaced by this pattern:
//     %0 = collapse(filters[KCHW]) -> [KC]
//     %1 = expand(%0[KC]) -> [KHWC]
// If this feeds into a conv as filter, we will drop the collapse/expand and
// update the filter_layout attribute.
struct CollapseExpandRewritePattern
    : public OpRewritePattern<tensor::ExpandShapeOp> {
  using OpRewritePattern<tensor::ExpandShapeOp>::OpRewritePattern;

  bool checkExpand(tensor::ExpandShapeOp expOp) const {
    auto srcSh = cast<ShapedType>(expOp.getOperand(0).getType()).getShape();
    auto resSh = cast<ShapedType>(expOp.getResultType()).getShape();
    // [[0, 1, 2], [3]]
    // NC -> NHWC
    if (srcSh.size() == 2 && resSh.size() == 4 && srcSh[0] == resSh[0] &&
        srcSh[1] == resSh[3] && resSh[1] == 1 && resSh[2] == 1) {
      return true;
    }
    return false;
  }

  bool checkCollapse(tensor::CollapseShapeOp colOp) const {
    auto srcSh = cast<ShapedType>(colOp.getOperand().getType()).getShape();
    auto resSh = cast<ShapedType>(colOp.getResultType()).getShape();
    // [[0], [1, 2, 3]]
    // NCHW -> NC
    if (srcSh.size() == 4 && resSh.size() == 2 && srcSh[0] == resSh[0] &&
        srcSh[1] == resSh[1] && srcSh[2] == 1 && srcSh[3] == 1) {
      return true;
    }
    return false;
  }

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expOp,
                                PatternRewriter &b) const final {
    LogicalResult lres = failure();
    Value expInp = expOp.getOperand(0);
    Value expOut = expOp.getResult();

    if (!checkExpand(expOp))
      return failure();

    auto colOp = expInp.getDefiningOp<tensor::CollapseShapeOp>();
    if (colOp && checkCollapse(colOp)) {
      auto colInp = colOp.getOperand();

      for (Operation *usr : expOut.getUsers()) {
        if (isa<tosa::TransposeConv2DOp>(usr) || isa<tosa::Conv2DOp>(usr)) {
          if (usr->getOperand(1) == expOut) {
            // update filter_layout
            SmallVector<int32_t> dims{0, 2, 3, 1};
            permuteLayout(usr, "filter_layout", "kyxc", dims, true);
            // replace filter input with collapse source
            usr->replaceUsesOfWith(expOut, colInp);

            lres = success();
          }
        }
      }
    }

    return lres;
  }
};

struct ConvElementwiseGemmRewritePattern
    : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<ElementwiseRegionFinder<tosa::Conv2DOp>>
  match(tosa::MatMulOp op) const {
    ElementwiseRegionFinder<tosa::Conv2DOp> elementwiseRegionFinder;
    elementwiseRegionFinder.visit(op.getA());
    FailureOr<tosa::Conv2DOp> maybeConv =
        elementwiseRegionFinder.getFirstGemmBasedOp();

    if (succeeded(maybeConv))
      LLVM_DEBUG(llvm::dbgs() << "conv = " << maybeConv.value() << "\n");
    else {
      LLVM_DEBUG(llvm::dbgs() << "conv not found\n");
      return failure();
    }

    tosa::Conv2DOp firstConv = maybeConv.value();
    // bias not supported
    if (!mlir::rock::isConstantZero(firstConv.getBias())) {
      op.emitOpError("bias not supported yet");
      return failure();
    }
    return elementwiseRegionFinder;
  }

  void rewrite(
      tosa::MatMulOp op,
      const ElementwiseRegionFinder<tosa::Conv2DOp> &elementwiseRegionFinder,
      PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto outputType = cast<RankedTensorType>(op.getType());
    Value output = bufferization::AllocTensorOp::create(
        rewriter, loc, outputType, ValueRange{});

    // This is guaranteed by the matcher
    tosa::Conv2DOp firstConv =
        elementwiseRegionFinder.getFirstGemmBasedOp().value();

    SmallVector<Value> elementwiseOtherArgs =
        elementwiseRegionFinder.getElementwiseArgs();

    int64_t group = 1;
    if (auto attr = op->template getAttrOfType<IntegerAttr>("group"))
      group = attr.getInt(); // Use op.getGroup() when all OpT have it.
    ConvFields convFields =
        commonConv(rewriter, op, firstConv.getInput(), firstConv.getWeight(),
                   output, firstConv.getPadAttr(), firstConv.getStrideAttr(),
                   firstConv.getDilationAttr(), group);
    auto firstGemmBlockIndex = elementwiseRegionFinder.getFirstGemmBlockIndex();

    rock::GemmFeatures featuresA =
        getGemmFeaturesFromOp(op, convFields.filterExp.getType());
    rock::GemmFeatures featuresC =
        getGemmFeaturesFromOp(op, op.getB().getType());
    rock::GemmFeatures features = intersectGemmFeatures(featuresA, featuresC);

    if (failed(setSplitKAttrs(op, features, rewriter)))
      return;

    auto convElentwiseGemmOp = rock::ConvElementwiseGemmOp::create(
        rewriter, loc, outputType, convFields.filterExp, convFields.inputExp,
        op.getB(), elementwiseOtherArgs, output,
        /*cTransposed=*/nullptr,
        /*oTransposed=*/nullptr, /*features=*/nullptr,
        rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        convFields.pad, convFields.stride, convFields.dilation,
        /*params0=*/nullptr, /*params1=*/nullptr,
        /*firstGemmIndices=*/
        rewriter.getDenseI64ArrayAttr(firstGemmBlockIndex));

    addConvAttributes(rewriter, convElentwiseGemmOp, convFields);

    Block *preSecondGemmElemwiseBlock =
        &convElentwiseGemmOp.getPreSecondGemmBody().emplaceBlock();
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(preSecondGemmElemwiseBlock);
      elementwiseRegionFinder.rewrite(op.getA(), rewriter,
                                      preSecondGemmElemwiseBlock, loc);
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      convElentwiseGemmOp->setAttr("perf_config", attr);

    rewriter.replaceOp(op, convElentwiseGemmOp.getResult());
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ElementwiseRegionFinder<tosa::Conv2DOp>> elemwiseFinder =
        match(op);
    if (succeeded(elemwiseFinder)) {
      rewrite(op, elemwiseFinder.value(), rewriter);
    }
    return elemwiseFinder;
  }
};

struct GemmElementwiseGemmRewritePattern
    : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<ElementwiseRegionFinder<tosa::MatMulOp>>
  match(tosa::MatMulOp op) const {
    ElementwiseRegionFinder<tosa::MatMulOp> elemwiseRegionFinder;
    elemwiseRegionFinder.visit(op.getA());
    FailureOr<tosa::MatMulOp> maybeFirstMatMul =
        elemwiseRegionFinder.getFirstGemmBasedOp();
    if (succeeded(maybeFirstMatMul))
      LLVM_DEBUG(llvm::dbgs()
                 << "first matmul = " << maybeFirstMatMul.value() << "\n");
    else {
      LLVM_DEBUG(llvm::dbgs() << "first matmul not found\n");
      return failure();
    }
    return elemwiseRegionFinder;
  }

  void rewrite(tosa::MatMulOp op,
               const ElementwiseRegionFinder<tosa::MatMulOp> &elemwiseFinder,
               PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    auto outputType = cast<RankedTensorType>(op.getType());
    Value output = bufferization::AllocTensorOp::create(
        rewriter, loc, outputType, ValueRange{});
    SmallVector<Value> elementwiseOtherArgs =
        elemwiseFinder.getElementwiseArgs();
    // This is guranteed by the matcher
    tosa::MatMulOp firstMatMulOp = elemwiseFinder.getFirstGemmBasedOp().value();
    int64_t firstGemmBlockIndex = elemwiseFinder.getFirstGemmBlockIndex();

    rock::GemmFeatures featuresA =
        getGemmFeaturesFromOp(op, firstMatMulOp.getA().getType());
    rock::GemmFeatures featuresC =
        getGemmFeaturesFromOp(op, op.getB().getType());
    rock::GemmFeatures features = intersectGemmFeatures(featuresA, featuresC);

    if (failed(setSplitKAttrs(op, features, rewriter)))
      return;

    rock::GemmElementwiseGemmOp gemmElentwiseGemmOp =
        rock::GemmElementwiseGemmOp::create(
            rewriter, loc, outputType, firstMatMulOp.getA(),
            firstMatMulOp.getB(), op.getB(), elementwiseOtherArgs, output,
            /*qTransposed=*/nullptr,
            /*kTransposed=*/nullptr,
            /*vTransposed=*/nullptr,
            /*oTransposed=*/nullptr,
            /*features=*/nullptr,
            rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
            /*params0=*/nullptr, /*params1=*/nullptr,
            /*firstGemmIndices=*/
            rewriter.getDenseI64ArrayAttr(firstGemmBlockIndex));
    Block *preSecondGemmElemwiseBlock =
        &gemmElentwiseGemmOp.getPreSecondGemmBody().emplaceBlock();
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(preSecondGemmElemwiseBlock);
      elemwiseFinder.rewrite(op.getA(), rewriter, preSecondGemmElemwiseBlock,
                             loc);
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      gemmElentwiseGemmOp->setAttr("perf_config", attr);

    rewriter.replaceOp(op, gemmElentwiseGemmOp.getResult());
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<ElementwiseRegionFinder<tosa::MatMulOp>> elemwiseFinder =
        match(op);
    if (succeeded(elemwiseFinder)) {
      rewrite(op, elemwiseFinder.value(), rewriter);
    }
    return elemwiseFinder;
  }
};

struct SoftmaxMatcherValues {
  Value softmaxInput;
  Operation *subOp;
  Value exp;
  Operation *reduceMaxOp;
  Operation *reduceSumOp;
  bool hasReduceOp;
};

struct AttentionMatcherValues {
  SoftmaxMatcherValues softmaxValues;
  Value lse;
  Value causalMaskInput;
  Value currentSeqLen;
  bool isCausal;
  Value prefixOffset;
  Type softmaxType;
  ElementwiseRegionFinder<tosa::MatMulOp> preSoftmaxElementwiseFinder;
};

struct AttentionRewritePattern : public OpRewritePattern<tosa::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  // This function checks if a given input is a constant range. There are two
  // possible cases that we can handle:
  // - The input is broadcasted -> we can then check the resulting broadcasted
  //   value to see if it's a constant range
  // - The input is already a constant range
  LogicalResult isConstantRange(TypedValue<TensorType> input,
                                size_t nonOneDimFromEnd) const {
    // Lambda to get constant result from either tosa.const or arith.constant
    auto getConstantResult = [&](Value input) -> FailureOr<Value> {
      DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                    tensor::ExpandShapeOp::getOperationName()};
      auto maybeTosaConst =
          getDefiningOpSkipping<tosa::ConstOp>(input, opsToSkip);
      auto maybeArithConst =
          getDefiningOpSkipping<arith::ConstantOp>(input, opsToSkip);
      if (succeeded(maybeTosaConst)) {
        return maybeTosaConst.value().getResult();
      } else if (succeeded(maybeArithConst)) {
        return maybeArithConst.value().getResult();
      }
      return failure();
    };

    // input is a constant with a range from 0 to maxSeqLen
    Value rangeInput = mulBroadcast(input).value_or(input);

    // check that rangeInput is a const with range 0..maxSeqLen
    Value rangeResult;
    auto rangeConstantResult = getConstantResult(rangeInput);
    bool isRange = succeeded(rangeConstantResult) &&
                   rock::isConstRange(rangeConstantResult.value());

    if (!isRange)
      return failure();

    rangeResult = rangeConstantResult.value();

    auto shapedType = dyn_cast<ShapedType>(rangeResult.getType());
    if (!shapedType)
      return failure();

    auto shape = shapedType.getShape();
    assert(nonOneDimFromEnd < shape.size());
    size_t couldBeDiffOne = shape.size() - nonOneDimFromEnd - 1;
    for (auto [i, dim] : llvm::enumerate(shape)) {
      if (i != couldBeDiffOne && dim != 1) {
        return failure();
      }
    }
    return success();
  }

  // Validates that a constant mask follows the causal mask pattern:
  // - Each row i should have zeros at positions 0 through i (lower triangular
  //   part)
  // - The upper triangular part (positions > i) depends on the pattern:
  //     * For select-based masks (expectOnesInUpperTriangle=true): 1's
  //     * For non-select based masks (expectOnesInUpperTriangle=false): -infs
  // - In the select-based mask case, the upper triangular part that is all 1's
  //   is combined with the -inf values that are fed into the select op,
  //   resulting in the same type of mask pattern that we see in the
  //   non-select based mask case. In the non-select case the mask is added
  //   directly to the result of the first gemm.
  bool isValidCausalMask(Operation *op, bool expectOnesInUpperTriangle) const {
    // Get the constant value
    DenseElementsAttr constAttr;
    if (auto tosaConst = dyn_cast<tosa::ConstOp>(op)) {
      constAttr = dyn_cast<DenseElementsAttr>(tosaConst.getValuesAttr());
    } else if (auto arithConst = dyn_cast<arith::ConstantOp>(op)) {
      constAttr = dyn_cast<DenseElementsAttr>(arithConst.getValue());
    }

    if (!constAttr)
      return false;

    auto shapedType = cast<ShapedType>(constAttr.getType());
    auto shape = shapedType.getShape();

    // First two dims must be 1 (broadcast dimensions)
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1)
      return false;

    // Sanity check that this is an integer or float
    bool isInt = constAttr.getElementType().isIntOrIndex();
    bool isFloat = isa<FloatType>(constAttr.getElementType());
    if (!isInt && !isFloat)
      return false;

    // If this is an integer type, and we don't expect all ones in the upper
    // triangle portion, then we cannot match this as a causal mask.
    if (isInt && !expectOnesInUpperTriangle)
      return false;

    int64_t seqLen = shape[2];
    int64_t maxSeqLen = shape[3];

    // Generic validation function that works with any value type
    auto validateMask = [&](auto values, auto isZero, auto isOne,
                            auto isNegInf) -> bool {
      for (int64_t row = 0; row < seqLen; ++row) {
        for (int64_t col = 0; col < maxSeqLen; ++col) {
          auto val = values[row * maxSeqLen + col];

          // Validate that the lower triangular portion is all zeros
          if (col <= row && !isZero(val))
            return false;

          // Check that the upper triangular portion is correct
          bool validUpperTriangleVal =
              expectOnesInUpperTriangle ? isOne(val) : isNegInf(val);
          if (col > row && !validUpperTriangleVal)
            return false;
        }
      }
      return true;
    };

    if (isInt) {
      auto intValues = constAttr.getValues<APInt>();
      return validateMask(
          intValues, [](const APInt &v) { return v.isZero(); },
          [](const APInt &v) { return v.isOne(); },
          [](const APInt &v) { return v.isMinSignedValue(); });
    } else {
      auto floatValues = constAttr.getValues<APFloat>();
      return validateMask(
          floatValues, [](const APFloat &v) { return v.isZero(); },
          [](const APFloat &v) { return v.convertToDouble() == 1.0; },
          [](const APFloat &v) { return v.isInfinity() && v.isNegative(); });
    }
  }

  // Helper function to detect if a given value is used by a tosa.exp op.
  bool isUsedByExp(Value value) const {
    // Use iterative DFS with a worklist to search through the use chain
    SmallVector<Value, 8> worklist{value};
    DenseSet<Operation *> visited;

    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();

      for (Operation *user : current.getUsers()) {
        // Insert the op into the visited set. Insert will return a pair where
        // .second is true if the insertion was successful.
        if (!visited.insert(user).second)
          continue;

        // Check if this user is a tosa.exp op
        if (isa<tosa::ExpOp>(user))
          return true;

        // Add all results of this operation to the worklist
        for (Value result : user->getResults())
          worklist.push_back(result);
      }
    }

    return false;
  }

  // Helper to check if a value is a -inf constant (skipping shape ops)
  bool isNegInfConstant(Value val) const {
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};
    auto maybeTosaConst =
        getDefiningOpSkipping<tosa::ConstOp>(val, expandAndCollapse);
    auto maybeArithConst =
        getDefiningOpSkipping<arith::ConstantOp>(val, expandAndCollapse);
    if (succeeded(maybeTosaConst))
      return rock::isConstNegInf(maybeTosaConst.value().getResult());
    if (succeeded(maybeArithConst))
      return rock::isConstNegInf(maybeArithConst.value().getResult());
    return false;
  }

  // Helper to get a select op where the onTrue branch is -inf
  // Returns the select op if found, failure otherwise
  FailureOr<tosa::SelectOp> getSelectWithNegInf(Value input) const {
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::CastOp::getOperationName()};
    auto maybeSelect = getDefiningOpSkipping<tosa::SelectOp>(input, opsToSkip);
    if (failed(maybeSelect))
      return failure();
    if (!isNegInfConstant(maybeSelect.value().getInput2()))
      return failure();
    return maybeSelect;
  }

  // Helper to verify a value is i32 and traces back to a block argument
  bool isI32BlockArgument(Value val,
                          const DenseSet<StringRef> &seqLenSkip) const {
    auto shape = dyn_cast<ShapedType>(val.getType());
    if (!shape || !shape.getElementType().isInteger(32))
      return false;

    FailureOr<Value> maybeBlockArg = getValueSkipping(val, seqLenSkip);
    return succeeded(maybeBlockArg) &&
           isa<BlockArgument>(maybeBlockArg.value());
  }

  // Helper function to detect select-based causal mask pattern:
  //   - true branch is a splat -inf constant
  //   - false branch is the tensor value that we want to return
  //   - pred is either:
  //       (1) tosa.greater(const1, const2) comparing two constant 0..N range
  //           tensors
  //       (2) A pre-folded broadcasted constant 1 upper‑triangular mask tensor
  FailureOr<Value> getCausalFromSelect(Value input) const {
    auto maybeSelect = getSelectWithNegInf(input);
    if (failed(maybeSelect))
      return failure();

    auto select = maybeSelect.value();
    auto pred = select.getInput1();

    // There are two cases that we need to be able to handle for the pred:
    // 1. We have a greater op that is doing a comparison between two
    //    constants
    // 2. The greater op has already been constant folded by MIGraphX, so we
    //    find the broadcast input and then do the necessary constant checks
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::CastOp::getOperationName()};
    auto maybeBroadcast = getDefiningOpSkipping<tosa::MulOp>(pred, opsToSkip);
    opsToSkip.insert(tosa::MulOp::getOperationName());
    auto maybeGreater = getDefiningOpSkipping<tosa::GreaterOp>(pred, opsToSkip);
    if (succeeded(maybeGreater)) {
      auto greater = maybeGreater.value();
      // input1 is a constant with a range from 0 to maxSeqLen (KV)
      if (failed(isConstantRange(greater.getInput1(), 0)))
        return failure();

      // input2 is a constant with a range from 0 to seqLenQ
      if (failed(isConstantRange(greater.getInput2(), 1)))
        return failure();

      Value result = select.getInput3();
      return result;
    } else if (succeeded(maybeBroadcast)) {
      // The input from MIGraphX will not be a constant range, so we cannot
      // use the isConstantRange function. Instead we need to check that
      // the constant is a valid causal mask pattern.
      auto maybeNonOne = mulBroadcast(maybeBroadcast.value());
      if (failed(maybeNonOne))
        return failure();

      // Validate the causal mask pattern (select uses 1's in upper triangle)
      Operation *defOp = maybeNonOne.value().getDefiningOp();
      if (!isValidCausalMask(defOp, /*expectOnesInUpperTriangle=*/true))
        return failure();

      Value result = select.getInput3();
      return result;
    }
    return failure();
  }

  // Helper function to detect add-based causal mask pattern:
  //   - Looks for tosa.add(scores, mask) where mask is a constant with:
  //     * 0s in lower triangle (allow attention)
  //     * -inf values in upper triangle (block attention)
  FailureOr<Value> getCausalFromAdd(Value input) const {
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::CastOp::getOperationName()};
    auto maybeAdd = getDefiningOpSkipping<tosa::AddOp>(input, opsToSkip);
    if (failed(maybeAdd))
      return failure();

    auto add = maybeAdd.value();
    Value input1 = add.getInput1();
    Value input2 = add.getInput2();

    // Try to find the causal mask constant in either input
    // Check if input2 is a causal mask (broadcasted via mul)
    auto maybeNonOne2 = mulBroadcast(input2);
    if (succeeded(maybeNonOne2)) {
      Operation *defOp = maybeNonOne2.value().getDefiningOp();
      if (defOp && isValidCausalMask(defOp,
                                     /*expectOnesInUpperTriangle=*/false)) {
        return input1;
      }
    }

    // Check if input1 is a causal mask (broadcasted via mul)
    auto maybeNonOne1 = mulBroadcast(input1);
    if (succeeded(maybeNonOne1)) {
      Operation *defOp = maybeNonOne1.value().getDefiningOp();
      if (defOp && isValidCausalMask(defOp,
                                     /*expectOnesInUpperTriangle=*/false)) {
        return input2;
      }
    }

    return failure();
  }

  // Detects a standard causal mask for attention ops.
  // Tries both select-based and add-based patterns.
  FailureOr<Value> getCausal(Value input) const {
    // Check that the input that comes from the causal mask (verified to be
    // -inf values in getCausalFromSelect or getCausalFromAdd) is used by an
    // exp op.
    if (!isUsedByExp(input))
      return failure();

    // Try select-based pattern first (most common)
    auto selectResult = getCausalFromSelect(input);
    if (succeeded(selectResult))
      return selectResult;

    // Try add-based pattern
    auto addResult = getCausalFromAdd(input);
    if (succeeded(addResult))
      return addResult;

    return failure();
  }

  // Result struct for sequence length mask detection
  struct SeqLenMaskResult {
    Value inputToContinue; // The value to continue pattern matching with
    Value seqLen;          // The sequence length
    Value prefixOffset;    // The prefix offset value
  };

  // Helper to try detecting prefix causal pattern: add(row_indices, offset)
  // Returns the offset value if successful
  FailureOr<Value>
  tryPrefixCausalPattern(Value input,
                         const DenseSet<StringRef> &seqLenSkip) const {
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};
    FailureOr<Value> maybeNonOne = mulBroadcast(input);
    if (failed(maybeNonOne))
      return failure();

    // Look for add(row_indices, offset)
    auto maybeAdd = getDefiningOpSkipping<tosa::AddOp>(maybeNonOne.value(),
                                                       expandAndCollapse);
    if (failed(maybeAdd))
      return failure();

    auto add = maybeAdd.value();
    Value addInput1 = add.getInput1();
    Value addInput2 = add.getInput2();

    // One input should be row indices (constant range on dimension 1)
    // Other input should trace to block arg (the prefix offset)
    Value offset;
    if (succeeded(
            isConstantRange(cast<TypedValue<TensorType>>(addInput1), 1))) {
      offset = addInput2;
    } else if (succeeded(isConstantRange(
                   cast<TypedValue<TensorType>>(addInput2), 1))) {
      offset = addInput1;
    } else {
      return failure();
    }

    // Trace offset back through broadcasts to find the original value
    FailureOr<Value> maybeOffset = mulBroadcast(offset);
    if (failed(maybeOffset))
      maybeOffset = offset;

    FailureOr<Value> maybeOffsetUnwrapped =
        getValueSkipping(maybeOffset.value(), expandAndCollapse);
    if (failed(maybeOffsetUnwrapped))
      return failure();

    Value unwrappedOffset = maybeOffsetUnwrapped.value();

    // Verify offset is i32 and traces back to a block argument
    if (!isI32BlockArgument(unwrappedOffset, seqLenSkip))
      return failure();

    return unwrappedOffset;
  }

  // Helper to try detecting KV-cache pattern
  // Returns the seqLen value if successful
  FailureOr<Value>
  tryKVCachePattern(Value input, const DenseSet<StringRef> &seqLenSkip) const {
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};
    FailureOr<Value> maybeNonOne = mulBroadcast(input);
    if (failed(maybeNonOne))
      return failure();

    // Check that the right dimensions are broadcasted (scalar-like)
    auto beforeBroadcastShape = dyn_cast<ShapedType>(maybeNonOne->getType());
    if (!beforeBroadcastShape)
      return failure();

    auto shape = beforeBroadcastShape.getShape();
    if (beforeBroadcastShape.getRank() > 2 &&
        !llvm::all_of(shape.slice(2), [](int32_t v) { return v == 1; }))
      return failure();

    auto maybeCurrentSeqLen =
        getValueSkipping(maybeNonOne.value(), expandAndCollapse);
    assert(succeeded(maybeCurrentSeqLen) && "Must have non-reshape op");
    Value currentSeqLen = maybeCurrentSeqLen.value();

    // Verify currentSeqLen is i32 and traces back to a block argument
    if (!isI32BlockArgument(currentSeqLen, seqLenSkip))
      return failure();

    return currentSeqLen;
  }

  /*
  LSE pattern for seqLen1 would be simplified from
  log(sum(exp(sub(x, x)))) + max(x)
  = log(exp(sub(x, x))) + x
  = sub(x, x) + x

  Upstream disabled folding of log(exp(..)) by default, so we need to match the
  following two patterns:
  1. The folded pattern: sub(x, x) + x
  2. The unfolded pattern: log(exp(sub(x, x))) + x
  */
  Value getLSESeqLen1(tosa::SubOp subOp) const {
    if (subOp.getInput1() != subOp.getInput2()) {
      // this is a sub of two different values, we cannot match LSE
      return nullptr;
    }
    Value subInput = subOp.getInput1();
    for (Operation *user : subOp->getUsers()) {
      // Pattern 1: Check for direct add: sub(x, x) + x
      if (tosa::AddOp addOp = dyn_cast<tosa::AddOp>(user)) {
        Value addOpInput1 = addOp.getInput1();
        Value addOpInput2 = addOp.getInput2();
        if (tosa::SubOp addOperandSubOp =
                addOpInput1.getDefiningOp<tosa::SubOp>()) {
          if (addOperandSubOp == subOp && addOpInput2 == subInput)
            return addOp.getOutput();
        } else if (tosa::SubOp addOperandSubOp =
                       addOpInput2.getDefiningOp<tosa::SubOp>()) {
          if (addOperandSubOp == subOp && addOpInput1 == subInput) {
            return addOp.getOutput();
          }
        }
      }

      // Pattern 2: Check for log(exp(sub(x, x))) + x
      tosa::ExpOp expOp = dyn_cast<tosa::ExpOp>(user);
      if (!expOp)
        continue;

      for (Operation *expUser : expOp->getUsers()) {
        tosa::LogOp logOp = dyn_cast<tosa::LogOp>(expUser);
        if (!logOp)
          continue;

        for (Operation *logUser : logOp->getUsers()) {
          tosa::AddOp addOp = dyn_cast<tosa::AddOp>(logUser);
          if (!addOp)
            continue;

          Value addOpInput1 = addOp.getInput1();
          Value addOpInput2 = addOp.getInput2();
          // Check if one input is the log result and the other is the
          // original subInput (x)
          if ((addOpInput1 == logOp.getOutput() && addOpInput2 == subInput) ||
              (addOpInput2 == logOp.getOutput() && addOpInput1 == subInput)) {
            return addOp.getOutput();
          }
        }
      }
    }
    return nullptr;
  }

  /**
   * Attempts to match and extract a Log-Sum-Exp (LSE) pattern from TOSA
   * operations.
   *
   * This function traverses the users of a reduce sum operation to identify a
   * complete LSE computation pattern, which typically consists of:
   * 1. A reduce_max operation to find the maximum values (in some cases this
   * might not exist)
   * 2. Subtraction of the max from original values (implicit in the pattern)
   * 3. Exponential and sum operations (represented by reduceSum)
   * 4. A logarithm operation on the result
   * 5. Addition of the original max values back
   *
   * Note that reduceSum and reduceMax are given.
   *
   * The LSE pattern: log(sum(exp(x - max(x)))) + max(x)
   */
  Value getLSE(Operation *reduceSum, Operation *reduceMax,
               tosa::LogOp logOp = nullptr) const {
    for (auto *user : reduceSum->getUsers()) {
      if (auto op = dyn_cast<tosa::CastOp>(user)) {
        // we already found a log
        if (logOp != nullptr)
          return nullptr;
        Value val = getLSE(op, reduceMax);
        if (val)
          return val;
      } else if (auto op = dyn_cast<tosa::LogOp>(user)) {
        // we already found a log
        if (logOp != nullptr)
          return nullptr;
        Value val = getLSE(op, reduceMax, op);
        if (val)
          return val;
      } else if (auto addOp = dyn_cast<tosa::AddOp>(user)) {
        if (!logOp)
          continue;

        DenseSet<StringRef> expandAndCollapse{
            tensor::CollapseShapeOp::getOperationName(),
            tensor::ExpandShapeOp::getOperationName()};
        auto maybeLogOp = getDefiningOpSkipping<tosa::LogOp>(addOp.getInput1(),
                                                             expandAndCollapse);
        if (failed(maybeLogOp))
          maybeLogOp = getDefiningOpSkipping<tosa::LogOp>(addOp.getInput2(),
                                                          expandAndCollapse);

        assert(succeeded(maybeLogOp) && "Expected to find log op");
        auto logOpFromAdd = maybeLogOp.value();

        // must match the logOp
        if (logOp != logOpFromAdd)
          return nullptr;

        // ReduceMax could be gone if there's only one dim, then, we don't
        // know the previous op, because it could be anything we want to fuse
        auto maybeReduceMaxOpFromAdd = getDefiningOpSkipping<Operation *>(
            addOp.getInput1(), expandAndCollapse);
        if (failed(maybeReduceMaxOpFromAdd) ||
            isa<tosa::LogOp>(maybeReduceMaxOpFromAdd.value()))
          maybeReduceMaxOpFromAdd = getDefiningOpSkipping<Operation *>(
              addOp.getInput2(), expandAndCollapse);

        assert(succeeded(maybeReduceMaxOpFromAdd) &&
               "Expected to find reduce max op");
        auto *reduceMaxOpFromAdd = maybeReduceMaxOpFromAdd.value();

        if (auto castOp = dyn_cast<tosa::CastOp>(reduceMaxOpFromAdd)) {
          // if the reduceMax is a cast, we need to get the input of the cast
          auto maybeCast = getDefiningOpSkipping<Operation *>(
              castOp.getInput(), expandAndCollapse);
          assert(succeeded(maybeCast) && "Expected a castOp");
          reduceMaxOpFromAdd = maybeCast.value();
        }

        // must match the reduceMax
        if (!reduceMax || reduceMax != reduceMaxOpFromAdd)
          return nullptr;

        return addOp.getOutput();
      } else if (isa<tensor::CollapseShapeOp>(user) ||
                 isa<tensor::ExpandShapeOp>(user)) {
        Value val = getLSE(user, reduceMax, logOp);
        if (val)
          return val;
      }
    }
    return nullptr;
  }

  // Detects sequence length masking patterns:
  //   - KV-cache: select(greater(col_indices, seqLen), -inf, value)
  //   - Prefix causal: select(greater(col_indices, row_indices + offset), -inf,
  //   value)
  // Updates SeqLenMaskResult with the detected pattern type
  void analyzeSelectForSeqLenMask(tosa::SelectOp select,
                                  SeqLenMaskResult &result,
                                  const DenseSet<StringRef> &opsToSkip,
                                  const DenseSet<StringRef> &seqLenSkip) const {
    auto pred = select.getInput1();
    auto maybeGreater = getDefiningOpSkipping<tosa::GreaterOp>(pred, opsToSkip);
    if (failed(maybeGreater))
      return;

    auto greater = maybeGreater.value();

    // input1 must be column indices (constant range from 0)
    if (failed(isConstantRange(greater.getInput1(), 0)))
      return;

    Value input2 = greater.getInput2();

    // Try KV-cache pattern (scalar seqLen) if not already found
    if (!result.seqLen) {
      auto maybeKVCache = tryKVCachePattern(input2, seqLenSkip);
      if (succeeded(maybeKVCache)) {
        result.seqLen = maybeKVCache.value();
      }
    }

    // Try prefix causal pattern (row_indices + offset) if not already found
    if (!result.prefixOffset) {
      auto maybePrefixCausal = tryPrefixCausalPattern(input2, seqLenSkip);
      if (succeeded(maybePrefixCausal)) {
        result.prefixOffset = maybePrefixCausal.value();
      }
    }
  }

  FailureOr<SeqLenMaskResult> getSeqLenMask(Value softmaxInput) const {
    auto maybeSelect = getSelectWithNegInf(softmaxInput);
    if (failed(maybeSelect))
      return failure();

    auto select = maybeSelect.value();

    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::CastOp::getOperationName(),
                                  tosa::MulOp::getOperationName()};

    // Common set used by both pattern detectors
    DenseSet<StringRef> seqLenSkip{tensor::CollapseShapeOp::getOperationName(),
                                   tensor::ExpandShapeOp::getOperationName(),
                                   tosa::TransposeOp::getOperationName(),
                                   tosa::MulOp::getOperationName()};

    Value inputToContinue = select.getInput3();
    SeqLenMaskResult currentResult{inputToContinue, nullptr, nullptr};

    // Analyze the first (outer) select
    analyzeSelectForSeqLenMask(select, currentResult, opsToSkip, seqLenSkip);

    // Check if the inputToContinue (input3) is another chained select with -inf
    // This handles the case where KVCache and prefix causal use separate
    // selects.
    bool haveSeqLen = currentResult.seqLen != nullptr;
    bool havePrefixOffset = currentResult.prefixOffset != nullptr;

    if (haveSeqLen != havePrefixOffset) {
      auto maybeChainedSelect = getSelectWithNegInf(inputToContinue);
      if (succeeded(maybeChainedSelect)) {
        auto chainedSelect = maybeChainedSelect.value();
        // Try to analyze the chained select for the missing pattern
        analyzeSelectForSeqLenMask(chainedSelect, currentResult, opsToSkip,
                                   seqLenSkip);
        // Only update inputToContinue if we found the complementary pattern
        bool foundComplementary =
            (!haveSeqLen && currentResult.seqLen) ||
            (!havePrefixOffset && currentResult.prefixOffset);
        if (foundComplementary) {
          currentResult.inputToContinue = chainedSelect.getInput3();
        }
      }
    }

    // We need at least one pattern to be detected
    if (!currentResult.seqLen && !currentResult.prefixOffset)
      return failure();

    return currentResult;
  }

  /*
  return true if there is path from `fromVal` to `toVal`
  */
  bool areConnected(Value fromVal, Value toVal,
                    llvm::SmallDenseMap<Value, bool> &pathMap) const {
    if (fromVal == toVal) {
      pathMap[fromVal] = true;
    }
    if (pathMap.contains(fromVal))
      return pathMap[fromVal];
    pathMap[fromVal] = false;
    for (Operation *user : fromVal.getUsers()) {
      for (Value userResultVal : user->getResults()) {
        if (areConnected(userResultVal, toVal, pathMap))
          pathMap[fromVal] = true;
      }
    }
    return pathMap[fromVal];
  }

  /*
  if softmax happens in a different datatype/precision compared to the first
  gemm output, then first gemm output type would have a cast operation that
  converts input to softmax data type. This function traces from first gemm
  output to cast operation and then traces path from cast to softmax input.
  Later during `match()` types of the casts on both softmax input and outputs
  are compared to ensure that cast op is indeed to change type of the softmax
  and it is not part of the fusion.
  */
  FailureOr<Type> getSoftmaxType(Value firstGemmOutput,
                                 Value softmaxInput) const {
    llvm::SmallDenseSet<Operation *> visited;
    llvm::SmallVector<Operation *> worklist = {
        firstGemmOutput.getUsers().begin(), firstGemmOutput.getUsers().end()};
    Type softmaxInputType =
        cast<ShapedType>(softmaxInput.getType()).getElementType();
    Type lastCastOutputType = nullptr;
    while (!worklist.empty()) {
      Operation *user = worklist.pop_back_val();
      if (visited.contains(user))
        continue;
      visited.insert(user);
      if (isa<tosa::CastOp>(user)) {
        // trace cast op to softmax input
        llvm::SmallDenseMap<Value, bool> pathMap;
        Value castOutput = user->getResult(0);
        Type castOutputType =
            cast<ShapedType>(castOutput.getType()).getElementType();
        if (areConnected(castOutput, softmaxInput, pathMap) &&
            castOutputType == softmaxInputType) {
          lastCastOutputType = castOutputType;
        }
      }
      worklist.insert(worklist.end(), user->getUsers().begin(),
                      user->getUsers().end());
    }
    if (lastCastOutputType == nullptr)
      return failure();
    return lastCastOutputType;
  }

  FailureOr<SoftmaxMatcherValues> maybeSoftmaxNumerator(Value val,
                                                        Operation *rsum) const {
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};
    auto maybeExp = getDefiningOpSkipping<tosa::ExpOp>(val, expandAndCollapse);
    if (failed(maybeExp))
      return failure();
    tosa::ExpOp exp = maybeExp.value();

    auto maybeSub =
        getDefiningOpSkipping<tosa::SubOp>(exp.getInput1(), expandAndCollapse);
    if (failed(maybeSub))
      return failure();
    tosa::SubOp sub = maybeSub.value();

    bool hasTosaReduce = false;
    Value result;
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::MulOp::getOperationName()};
    auto maybeRmax =
        getDefiningOpSkipping<tosa::ReduceMaxOp>(sub.getInput2(), opsToSkip);
    tosa::ReduceMaxOp rmax = nullptr;
    if (succeeded(maybeRmax)) {
      rmax = maybeRmax.value();
      if (rmax.getInput() != sub.getInput1())
        return failure();

      hasTosaReduce = true;
      result = rmax.getInput();
    } else {
      // this case happens when we have seq_len=1. in that case reduction size
      // would be one and both reduceMax and reduceSum would have been
      // const-folded
      if (sub.getInput1() != sub.getInput2())
        return failure();

      hasTosaReduce = false;
      result = sub.getInput1();
    }
    return SoftmaxMatcherValues{result, sub, exp, rmax, rsum, hasTosaReduce};
  }

  FailureOr<SoftmaxMatcherValues> maybeSoftmaxDenominator(Value val) const {
    FailureOr<SoftmaxMatcherValues> result;
    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::MulOp::getOperationName()};
    auto maybeRsum = getDefiningOpSkipping<tosa::ReduceSumOp>(val, opsToSkip);
    if (succeeded(maybeRsum)) {
      auto rsum = maybeRsum.value();
      result = maybeSoftmaxNumerator(rsum.getInput(), rsum);
      if (succeeded(result) && !result.value().hasReduceOp) {
        // if we see tosa::Reduce Op in the denominator then we expect to see
        // tosa::Reduce Op in the numerator as well
        return failure();
      }
    } else {
      result = maybeSoftmaxNumerator(val, val.getDefiningOp());
      if (succeeded(result) && result.value().hasReduceOp) {
        // if we don't see tosa::Reduce Op in the denominator then we expect
        // to not see any tosa::Reduce Op in the numerator as well
        return failure();
      }
    }
    return result;
  }

  FailureOr<SoftmaxMatcherValues> maybeSoftmax(Value val) const {
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};
    auto maybeMul = getDefiningOpSkipping<tosa::MulOp>(val, expandAndCollapse);
    if (failed(maybeMul))
      return failure();
    auto mul = maybeMul.value();

    DenseSet<StringRef> opsToSkip{tensor::CollapseShapeOp::getOperationName(),
                                  tensor::ExpandShapeOp::getOperationName(),
                                  tosa::MulOp::getOperationName()};
    auto maybeRecIn1 =
        getDefiningOpSkipping<tosa::ReciprocalOp>(mul.getInput1(), opsToSkip);
    if (succeeded(maybeRecIn1)) {
      return maybeSoftmaxDenominator(maybeRecIn1.value().getInput1());
    }

    auto maybeRecIn2 =
        getDefiningOpSkipping<tosa::ReciprocalOp>(mul.getInput2(), opsToSkip);
    if (succeeded(maybeRecIn2)) {
      return maybeSoftmaxDenominator(maybeRecIn2.value().getInput1());
    }
    return failure();
  }

  Value normalizeInputTensor(PatternRewriter &rewriter, Location loc,
                             TypedValue<TensorType> inputTensor) const {
    if (!inputTensor) {
      return inputTensor;
    }
    ArrayRef<int64_t> shape = inputTensor.getType().getShape();
    SmallVector<int64_t, 4> reverseInputShape =
        llvm::to_vector<4>(llvm::reverse(shape));
    SmallVector<int64_t, 4> normalizedShape;
    int collapsedBatchLen = 1;
    for (int64_t dimLen : ArrayRef<int64_t>{reverseInputShape}.slice(2)) {
      collapsedBatchLen *= dimLen;
    }
    normalizedShape.push_back(collapsedBatchLen);
    normalizedShape.push_back(reverseInputShape[1]);
    normalizedShape.push_back(reverseInputShape[0]);
    auto normalizedType = RankedTensorType::get(
        normalizedShape, inputTensor.getType().getElementType());
    auto normalizedShapeValue =
        tosa::getTosaConstShape(rewriter, loc, normalizedShape);
    auto reshapeOp = tosa::ReshapeOp::create(rewriter, loc, normalizedType,
                                             inputTensor, normalizedShapeValue);
    return reshapeOp;
  }

  void moveUsersAfterExpandShape(PatternRewriter &rewriter, Location loc,
                                 Operation *expandedOutLse,
                                 tosa::AddOp addOp) const {
    llvm::SmallVector<Operation *> toMove;
    llvm::SmallDenseSet<Operation *> visited;
    llvm::SmallVector<Operation *> worklist;

    // Seed the worklist with direct users
    for (Operation *user : addOp->getUsers()) {
      if (!isa<func::ReturnOp>(user))
        worklist.push_back(user);
    }

    // Collect all transitive users (BFS)
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      if (!visited.insert(op).second)
        continue;
      toMove.push_back(op);
      for (Operation *user : op->getUsers()) {
        if (!isa<func::ReturnOp>(user))
          worklist.push_back(user);
      }
    }
    // Sort by IR order
    llvm::sort(toMove, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });

    // Move in reverse order to preserve dependencies
    for (Operation *op : llvm::reverse(toMove))
      op->moveAfter(expandedOutLse);
  }

  // This function identifies when the currentSeqLen is a block argument
  // that is one dimensional, and broadcasts it to the correct shape, and with
  // the correct batch, numHeads values
  FailureOr<Value> addBroadcastForBlockArg(PatternRewriter &rewriter,
                                           Value currentSeqLen,
                                           Value matrixQ) const {
    // Exit early if there is no currentSeqLen (no kv-cache)
    if (!currentSeqLen)
      return failure();

    // Exit early if currentSeqLen is not a 1D block argument
    if (!isa<BlockArgument>(currentSeqLen) ||
        cast<ShapedType>(currentSeqLen.getType()).getRank() != 1)
      return failure();

    // Extract the shape information
    auto origShape = cast<ShapedType>(currentSeqLen.getType()).getShape()[0];

    // Find the original shape of matrixQ (before reshaping) to get the batch
    // and numHeads values
    if (!isa<tensor::CollapseShapeOp>(matrixQ.getDefiningOp())) {
      // If we didn't find a collapse op, we can't determine the original shape
      return failure();
    }

    auto collapse = cast<tensor::CollapseShapeOp>(matrixQ.getDefiningOp());
    auto reassocIndices = collapse.getReassociationIndices();

    // Check if the first reassociation merges two dimensions [0, 1]
    if (reassocIndices.empty() || reassocIndices[0].size() != 2)
      return failure();

    // Get the original shape before collapse
    auto srcShape = collapse.getSrcType().getShape();

    if (srcShape.size() < 2)
      return failure();

    int64_t batch = srcShape[0];
    int64_t numHeads = srcShape[1];

    // Create a tensor.expand_shape from 1D to 2D
    auto loc = currentSeqLen.getLoc();
    auto elemTy = cast<ShapedType>(currentSeqLen.getType()).getElementType();
    SmallVector<int64_t, 2> expandedShape{origShape, 1};
    auto expandedType = RankedTensorType::get(expandedShape, elemTy);
    SmallVector<ReassociationIndices, 1> reassoc{{0, 1}};
    Value expanded = tensor::ExpandShapeOp::create(rewriter, loc, expandedType,
                                                   currentSeqLen, reassoc);

    // Create a tosa.const that is all zeros, but in our desired shape of
    // batch x numHeads
    auto broadcastTy = RankedTensorType::get({batch, numHeads}, elemTy);
    auto oneElems = cast<ElementsAttr>(rewriter.getOneAttr(broadcastTy));
    auto constOp = tosa::ConstOp::create(rewriter, loc, broadcastTy, oneElems);

    // Create a tosa.mul (broadcast) to our desired batch and numHeads values.
    auto mul =
        rock::tosa::getMulOp(rewriter, loc, expanded, constOp, broadcastTy);
    return mul.getOutput();
  }

  FailureOr<std::pair<int64_t, int64_t>> getNumHeadsGQA(Value value,
                                                        bool isQ) const {
    // this size is = batch*numHeads
    auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse)
      return failure();

    auto reassociationIdx = collapse.getReassociationIndices();

    // expected to reshape to three dimensions (input to tosa.matmul)
    if (reassociationIdx.size() != 3)
      return failure();
    size_t expectedGroupSize = isQ ? 2 : 3;
    if (reassociationIdx[0].size() != expectedGroupSize ||
        reassociationIdx[1].size() != 1 || reassociationIdx[2].size() != 1)
      return failure();

    // group size must match groupSizeQ
    int64_t count = 0;
    for (const auto &reassociation : reassociationIdx) {
      for (auto idx : reassociation) {
        if (count != idx)
          return failure();
        count++;
      }
    }

    auto reshapeInputShape =
        cast<ShapedType>(collapse.getSrc().getType()).getShape();
    // we expect the input to be batch x num_heads x D x K (or K x D)
    size_t expectedSize = isQ ? 4 : 5;
    if (reshapeInputShape.size() != expectedSize)
      return failure();

    int64_t batch = reshapeInputShape[0];
    int64_t numHeads = reshapeInputShape[1];
    return std::make_pair(batch, numHeads);
  }

  LogicalResult checkBroadcastGQA(Value value, int64_t expectedRepeat) const {
    auto collapse = value.getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse)
      return failure();
    Value collapseVal = collapse.getSrc();

    auto maybeNonOne = mulBroadcast(collapseVal, /*skipCollapseExpand=*/false);
    if (failed(maybeNonOne))
      return failure();

    // we should be doing batch x num_heads x 1 x D x K -> batch x num_heads x
    // REPEAT x D x K
    Value nonOne = maybeNonOne.value();
    auto shapeBeforeBroadcast = cast<ShapedType>(nonOne.getType()).getShape();
    auto shapeAfterBroadcast =
        cast<ShapedType>(
            collapseVal.getDefiningOp<tosa::MulOp>().getOutput().getType())
            .getShape();
    if (shapeBeforeBroadcast.size() != shapeAfterBroadcast.size())
      return failure();

    // we expect five dimensions
    if (shapeBeforeBroadcast.size() != 5)
      return failure();

    // dimension we are broadcasting
    if (shapeBeforeBroadcast[2] != 1 ||
        shapeAfterBroadcast[2] != expectedRepeat)
      return failure();

    // rest of dimensions must be the same
    for (size_t idx = 0; idx < shapeBeforeBroadcast.size(); idx++) {
      if (idx != 2 && shapeBeforeBroadcast[idx] != shapeAfterBroadcast[idx])
        return failure();
    }

    return success();
  }

  FailureOr<Value> sliceTensorGQA(PatternRewriter &rewriter, Value value,
                                  int64_t batch, int64_t numHeads,
                                  int64_t repeat) const {
    Location loc = value.getLoc();
    ArrayRef<int64_t> shape = cast<ShapedType>(value.getType()).getShape();
    if (shape.size() != 3)
      return failure();

    if (shape[0] != (batch * numHeads * repeat))
      return failure();

    // reshape group x D x K -> batch x num_heads x repeat x D x K
    rock::BottomUpTMBuilder unmergeDims(rewriter, {"group", "dim0", "dim1"},
                                        shape, loc);
    unmergeDims.unmerge({"batch", "num_heads", "repeat"}, {0, 1, 2}, "group",
                        {batch, numHeads, repeat});
    unmergeDims.passThrough({3, 4}, {1, 2});
    rock::TransformMapAttr unmergeDimsAttr = unmergeDims.get();

    // slice repeat to 1
    auto sliceRepeat =
        rock::BottomUpTMBuilder::above(unmergeDims, unmergeDimsAttr);
    sliceRepeat.slice({"repeat"}, {"repeat"}, {0}, {1});
    sliceRepeat.passThrough({"batch", "num_heads", "dim0", "dim1"});
    rock::TransformMapAttr sliceRepeatAttr = sliceRepeat.get();

    // reshape back to group/repeat x D x K
    auto finalMerge =
        rock::BottomUpTMBuilder::above(sliceRepeat, sliceRepeatAttr);
    finalMerge.merge("group", 0, {"batch", "num_heads", "repeat"});
    finalMerge.passThrough({"dim0", "dim1"}, {1, 2}, {"dim0", "dim1"});
    rock::TransformMapAttr finalMergeAttr = finalMerge.get();

    ArrayAttr transformsAttr = rewriter.getArrayAttr(
        {finalMergeAttr, sliceRepeatAttr, unmergeDimsAttr});
    return rock::transform(rewriter, value, transformsAttr);
  }

  /*
  This tries to identify if GQA is used, and undoes the broadcast. The expected
  IR is:

  // clang-format off
  ```
  %q = tensor.collapse %q [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into
  tensor<32x1x128xf16>

  // broadcast from numHeadsK, 1 -> numHeadsK, repeat where
  numHeadsQ=numHeadsK*repeat %k = tosa.mul %k, constant=1, constant=0 :
  (tensor<1x8x1x128x64xf16>, tensor<1x8x4x128x64xf16>, tensor<1xi8>) ->
  tensor<1x8x4x128x64xf16>
  // collapse batch, numHeadsK and repeat into group dimension,
  group=batch*numHeadsK*repeat %k = tensor.collapse_shape %k [[0, 1, 2], [3],
  [4]] : tensor<1x8x4x128x64xf16> into tensor<32x128x64xf16>

  %v = same transforms as %k
  rock.attention(%q, %k, %v)
  ```
  // clang-format on

  Note that if we identify the GQA pattern, we slice the K and V tensors
  and pass numHeadsQ and numHeadsKV to rock.attention. Otherwise, K and V
  tensors are left untouched and numHeadsQ=1, numHeadsKV=1.
  */
  std::tuple<Value, Value, Value, IntegerAttr, IntegerAttr>
  getGQAValues(PatternRewriter &rewriter, Value queries, Value keys,
               Value values) const {
    // default values in case GQA is not pattern matched
    IntegerAttr numHeadsQAttr = rewriter.getI32IntegerAttr(1);
    IntegerAttr numHeadsKVAttr = rewriter.getI32IntegerAttr(1);
    auto defaultValues =
        std::make_tuple(queries, keys, values, numHeadsQAttr, numHeadsKVAttr);

    FailureOr<std::pair<int64_t, int64_t>> reshapeQResults =
        getNumHeadsGQA(queries, true);
    if (failed(reshapeQResults))
      return defaultValues;
    int64_t batchQ = reshapeQResults->first;
    int64_t numHeadsQ = reshapeQResults->second;

    FailureOr<std::pair<int64_t, int64_t>> reshapeKResults =
        getNumHeadsGQA(keys, false);
    if (failed(reshapeKResults))
      return defaultValues;
    int64_t batchK = reshapeKResults->first;
    int64_t numHeadsK = reshapeKResults->second;

    FailureOr<std::pair<int64_t, int64_t>> reshapeVResults =
        getNumHeadsGQA(values, false);
    if (failed(reshapeVResults))
      return defaultValues;
    int64_t batchV = reshapeVResults->first;
    int64_t numHeadsV = reshapeVResults->second;

    // batch must be equal for all tensors
    if (batchQ != batchK || batchQ != batchV)
      return defaultValues;

    // num heads of K and V must be equal
    if (numHeadsK != numHeadsV)
      return defaultValues;

    // numHeadsQ must be divisible by numHeadsKV
    if (numHeadsQ % numHeadsK != 0)
      return defaultValues;

    int64_t expectedRepeat = numHeadsQ / numHeadsK;
    // check we are doing the expected broadcast for K and V
    LogicalResult kCorrect = checkBroadcastGQA(keys, expectedRepeat);
    LogicalResult vCorrect = checkBroadcastGQA(values, expectedRepeat);
    if (failed(kCorrect) || failed(vCorrect))
      return defaultValues;

    // update keys and values (slicing the repeats)
    auto maybeKeys =
        sliceTensorGQA(rewriter, keys, batchK, numHeadsK, expectedRepeat);
    auto maybeValues =
        sliceTensorGQA(rewriter, values, batchV, numHeadsV, expectedRepeat);
    if (failed(maybeKeys) || failed(maybeValues))
      return defaultValues;

    keys = maybeKeys.value();
    values = maybeValues.value();

    numHeadsQAttr = rewriter.getI32IntegerAttr(numHeadsQ);
    numHeadsKVAttr = rewriter.getI32IntegerAttr(numHeadsK);
    LLVM_DEBUG(llvm::dbgs() << "Found GQA pattern, numHeadsQ=" << numHeadsQ
                            << " numHeadsKV=" << numHeadsK << "\n");
    return std::make_tuple(queries, keys, values, numHeadsQAttr,
                           numHeadsKVAttr);
  }

  FailureOr<AttentionMatcherValues> match(tosa::MatMulOp op) const {
    Value softmaxOutput = op.getA();
    DenseSet<StringRef> expandAndCollapse{
        tensor::CollapseShapeOp::getOperationName(),
        tensor::ExpandShapeOp::getOperationName()};

    // check if the softmax is done in different precision compared to GEMMs
    Type softmaxType =
        cast<ShapedType>(softmaxOutput.getType()).getElementType();
    auto maybesoftmaxOutputCastOp =
        getDefiningOpSkipping<tosa::CastOp>(softmaxOutput, expandAndCollapse);
    if (succeeded(maybesoftmaxOutputCastOp)) {
      softmaxOutput = maybesoftmaxOutputCastOp.value().getInput();
      if (succeeded(getDefiningOpSkipping<tosa::CastOp>(softmaxOutput,
                                                        expandAndCollapse))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "softmax output has multiple casts. rocMLIR only allows "
                      "one cast between softmax and gemm2\n");
        return failure();
      }
      softmaxType = cast<ShapedType>(softmaxOutput.getType()).getElementType();
    }

    // pattern match for softmax operation
    FailureOr<SoftmaxMatcherValues> softmaxMatcherResults =
        maybeSoftmax(softmaxOutput);

    if (failed(softmaxMatcherResults))
      return failure();
    SoftmaxMatcherValues softmaxMatcherValues = softmaxMatcherResults.value();

    Value softmaxInput = softmaxMatcherValues.softmaxInput;
    bool hasReduceOp = softmaxMatcherValues.hasReduceOp;
    Operation *sub = softmaxMatcherValues.subOp;
    Operation *rmax = softmaxMatcherValues.reduceMaxOp;
    Operation *rsum = softmaxMatcherValues.reduceSumOp;
    Value lse;
    if (hasReduceOp) {
      lse = getLSE(rsum, rmax);
    } else {
      // if there is no reduce op, then we have seq_len=1 and lse is either
      // sub(x, x) + x or log(exp(sub(x, x))) + x
      lse = getLSESeqLen1(cast<tosa::SubOp>(sub));
    }
    // lse has three, four, or five dimensions depending on the attention type:
    // - Rank 5: [batch, heads, splitKV, seq_q, 1]
    // - Rank 4: [batch, heads, seq_q, 1]
    // - Rank 3: [batch*heads, seq_q, 1]
    if (lse) {
      auto type = cast<ShapedType>(lse.getType());
      if (type.getRank() != 5 && type.getRank() != 4 && type.getRank() != 3)
        return failure();
      // last dimension must be 1
      if (type.getDimSize(type.getRank() - 1) != 1)
        return failure();
    }

    // Detect sequence length masking patterns (KV-cache or prefix causal)
    // Note that non KV-Cache fusions might have tosa.select
    // so, if the checks fail, we just keep going
    Value kvCacheInput, currentSeqLen, prefixOffset;
    auto maybeSeqLenMask = getSeqLenMask(softmaxInput);
    if (succeeded(maybeSeqLenMask)) {
      auto result = maybeSeqLenMask.value();
      kvCacheInput = result.inputToContinue;
      currentSeqLen = result.seqLen;
      prefixOffset = result.prefixOffset;
    } else {
      kvCacheInput = softmaxInput;
    }

    // currentSeqLen and prefixOffset need one or two dimensions
    auto hasInvalidRank = [](Value v, StringRef name) {
      if (v && cast<ShapedType>(v.getType()).getRank() > 2) {
        LLVM_DEBUG(llvm::dbgs() << name << " has more than 2 dimensions\n");
        return true;
      }
      return false;
    };
    if (hasInvalidRank(currentSeqLen, "currentSeqLen") ||
        hasInvalidRank(prefixOffset, "prefixOffset"))
      return failure();

    // Try standard causal detection if not prefix causal
    auto causal = getCausal(kvCacheInput);
    bool isCausal = succeeded(causal) || prefixOffset;
    // Use causal input if standard causal, otherwise use kvCacheInput
    // (which is also set for prefix causal pattern)
    Value causalMaskInput =
        (succeeded(causal) && !prefixOffset) ? causal.value() : kvCacheInput;

    OpBuilder b{op};
    ElementwiseRegionFinder<tosa::MatMulOp> preSoftmaxElementwiseFinder;
    preSoftmaxElementwiseFinder.visit(causalMaskInput);
    FailureOr<tosa::MatMulOp> maybeFirstMatMul =
        preSoftmaxElementwiseFinder.getFirstGemmBasedOp();
    if (failed(maybeFirstMatMul)) {
      LLVM_DEBUG(llvm::dbgs() << "first matmul not found\n");
      return failure();
    }

    TypedValue<TensorType> matC = maybeFirstMatMul.value().getOutput();
    ArrayRef<int64_t> shapeC = matC.getType().getShape();
    bool isDotProduct = *(std::prev(shapeC.end(), 1)) == 1;
    isDotProduct &= *(std::prev(shapeC.end(), 2)) == 1;

    LLVM_DEBUG(llvm::dbgs()
               << "first matmul = " << maybeFirstMatMul.value() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "hasReduceOp = " << hasReduceOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "isKVCache: " << (bool)currentSeqLen << "\n");
    LLVM_DEBUG(llvm::dbgs() << "isCausal = " << isCausal << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "isPrefixCausal = " << (bool)prefixOffset << "\n");
    if (isDotProduct && hasReduceOp)
      return failure();
    if (!isDotProduct && !hasReduceOp)
      return failure();

    // if softmax is done in different precision than GEMMs then there must be
    // cast operation on one of the uses of first GEMM
    if (succeeded(maybesoftmaxOutputCastOp)) {
      FailureOr<Type> softmaxInputCast = getSoftmaxType(matC, softmaxInput);
      if (failed(softmaxInputCast)) {
        LLVM_DEBUG(llvm::dbgs() << "softmax input cast not found\n");
        return failure();
      }
      if (softmaxInputCast.value() != softmaxType) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "softmax type on input cast and output cast does not match\n");
        return failure();
      }
    }

    // populate struct to aggregate attention matcher values and pass it to
    // rewriter
    AttentionMatcherValues attentionMatcherValues;
    attentionMatcherValues.isCausal = isCausal;
    attentionMatcherValues.prefixOffset = prefixOffset;
    attentionMatcherValues.softmaxType = softmaxType;
    attentionMatcherValues.softmaxValues = softmaxMatcherValues;
    attentionMatcherValues.lse = lse;
    attentionMatcherValues.causalMaskInput = causalMaskInput;
    attentionMatcherValues.currentSeqLen = currentSeqLen;
    attentionMatcherValues.preSoftmaxElementwiseFinder =
        preSoftmaxElementwiseFinder;
    return attentionMatcherValues;
  }

  void rewrite(tosa::MatMulOp op,
               const AttentionMatcherValues &attentionMatcherValues,
               PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto outputType = cast<RankedTensorType>(op.getType());
    Value output = bufferization::AllocTensorOp::create(
        rewriter, loc, outputType, ValueRange{});
    RankedTensorType lseType;
    Value lse = attentionMatcherValues.lse;
    Value lseOut, lseOrig;
    SmallVector<ReassociationIndices> reassocIndicesLSE;

    if (lse) {
      // rock.attention expects lse to have the shape = {B, SEQ_LEN_Q}
      // Collapse all leading dimensions into one, and last two dimensions
      // into another
      // Rank 5: [batch, heads, splitKV, seq_q, 1] ->
      //         [batch*heads*splitKV, seq_q]
      // Rank 4: [batch, heads, seq_q, 1] -> [batch*heads, seq_q]
      // Rank 3: [batch*heads, seq_q, 1] -> [batch*heads, seq_q]
      int rank = cast<ShapedType>(lse.getType()).getRank();
      ReassociationIndices leadingDims, trailingDims;
      for (int i = 0; i < rank - 2; ++i)
        leadingDims.push_back(i);
      trailingDims = {rank - 2, rank - 1};
      reassocIndicesLSE = {leadingDims, trailingDims};

      lseOrig = lse;
      lse = tensor::CollapseShapeOp::create(rewriter, op.getLoc(), lse,
                                            reassocIndicesLSE);

      lseType = cast<RankedTensorType>(lse.getType());
      lseOut = bufferization::AllocTensorOp::create(rewriter, loc, lseType,
                                                    ValueRange{});
    }
    ElementwiseRegionFinder<tosa::MatMulOp> preSoftmaxElementwiseFinder =
        attentionMatcherValues.preSoftmaxElementwiseFinder;
    SmallVector<Value> elementwiseOtherArgs =
        preSoftmaxElementwiseFinder.getElementwiseArgs();
    // causalMaskInput would be equal to kvCacheInput if there is no causal
    // mask and kvCacheInput would be same as softmaxInput if there is no
    // kv-cache. see match() for details
    Value causalMaskInput = attentionMatcherValues.causalMaskInput;
    tosa::MatMulOp firstMatMulOp =
        preSoftmaxElementwiseFinder.getFirstGemmBasedOp().value();
    Value currentSeqLen = attentionMatcherValues.currentSeqLen;
    Value prefixOffset = attentionMatcherValues.prefixOffset;
    bool isCausal = attentionMatcherValues.isCausal;
    TypeAttr softmaxTypeAttr =
        TypeAttr::get(attentionMatcherValues.softmaxType);

    // Helper to broadcast and reshape a block arg tensor to match output shape
    auto prepareBlockArgTensor = [&](Value &val) {
      if (!val)
        return;
      // Broadcast if dimension doesn't match output
      if (cast<ShapedType>(val.getType()).getShape()[0] !=
          outputType.getShape()[0]) {
        auto maybeNew =
            addBroadcastForBlockArg(rewriter, val, firstMatMulOp.getA());
        if (succeeded(maybeNew))
          val = maybeNew.value();
      }
      // Reshape {batch, numHeads} -> {batch * numHeads}
      if (cast<ShapedType>(val.getType()).getRank() == 2) {
        SmallVector<ReassociationIndices> reassocIndices = {{0, 1}};
        val = tensor::CollapseShapeOp::create(rewriter, op.getLoc(), val,
                                              reassocIndices);
      }
    };

    prepareBlockArgTensor(currentSeqLen);
    prepareBlockArgTensor(prefixOffset);

    UnitAttr causalAttr = isCausal ? rewriter.getUnitAttr() : nullptr;
    ElementwiseRegionFinder<tosa::MatMulOp> elemwiseRegion =
        attentionMatcherValues.preSoftmaxElementwiseFinder;
    int64_t firstGemmBlockIndex = elemwiseRegion.getFirstGemmBlockIndex();

    IntegerAttr numHeadsQ, numHeadsKV;
    Value queries, keys, values;
    std::tie(queries, keys, values, numHeadsQ, numHeadsKV) = getGQAValues(
        rewriter, firstMatMulOp.getA(), firstMatMulOp.getB(), op.getB());

    rock::AttentionOp attnOp = rock::AttentionOp::create(
        rewriter, loc, outputType, lseType, queries, keys, values,
        elementwiseOtherArgs, currentSeqLen, prefixOffset, output, lseOut,
        /*numHeadsQ=*/numHeadsQ,
        /*numHeadsKV=*/numHeadsKV,
        /*qTransposed=*/nullptr,
        /*kTransposed=*/nullptr,
        /*vTransposed=*/nullptr,
        /*oTransposed=*/nullptr, causalAttr,
        /*splitKV=*/rewriter.getI32IntegerAttr(1),
        /*features=*/nullptr,
        rewriter.getAttr<rock::StoreMethodAttr>(rock::StoreMethod::Set),
        softmaxTypeAttr,
        /*params0=*/nullptr, /*params1=*/nullptr,
        /*firstGemmIndices=*/
        rewriter.getDenseI64ArrayAttr(firstGemmBlockIndex));
    Block *preSoftmaxElemwiseBlock = &attnOp.getPreSoftmaxBody().emplaceBlock();
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(preSoftmaxElemwiseBlock);
      elemwiseRegion.rewrite(causalMaskInput, rewriter, preSoftmaxElemwiseBlock,
                             loc);
    }
    tosa::AddOp addOp;
    Value expandedOutLse;
    if (lse) {
      // Reverse the collapse operation
      expandedOutLse = tensor::ExpandShapeOp::create(
          rewriter, op.getLoc(), lseOrig.getType(), attnOp->getResult(1),
          reassocIndicesLSE);

      // collecting AddOp before the first replace
      addOp = lseOrig.getDefiningOp<tosa::AddOp>();

      // all users have to be moved after the expand shape
      moveUsersAfterExpandShape(rewriter, op.getLoc(),
                                expandedOutLse.getDefiningOp(), addOp);
    }
    if (auto attr = op->getAttrOfType<StringAttr>("perf_config"))
      attnOp->setAttr("perf_config", attr);

    rewriter.replaceOp(op, attnOp->getResult(0));
    if (lse) {
      rewriter.replaceOp(addOp, expandedOutLse);
    }
  }

  LogicalResult matchAndRewrite(tosa::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<AttentionMatcherValues> attentionMatcherResult = match(op);
    if (failed(attentionMatcherResult)) {
      return failure();
    }
    const AttentionMatcherValues &attentionMatcherValues =
        attentionMatcherResult.value();
    rewrite(op, attentionMatcherValues, rewriter);
    return success();
  }
};

template <typename TosaReduceOp>
typename std::enable_if_t<
    std::is_same<TosaReduceOp, tosa::ReduceSumOp>::value ||
        std::is_same<TosaReduceOp, tosa::ReduceMaxOp>::value,
    LogicalResult> static matchAndRewriteReductions(TosaReduceOp op,
                                                    rock::ReduceMethod rMethod,
                                                    Attribute outputInitVal,
                                                    ConversionPatternRewriter
                                                        &rw) {
  Location loc = op->getLoc();
  auto outputType = cast<RankedTensorType>(op.getType());
  Value output =
      bufferization::AllocTensorOp::create(rw, loc, outputType, ValueRange{});

  int32_t blockSize = 256;
  auto elementCount =
      cast<ShapedType>(op.getInput().getType()).getNumElements();
  int32_t gridSize = (elementCount + blockSize - 1) / blockSize;
  auto numCU = rock::getNumCU(op);
  if (succeeded(numCU)) {
    gridSize = std::min((int32_t)(20 * numCU.value()), gridSize);
  }

  auto rockReduce = rock::ReduceOp::create(
      rw, loc, outputType, op.getInput(), output,
      rw.getAttr<rock::ReduceMethodAttr>(rMethod),
      rw.getIndexAttr(op.getAxis()), rw.getI32IntegerAttr(blockSize),
      rw.getI32IntegerAttr(gridSize),
      /*useLDS=*/nullptr,
      /*useDPP=*/nullptr);

  func::FuncOp func = op->template getParentOfType<func::FuncOp>();
  SetVector<int64_t> resIndices = traceToRes(op.getOutput(), func);
  if (resIndices.empty())
    return op.emitOpError(
        "can't trace the reduction output to a kernel result");

  for (int64_t resNumber : resIndices) {
    func.setResultAttr(resNumber, rock::PrefillAttr::getMnemonic(),
                       outputInitVal);
    func.setResultAttr(resNumber, "read_access", rw.getUnitAttr());
    // The original function also need the read access attr for the output.
    if (func->hasAttr("original_func")) {
      if (ModuleOp rootMod =
              func->getParentOfType<ModuleOp>()->getParentOfType<ModuleOp>()) {
        SymbolTable symTable(rootMod);
        SymbolRefAttr originalFuncAttr =
            func->getAttrOfType<SymbolRefAttr>("original_func");
        if (func::FuncOp originalFunc = dyn_cast<func::FuncOp>(
                symTable.lookupSymbolIn(rootMod, originalFuncAttr))) {
          originalFunc.setResultAttr(resNumber, "read_access",
                                     rw.getUnitAttr());
        }
      }
    }
  }
  rw.replaceOp(op, rockReduce.getResult());
  return success();
}

class ReduceSumConverter final : public OpConversionPattern<tosa::ReduceSumOp> {
public:
  using OpConversionPattern<tosa::ReduceSumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceSumOp op,
                                tosa::ReduceSumOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Type elementType =
        cast<ShapedType>(op.getInput().getType()).getElementType();
    if (!isa<Float32Type, Float16Type, BFloat16Type>(elementType)) {
      return rw.notifyMatchFailure(
          op, "We only support F32, F16 and BF16 reductions, yet.");
    }
    Attribute outputInitVal = rw.getFloatAttr(elementType, 0.0000);
    return matchAndRewriteReductions(op, rock::ReduceMethod::Sum, outputInitVal,
                                     rw);
  }
};

class ReduceMaxConverter final : public OpConversionPattern<tosa::ReduceMaxOp> {
public:
  using OpConversionPattern<tosa::ReduceMaxOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(tosa::ReduceMaxOp op,
                                tosa::ReduceMaxOp::Adaptor adaptor,
                                ConversionPatternRewriter &rw) const final {
    Type elementType =
        cast<ShapedType>(op.getInput().getType()).getElementType();
    Attribute outputInitVal;
    if (elementType.isF32()) {
      outputInitVal = rw.getFloatAttr(
          elementType, APFloat::getInf(APFloat::IEEEsingle(), true));
    } else {
      return rw.notifyMatchFailure(op, "We only support F32 reductions, yet.");
    }
    return matchAndRewriteReductions(op, rock::ReduceMethod::Max, outputInitVal,
                                     rw);
  }
};

// We identify the dummy pattern tosa.mul with implicit broadcasting
// and rewrite it to be rock.transform broadcast
class MulSplatOneRewritePattern final : public OpRewritePattern<tosa::MulOp> {
public:
  using OpRewritePattern<tosa::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(tosa::MulOp op,
                                PatternRewriter &rw) const final {
    Location loc = op.getLoc();
    TypedValue<TensorType> inp1 = op.getInput1();
    TypedValue<TensorType> inp2 = op.getInput2();
    TypedValue<TensorType> out = op.getOutput();

    TypedValue<TensorType> bcastInput;
    if (mlir::rock::isConstantOne(inp1))
      bcastInput = inp2;
    if (mlir::rock::isConstantOne(inp2)) {
      if (bcastInput) {
        return rw.notifyMatchFailure(op, "both inputs are splat ones");
      }
      bcastInput = inp1;
    }
    if (bcastInput) {
      Value bcast =
          insertBroadcast(bcastInput, out.getType().getShape(), loc, rw);
      rw.replaceOp(op, bcast);
      return success();
    }
    return rw.notifyMatchFailure(op, "none of the inputs are splat ones");
  }
};

} // namespace

void tosa::populateTosaToRockConversionPatterns(MLIRContext *context,
                                                RewritePatternSet &patterns) {
  patterns.add<ForwardConvConverter<tosa::Conv2DOp>,
               ForwardConvConverter<tosa::Conv3DOp>, BackwardConvConverter,
               MatMulConverter, ReduceSumConverter, ReduceMaxConverter>(
      context);
}

void tosa::populateTosaToRockAttentionConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<AttentionRewritePattern>(context);
}

void tosa::populateTosaToRockGemmGemmConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<GemmElementwiseGemmRewritePattern>(context);
}

void tosa::populateTosaToRockConvGemmConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<ConvElementwiseGemmRewritePattern>(context);
}

void tosa::populateTosaToRockTensorConversionPatterns(
    MLIRContext *context, RewritePatternSet &patterns) {
  patterns.add<TransposeRewritePattern, CollapseExpandRewritePattern,
               MulSplatOneRewritePattern>(context);
}
