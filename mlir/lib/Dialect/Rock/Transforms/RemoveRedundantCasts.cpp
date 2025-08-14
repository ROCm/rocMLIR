//===------- RemoveRedundantCasts - MLIR Rock ops lowering passes ---------===//
//
// Copyright 2025 The MLIR Authors.
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
// This pass identifies and removes redundant cast chains in the
// IR, specifically patterns where a value is converted from f32 to a smaller
// type (e.g., f16) and then immediately extended back to f32. By eliminating
// these unnecessary conversions, the pass simplifies the IR and can improve
// performance by reducing superfluous operations and memory traffic.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {

// Pattern to remove redundant dtype -> newDtype -> dtype cast chains
struct RemoveRedundantTruncExtfPattern
    : public OpRewritePattern<linalg::GenericOp> {
  func::FuncOp func;

  RemoveRedundantTruncExtfPattern(MLIRContext *context, func::FuncOp func)
      : OpRewritePattern<linalg::GenericOp>(context), func(func) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs()
               << "Looking at generic: " << generic.getLoc() << "\n");

    // Check if the generic operation only contains an ext operation.
    if (!(isGenericWithSingleOp<arith::ExtFOp>(generic) ||
          isGenericWithSingleOp<arith::ExtSIOp>(generic) ||
          isGenericWithSingleOp<arith::ExtUIOp>(generic))) {
      LLVM_DEBUG(llvm::dbgs() << "Generic doesn't contain only extf\n");
      return failure();
    }

    bool changed = false;

    auto extInput = getExtInput(generic);
    if (!extInput)
      return failure();
    auto input = extInput.getDefiningOp();

    if (isa<RockGemmWrapperInterface>(input) ||
        isa<RockGemmGemmWrapperInterface>(input)) {
      // Now we need to check the input of the cast (linalg.generic with ext)
      // to see if it is a RockOp that makes use of MFMA. MFMA in rocmlir does
      // accumulation in higher precision, so if our rock op is using MFMA and
      // the return type isn't F32 or I32 (highest level of precision), then it
      // means that there will be a trunc op inserted in RockGemmToGridwise
      // that will potentially be redundant.
      changed |= handleRockGemmWrapper(input, generic, rewriter);
    } else if (auto inputGeneric = dyn_cast<linalg::GenericOp>(input)) {
      // We also want to handle an arbitrary trunc op followed by ext ops
      if (isGenericWithSingleOp<arith::TruncFOp>(inputGeneric) ||
          isGenericWithSingleOp<arith::TruncIOp>(inputGeneric)) {
        changed |= handleLinalgGeneric(inputGeneric, generic, rewriter);
      }
    } else {
      // We have come across a pattern that we do not know how to handle, or
      // does not require changing.
      return failure();
    }

    return changed ? success() : failure();
  }

private:
  // This is a helper function that checks if the generic operation contains
  // only a single operation of type OpType, and that the yield returns that
  // value.
  // NOTE: A convert in migraphx, will eventually be lowered to a linalg
  // with either a single trunc or ext op. e.g.,:
  // %1 = migraphx.convert %0 : <1x5x3xf16, 15x3x1> to <1x5x3xf32, 15x3x1>
  template <typename OpType>
  bool isGenericWithSingleOp(linalg::GenericOp generic) const {
    Block &body = generic.getRegion().front();

    // Check that the body has exactly 2 operations: OpType and yield
    if (std::distance(body.begin(), body.end()) != 2) {
      return false;
    }

    // First operation should be an OpType
    auto firstOp = body.begin();
    if (!isa<OpType>(firstOp)) {
      return false;
    }
    auto requiredOp = cast<OpType>(firstOp);

    // Second operation should be yield
    auto secondOp = std::next(body.begin());
    if (!isa<linalg::YieldOp>(secondOp)) {
      return false;
    }
    auto yieldOp = cast<linalg::YieldOp>(secondOp);

    // Verify that the yield operation yields the result of the required op
    return yieldOp.getValues().size() == 1 &&
           yieldOp.getValues()[0] == requiredOp.getResult();
  }

  Value getExtInput(linalg::GenericOp generic) const {
    // Check that there is only one input to the generic operation
    if (generic.getInputs().size() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tGeneric does not have exactly one input, skipping\n");
      return nullptr;
    }

    // We don't need to investigate BlockArguments any further
    Value input = generic.getInputs()[0];
    if (isa<BlockArgument>(input))
      return nullptr;

    // If the input is a rock.transform op, we need to untransform it to get
    // to the original input value.
    if (auto rockTransformOp =
            dyn_cast<rock::TransformOp>(input.getDefiningOp())) {
      // We need to traverse the rock.transform ops to find the original input
      SmallVector<rock::TransformOp> inputTransforms;
      return std::get<0>(untransform(input, inputTransforms));
    }

    return input;
  }

  // Helper function to create a new output value for a
  // RockGemmWrapperInterface/RockGemmGemmwrapperInterface Op or LinalgGeneric
  // TruncOp, and any corresponding rock.transformOps
  Value createNewOutput(Value prevValue,
                        SmallVector<rock::TransformOp> outputTransforms,
                        OpBuilder &builder) const {
    for (auto &transform : outputTransforms) {
      auto newTransformOp = builder.create<rock::TransformOp>(
          transform.getLoc(), prevValue, transform.getTransform());
      prevValue = newTransformOp.getResult();
    }

    return prevValue;
  }

  // Helper function to get all non-transform uses from a base value. This gives
  // us the true number of uses of the base value without all of the
  // rock.TransformOps. The boolean value returned tells us if there is a single
  // non-TransformOp user that is an ExtFOp
  std::tuple<SmallVector<Operation *>, bool>
  getAllNonTransformedUses(Operation *baseOp) const {
    SmallVector<Operation *> finalValues;
    bool allExtFOps = true;
    int extFCount = 0;
    // Start with just the base value
    SmallVector<Operation *> worklist = {baseOp};
    while (!worklist.empty()) {
      Operation *currentOp = worklist.pop_back_val();

      for (Operation *user : currentOp->getUsers()) {
        if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
          // This value is used by a transform op, add the result to worklist
          worklist.push_back(transformOp.getOperation());
        } else if (isa<linalg::GenericOp>(user) &&
                   (isGenericWithSingleOp<arith::ExtFOp>(
                        cast<linalg::GenericOp>(user)) ||
                    isGenericWithSingleOp<arith::ExtSIOp>(
                        cast<linalg::GenericOp>(user)) ||
                    isGenericWithSingleOp<arith::ExtUIOp>(
                        cast<linalg::GenericOp>(user)))) {
          finalValues.push_back(user);
          extFCount++;
        } else {
          finalValues.push_back(user);
          allExtFOps = false;
        }
      }
    }

    return std::make_tuple(finalValues, (extFCount == 1 && allExtFOps));
  }

  bool handleLinalgGeneric(linalg::GenericOp truncGen, linalg::GenericOp extGen,
                           PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "\tFound truncf Linalg GenericOp: " << *truncGen << "\n");

    // Check to make sure that we are casting to and from the same type
    Value truncGenResult = truncGen.getResult(0);
    Value truncGenInput = truncGen.getInputs()[0];
    Value extGenResult = extGen.getResult(0);
    Value extGenInput = extGen.getInputs()[0];

    auto truncOutputType = dyn_cast<RankedTensorType>(truncGenResult.getType());
    auto truncInputType = dyn_cast<RankedTensorType>(truncGenInput.getType());
    auto extOutputType = dyn_cast<RankedTensorType>(extGenResult.getType());
    auto extInputType = dyn_cast<RankedTensorType>(extGenInput.getType());

    if (!truncOutputType || !extOutputType || !truncInputType ||
        !extInputType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tOne or more types are not RankedTensorType, skipping\n");
      return false;
    }

    if (truncInputType.getElementType() != extOutputType.getElementType() ||
        truncOutputType.getElementType() != extInputType.getElementType()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "\tTrunc input type doesn't match ext output type, or trunc "
          << "output type doesn't match ext input type, skipping\n");
      return false;
    }

    // Check to see if the extGen is a direct use of the truncGen
    SmallVector<rock::TransformOp> transforms;
    auto trueInput = std::get<0>(untransform(extGenInput, transforms));
    if (trueInput != truncGenResult) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\textGen input is not the truncGen result, skipping\n");
      return false;
    }

    // Replace and remove the ext op
    OpBuilder builder(truncGen->getContext());
    auto tup = getAllNonTransformedUses(truncGen);
    builder.setInsertionPoint(truncGen);
    Value newOutput = createNewOutput(truncGenInput, transforms, builder);
    rewriter.replaceAllUsesWith(extGen.getResult(0), newOutput);
    rewriter.eraseOp(extGen);

    // If there is only a single ExtOp use, then we can go ahead and remove
    // all of the TransformOps and the original truncf
    auto singleExtOp = std::get<1>(tup);
    if (singleExtOp) {
      for (auto &t : transforms) {
        rewriter.eraseOp(t);
      }
    }

    return false;
  }

  linalg::GenericOp createTruncGeneric(Operation *input, Operation *output,
                                       PatternRewriter &rewriter) const {
    // Get the input and output types
    auto inputType = cast<RankedTensorType>(input->getResult(0).getType());
    auto outputType = cast<RankedTensorType>(output->getResult(0).getType());

    // Verify that the output element type is smaller than the input
    if (isa<IntegerType>(inputType.getElementType())) {
      assert(cast<IntegerType>(outputType.getElementType()).getWidth() < 32 &&
             "Output element type must be smaller than i32");
    } else {
      assert(cast<FloatType>(outputType.getElementType()).getWidth() < 32 &&
             "Output element type must be smaller than f32");
    }

    // Create indexing maps - both input and output use the same identity map
    MLIRContext *ctx = rewriter.getContext();
    int64_t rank = inputType.getRank();
    SmallVector<AffineExpr> exprs;
    for (int64_t i = 0; i < rank; ++i) {
      exprs.push_back(getAffineDimExpr(i, ctx));
    }
    auto identityMap = AffineMap::get(rank, 0, exprs, ctx);
    SmallVector<AffineMap> indexingMaps = {identityMap, identityMap};

    // Create iterator types - all parallel
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Create the linalg.generic operation
    auto loc = input->getLoc();
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTypes=*/TypeRange{outputType},
        /*inputs=*/ValueRange{input->getResult(0)},
        /*outputs=*/ValueRange{output->getResult(0)},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/"",
        /*library_call=*/"",
        /*bodyBuild=*/
        [](OpBuilder &builder, Location loc, ValueRange args) {
          Value blockArg = args[0];
          Value outputArg = args[1];
          Type oType = outputArg.getType();
          Value truncResult =
              builder.create<arith::TruncFOp>(loc, oType, blockArg);
          builder.create<linalg::YieldOp>(loc, truncResult);
        });

    return genericOp;
  }

  bool handleRockGemmWrapper(Operation *rockOp, linalg::GenericOp extGen,
                             PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs()
               << "\tFound "
               << "RockGemmWrapperInterface/RockGemmGemmWrapperInterface: "
               << *rockOp << "\n");

    Value extGenResult = extGen.getResult(0);
    Value extGenInput = extGen.getInputs()[0];

    auto extOutputType = dyn_cast<RankedTensorType>(extGenResult.getType());
    auto extInputType = dyn_cast<RankedTensorType>(extGenInput.getType());

    if (!extOutputType || !extInputType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tOne or more types are not RankedTensorType, skipping\n");
      return false;
    }

    Type rockOutputElementType;
    Value rockOutputOp;
    if (auto rockI = dyn_cast<RockGemmWrapperInterface>(rockOp)) {
      rockOutputElementType = rockI.getCType();
      rockOutputOp = rockI.getOutArgument()->get();
    } else if (auto rockGemmI =
                   dyn_cast<RockGemmGemmWrapperInterface>(rockOp)) {
      rockOutputElementType = rockGemmI.getCType();
      rockOutputOp = rockGemmI.getOutArgument()->get();
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tNot a RockGemmWrapperInterface, skipping\n");
      return false;
    }

    auto rockOutputType = dyn_cast<RankedTensorType>(rockOutputOp.getType());
    if (!rockOutputType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tOutput type is not RankedTensorType, skipping\n");
      return false;
    }

    if (rockOutputType.getElementType() != extInputType.getElementType()) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "\tTrunc input type doesn't match ext output type, or trunc "
          << "output type doesn't match ext input type, skipping\n");
      return false;
    }

    OpBuilder builder(rockOp->getContext());
    builder.setInsertionPoint(rockOp);

    // If this op uses mfma, it will accumulate in higher precision (F32 or I32)
    auto features = rock::getFeatures(rockOp);
    bool isMfma = bitEnumContainsAll(features, GemmFeatures::mfma);
    if (!isMfma ||
        !((cast<FloatType>(rockOutputElementType).getWidth() < 32) ||
          (cast<IntegerType>(rockOutputElementType).getWidth() < 32))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tNot an op that uses mfma with an output "
                    "that has a type smaller than F32/I32, skipping\n");
      return false;
    }

    // Now we can create a clone of the original Interface op with
    // the new output type/value
    SmallVector<rock::TransformOp> resultTransforms;
    Value untransformedOutput =
        std::get<0>(untransform(rockOutputOp, resultTransforms));
    assert(isa<bufferization::AllocTensorOp>(
               untransformedOutput.getDefiningOp()) &&
           "Expected output to be a bufferization.alloc_tensor op");

    Type newRockOutputType;
    if (isa<IntegerType>(rockOutputElementType)) {
      newRockOutputType =
          rockOutputType.cloneWith(std::nullopt, builder.getI32Type());
    } else {
      newRockOutputType =
          rockOutputType.cloneWith(std::nullopt, builder.getF32Type());
    }

    auto newAllocTensorOp = builder.create<bufferization::AllocTensorOp>(
        untransformedOutput.getLoc(), cast<RankedTensorType>(newRockOutputType),
        ValueRange{});
    Value newOutput =
        createNewOutput(newAllocTensorOp, resultTransforms, builder);
    Operation *clonedOp = builder.clone(*rockOp);
    int outArgIndex = -1;
    if (auto rockI = dyn_cast<RockGemmWrapperInterface>(clonedOp)) {
      outArgIndex = rockI.getOutArgument()->getOperandNumber();
    } else if (auto rockGemmI =
                   dyn_cast<RockGemmGemmWrapperInterface>(clonedOp)) {
      outArgIndex = rockGemmI.getOutArgument()->getOperandNumber();
    }
    assert(outArgIndex != -1 && "outArgIndex was not initialized");
    clonedOp->setOperand(outArgIndex, newOutput);
    Type newResultType;
    if (isa<IntegerType>(rockOutputElementType)) {
      newResultType =
          rockOutputType.cloneWith(std::nullopt, builder.getI32Type());
    } else {
      newResultType =
          rockOutputType.cloneWith(std::nullopt, builder.getF32Type());
    }
    clonedOp->getResult(0).setType(newResultType);

    // If the Interface op had rock.transforms between it and
    // the extf generic use, then we need to create those transforms for our
    // new Rock op
    SmallVector<rock::TransformOp> extInputTransforms;
    untransform(extGenInput, extInputTransforms);
    Value prevValue = clonedOp->getResult(0);
    for (auto &transform : extInputTransforms) {
      auto newTransformOp = builder.create<rock::TransformOp>(
          transform.getLoc(), prevValue, transform.getTransform());
      prevValue = newTransformOp.getResult();
    }

    // Replace and remove the extf op
    rewriter.setInsertionPoint(rockOp);
    auto tup = getAllNonTransformedUses(rockOp);
    rewriter.replaceAllUsesWith(extGen.getResult(0), prevValue);
    rewriter.eraseOp(extGen);

    // If there is only a single ExtFOp use, then we can go ahead and remove
    // all of the TransformOps and the original rockOp
    auto singleExtFOp = std::get<1>(tup);
    if (singleExtFOp) {
      for (auto &transform : extInputTransforms) {
        rewriter.eraseOp(transform);
      }
    } else {
      // Otherwise, we need to create a truncF from our new rock op, and update
      // all of the users of the old rockOp
      auto newTruncF =
          createTruncGeneric(clonedOp, rockOutputOp.getDefiningOp(), rewriter);
      rewriter.replaceAllUsesWith(rockOp->getResult(0), newTruncF.getResult(0));
    }

    // Now we can safely clean up the original rock op
    rewriter.eraseOp(rockOp);

    return true;
  }
};

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Running RemoveRedundantCasts\n");

    func::FuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRedundantTruncExtfPattern>(&getContext(), func);

    if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Finished RemoveRedundantCasts\n");
  }
};

} // end anonymous namespace
