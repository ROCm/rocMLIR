//===- RemoveRedundantCasts - MLIR Rock ops lowering passes -----===//
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
// ============================================================
//
// This pass identifies and removes redundant floating-point cast chains in the
// IR, specifically patterns where a value is converted from f32 to a smaller
// type (e.g., f16) and then immediately extended back to f32. By eliminating
// these unnecessary conversions, the pass simplifies the IR and can improve
// performance by reducing superfluous operations and memory traffic.
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/IR/AmdArchDb.h"
#include "mlir/Dialect/Rock/IR/GetRockInfo.h"
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

// Pattern to remove redundant f32 -> dtype -> f32 cast chains
struct RemoveRedundantTruncExtfPattern
    : public OpRewritePattern<linalg::GenericOp> {
  func::FuncOp func;

  RemoveRedundantTruncExtfPattern(MLIRContext *context, func::FuncOp func)
      : OpRewritePattern<linalg::GenericOp>(context), func(func) {}

  LogicalResult matchAndRewrite(linalg::GenericOp generic,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Looking at generic: " << generic.getLoc() << "\n");

    // Check if the generic operation only contains an extf operation.
    if (!isGenericWithSingleOp<arith::ExtFOp>(generic)) {
      return rewriter.notifyMatchFailure(generic,
                                         "Generic doesn't contain only extf");
    }

    bool changed = false;

    // Now we need to check the input of the cast (linalg.generic with extf) to
    // see if it is a RockOp that makes use of MFMA. MFMA in rocmlir does
    // accumulation in higher precision, so if our rock op is using MFMA and
    // the return type is F16, then it means that there will be a truncf op
    // inserted in RockGemmToGridwise that will potentially be redundant.
    auto input = getExtfInput(generic).getDefiningOp();
    if (auto rockI = dyn_cast<RockGemmWrapperInterface>(input)) {
      changed |= handleRockGemmWrapper(input, generic, rewriter);
    } else if (auto inputGeneric = dyn_cast<linalg::GenericOp>(input)) {
      if (isGenericWithSingleOp<arith::TruncFOp>(inputGeneric))
        changed |= handleLinalgGeneric(inputGeneric, generic, rewriter);
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
  template <typename OpType>
  bool isGenericWithSingleOp(linalg::GenericOp generic) const {
    Block &body = generic.getRegion().front();
    
    // Check that the body has exactly 2 operations: OpType and yield
    if (std::distance(body.begin(), body.end())  != 2) {
      return false;
    }
    
    // First operation should be an OpType
    auto firstOp = body.begin();
    if (!isa<OpType>(firstOp)) {
      return false;
    }
    auto requiredOp = cast<OpType>(firstOp);

    // For the time being we only want to check ExtFOps which extend to F32,
    // and TruncFOps which truncate from F32
    if (isa<arith::ExtFOp>(requiredOp)) {
      Type outputType = requiredOp.getOut().getType();
      Type elementType = outputType;
      if (auto vectorType = dyn_cast<VectorType>(outputType)) {
        elementType = vectorType.getElementType();
      }

      if (!elementType.isF32()) {
        LLVM_DEBUG(llvm::dbgs() << "\tExtf does not extend to f32, skipping\n");
        return false;
      }
    } else if (isa<arith::TruncFOp>(requiredOp)) {
      // We only want to handle truncf ops that truncate from f32
      if (!requiredOp.getIn().getType().isF32()) {
        LLVM_DEBUG(llvm::dbgs() << "\tTruncf does not truncate from f32, skipping\n");
        return false;
      }
    }
    
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

  Value getExtfInput(linalg::GenericOp generic) const {
    // Check that there is only one input to the generic operation
    if (generic.getInputs().size() != 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tGeneric does not have exactly one input, skipping\n");
      return nullptr;
    }

    // If the input is a rock.transform op, we need to untransform it to get
    // to the original input value. 
    Value input = generic.getInputs()[0];
    if (auto rockTransformOp = dyn_cast<rock::TransformOp>(input.getDefiningOp())) {
      // We need to traverse the rock.transform ops to find the original input
      SmallVector<rock::TransformOp> inputTransforms;
      return std::get<0>(untransform(input, inputTransforms));
    }

    return input;
  }

  // Helper function to create a new f32 output value for a
  // RockGemmWrapperInterface Op or LinalgGeneric TruncFOp, and any
  // corresponding rock.transformOps
  Value createNewF32Output(Value prevValue, 
                           SmallVector<rock::TransformOp> outputTransforms,
                           OpBuilder &builder) const {
    for (auto &transform : outputTransforms) {
      auto newTransformOp = builder.create<rock::TransformOp>(
          transform.getLoc(),
          prevValue,
          transform.getTransform()
      );
      prevValue = newTransformOp.getResult();
    }
    
    return prevValue;
  }

  // Helper function to get all transformed values from a base value. This gives
  // us the true number of uses of the base value without all of the
  // rock.TransformOps
  SmallVector<Value> getAllTransformedValues(Value baseValue) const {
    SmallVector<Value> finalValues;

    // Start with just the base value
    SmallVector<Value> worklist = {baseValue};
    while (!worklist.empty()) {
      Value currentValue = worklist.pop_back_val();
      
      bool hasTransformUsers = false;
      for (Operation *user : currentValue.getUsers()) {
        if (auto transformOp = dyn_cast<rock::TransformOp>(user)) {
          // This value is used by a transform op, add the result to worklist
          worklist.push_back(transformOp.getResult());
          hasTransformUsers = true;
        }
      }
      
      // If this value has no transform users, it's a final value
      if (!hasTransformUsers) {
        finalValues.push_back(currentValue);
      }
    }
    
    return finalValues;
  }

  bool handleLinalgGeneric(linalg::GenericOp truncfGen,
                           linalg::GenericOp extfGen,
                           PatternRewriter &rewriter) const {
    LLVM_DEBUG(llvm::dbgs() << "\tFound truncf Linalg GenericOp: " << *truncfGen
                            << "\n");
    Value truncfGenResult = truncfGen.getResult(0);
    Value truncfGenInput = truncfGen.getInputs()[0];
    auto outputType = dyn_cast<RankedTensorType>(truncfGenResult.getType());
    if (!outputType) {
      LLVM_DEBUG(llvm::dbgs() << "\tOutput type is not RankedTensorType, skipping\n");
      return false;
    }

    // Check to see if the extfGen is a direct use of the truncfGen
    SmallVector<rock::TransformOp> transforms;
    Value extfGenInput = extfGen.getInputs()[0];
    auto trueInput = std::get<0>(untransform(extfGenInput, transforms));
    if (trueInput != truncfGenResult) {
      LLVM_DEBUG(llvm::dbgs() << "\tExtfGen input is not the truncfGen result, skipping\n");
      return false;
    }

    // Create any Rock.TransFormOps if necessary and replace all users of the
    // ExtfGen op with newF32Output
    OpBuilder builder(truncfGen->getContext());
    builder.setInsertionPoint(truncfGen);
    Value newF32Output = createNewF32Output(truncfGenInput, transforms,
                                            builder);
    rewriter.replaceAllUsesWith(extfGen.getResult(0), newF32Output);
    rewriter.eraseOp(extfGen);

    // TODO: Handle the general case where there are an arbitrary number of
    // ops that make use of the initial truncated result. This will be part of
    // the decision on whether or not we can remove the truncf op entirely.
    return false;
  }

  bool handleRockGemmWrapper(Operation *rockOp,
                             linalg::GenericOp extfGen,
                             PatternRewriter &rewriter) const {
    auto rockI = cast<RockGemmWrapperInterface>(rockOp);
    LLVM_DEBUG(llvm::dbgs() << "\tFound RockGemmWrapperInterface: " << *rockI
                            << "\n");
    auto outputElementType = rockI.getCType();
    auto outputOp = rockI.getOutArgument()->get();
    auto outputType = dyn_cast<RankedTensorType>(outputOp.getType());
    if (!outputType) {
      LLVM_DEBUG(llvm::dbgs() << "\tOutput type is not RankedTensorType, skipping\n");
      return false;
    }

    OpBuilder builder(rockOp->getContext());
    builder.setInsertionPoint(rockOp);

    // If this op uses mfma, it will accumulate in higher precision (F32)
    auto features = rock::getFeatures(rockI);
    bool isMfma = bitEnumContainsAll(features, GemmFeatures::mfma);
    if (!isMfma || !(cast<FloatType>(outputElementType).getWidth() < 32)) {
      LLVM_DEBUG(llvm::dbgs() << "\tNot an op that uses mfma with an output "
                                 "that has a type smaller than F32, skipping\n");
      return false;
    }

    // Now we can create a clone of the original RockGemmWrapperInterface with
    // the new output type/value
    SmallVector<rock::TransformOp> resultTransforms;
    Value untransformedOutput = std::get<0>(untransform(outputOp,
                                                        resultTransforms));
    assert(isa<bufferization::AllocTensorOp>(untransformedOutput.getDefiningOp()) &&
            "Expected output to be a bufferization.alloc_tensor op");
    
    Type newOutputType = outputType.cloneWith(std::nullopt, builder.getF32Type());
    auto newAllocTensorOp = builder.create<bufferization::AllocTensorOp>(
        untransformedOutput.getLoc(),
        cast<RankedTensorType>(newOutputType),
        ValueRange{}
    );
    Value newF32Output = createNewF32Output(newAllocTensorOp, resultTransforms,
                                            builder);
    Operation *clonedOp = builder.clone(*rockOp);
    auto clonedRockI = cast<RockGemmWrapperInterface>(clonedOp);
    unsigned outArgIndex = clonedRockI.getOutArgument()->getOperandNumber();
    clonedOp->setOperand(outArgIndex, newF32Output);
    Type newResultType = outputType.cloneWith(std::nullopt, builder.getF32Type());
    clonedOp->getResult(0).setType(newResultType);

    // If the RockGemmWrapperInterface op had rock.transforms between it and
    // the extf generic use, then we need to create those transforms for our
    // new Rock op
    SmallVector<rock::TransformOp> extfInputTransforms;
    Value extfGenInput = extfGen.getInputs()[0];
    untransform(extfGenInput, extfInputTransforms);
    Value prevValue = clonedOp->getResult(0);
    for (auto &transform : extfInputTransforms) {
      auto newTransformOp = builder.create<rock::TransformOp>(
          transform.getLoc(),
          prevValue,
          transform.getTransform()
      );
      prevValue = newTransformOp.getResult();
    }

    // Replace and remove the original extf generic operation and the original
    // RockGemmWrapperInterface operation
    rewriter.replaceAllUsesWith(extfGen.getResult(0), prevValue);
    rewriter.eraseOp(extfGen);
    for (auto &transform : extfInputTransforms) {
      rewriter.eraseOp(transform);
    }
    rewriter.eraseOp(rockOp);

    // TODO: I need to handle the general case where there can be an arbitrary
    // number of uses of the RockGemmWrapperInterface op, and it's not just
    // directly used by the extf generic

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