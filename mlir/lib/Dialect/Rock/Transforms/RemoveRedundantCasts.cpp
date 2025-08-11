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
    if (!isGenericWithOnlyExtf(generic)) {
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
      LLVM_DEBUG(llvm::dbgs() << "\tFound Linalg GenericOp: " << *inputGeneric
                 << "\n");
      // Update the uses of the extf to use the value before the upcast

      // If there are no other uses, then we are safe to clean up and remove
      // the input (linalg.generic op that just does a truncation).

      // If this input has other uses, then we can keep the input around and 
      // there is no further modification that needs to be done.
    } else {
      // We have come across a pattern that we do not know how to handle, or
      // does not require changing.
      return failure();
    }

    // At this point, the upcast (linalg.generic with the extf op) should have
    // no more uses and we are safe to clean it up.

    return changed ? success() : failure();
  }

private:
  bool isGenericWithOnlyExtf(linalg::GenericOp generic) const {
    Block &body = generic.getRegion().front();
    
    // Check that the body has exactly 2 operations: extf and yield
    if (std::distance(body.begin(), body.end())  != 2) {
      return false;
    }
    
    // First operation should be extf
    auto firstOp = body.begin();
    if (!isa<arith::ExtFOp>(firstOp)) {
      return false;
    }
    auto extfOp = cast<arith::ExtFOp>(firstOp);
    
    // Second operation should be yield
    auto secondOp = std::next(body.begin());
    if (!isa<linalg::YieldOp>(secondOp)) {
      return false;
    }
    auto yieldOp = cast<linalg::YieldOp>(secondOp);

    // Ensure that the extf operation cast up to f32
    if (!extendsToF32(extfOp)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tNot an extf to f32 from a smaller type, skipping\n");
      return false;
    }
    
    // Verify that the yield operation yields the result of the extf
    return yieldOp.getValues().size() == 1 && 
           yieldOp.getValues()[0] == extfOp.getResult();
  }

  // Helper function that checks if the ExtfOp is converting to f32 from a
  // smaller type (e.g., f16).
  bool extendsToF32(arith::ExtFOp extfOp) const {
    Type outputType = extfOp.getOut().getType();
    Type elementType = outputType;

    if (auto vectorType = dyn_cast<VectorType>(outputType)) {
      elementType = vectorType.getElementType();
    }

    return elementType.isF32();
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
  // RockGemmWrapperInterface Op, and any corresponding rock.transformOps
  Value createNewF32Output(Value originalOutput, 
                           RankedTensorType originalType,
                           OpBuilder &builder) const {
    // Get the value that is used as output for the RockGemmWrapperInterface op
    SmallVector<rock::TransformOp> outputTransforms;
    Value untransformedOutput = std::get<0>(untransform(originalOutput, outputTransforms));

    // Assert that this is a bufferization.alloc_tensor op, and then create
    // a clone of this that uses the newOutputType
    assert(isa<bufferization::AllocTensorOp>(untransformedOutput.getDefiningOp()) &&
            "Expected output to be a bufferization.alloc_tensor op");
    
    Type newOutputType = originalType.cloneWith(std::nullopt, builder.getF32Type());
    auto newAllocTensorOp = builder.create<bufferization::AllocTensorOp>(
        untransformedOutput.getLoc(),
        cast<RankedTensorType>(newOutputType),
        ValueRange{}
    );
  
    // Create TransformOps for this new AllocTensorOp if needed
    Value prevValue = newAllocTensorOp;
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

    // If this op uses mfma, it will accumulate in higher precision
    auto features = rock::getFeatures(rockI);
    bool isMfma = bitEnumContainsAll(features, GemmFeatures::mfma);
    if (!(isMfma && outputElementType.isF16())) {
      LLVM_DEBUG(llvm::dbgs() << "\tNot an op that uses mfma with f16 output, skipping\n");
      return false;
    }

    // Now we can create a clone of the original RockGemmWrapperInterface with
    // the new output type/value
    Value newF32Output = createNewF32Output(outputOp, outputType, builder);
    Operation *clonedOp = builder.clone(*rockOp);
    auto clonedRockI = cast<RockGemmWrapperInterface>(clonedOp);
    unsigned outArgIndex = clonedRockI.getOutArgument()->getOperandNumber();
    clonedOp->setOperand(outArgIndex, newF32Output);
    Type newResultType = outputType.cloneWith(std::nullopt, builder.getF32Type());
    clonedOp->getResult(0).setType(newResultType);

    // If the RockGemmWrapperInterface op had rock.transforms between it and
    // the extf generic use, then we need to create those transforms for our
    // new Rock op
    SmallVector<rock::TransformOp> resultTransforms;
    Value extfGenInput = extfGen.getInputs()[0];
    untransform(extfGenInput, resultTransforms);
    Value prevValue = clonedOp->getResult(0);
    for (auto &transform : resultTransforms) {
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
    for (auto &transform : resultTransforms) {
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