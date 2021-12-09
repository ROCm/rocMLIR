//===- TosaPartition.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace conv2d followed by elementwise with call to function containing them.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <deque>
#include <iostream>

using llvm::SmallVector;

using namespace mlir;

namespace {

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// Elementwise trait is strict about matching dimensions, but the
// AbstractElementwise trait is specific to applicable Tosa ops.  In
// practice, broadcastable and same-type Tosa ops are also element-wise.
bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::tosa::AbstractElementwise>() ||
         op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
         op->hasTrait<OpTrait::SameOperandsAndResultType>();
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp.
// Given a convolution op and its fuse-able trailing (second) and leading
// (front) ops, remove them into a separate function.
void outlineConvPartOps(Operation *convOp, SetVector<Operation *> &secondOps,
                        ArrayRef<Operation *> frontOps, ArrayRef<Value> params,
                        ArrayRef<Value> returnVals, StringRef partFnName) {
  OpBuilder b(convOp);
  Location loc = convOp->getLoc();
  MLIRContext *ctx = convOp->getContext();

  // Insert outlined function before current function.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(convOp->getParentOfType<FuncOp>());

  // Make FuncOp from convOp's operand types and secondOp's result type.
  ValueRange values(params);
  ValueRange results(returnVals);
  FunctionType type =
      FunctionType::get(ctx, values.getTypes(), results.getTypes());
  SmallVector<NamedAttribute, 1> kernelAttrs{
      b.getNamedAttr("kernel", b.getUnitAttr()),
  };
  auto outlinedFunc = b.create<FuncOp>(loc, partFnName, type,
                                       ArrayRef<NamedAttribute>(kernelAttrs));
  outlinedFunc->setAttr("sym_visibility", StringAttr::get(ctx, "private"));

  // Clone convOp and secondOps into the body of the new function.
  b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(values, outlinedFunc.getArguments()))
    bvm.map(std::get<0>(it), std::get<1>(it));

  for (auto *op : llvm::reverse(frontOps)) {
    b.clone(*op, bvm);
  }

  b.clone(*convOp, bvm);

  // Since secondOps is a SetVector, it's in the order that the ops were
  // seen in the main function, which is safe to iterate through.
  Operation *lastOp = convOp;
  for (auto *op : secondOps) {
    assert(llvm::all_of(op->getOperands(),
                        [&](Value v) { return bvm.lookupOrNull(v); }));
    b.clone(*op, bvm);
    lastOp = op;
  }

  // Make ReturnOp from secondOps' results.
  SmallVector<Value> returnOperands;
  for (auto op : returnVals) {
    returnOperands.push_back(bvm.lookup(op));
  }
  // Can't also supply return types, because it'll see a mismatch
  // in numbers where there isn't one.
  b.create<ReturnOp>(loc, returnOperands);

  // Replace convOp and secondOps with CallOp to new function.
  b.setInsertionPointAfter(lastOp);
  CallOp callOp = b.create<CallOp>(loc, outlinedFunc, values);

  for (auto it : llvm::zip(returnVals, callOp->getResults())) {
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  // Erase the ops we outlined, which should be safe now.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(secondOps))) {
    assert(op->use_empty() && "expected 'op' to have no uses");
    op->erase();
  }
  assert(convOp->use_empty() && "expected 'op' to have no uses");
  convOp->erase();
  for (auto &op : llvm::make_early_inc_range(frontOps)) {
    assert(op->use_empty() && "expected 'op' to have no uses");
    op->erase();
  }
}

// Can't just use an equality test on attributes, because the sym_name
// attribute will always be different.  Here we get NamedAttrLists,
// which are implicitly sorted, and walk down them in parallel, ignoring
// sym_name.  If they're equivalent, then all the attributes will pair
// up.
bool areAttributesEffectivelyEqual(Operation *lhs, Operation *rhs) {
  NamedAttrList lhsAttrs(lhs->getAttrs());
  NamedAttrList rhsAttrs(rhs->getAttrs());

  const auto *lhsIterator = lhsAttrs.begin();
  const auto *rhsIterator = rhsAttrs.begin();

  while (lhsIterator != lhsAttrs.end() && rhsIterator != rhsAttrs.end()) {
    NamedAttribute lhsAttr = *lhsIterator;
    NamedAttribute rhsAttr = *rhsIterator;

    // Skip names, which always differ.
    if (lhsAttr.getName() == "sym_name")
      lhsAttr = *(++lhsIterator);
    if (rhsAttr.getName() == "sym_name")
      rhsAttr = *(++rhsIterator);

    if (lhsAttr != rhsAttr)
      return false;

    ++lhsIterator;
    ++rhsIterator;
  }

  return (lhsIterator == lhsAttrs.end() && rhsIterator == rhsAttrs.end());
}

// OperationEquivalence::isEquivalentTo() doesn't do what I want because
// the attributes it compares also include the function names, which will
// always be different.  Another approach would be to temporarily modify
// the functions to remove the names (and anything else unimportant).
bool isFunctionallyEquivalentTo(Operation *lhs, Operation *rhs) {
  if (lhs == rhs)
    return true;

  // Compare the operation name.
  if (lhs->getName() != rhs->getName())
    return false;
  // Check operand counts.
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;
  SmallVector<Type> lhsOperandTypes(lhs->getOperandTypes().begin(),
                                    lhs->getOperandTypes().end());
  SmallVector<Type> rhsOperandTypes(rhs->getOperandTypes().begin(),
                                    rhs->getOperandTypes().end());
  if (lhsOperandTypes.size() != rhsOperandTypes.size())
    return false;
  for (auto it : llvm::zip(lhsOperandTypes, rhsOperandTypes)) {
    if (std::get<0>(it) != std::get<1>(it))
      return false;
  }
  // Compare attributes.
  if (!areAttributesEffectivelyEqual(lhs, rhs))
    return false;
  // Compare result types.
  SmallVector<Type> lhsResultTypes(lhs->getResultTypes());
  SmallVector<Type> rhsResultTypes(rhs->getResultTypes());
  if (lhsResultTypes.size() != rhsResultTypes.size())
    return false;
  switch (lhsResultTypes.size()) {
  case 0:
    break;
  case 1:
    // Compare the single result type.
    if (lhsResultTypes.front() != rhsResultTypes.front())
      return false;
    break;
  default:
    // Use the type buffer for the comparison, as we can guarantee it is the
    // same for any given range of result types. This takes advantage of the
    // fact the result types >1 are stored in a TupleType and uniqued.
    if (lhsResultTypes.data() != rhsResultTypes.data())
      return false;
    break;
  }

  auto isExternFunc = [](Operation *op) {
    FuncOp f = dyn_cast<FuncOp>(op);
    return isa<FuncOp>(op) &&
           (f->getRegions().size() != 1 || f->getRegions()[0].empty());
  };
  if (isExternFunc(lhs) || isExternFunc(rhs)) {
    return false;
  }

  // TODO: Allow commutative operations to have different ordering.
  for (auto itRegion : llvm::zip(lhs->getRegions(), rhs->getRegions())) {
    for (auto itOperation : llvm::zip(std::get<0>(itRegion).getOps(),
                                      std::get<1>(itRegion).getOps())) {
      if (!isFunctionallyEquivalentTo(&std::get<0>(itOperation),
                                      &std::get<1>(itOperation))) {
        return false;
      }
    }
  }
  return true;
}

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
class TosaPartitionPass : public TosaPartitionBase<TosaPartitionPass> {
public:
  void runOnFunction() override {
    int count = 0;
    FuncOp func = getFunction();

    auto callback = [&](tosa::Conv2DOp convOp) {
      auto strCount = std::string("_outlined_part_") + std::to_string(count++);

      // Given a Conv2DOp, gather all the element-wise ops that are reachable
      // from its results, contiguously.
      //
      // The ops after the Conv2D are "second" ops.
      // inputNodes gathers what will become the parameters of the outlined
      // function;  initially it's the Conv2D's arguments, and it accumulates
      // arguments to other ops that don't come from inside the outlined
      // function.
      // resultNodes will become the results of the outlined function.  It
      // starts with Conv2D's result(s) and gains the results of each new
      // secondOp.  When all a resultNode's users can be determined to lie
      // within the outlined function, it's removed from the set.
      //
      // These are SetVectors because we test with contains() a lot, but still
      // want to preserve order.
      SetVector<Operation *> secondOps;
      SetVector<Value> inputNodes(convOp->getOperands().begin(),
                                  convOp->getOperands().end());
      SetVector<Value> resultNodes(convOp->getResults().begin(),
                                   convOp->getResults().end());

      // Grab a useful set of leading ops, like we do for trailing.
      // Let's limit it to only first arguments, with single uses.
      SmallVector<Operation *> frontOps;
      if (!nofront) {
        Operation *op = convOp;
        while (true) {
          if (op->getNumOperands() < 1)
            break;
          Operation *usedOp = op->getOpOperand(0).get().getDefiningOp();
          if (!usedOp)
            break;
          if (!isElementwiseOp(usedOp) || !usedOp->hasOneUse())
            break;
          // Remove the first operand from the function inputs, if present.
          // Add usedOp's operands to the function inputs.
          inputNodes.remove(op->getOpOperand(0).get());
          inputNodes.insert(usedOp->getOperands().begin(),
                            usedOp->getOperands().end());
          frontOps.push_back(usedOp);
          op = usedOp;
        }
      }

      DominanceInfo domInfo(func);
      std::deque<Operation *> worklist; // cuz I want to pull from the front.

      worklist.push_back(convOp);
      while (!worklist.empty()) {
        Operation *op = worklist.front();
        worklist.pop_front();
        for (auto *userOp : op->getUsers()) {
          if (isElementwiseOp(userOp) /* && userOp->hasOneUse() */) {
            bool skip = false;
            // First criterion is that the op is element-wise.  Second criterion
            // is that the op dominates all the users of the accumulated results
            // of the outlined function.  In other words, we can't take an op
            // that comes "after" a user of the result from the eventual call,
            // because the call needs to dominate all its users.
            for (const Value &val : resultNodes) {
              for (auto *user : val.getDefiningOp()->getUsers()) {
                if (user != userOp &&
                    !domInfo.properlyDominates(userOp, user)) {
                  skip = true;
                }
              }
            }

            // userOp is acceptable.  Keep it as a secondOp, put it on the
            // worklist.  Add its operands to inputNodes unless they come
            // from other secondOps (indicated by being in resultNodes).
            // If all the users of any resultNode are in secondOps, there's
            // no need to return it so remove from resultNodes.  Finally,
            // add all userOp's results to resultNodes.
            if (!skip) {
              secondOps.insert(userOp);
              worklist.push_back(userOp);
              for (Value opnd : userOp->getOperands()) {
                if (!resultNodes.contains(opnd)) {
                  inputNodes.insert(opnd);
                }
              }
              for (const Value &val : resultNodes) {
                if (llvm::all_of(val.getUsers(), [&](Operation *u) {
                      return secondOps.contains(u);
                    })) {
                  resultNodes.remove(val);
                }
              }
              for (auto res : userOp->getResults()) {
                resultNodes.insert(res);
              }
            }
          }
        }
      }

      if (!secondOps.empty() || !frontOps.empty()) {
        // Make the outlined function from the ops we've gathered.
        outlineConvPartOps(convOp, secondOps, frontOps,
                           inputNodes.getArrayRef(), resultNodes.getArrayRef(),
                           std::string(func.sym_name()) + strCount);
        // Outlining will erase nodes and thus perturb the walk, so
        // signal interrupted to exit it and restart.
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    // Walk until we've outlined all the Conv2D ops we can.
    while (func.walk(callback).wasInterrupted()) {
    }

    // Count on the SymbolDCE pass to clean up the dead functions.
  }
};

// Replace calls to identical functions with calls to the first one,
// allowing the others to be dead-coded.
struct PostPartitionCollapsePass
    : public PostPartitionCollapseBase<PostPartitionCollapsePass> {
  void runOnOperation() override;
};

void PostPartitionCollapsePass::runOnOperation() {
  ModuleOp module = getOperation();
  // For all FuncOps, make a mapping to replace those that are identical to
  // another.
  SmallVector<Operation *> opsSeen;
  DenseMap<StringRef, StringRef> replacements;
  for (auto f : module.getOps<FuncOp>()) {
    bool replaced = false;
    for (Operation *o : opsSeen) {
      if (isFunctionallyEquivalentTo(f, o)) {
        replacements[f.sym_name()] = dyn_cast<FuncOp>(o).sym_name();
        replaced = true;
      }
    }
    if (!replaced) {
      opsSeen.push_back(f);
    }
  }
  // Then walk all the CallOps and remap callees where appropriate.
  module.walk([&](CallOp call) {
    if (replacements.find(call.getCallee()) != replacements.end()) {
      call.calleeAttr(FlatSymbolRefAttr::get(module->getContext(),
                                             replacements[call.getCallee()]));
    }
  });
}

} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaPartitionPass() {
  return std::make_unique<TosaPartitionPass>();
}

std::unique_ptr<Pass> mlir::tosa::createPostPartitionCollapsePass() {
  return std::make_unique<PostPartitionCollapsePass>();
}

////////////////////////////////////////////////////////////////////////////////

static void tosaPartitionPipeline(OpPassManager &pm) {
  pm.addPass(std::make_unique<TosaPartitionPass>());
  pm.addPass(std::make_unique<PostPartitionCollapsePass>());
  pm.addPass(mlir::createSymbolDCEPass());
}

namespace mlir {
namespace tosa {
void registerTosaPartitionPipelinePass() {
  PassPipelineRegistration<>("tosa-partition-pipeline",
                             "Partition around Conv2D ops and clean up after",
                             tosaPartitionPipeline);
}
} // namespace tosa
} // namespace mlir
