//===- TosaPartition1.cpp ------------------------------------------===//
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

#include <iostream>
#include <algorithm>
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tosa/IR//TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/PassDetail.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

using llvm::SmallVector;
using llvm::SetVector;
using llvm::errs;

using namespace mlir;


namespace {

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// ElementwiseMappable trait is strict about matching dimensions.  In
// practice, broadcastable and same-type Tosa ops are also element-wise.
bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::ElementwiseMappable>()
    ||   op->hasTrait<OpTrait::ResultsBroadcastableShape>()
    ||   op->hasTrait<OpTrait::SameOperandsAndResultType>();
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp.
void outlineConvPartOps(Operation *convOp,
                        SetVector<Operation*> &secondOps,
                        ArrayRef<Operation*> frontOps,
                        ArrayRef<Value> params,
                        ArrayRef<Value> returnVals,
                        StringRef partFnName) {
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
    assert(llvm::all_of(op->getOperands(), [&](Value v) {
          return bvm.lookupOrNull(v);
        }));
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
}

// OperationEquivalence::isEquivalentTo() doesn't do what I want because
// the attributes its compares also include the function names, which will
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
  SmallVector<Type> lhsOperandTypes(lhs->getOperandTypes().begin(), lhs->getOperandTypes().end());
  SmallVector<Type> rhsOperandTypes(rhs->getOperandTypes().begin(), rhs->getOperandTypes().end());
  if (lhsOperandTypes.size() != rhsOperandTypes.size())
    return false;
  for (auto it : llvm::zip(lhsOperandTypes, rhsOperandTypes)) {
    if (std::get<0>(it) != std::get<1>(it))
      return false;
  }
  // Compare attributes.
//   if (lhs->getAttrDictionary() != rhs->getAttrDictionary())
//     return false;
  // Compare result types.
  ArrayRef<Type> lhsResultTypes = lhs->getResultTypes();
  ArrayRef<Type> rhsResultTypes = rhs->getResultTypes();
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


void collapseIdenticalFunctions(ModuleOp module) {
  SmallVector<Operation *> opsSeen;
  SmallVector<Operation *> opsToDelete;
  for (auto f : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    if (f->getRegions().size() == 1 && !f->getRegions()[0].empty()) {
      bool erase = false;
      for (Operation *o : opsSeen) {
        if (isFunctionallyEquivalentTo(f, o)) {
          CallOp callToF = nullptr;
          module.walk([&] (CallOp call) {
                        if (call.getCallee() == f.sym_name()) {
                          callToF = call;
                          return WalkResult::interrupt();
                        }
                        return WalkResult::advance();
                      });
          assert(callToF);
          callToF.calleeAttr(FlatSymbolRefAttr::get(module->getContext(),
                                                    dyn_cast<FuncOp>(o).sym_name()));
          erase = true;
          break;
        }
      }
      if (erase) {
        opsToDelete.push_back(f);
      } else {
        opsSeen.push_back(f);
      }
    }
  }
  for (auto &op : llvm::make_early_inc_range(opsToDelete)) {
    op->erase();
  }
}


// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
class TosaPartition1Pass
    : public PassWrapper<TosaPartition1Pass, FunctionPass> {
public:
  explicit TosaPartition1Pass() {}

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



      // Experiment:  can we grab a useful set of leading ops, like we do
      // for trailing?
      //
      // Let's limit it to only first arguments, with single uses.
      SmallVector<Operation*> frontOps;
      Operation *op = convOp;
      while (true) {
        if (op->getNumOperands() < 1) break;
        Operation *usedOp = op->getOpOperand(0).get().getDefiningOp();
        if (!usedOp) break;
        if (!isElementwiseOp(usedOp) || !usedOp->hasOneUse()) break;
        // Remove the first operand from the function inputs, if present.
        // Add usedOp's operands to the function inputs.
        inputNodes.remove(op->getOpOperand(0).get());
        inputNodes.insert(usedOp->getOperands().begin(),
                          usedOp->getOperands().end());
        frontOps.push_back(usedOp);
        op = usedOp;
      }



      DominanceInfo domInfo(func);
      std::deque<Operation *> worklist; // cuz I want to pull from the front.

      worklist.push_back(convOp);
      while (!worklist.empty()) {
        Operation *op = worklist.front();
        worklist.pop_front();
        for (auto *userOp : op->getUsers()) {
          if (isElementwiseOp(userOp) /* && userOp->hasOneUse() */ ) {
            bool skip = false;
            // First criterion is that the op is element-wise.  Second criterion
            // is that the op dominates all the users of the accumulated results
            // of the outlined function.  In other words, we can't take an op
            // that comes "after" a user of the result from the eventual call,
            // because the call needs to dominate all its users.
            for (const Value& val : resultNodes) {
              for (auto *user : val.getDefiningOp()->getUsers()) {
                if (user != userOp && !domInfo.properlyDominates(userOp, user)) {
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
              for (const Value& val : resultNodes) {
                if (llvm::all_of(val.getUsers(),
                                 [&](Operation *u) {
                                   return secondOps.contains(u);
                                 } )) {
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
                           inputNodes.getArrayRef(),
                           resultNodes.getArrayRef(),
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

    // Then get rid of outlined functions that duplicate each other.
//    collapseIdenticalFunctions(func->getParentOfType<ModuleOp>());
  }
};

////////////////////////////////////////////////////////////////////////////////

struct PartitionConv2D : public RewritePattern {
  explicit PartitionConv2D(MLIRContext* context)
      : RewritePattern(tosa::Conv2DOp::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter& rewriter) const override;
};

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
class TosaPartition2Pass
    : public PassWrapper<TosaPartition2Pass, FunctionPass> {
public:
  explicit TosaPartition2Pass() {}

  void runOnFunction() override {
    FuncOp func = getFunction();
    OwningRewritePatternList patterns;
    auto* ctx = func->getContext();

    // Add the generated patterns to the list.
    patterns.insert<PartitionConv2D>(ctx);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

LogicalResult PartitionConv2D::matchAndRewrite(Operation *op,
                                               PatternRewriter& rw) const {
  auto convOp = cast<tosa::Conv2DOp>(op);

  // Insert outlined function before current function.
  auto func = op->getParentOfType<FuncOp>();

  errs() << "inside function: ";
  func->dump();

  // We're using the PatternRewriter for changes to the current function,
  // and the OpBuilder for nodes in the outlined function.  One thing that
  // split avoids is notifyOperationInserted() calls on nodes cloned into
  // the outlined functions.
  OpBuilder b(func->getContext());
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(func);
//   // Need arrays for params and returnVals, so they can become
//   // ValueRanges to make the FuncOp.
//   SmallVector<Value> params;
//   SmallVector<Value> returnVals;
//   ValueRange values(params);
//   ValueRange results(returnVals);

  llvm::SetVector<mlir::Value> functionInputs(convOp->getOperands().begin(),
                                              convOp->getOperands().end());
  auto typesRange = llvm::map_range(
      functionInputs, [](Value value) { return value.getType(); });
  SmallVector<Type, 4> inputTypes(typesRange.begin(), typesRange.end());
  auto outputTypes = convOp->getResultTypes();

  FunctionType type = FunctionType::get(func->getContext(), inputTypes, outputTypes);
  SmallVector<NamedAttribute, 1> kernelAttrs{b.getNamedAttr("kernel", b.getUnitAttr())};
  auto outlinedFunc = b.create<FuncOp>(op->getLoc(),
                                       std::string(func.sym_name()) + "_foo" + std::to_string(rand()),
                                       type, ArrayRef<NamedAttribute>(kernelAttrs));
  b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
  BlockAndValueMapping bvm;
  for (auto it : llvm::zip(functionInputs, outlinedFunc.getArguments()))
    bvm.map(std::get<0>(it), std::get<1>(it));
  auto* newOp = b.clone(*convOp, bvm);

  // for each use
  //    if it qualifies
  //       clone into new function
  //       check its uses
  // look for operands that aren't included in the parameters
  // look for users outside the function
  // insert call

#if 0
  std::deque<Operation *> worklist; // cuz I want to pull from the front.
  worklist.push_back(convOp);
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    for (auto *userOp : op->getUsers()) {
      // If it has no uses, it was part of another convOp's chain.
      // (Need a better test here.)
      // All three of these traits indicate element-wise ops.
      if ((   userOp->hasTrait<OpTrait::ElementwiseMappable>()
              || userOp->hasTrait<OpTrait::ResultsBroadcastableShape>()
              || userOp->hasTrait<OpTrait::SameOperandsAndResultType>())
          && !op->use_empty()) {
        secondOps.insert(userOp);
        worklist.push_back(userOp);
      }
    }
  }
#endif  /* 0 */




  Operation *lastOp = convOp;
  for (auto *userOp : convOp->getUsers()) {
    errs() << "user?\n";
    if (isElementwiseOp(userOp)) {
      errs() << "  cloned\n";
      newOp  = b.clone(*userOp, bvm);
      lastOp = userOp;
    }
  }

  llvm::SetVector<Value> valuesSet;
  mlir::getUsedValuesDefinedAbove(outlinedFunc.getBody(), valuesSet);
  errs() << "defined above:\n";
  for (Value val : valuesSet) {
    errs() << "  ";
    val.dump();
  }
  if (!valuesSet.empty()) {
    errs() << "will need to add parameters\n";
  }

  b.create<ReturnOp>(outlinedFunc->getLoc(), newOp->getResults());
  rw.setInsertionPoint(convOp);
  CallOp callOp = rw.create<CallOp>(convOp->getLoc(), outlinedFunc, functionInputs.getArrayRef());
  rw.replaceOp(lastOp, callOp->getResults());

//  for (auto it : llvm::zip(returnVals, callOp->getResults())) {
//    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
//  }

//   SmallVector<Operation*> ops;
//   for (auto &op : func.getBody().getOps()) {
//     ops.push_back(&op);
//   }
//   for (auto &op : llvm::make_early_inc_range(llvm::reverse(ops))) {
//     if (op->use_empty() && !op->isKnownTerminator())
//       rw.eraseOp(op);
//   }




  errs() << "main function:\n";
  func->dump();
  errs() << "outlined function:\n";
  outlinedFunc->dump();


//   if (!result) {
//     return failure();
//   }

//  rewriter.replaceOp(convOp, {result.getValue()});
  return success();
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
class TosaPartition3Pass
    : public PassWrapper<TosaPartition3Pass, FunctionPass> {
public:
  explicit TosaPartition3Pass() {}

  void runOnFunction() override {
    FuncOp func = getFunction();
    OpBuilder b(func->getContext());

    SmallVector<Operation*> convOps;

    func.walk([&](tosa::Conv2DOp convOp) {
                if (llvm::any_of(convOp->getUsers(), isElementwiseOp)) {
                  convOps.push_back(convOp);
                }
              });

    for (Operation *op : convOps) {
      Block *block, *beforeBlock, *afterBlock;
      if (op->getPrevNode()) {
        beforeBlock = op->getBlock();
        block = beforeBlock->splitBlock(op);
      } else {
        beforeBlock = nullptr;
        block = op->getBlock();
      }
      assert(op->getNextNode());
      afterBlock = block->splitBlock(op->getNextNode());

      // Insert outlined function before current function.
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(op->getParentOfType<FuncOp>());
      // Need arrays for params and returnVals, so they can become
      // ValueRanges to make the FuncOp.
      SmallVector<Value> params;
      SmallVector<Value> returnVals;
      ValueRange values(params);
      ValueRange results(returnVals);
//       FunctionType type =
//         FunctionType::get(func->getContext(), values.getTypes(), results.getTypes());
      SmallVector<NamedAttribute, 1> kernelAttrs{b.getNamedAttr("kernel", b.getUnitAttr())};
//       auto outlinedFunc =
//           b.create<FuncOp>(op->getLoc(), std::string(func.sym_name()) + "_out",
//                            type, ArrayRef<NamedAttribute>(kernelAttrs));

//       // Clone convOp and secondOps into the body of the new function.
//       b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
//       BlockAndValueMapping bvm;
//       b.clone(*convOp, bvm);


      if (beforeBlock) {
        b.setInsertionPointToEnd(beforeBlock);
        b.create<BranchOp>(op->getLoc(), block);
      }
      b.setInsertionPointToEnd(block);
      b.create<BranchOp>(afterBlock->begin()->getLoc(), afterBlock);

      for (auto *userOp : op->getUsers()) {
        if (isElementwiseOp(userOp)) {
          userOp->moveAfter(op);
        }
      }

    }

    func.dump();
  }
};
} // namespace

////////////////////////////////////////////////////////////////////////////////

namespace mlir {
namespace test {
void registerTosaPartition1Pass() {
  PassRegistration<TosaPartition1Pass>("tosa-partition1",
                                       "tosa partition1");
}
void registerTosaPartition2Pass() {
  PassRegistration<TosaPartition2Pass>("tosa-partition2",
                                       "tosa partition2");
}
void registerTosaPartition3Pass() {
  PassRegistration<TosaPartition3Pass>("tosa-partition3",
                                       "tosa partition3");
}
} // namespace test
} // namespace mlir
