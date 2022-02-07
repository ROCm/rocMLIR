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
#include "mlir/IR/Matchers.h"
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

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp
// and mergeIdenticalBlocks() in Utils/RegionUtils.cpp.

struct OutliningCandidate {
  OutliningCandidate(Operation *_convOp, ArrayRef<Operation *> &_secondOps,
                     ArrayRef<Operation *> &_frontOps, ArrayRef<Value> &_params,
                     ArrayRef<Value> &_returnVals, StringRef _partFnName);

  unsigned addOp(Operation *op, unsigned orderIt);

  Operation *convOp;
  SmallVector<Operation *> secondOps;
  SmallVector<Operation *> frontOps;
  SmallVector<Value> params;
  SmallVector<Value> returnVals;
  std::string partFnName;
  llvm::hash_code hash;
  FuncOp function;

  /// Return the order index for the given value that is within the block of
  /// this data.
  unsigned getOrderOf(Value value) const;

  /// A map of result producing operations to their relative orders within this
  /// block. The order of an operation is the number of defined values that are
  /// produced within the block before this operation.
  DenseMap<Operation *, unsigned> opOrderIndex;
};

unsigned OutliningCandidate::addOp(Operation *op, unsigned orderIt) {
  if (unsigned numResults = op->getNumResults()) {
    opOrderIndex.try_emplace(op, orderIt);
    orderIt += numResults;
  }

  auto opHash = OperationEquivalence::computeHash(
      op, OperationEquivalence::ignoreHashValue,
      OperationEquivalence::ignoreHashValue,
      OperationEquivalence::IgnoreLocations);
  hash = llvm::hash_combine(hash, opHash);

  return orderIt;
}

OutliningCandidate::OutliningCandidate(Operation *convOp_,
                                       ArrayRef<Operation *> &secondOps_,
                                       ArrayRef<Operation *> &frontOps_,
                                       ArrayRef<Value> &params_,
                                       ArrayRef<Value> &returnVals_,
                                       StringRef partFnName_)
    : convOp(convOp_), partFnName(partFnName_), hash(0), function(nullptr) {
  // We'll need to grab the cloned ops to avoid use-after-free.
  for (auto *op : secondOps_) {
    secondOps.push_back(op);
  }
  for (auto *op : frontOps_) {
    frontOps.push_back(op);
  }
  for (auto val : params_) {
    params.push_back(val);
  }
  for (auto val : returnVals_) {
    returnVals.push_back(val);
  }

  unsigned orderIt = params.size();
  for (auto *op : frontOps) {
    orderIt = addOp(op, orderIt);
  }
  orderIt = addOp(convOp, orderIt);
  for (auto *op : secondOps) {
    orderIt = addOp(op, orderIt);
  }
}

unsigned OutliningCandidate::getOrderOf(Value value) const {
  // Arguments use the argument number as the order index.
  if (BlockArgument arg = value.dyn_cast<BlockArgument>())
    return arg.getArgNumber();
  for (unsigned i = 0; i < params.size(); i++) {
    if (params[i] == value)
      return i;
  }

  // Otherwise, the result order is offset from the parent op's order.
  auto *definingOp = value.getDefiningOp();
  if (definingOp) {
    auto opOrderIt = opOrderIndex.find(definingOp);
    // Candidate arguments will have a definingOp that won't be in opOrderIndex.
    assert(opOrderIt != opOrderIndex.end() && "expected op to have an order");
    return opOrderIt->second + value.cast<OpResult>().getResultNumber();
  }

  return 0;
}

bool opsMatch(Operation *lhs, Operation *rhs, OutliningCandidate &one,
              OutliningCandidate &two) {
  // Check that the operations are equivalent.
  if (!OperationEquivalence::isEquivalentTo(
          lhs, rhs, OperationEquivalence::ignoreValueEquivalence,
          OperationEquivalence::ignoreValueEquivalence,
          OperationEquivalence::Flags::IgnoreLocations))
    return false;

  // Compare the operands of the two operations. If the operand is within
  // the block, it must refer to the same operation.
  auto lhsOperands = lhs->getOperands(), rhsOperands = rhs->getOperands();
  if (lhs->getNumOperands() != rhs->getNumOperands()) {
    return false;
  }
  for (int operand : llvm::seq<int>(0, lhs->getNumOperands())) {
    Value lhsOperand = lhsOperands[operand];
    Value rhsOperand = rhsOperands[operand];
    if (lhsOperand == rhsOperand)
      continue;
    // Check that the types of the operands match.
    if (lhsOperand.getType() != rhsOperand.getType())
      return false;

    // Otherwise, these operands must have the same logical order within the
    // parent block.
    if (one.getOrderOf(lhsOperand) != two.getOrderOf(rhsOperand)) {
      return false;
    }
  }

  return true;
}

bool outliningCandidatesEquivalent(OutliningCandidate &one,
                                   OutliningCandidate &two) {
  if (one.hash != two.hash) {
    return false;
  }

  if (one.params.size() != two.params.size()) {
    return false;
  }
  for (unsigned i = 0; i < one.params.size(); i++) {
    if (one.params[i].getType() != two.params[i].getType()) {
      return false;
    }
  }

  for (auto ops : llvm::zip(one.frontOps, two.frontOps)) {
    if (!opsMatch(std::get<0>(ops), std::get<1>(ops), one, two)) {
      return false;
    }
  }
  if (!opsMatch(one.convOp, two.convOp, one, two)) {
    return false;
  }
  for (auto ops : llvm::zip(one.secondOps, two.secondOps)) {
    if (!opsMatch(std::get<0>(ops), std::get<1>(ops), one, two)) {
      return false;
    }
  }
  return true;
}

OutliningCandidate *
findOutliningCandidate(OutliningCandidate &newCandidate,
                       std::vector<OutliningCandidate> &candidates) {
  for (auto &candidate : candidates) {
    if (outliningCandidatesEquivalent(candidate, newCandidate)) {
      return &candidate;
    }
  }
  return nullptr;
}

// Given a convolution op and its fuse-able trailing (second) and leading
// (front) ops, remove them into a separate function.
void outlineConvPartOps(Operation *convOp, ArrayRef<Operation *> secondOps,
                        ArrayRef<Operation *> frontOps, ArrayRef<Value> params,
                        ArrayRef<Value> returnVals, StringRef partFnName,
                        std::vector<OutliningCandidate> &candidates) {
  ValueRange values(params);
  Operation *lastOp = secondOps[secondOps.size() - 1];
  OpBuilder b(convOp);
  Location loc = convOp->getLoc();
  FuncOp outlinedFunc;

  // ------------------------------------------------------------
  // Merging part.

  OutliningCandidate newCandidate(convOp, secondOps, frontOps, params,
                                  returnVals, partFnName);

  if (OutliningCandidate *found =
          findOutliningCandidate(newCandidate, candidates)) {
    // Matches one we already have.
    outlinedFunc = found->function;
  } else {
    // ------------------------------------------------------------
    // Construction part.

    // Insert outlined function before current function.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(convOp->getParentOfType<FuncOp>());

    // Make FuncOp from convOp's operand types and secondOp's result type.
    MLIRContext *ctx = convOp->getContext();
    ValueRange results(returnVals);
    FunctionType type =
        FunctionType::get(ctx, values.getTypes(), results.getTypes());
    SmallVector<NamedAttribute, 1> kernelAttrs{
        b.getNamedAttr("kernel", b.getUnitAttr()),
    };
    outlinedFunc = b.create<FuncOp>(loc, partFnName, type,
                                    ArrayRef<NamedAttribute>(kernelAttrs));
    outlinedFunc->setAttr("sym_visibility", StringAttr::get(ctx, "private"));
    newCandidate.function = outlinedFunc;

    // Clone frontOps, convOp, and secondOps into the body of the new function,
    // while also updating the comparison details for future candidates.
    b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
    BlockAndValueMapping bvm;
    for (auto it : llvm::zip(values, outlinedFunc.getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));

    newCandidate.frontOps.clear();
    for (auto *op : llvm::reverse(frontOps)) {
      newCandidate.frontOps.push_back(b.clone(*op, bvm));
      newCandidate.opOrderIndex[newCandidate.frontOps.back()] =
          newCandidate.opOrderIndex[op];
    }
    std::reverse(newCandidate.frontOps.begin(), newCandidate.frontOps.end());

    newCandidate.convOp = b.clone(*convOp, bvm);
    newCandidate.opOrderIndex[newCandidate.convOp] =
        newCandidate.opOrderIndex[convOp];

    newCandidate.secondOps.clear();
    for (auto *op : secondOps) {
      // All operands should already be in bvm.
      assert(llvm::all_of(op->getOperands(),
                          [&](Value v) { return bvm.lookupOrNull(v); }));
      newCandidate.secondOps.push_back(b.clone(*op, bvm));
      newCandidate.opOrderIndex[newCandidate.secondOps.back()] =
          newCandidate.opOrderIndex[op];
    }

    // Make ReturnOp from secondOps' results.
    SmallVector<Value> returnOperands;
    for (auto op : returnVals) {
      returnOperands.push_back(bvm.lookup(op));
    }
    // Can't also supply return types, because it'll see a mismatch
    // in numbers where there isn't one.
    b.create<ReturnOp>(loc, returnOperands);

    candidates.push_back(newCandidate);
  }

  // ------------------------------------------------------------
  // Replacement part.

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

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
class TosaPartitionPass : public TosaPartitionBase<TosaPartitionPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto funcOps = module.getOps<FuncOp>();
    for (auto func : llvm::make_early_inc_range(funcOps)) {
      int count = 0;
      // (Problems with node mismatches and unexpected uses if we have the
      // candidates list at module level.)
      std::vector<OutliningCandidate> candidates;
      auto callback = [&](tosa::Conv2DOp convOp) {
        auto strCount =
            std::string("_outlined_part_") + std::to_string(count++);

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
            // Special loop for constant operands.
            for (const auto &opnd : op->getOperands()) {
              Operation *usedOp = opnd.getDefiningOp();
              if (usedOp) {
                if (detail::isConstantLike(usedOp) && usedOp->hasOneUse()) {
                  // Remove the operand from the function inputs, if present.
                  inputNodes.remove(opnd);
                  frontOps.push_back(usedOp);
                }
              }
            }

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
            if (isElementwiseOp(userOp)) {
              bool skip = false;
              // First criterion is that the op is element-wise.  Second
              // criterion is that the op dominates all the users of the
              // accumulated results of the outlined function.  In other words,
              // we can't take an op that comes "after" a user of the result
              // from the eventual call, because the call needs to dominate all
              // its users.
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
          outlineConvPartOps(
              convOp, secondOps.getArrayRef(), frontOps,
              inputNodes.getArrayRef(), resultNodes.getArrayRef(),
              std::string(func.sym_name()) + strCount, candidates);
          // Outlining will erase nodes and thus perturb the walk, so
          // signal interrupted to exit it and restart.
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      };

      // Walk until we've outlined all the Conv2D ops we can.
      while (func.walk(callback).wasInterrupted()) {
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::tosa::createTosaPartitionPass() {
  return std::make_unique<TosaPartitionPass>();
}
