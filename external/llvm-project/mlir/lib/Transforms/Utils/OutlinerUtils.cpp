//===- OutlinerUtils.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace conv2d followed by elementwise op with call to function containing
// them.  Generalised, outline any anchor op, all its trailing elementwise ops,
// and all its leading elementwise ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/OutlinerUtils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/TopologicalSortUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <deque>
#include <iostream>

using llvm::SmallVector;
using namespace mlir;

static bool isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isZero();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<DenseResourceElementsAttr>())
    return false;
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

bool mlir::isConstantZero(Operation *op) {
  // Cheating, by assuming that constants will have "value" attribute.
  if (op->hasTrait<OpTrait::ConstantLike>() && op->hasAttr("value"))
    return isZeroAttribute(op->getAttr("value"));
  return false;
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp
// and mergeIdenticalBlocks() in Utils/RegionUtils.cpp.

struct OutliningCandidate {
  OutliningCandidate(Operation *anchorOp, ArrayRef<Operation *> &trailingOps,
                     ArrayRef<Operation *> &leadingOps, ArrayRef<Value> &params,
                     ArrayRef<Value> &returnVals, StringRef partFnName);

  unsigned addOp(Operation *op, unsigned orderIt);

  Operation *anchorOp;
  SmallVector<Operation *> trailingOps;
  SmallVector<Operation *> leadingOps;
  SmallVector<Type> params;
  SmallVector<Value> returnVals;
  std::string partFnName;
  llvm::hash_code hash;
  func::FuncOp function;
  Location fusedLoc;

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

OutliningCandidate::OutliningCandidate(Operation *anchorOp_,
                                       ArrayRef<Operation *> &trailingOps_,
                                       ArrayRef<Operation *> &leadingOps_,
                                       ArrayRef<Value> &params_,
                                       ArrayRef<Value> &returnVals_,
                                       StringRef partFnName_)
    : anchorOp(anchorOp_), trailingOps(trailingOps_), leadingOps(leadingOps_),
      returnVals(returnVals_), partFnName(partFnName_), hash(0),
      function(nullptr), fusedLoc(UnknownLoc::get(anchorOp_->getContext())) {
  for (auto val : params_) {
    params.push_back(val.getType());
  }
  unsigned orderIt = params.size();
  for (auto *op : leadingOps) {
    orderIt = addOp(op, orderIt);
  }
  orderIt = addOp(anchorOp, orderIt);
  for (auto *op : trailingOps) {
    orderIt = addOp(op, orderIt);
  }
}

unsigned OutliningCandidate::getOrderOf(Value value) const {
  // Otherwise, the result order is offset from the parent op's order.
  auto *definingOp = value.getDefiningOp();
  if (definingOp) {
    auto opOrderIt = opOrderIndex.find(definingOp);
    // Candidate arguments will have a definingOp that won't be in opOrderIndex.
    if (opOrderIt != opOrderIndex.end())
      return opOrderIt->second + value.cast<OpResult>().getResultNumber();

    for (unsigned i = 0; i < params.size(); i++) {
      if (params[i] == value.getType())
        return i;
    }
  }

  return 0;
}

static bool opsMatch(Operation *lhs, Operation *rhs, OutliningCandidate &one,
                     OutliningCandidate &two) {
  // Check that the operations are equivalent.
  if (!OperationEquivalence::isEquivalentTo(
          lhs, rhs, OperationEquivalence::ignoreValueEquivalence, nullptr,
          OperationEquivalence::Flags::IgnoreLocations))
    return false;

  // Compare the operands of the two operations. If the operand is within
  // the block, it must refer to the same operation.
  auto lhsOperands = lhs->getOperands(), rhsOperands = rhs->getOperands();
  if (lhs->getNumOperands() != rhs->getNumOperands()) {
    return false;
  }
  for (auto opnds : llvm::zip(lhsOperands, rhsOperands)) {
    Value lhsOperand = std::get<0>(opnds);
    Value rhsOperand = std::get<1>(opnds);
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

static bool outliningCandidatesEquivalent(OutliningCandidate &one,
                                          OutliningCandidate &two) {
  if (one.hash != two.hash) {
    return false;
  }

  if (one.params.size() != two.params.size()) {
    return false;
  }
  for (auto params : llvm::zip(one.params, two.params)) {
    if (std::get<0>(params) != std::get<1>(params)) {
      return false;
    }
  }

  for (auto ops : llvm::zip(one.leadingOps, two.leadingOps)) {
    if (!opsMatch(std::get<0>(ops), std::get<1>(ops), one, two)) {
      return false;
    }
  }
  if (!opsMatch(one.anchorOp, two.anchorOp, one, two)) {
    return false;
  }
  for (auto ops : llvm::zip(one.trailingOps, two.trailingOps)) {
    if (!opsMatch(std::get<0>(ops), std::get<1>(ops), one, two)) {
      return false;
    }
  }
  return true;
}

static OutliningCandidate *
findOutliningCandidate(OutliningCandidate &newCandidate,
                       std::vector<OutliningCandidate> &candidates) {
  for (auto &candidate : candidates) {
    if (outliningCandidatesEquivalent(candidate, newCandidate)) {
      return &candidate;
    }
  }
  return nullptr;
}

// Given an op and its fuse-able trailing (second) and leading
// (front) ops, remove them into a separate function.
static void outlineOps(Operation *anchorOp, ArrayRef<Operation *> trailingOps,
                       ArrayRef<Operation *> leadingOps, ArrayRef<Value> params,
                       ArrayRef<Value> returnVals, StringRef partFnName,
                       StringRef attrName,
                       std::vector<OutliningCandidate> &candidates) {
  ValueRange values(params);
  OpBuilder b(anchorOp);
  Location loc = anchorOp->getLoc();
  func::FuncOp outlinedFunc;
  Location fusedLoc(loc);

  // ------------------------------------------------------------
  // Merging part.

  OutliningCandidate newCandidate(anchorOp, trailingOps, leadingOps, params,
                                  returnVals, partFnName);

  if (OutliningCandidate *found =
          findOutliningCandidate(newCandidate, candidates)) {
    // Matches one we already have.
    outlinedFunc = found->function;
    fusedLoc = found->fusedLoc;
  } else {
    // ------------------------------------------------------------
    // Construction part.

    // Insert outlined function before current function.
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPoint(anchorOp->getParentOfType<func::FuncOp>());

    // Make FuncOp from anchorOp's operand types and trailingOp's result type.
    MLIRContext *ctx = anchorOp->getContext();
    ValueRange results(returnVals);
    FunctionType type =
        FunctionType::get(ctx, values.getTypes(), results.getTypes());
    SmallVector<NamedAttribute, 1> kernelAttrs{
        b.getNamedAttr(attrName, b.getUnitAttr()),
    };
    outlinedFunc = b.create<func::FuncOp>(
        loc, partFnName, type, ArrayRef<NamedAttribute>(kernelAttrs));
    outlinedFunc->setAttr("sym_visibility", StringAttr::get(ctx, "private"));
    newCandidate.function = outlinedFunc;

    // Add access modes for parameters: read-only, write-only, read-write
    // All MemRef params are marked as 'read-write'
    // Non-MemRef inputs are added as 'read-only'
    auto readAccessAttr =
        b.getNamedAttr(func::FuncOp::getReadAccessAttrName(), b.getUnitAttr());
    auto writeAccessAttr =
        b.getNamedAttr(func::FuncOp::getWriteAccessAttrName(), b.getUnitAttr());
    for (auto pair : llvm::enumerate(values)) {
      auto vtype = pair.value().getType();
      if (vtype.isa<VectorType, RankedTensorType, UnrankedTensorType>())
        outlinedFunc.setArgAttrs(pair.index(),
                                 b.getDictionaryAttr({readAccessAttr}));
      else if (vtype.isa<MemRefType>())
        outlinedFunc.setArgAttrs(
            pair.index(),
            b.getDictionaryAttr({readAccessAttr, writeAccessAttr}));
    }
    // Non-MemRef results are added as 'write-only'
    for (auto pair : llvm::enumerate(results)) {
      auto vtype = pair.value().getType();
      if (vtype.isa<VectorType, RankedTensorType, UnrankedTensorType>())
        outlinedFunc.setResultAttrs(pair.index(),
                                    b.getDictionaryAttr({writeAccessAttr}));
      else if (vtype.isa<MemRefType>())
        outlinedFunc.setResultAttrs(
            pair.index(),
            b.getDictionaryAttr({readAccessAttr, writeAccessAttr}));
    }

    // Clone leadingOps, anchorOp, and trailingOps into the body of the new
    // function, while also updating the comparison details for future
    // candidates.
    b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
    IRMapping bvm;
    for (auto it : llvm::zip(values, outlinedFunc.getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));

    SmallVector<Location> collectedLocs{anchorOp->getLoc()};
    newCandidate.leadingOps.clear();
    for (auto *op : llvm::reverse(leadingOps)) {
      newCandidate.leadingOps.push_back(b.clone(*op, bvm));
      newCandidate.opOrderIndex[newCandidate.leadingOps.back()] =
          newCandidate.opOrderIndex[op];
      collectedLocs.push_back(op->getLoc());
    }
    std::reverse(newCandidate.leadingOps.begin(),
                 newCandidate.leadingOps.end());

    newCandidate.anchorOp = b.clone(*anchorOp, bvm);
    newCandidate.opOrderIndex[newCandidate.anchorOp] =
        newCandidate.opOrderIndex[anchorOp];

    newCandidate.trailingOps.clear();
    for (auto *op : trailingOps) {
      // All operands should already be in bvm.
      assert(llvm::all_of(op->getOperands(),
                          [&](Value v) { return bvm.lookupOrNull(v); }));
      newCandidate.trailingOps.push_back(b.clone(*op, bvm));
      newCandidate.opOrderIndex[newCandidate.trailingOps.back()] =
          newCandidate.opOrderIndex[op];
      collectedLocs.push_back(op->getLoc());
    }

    // Make ReturnOp from trailingOps' results.
    SmallVector<Value> returnOperands;
    for (auto op : returnVals) {
      returnOperands.push_back(bvm.lookup(op));
    }
    // Can't also supply return types, because it'll see a mismatch
    // in numbers where there isn't one.
    b.create<func::ReturnOp>(loc, returnOperands);

    newCandidate.fusedLoc = FusedLoc::get(ctx, collectedLocs);
    candidates.push_back(newCandidate);
  }

  // ------------------------------------------------------------
  // Replacement part.

  // Replace anchorOp, trailingOps, and leadingOps with CallOp to new function.
  Operation *lastOp = anchorOp;
  if (!trailingOps.empty())
    lastOp = trailingOps[trailingOps.size() - 1];
  b.setInsertionPointAfter(lastOp);
  func::CallOp callOp = b.create<func::CallOp>(fusedLoc, outlinedFunc, values);

  for (auto it : llvm::zip(returnVals, callOp->getResults())) {
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }

  // Erase the ops we outlined, which should be safe now.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(trailingOps))) {
    if (op->use_empty()) {
      op->erase();
    }
  }
  assert(anchorOp->use_empty() && "expected 'op' to have no uses");
  anchorOp->erase();
  for (auto &op : llvm::make_early_inc_range(leadingOps)) {
    if (op->use_empty()) {
      op->erase();
    }
  }
}

void Outliner::traceInputs(Operation *op, Operation *ignoreOp,
                           SetVector<Operation *> &predecessors,
                           SetVector<Value> &inputNodes) {
  for (const auto &opnd : op->getOperands()) {
    Operation *usedOp = opnd.getDefiningOp();
    if (usedOp == ignoreOp)
      continue;
    if (usedOp && isLeadingOp(usedOp)) {
      if (predecessors.contains(
              usedOp)) // If already present, move it for new use.
        predecessors.remove(usedOp);
      predecessors.insert(usedOp);
      if (!usedOp->hasTrait<OpTrait::ConstantLike>()) {
        // depth first
        traceInputs(usedOp, op, predecessors, inputNodes);
      }
    } else if (!predecessors.contains(
                   usedOp)) { // special-case consts aren't inputs
      inputNodes.insert(opnd);
    }
  }
}

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
void Outliner::outline(ModuleOp module) {
  auto funcOps = module.getOps<func::FuncOp>();
  for (auto func : llvm::make_early_inc_range(funcOps)) {
    // Don't outline a kernel;  it may already have been outlined.
    if (func->hasAttr(outlineTag))
      continue;

    std::vector<Operation *> anchors;
    auto callback = [&](Operation *op) {
      if (isAnchorOp(op))
        anchors.push_back(op);
    };
    // Gather the anchor ops so we can process them back-to-front.
    func.walk(callback);

    int count = 0;
    // (Problems with node mismatches and unexpected uses if we have the
    // candidates list at module level.)
    std::vector<OutliningCandidate> candidates;
    for (auto &anchorOp : llvm::make_early_inc_range(llvm::reverse(anchors))) {
      auto strCount = std::string("__part_") + std::to_string(count++);

      // Given a Conv2DOp (or other anchor op), gather all the
      // element-wise ops that are reachable from its results,
      // contiguously.
      //
      // The ops after the anchor are "trailing" ops.
      //
      // inputNodes gathers what will become the parameters of the
      // outlined function;  initially it's the anchor's arguments,
      // and it accumulates arguments to other ops that don't come
      // from inside the outlined function.
      //
      // resultNodes will become the results of the outlined function.
      // It starts with the anchor's result(s) and gains the results
      // of each new trailingOp.  When all a resultNode's users can be
      // determined to lie within the outlined function, it's removed
      // from the set.
      //
      // These are SetVectors because we test with contains() a lot,
      // but still want to preserve order.
      SetVector<Operation *> trailingOps;
      SetVector<Value> inputNodes;
      SetVector<Value> resultNodes(anchorOp->getResults().begin(),
                                   anchorOp->getResults().end());

      // Grab a useful set of leading ops, like we do for trailing.
      SetVector<Operation *> leadingOps;
      traceInputs(anchorOp, anchorOp, leadingOps, inputNodes);

      DominanceInfo domInfo(func);
      std::deque<Operation *> worklist; // cuz I want to pull from the front.

      worklist.push_back(anchorOp);
      while (!worklist.empty()) {
        Operation *op = worklist.front();
        worklist.pop_front();
        for (auto *userOp : op->getUsers()) {
          if (isTrailingOp(userOp)) {
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

            // userOp is acceptable.  Keep it as a trailingOp, put it on
            // the worklist.  Add its operands to inputNodes unless
            // they're suitable as leading ops or come from other
            // trailingOps (indicated by being in resultNodes).  If all
            // the users of any resultNode are in trailingOps, there's
            // no need to return it so remove from resultNodes.
            // Finally, add all userOp's results to resultNodes.
            if (!skip) {
              // Also accept inputs to userOp.
              // Put traced ops in leadingOps so they're always ahead of op.
              traceInputs(userOp, op, leadingOps, inputNodes);
              // General case.
              trailingOps.insert(userOp);
              worklist.push_back(userOp);
              for (const Value &val : resultNodes)
                if (llvm::all_of(val.getUsers(), [&](Operation *u) {
                      return trailingOps.contains(u);
                    }))
                  resultNodes.remove(val);
              for (auto res : userOp->getResults())
                resultNodes.insert(res);
            }
          }
        }
      }

      // Make the outlined function from the ops we've gathered.
      outlineOps(anchorOp, trailingOps.getArrayRef(), leadingOps.getArrayRef(),
                 inputNodes.getArrayRef(), resultNodes.getArrayRef(),
                 std::string(func.getSymName()) + strCount, outlineTag,
                 candidates);
    }
  }
}
