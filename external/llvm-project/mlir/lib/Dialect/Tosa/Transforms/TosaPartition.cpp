//===- TosaPartition.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replace conv2d followed by elementwise op with call to function containing
// them.  Generalised, outline any anchor op, all its trailing elementwise ops,
// and all its leading elementwise ops.  (Where "elementwise" itself is
// generalised to include transpose and reshape ops.)
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
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

// TODO(kdrewnia): Make it so list options can have defaults, and then get rid
// of needing to set defaults here
namespace mlir {
namespace tosa {
#define GEN_PASS_DEF_TOSAPARTITION
#include "mlir/Dialect/Tosa/Transforms/Passes.h.inc"
} // namespace tosa
} // namespace mlir

using namespace mlir;
using namespace mlir::tosa;

namespace {

class TosaPartitionPass
    : public tosa::impl::TosaPartitionBase<TosaPartitionPass> {
public:
  using tosa::impl::TosaPartitionBase<TosaPartitionPass>::TosaPartitionBase;

  bool isAnchorOp(Operation *op);
  bool isTransposeOp(Operation *op) const;
  bool isLeadingOp(Operation *op) const;
  bool isTrailingOp(Operation *op) const;
  StringRef partitionTag() const;
  void traceInputs(Operation *op, Operation *ignoreOp,
                   SetVector<Operation *> &predecessors,
                   SetVector<Value> &inputNodes);
  void runOnOperation() override;
};

// Tosa ops can broadcast values along axes, which allows for
// element-wise operations without fully-matching dimensions.  The
// Elementwise trait is strict about matching dimensions, but
// broadcastable ops are also element-wise, and we know that an
// additional set of ops are also element-wise.
bool isElementwiseOp(Operation *op) {
  return op->hasTrait<OpTrait::Elementwise>() ||
         op->hasTrait<OpTrait::ResultsBroadcastableShape>() ||
         // clang-format off
    isa<tosa::CastOp,
        tosa::ClampOp,
        tosa::SigmoidOp,
        tosa::TanhOp,
// ResultsBroadcastableShape
//         tosa::AddOp,
//         tosa::ArithmeticRightShiftOp,
//         tosa::BitwiseAndOp,
//         tosa::BitwiseOrOp,
//         tosa::BitwiseXorOp,
//         tosa::DivOp,
//         tosa::LogicalAndOp,
//         tosa::LogicalLeftShiftOp,
//         tosa::LogicalRightShiftOp,
//         tosa::LogicalOrOp,
//         tosa::LogicalXorOp,
//         tosa::MaximumOp,
//         tosa::MinimumOp,
//         tosa::MulOp,
//         tosa::PowOp,
//         tosa::SubOp,
        tosa::AbsOp,
//         tosa::BitwiseNotOp,
        tosa::CeilOp,
        tosa::ClzOp,
        tosa::ExpOp,
        tosa::FloorOp,
        tosa::LogOp,
        tosa::LogicalNotOp,
        tosa::NegateOp,
        tosa::ReciprocalOp,
        tosa::RsqrtOp,
        tosa::SelectOp
//         tosa::EqualOp,
//         tosa::GreaterOp,
//         tosa::GreaterEqualOp
       >(op);
  // clang-format on
}

bool isFuseableOp(Operation *op) { return isElementwiseOp(op); }

bool isZeroAttribute(Attribute value) {
  if (auto intValue = value.dyn_cast<IntegerAttr>())
    return intValue.getValue().isNullValue();
  if (auto fpValue = value.dyn_cast<FloatAttr>())
    return fpValue.getValue().isZero();
  if (auto splatValue = value.dyn_cast<SplatElementsAttr>())
    return isZeroAttribute(splatValue.getSplatValue<Attribute>());
  if (auto elementsValue = value.dyn_cast<ElementsAttr>())
    return llvm::all_of(elementsValue.getValues<Attribute>(), isZeroAttribute);
  if (auto arrayValue = value.dyn_cast<ArrayAttr>())
    return llvm::all_of(arrayValue.getValue(), isZeroAttribute);
  return false;
}

bool isConstantZero(Operation *op) {
  // test for zero
  // test for zero
  if (auto cst = dyn_cast<arith::ConstantOp>(op))
    return isZeroAttribute(cst.getValue());
  else if (auto cst = dyn_cast<tosa::ConstOp>(op))
    return isZeroAttribute(cst->getAttr("value"));
  return false;
}

bool isSmallishConstant(Operation *op) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    // In TOSA, should always be tensor and thus shaped, but just in case.
    if (auto cstType = op->getResult(0).getType().dyn_cast<ShapedType>())
      if (cstType.hasStaticShape() && cstType.getNumElements() <= 8)
        return true;
  return false;
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp
// and mergeIdenticalBlocks() in Utils/RegionUtils.cpp.

struct OutliningCandidate {
  OutliningCandidate(Operation *anchorOp_, ArrayRef<Operation *> &trailingOps_,
                     ArrayRef<Operation *> &leadingOps_,
                     ArrayRef<Value> &params_, ArrayRef<Value> &returnVals_,
                     StringRef partFnName_);

  unsigned addOp(Operation *op, unsigned orderIt);

  Operation *anchorOp;
  SmallVector<Operation *> trailingOps;
  SmallVector<Operation *> leadingOps;
  SmallVector<Value> params;
  SmallVector<Value> returnVals;
  std::string partFnName;
  llvm::hash_code hash;
  func::FuncOp function;

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
    : anchorOp(anchorOp_), partFnName(partFnName_), hash(0), function(nullptr) {
  // We'll need to grab the cloned ops to avoid use-after-free.
  for (auto *op : trailingOps_) {
    trailingOps.push_back(op);
  }
  for (auto *op : leadingOps_) {
    leadingOps.push_back(op);
  }
  for (auto val : params_) {
    params.push_back(val);
  }
  for (auto val : returnVals_) {
    returnVals.push_back(val);
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
void outlinePartitionOps(Operation *anchorOp, ArrayRef<Operation *> trailingOps,
                         ArrayRef<Operation *> leadingOps,
                         ArrayRef<Value> params, ArrayRef<Value> returnVals,
                         StringRef partFnName, StringRef attrName,
                         std::vector<OutliningCandidate> &candidates) {
  ValueRange values(params);
  OpBuilder b(anchorOp);
  Location loc = anchorOp->getLoc();
  func::FuncOp outlinedFunc;

  // ------------------------------------------------------------
  // Merging part.

  OutliningCandidate newCandidate(anchorOp, trailingOps, leadingOps, params,
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

    newCandidate.leadingOps.clear();
    for (auto *op : llvm::reverse(leadingOps)) {
      newCandidate.leadingOps.push_back(b.clone(*op, bvm));
      newCandidate.opOrderIndex[newCandidate.leadingOps.back()] =
          newCandidate.opOrderIndex[op];
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
    }

    // Make ReturnOp from trailingOps' results.
    SmallVector<Value> returnOperands;
    for (auto op : returnVals) {
      returnOperands.push_back(bvm.lookup(op));
    }
    // Can't also supply return types, because it'll see a mismatch
    // in numbers where there isn't one.
    b.create<func::ReturnOp>(loc, returnOperands);

    candidates.push_back(newCandidate);
  }

  // ------------------------------------------------------------
  // Replacement part.

  // Replace anchorOp, trailingOps, and leadingOps with CallOp to new function.
  Operation *lastOp = anchorOp;
  if (!trailingOps.empty())
    lastOp = trailingOps[trailingOps.size() - 1];
  b.setInsertionPointAfter(lastOp);
  func::CallOp callOp = b.create<func::CallOp>(loc, outlinedFunc, values);

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

} // namespace

bool TosaPartitionPass::isAnchorOp(Operation *op) {
  if (anchorOps.empty()) // ListOption doesn't have a default value.
    anchorOps = {"tosa.conv2d", "tosa.matmul", "tosa.depthwise_conv2d",
                 "tosa.fully_connected"};
  return llvm::is_contained(anchorOps, op->getName().getIdentifier().str());
}

bool TosaPartitionPass::isTransposeOp(Operation *op) const {
  return isa<tosa::TransposeOp, tosa::ReshapeOp>(op);
}

bool TosaPartitionPass::isLeadingOp(Operation *op) const {
  return isConstantZero(op) || isSmallishConstant(op) || isTransposeOp(op) ||
         (!trailingOnly && isFuseableOp(op));
}

bool TosaPartitionPass::isTrailingOp(Operation *op) const {
  return isTransposeOp(op) || isFuseableOp(op);
}

StringRef TosaPartitionPass::partitionTag() const { return partitionTagOpt; }

void TosaPartitionPass::traceInputs(Operation *op, Operation *ignoreOp,
                                    SetVector<Operation *> &predecessors,
                                    SetVector<Value> &inputNodes) {
  for (const auto &opnd : op->getOperands()) {
    Operation *usedOp = opnd.getDefiningOp();
    if (usedOp != ignoreOp) {
      if (usedOp && (isTransposeOp(op) ? isSmallishConstant(usedOp)
                                       : isLeadingOp(usedOp))) {
        if (predecessors.contains(
                usedOp)) // If already present, move it for new use.
          predecessors.remove(usedOp);
        predecessors.insert(usedOp);
        if (!usedOp->hasTrait<OpTrait::ConstantLike>()) {
          // depth first
          traceInputs(usedOp, op, predecessors, inputNodes);
        }
      } else if (!usedOp || !predecessors.contains(
                                usedOp)) { // special-case consts aren't inputs
        inputNodes.insert(opnd);
      }
    }
  }
}

// Inspired by / adapted from TestSCFIfUtilsPass in
// test/lib/Transforms/TestSCFUtils.cpp.
void TosaPartitionPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto funcOps = module.getOps<func::FuncOp>();
  for (auto func : llvm::make_early_inc_range(funcOps)) {
    // Don't partition a kernel;  it may be already partitioned.
    if (func->hasAttr(partitionTag()))
      continue;

    int count = 0;
    // (Problems with node mismatches and unexpected uses if we have the
    // candidates list at module level.)
    std::vector<OutliningCandidate> candidates;
    auto callback = [&](Operation *op) {
      if (!isAnchorOp(op))
        return false;
      Operation *anchorOp = op;
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

            // userOp is acceptable.  Keep it as a trailingOp, put it on the
            // worklist.  Add its operands to inputNodes unless they come
            // from other trailingOps (indicated by being in resultNodes).
            // If all the users of any resultNode are in trailingOps, there's
            // no need to return it so remove from resultNodes.  Finally,
            // add all userOp's results to resultNodes.
            if (!skip) {
              // Also accept small constant inputs to userOp.
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
      outlinePartitionOps(anchorOp, trailingOps.getArrayRef(),
                          leadingOps.getArrayRef(), inputNodes.getArrayRef(),
                          resultNodes.getArrayRef(),
                          std::string(func.getSymName()) + strCount,
                          partitionTag(), candidates);
      // Outlining will erase nodes and thus perturb the walk, so
      // signal interrupted to exit it and restart.
      return true;
    };

    // Walk until we've outlined all the anchor ops we can.
    auto walkBackward = [&](Block &b) -> bool {
      for (auto it = b.rbegin(); it != b.rend(); ++it) {
        if (callback(&*it))
          return true;
      }
      return false;
    };
    for (auto &block : func.getBody()) {
      while (walkBackward(block)) {
      }
    }
  }
}
