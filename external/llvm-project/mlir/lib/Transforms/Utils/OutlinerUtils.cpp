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

#define DEBUG_TYPE "outliner-utility"

#define FUSION_CAPACITY_DEFAULT 32

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
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    if (auto attr = op->getAttr("value"))
      return isZeroAttribute(attr);
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////

// Inspired by / adapted from outlineIfOp() in SCF/Transforms/Utils.cpp
// and mergeIdenticalBlocks() in Utils/RegionUtils.cpp.

class OutliningCandidate {
  SmallVector<Type, FUSION_CAPACITY_DEFAULT> params;
  SmallVector<Operation *, FUSION_CAPACITY_DEFAULT> ops;
  llvm::hash_code hash = 0;
  func::FuncOp function;

  void processOps();
  unsigned addOp(Operation *op, unsigned orderIt);

  /// Return the order index for the given value that is within the block of
  /// this data.
  unsigned getOrderOf(Value value) const;

  /// A map of result producing operations to their relative orders within this
  /// block. The order of an operation is the number of defined values that are
  /// produced within the block before this operation.
  DenseMap<Operation *, unsigned> opOrderIndex;

  bool equivalent(const OutliningCandidate &that) const;
  bool opsMatch(Operation *lhs, Operation *rhs,
                const OutliningCandidate &two) const;

public:
  OutliningCandidate() = default;
  OutliningCandidate(ArrayRef<Operation *> ops_, ArrayRef<Value> params_)
      : ops(ops_) {
    for (auto val : params_)
      params.push_back(val.getType());
    processOps();
  }

  void setFunction(func::FuncOp f) {
    function = f;
    auto type = f.getFunctionType();
    params.assign(type.getInputs().begin(), type.getInputs().end());
    ops.clear();
    for (auto &op : f.getBody().getOps())
      ops.push_back(&op);
    ops.pop_back(); // drop the return
    processOps();
  }

  llvm::hash_code getHash() const { return hash; }
  func::FuncOp getFunction() const { return function; }

  bool operator==(const OutliningCandidate &that) const {
    return hash == that.hash && equivalent(that);
  }
  bool operator!=(const OutliningCandidate &that) const {
    return !operator==(that);
  }
};

/// A DenseMapInfo
namespace llvm {
template <>
struct DenseMapInfo<OutliningCandidate, void> {
  static OutliningCandidate getEmptyKey() { return OutliningCandidate(); }
  static OutliningCandidate getTombstoneKey() { return OutliningCandidate(); }
  static unsigned getHashValue(const OutliningCandidate &val) {
    return val.getHash();
  }
  static bool isEqual(const OutliningCandidate &lhs,
                      const OutliningCandidate &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

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

void OutliningCandidate::processOps() {
  hash = 0;
  opOrderIndex.clear();
  unsigned orderIt = params.size();
  for (auto *op : ops) {
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

bool OutliningCandidate::opsMatch(Operation *lhs, Operation *rhs,
                                  const OutliningCandidate &two) const {
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
    if (getOrderOf(lhsOperand) != two.getOrderOf(rhsOperand)) {
      return false;
    }
  }

  return true;
}

bool OutliningCandidate::equivalent(const OutliningCandidate &that) const {
  if (hash == that.hash) {
    if (params.size() != that.params.size() || ops.size() != that.ops.size()) {
      return false;
    }
    for (auto params : llvm::zip(params, that.params)) {
      if (std::get<0>(params) != std::get<1>(params)) {
        return false;
      }
    }
    // get Result Types

    for (auto opPair : llvm::zip(ops, that.ops)) {
      if (!opsMatch(std::get<0>(opPair), std::get<1>(opPair), that)) {
        return false;
      }
    }
    return true;
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////
////   OutlineBuilder
////   1) Collect all ops that can be fused
////   2) Look for match in Candidate Set
////   3) If no match, build outline func
////   4) Replace ops with call to outline func
////////////////////////////////////////////////////////////////////////////////////

class OutlineBuilder {

  // Debug messaging for traversal
  inline void debug(const char *tag, Operation *op) const {
    LLVM_DEBUG(llvm::dbgs() << tag << ": " << op << "\n");
  }

  inline void debug(const char *tag, Value v) const {
    LLVM_DEBUG(llvm::dbgs() << tag << ": "; v.print(
        llvm::dbgs(), OpPrintingFlags().elideLargeElementsAttrs());
               llvm::dbgs() << "\n");
  }

  // Add Value/Operation methods
  void addInput(Value v) {
    debug("INPUT", v);
    _inputs.insert(v);
  }

  void addResult(Value v) {
    debug("RESULT", v);
    _results.insert(v);
  }

  void addPredecessor(Operation *op) {
    if (!_leadingOps.contains(op)) {
      assert(!_trailingOps.contains(op));
      debug("PRED", op);
      _leadingOps.insert(op);
      _locs.push_back(op->getLoc());
    }
  }

  void addSuccessor(Operation *op) {
    if (!_trailingOps.contains(op)) {
      debug("SUCC", op);
      _trailingOps.insert(op);
      _locs.push_back(op->getLoc());
    }
  }

  bool contains(Operation *op) const {
    return _trailingOps.contains(op) || _leadingOps.contains(op);
  }

  // Recurse back full chain looking for anchorOp
  // - uses encountered set to reduce time
  bool inChainToAnchor(Operation *inOp,
                       DenseSet<Operation *> &encountered) const {
    if (inOp == nullptr)
      return false;
    else if (inOp == _anchorOp)
      return true;
    else if (encountered.contains(inOp))
      return false;
    encountered.insert(inOp);
    for (auto opr : inOp->getOperands()) {
      if (Operation *oprOp = opr.getDefiningOp()) {
        debug(">>>", oprOp);
        if (inChainToAnchor(oprOp, encountered))
          return true;
        debug("<<<", oprOp);
      }
    }
    return false;
  }

  // Make sure all operands are external or immediate in the chain from anchorOp
  bool isImmediateOp(Operation *inOp) const {
    for (auto opr : inOp->getOperands()) {
      if (Operation *oprOp = opr.getDefiningOp()) {
        debug("IN>", oprOp);
        DenseSet<Operation *> encountered;
        if (!contains(oprOp) && inChainToAnchor(oprOp, encountered))
          return false;
      }
    }
    return true;
  }

  // Recurse all inputs to op and capture predecessor ops if qualified
  void collectInputs(Operation *op, Operation *ignoreOp) {
    for (const auto &operand : op->getOperands()) {
      Operation *usedOp = operand.getDefiningOp();
      if (usedOp) {
        // Ignore input from anchor chain or already collected
        if (usedOp != ignoreOp && !contains(usedOp)) {
          if (_outliner.isLeadingOp(usedOp)) {
            // Depth first collection of Leading ops
            collectInputs(usedOp, op);
            addPredecessor(usedOp);
          } else if (!contains(usedOp)) {
            // Capture as input if not already captured as Trailing
            addInput(operand);
          }
        }
      } else {
        // Block parameter
        addInput(operand);
      }
    }
  }

public:
  OutlineBuilder(Outliner &outliner, Operation *anchorOp)
      : _outliner(outliner), _anchorOp(anchorOp) {
    _anchorBlock = anchorOp->getBlock();
    debug("ANCHOR", anchorOp);
    collectInputs(_anchorOp, _anchorOp);

    addSuccessor(_anchorOp);
  }

  ~OutlineBuilder() {
    // Erase the ops we outlined, which should be safe now.
    for (auto &op : llvm::make_early_inc_range(llvm::reverse(_leadingOps))) {
      if (op->use_empty())
        op->erase();
    }
  }

  bool collect() {
    // Given a Conv2DOp (or other anchor op), gather all Leading
    // and Trailing ops in the PBV chain up to and including Terminal
    // ops.
    //
    // _inputs gathers what will become the parameters of the
    // outlined function;  initially it's the anchor's arguments,
    // and it accumulates arguments to other ops that don't come
    // from inside the outlined function.
    //
    // _results will become the results of the outlined function.
    // These are gathered after all ops have been collected, if any
    // op has external uses.

    // BFS
    std::deque<Operation *> worklist;
    worklist.push_back(_anchorOp);
    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop_front();
      for (auto *userOp : op->getUsers()) {
        if (userOp->getBlock() == _anchorBlock && !contains(userOp)) {
          bool isTerminal = _outliner.isTerminatingOp(userOp);
          if ((_outliner.isTrailingOp(userOp) || isTerminal) &&
              isImmediateOp(userOp)) {
            addSuccessor(userOp);
            // Collect all inputs other than anchor-chain
            collectInputs(userOp, op);
            // Traverse if not a Terminal Op
            if (!isTerminal)
              worklist.push_back(userOp);
          }
        }
      }
    }

    // capture all result ops after anchor op that have external uses
    for (auto *op : _trailingOps) {
      for (auto res : op->getResults()) {
        if (!llvm::all_of(res.getUsers(),
                          [&](Operation *u) { return contains(u); }))
          addResult(res);
      }
    }

    // concat all ops into 1 list
    _leadingOps.insert(_trailingOps.begin(), _trailingOps.end());

    return true;
  }

  // Given an op and its fuse-able trailing (second) and leading
  // (front) ops, remove them into a separate function.
  func::FuncOp lookupOrCreate(DenseSet<OutliningCandidate> &candidates,
                              StringRef attrName) const {
    auto candidatePair = candidates.insert(
        OutliningCandidate(_leadingOps.getArrayRef(), _inputs.getArrayRef()));
    auto candidate = std::get<0>(candidatePair);
    if (std::get<1>(candidatePair)) {
      // Build outlined func.
      // And update the candidate with the cloned ops for comparison.
      candidate->setFunction(build(candidates.size() - 1, attrName));
    }
    return candidate->getFunction();
  }

  // Build the outlined function.
  func::FuncOp build(uint32_t idx, StringRef attrName) const {
    func::FuncOp anchorFunc = _anchorOp->getParentOfType<func::FuncOp>();
    ValueRange inputs(_inputs.getArrayRef());
    OpBuilder b(anchorFunc);
    Location loc = _anchorOp->getLoc();
    MLIRContext *ctx = _anchorOp->getContext();

    auto partFnName =
        anchorFunc.getSymName().str() + "__part_" + std::to_string(idx);

    // Make FuncOp from anchorOp's operand types and trailingOp's result type.
    ValueRange results(_results.getArrayRef());
    FunctionType type =
        FunctionType::get(ctx, inputs.getTypes(), results.getTypes());
    SmallVector<NamedAttribute, 1> kernelAttrs{
        b.getNamedAttr(attrName, b.getUnitAttr()),
    };
    func::FuncOp outlinedFunc = b.create<func::FuncOp>(
        loc, partFnName, type, ArrayRef<NamedAttribute>(kernelAttrs));
    outlinedFunc->setAttr("sym_visibility", StringAttr::get(ctx, "private"));

    // Add access modes for parameters: read-only, write-only, read-write
    // All MemRef params are marked as 'read-write'
    // Non-MemRef inputs are added as 'read-only'
    auto readAttr =
        b.getNamedAttr(func::FuncOp::getReadAccessAttrName(), b.getUnitAttr());
    auto writeAttr =
        b.getNamedAttr(func::FuncOp::getWriteAccessAttrName(), b.getUnitAttr());
    auto getAccessAttrs = [&](Type t,
                              bool inputs) -> std::optional<DictionaryAttr> {
      if (t.isa<VectorType, RankedTensorType, UnrankedTensorType>())
        return b.getDictionaryAttr({inputs ? readAttr : writeAttr});
      if (t.isa<MemRefType>())
        return b.getDictionaryAttr({readAttr, writeAttr});
      return {};
    };

    // Non-MemRef inputs are added as 'read-only'
    for (auto pair : llvm::enumerate(inputs)) {
      if (auto attrs = getAccessAttrs(pair.value().getType(), true))
        outlinedFunc.setArgAttrs(pair.index(), *attrs);
    }
    // Non-MemRef results are added as 'write-only'
    for (auto pair : llvm::enumerate(results)) {
      if (auto attrs = getAccessAttrs(pair.value().getType(), false))
        outlinedFunc.setResultAttrs(pair.index(), *attrs);
    }

    // Clone collected ops into the body of the new function.
    b.setInsertionPointToStart(outlinedFunc.addEntryBlock());
    IRMapping bvm;
    for (auto it : llvm::zip(inputs, outlinedFunc.getArguments()))
      bvm.map(std::get<0>(it), std::get<1>(it));

    for (auto *op : _leadingOps) {
      b.clone(*op, bvm);
    }

    // Make ReturnOp from trailingOps' results.
    SmallVector<Value> returnOperands;
    for (auto res : _results) {
      returnOperands.push_back(bvm.lookup(res));
    }
    // Can't also supply return types, because it'll see a mismatch
    // in numbers where there isn't one.
    b.create<func::ReturnOp>(loc, returnOperands);

    return outlinedFunc;
  }

  // Given an op and its fuse-able trailing (second) and leading
  // (front) ops, remove them into a separate function.
  void makeCall(func::FuncOp outlinedFunc) {
    OpBuilder b(_anchorOp);
    MLIRContext *ctx = _anchorOp->getContext();
    Location fusedLoc = FusedLoc::get(ctx, _locs);
    // ------------------------------------------------------------
    // Replacement part.

    // Replace anchorOp, trailingOps, and leadingOps with CallOp to new
    // function. ? Look for earliest result ?
    b.setInsertionPointAfter(_leadingOps.back());
    func::CallOp callOp =
        b.create<func::CallOp>(fusedLoc, outlinedFunc, _inputs.getArrayRef());

    for (auto it : llvm::zip(_results, callOp->getResults())) {
      std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
    }
  }

private:
  Outliner &_outliner;

  Operation *_anchorOp;
  Block *_anchorBlock;

  SmallVector<Location, FUSION_CAPACITY_DEFAULT> _locs;

  typedef SmallVector<Value, FUSION_CAPACITY_DEFAULT> ValVec;
  typedef SmallVector<Operation *, FUSION_CAPACITY_DEFAULT> OpVec;
  SetVector<Value, ValVec> _inputs;
  SetVector<Operation *, OpVec> _leadingOps;
  SetVector<Operation *, OpVec> _trailingOps;
  SetVector<Value, ValVec> _results;
};

// Walk each func outlining fusion opportunities, replacing with a call
// to a new (or matching) function containing the functionality. The call
// will be annotated the loc's of all fused ops.
void Outliner::outline(ModuleOp module) {
  auto funcOps = module.getOps<func::FuncOp>();

  for (auto func : llvm::make_early_inc_range(funcOps)) {
    // Don't outline a kernel;  it may already have been outlined.
    if (func->hasAttr(outlineTag))
      continue;

    bool hasMemrefs = false;
    std::vector<Operation *> anchors;
    auto callback = [&](Operation *op) {
      for (auto operand : op->getOperands()) {
        if (isa<MemRefType>(operand.getType()))
          hasMemrefs = true;
      }
      for (auto result : op->getResults()) {
        if (isa<MemRefType>(result.getType()))
          hasMemrefs = true;
      }
      if (isAnchorOp(op))
        anchors.push_back(op);
    };
    // Gather the anchor ops so we can process them back-to-front.
    func.walk(callback);

    if (!hasMemrefs) {
      // (Problems with node mismatches and unexpected uses if we have the
      // candidates list at module level.)
      DenseSet<OutliningCandidate> candidates;
      for (auto anchorOp : llvm::make_early_inc_range(llvm::reverse(anchors))) {
        // Create an OutlineBuilder for each anchor op.
        OutlineBuilder builder(*this, anchorOp);
        builder.collect();

        // Make the outlined function from the ops we've gathered.
        auto outlinedFunc = builder.lookupOrCreate(candidates, outlineTag);
        builder.makeCall(outlinedFunc);
      }
    }
  }
}
