//===- PropagateAlias - MLIR Rock ops lowering passes -----===//
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
// This pass propagates rock.noalias_view into memref.load, memref.store and
// other operations
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/Passes.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPROPAGATEALIASPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-propagate-alias"

using namespace mlir;
using namespace mlir::rock;

namespace {
struct RockPropagateAliasPass
    : public rock::impl::RockPropagateAliasPassBase<RockPropagateAliasPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Trace a memref value back to its NoAliasViewOp.
static Operation *
traceToNoAliasViewOrSelect(Value memref, DenseMap<Value, Operation *> &cache) {
  auto cached = cache.find(memref);
  if (cached != cache.end())
    return cached->second;
  Operation *res = nullptr;
  if (auto view = memref.getDefiningOp<rock::NoAliasViewOp>())
    res = view;
  else if (auto view = memref.getDefiningOp<ViewLikeOpInterface>())
    res = traceToNoAliasViewOrSelect(view.getViewSource(), cache);
  else if (auto select = memref.getDefiningOp<arith::SelectOp>())
    res = select;

  cache.insert({memref, res});
  return res;
}

static LogicalResult propagateAlias(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  // see RockPrepareLLVM for similar alias handling for global memory
  llvm::SmallDenseMap<Operation *, ArrayAttr> aliasScopes;
  auto domain = rewriter.getAttr<LLVM::AliasScopeDomainAttr>(
      rewriter.getStringAttr("LDS"));

  // create alias scopes for NoAliasViewOp
  func.walk([&](rock::NoAliasViewOp noAliasView) {
    gpu::AddressSpaceAttr memSpaceAttr =
        dyn_cast_or_null<gpu::AddressSpaceAttr>(
            noAliasView.getType().getMemorySpace());
    if (memSpaceAttr &&
        memSpaceAttr.getValue() == gpu::AddressSpace::Workgroup) {
      auto aliasScope = LLVM::AliasScopeAttr::get(
          domain, rewriter.getStringAttr("LDS" + Twine(aliasScopes.size())));
      aliasScopes[noAliasView] = rewriter.getArrayAttr(aliasScope);
    }
  });
  LLVM_DEBUG(llvm::dbgs() << "Found " << aliasScopes.size()
                          << " NoAliasViewOps\n");

  // find arith.selects of LDS buffers (from rock.extract_multibuffer)
  llvm::DenseMap<Value, Operation *> cache;
  func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
    // We will make the simplifying assumption that the last memref-valued
    // operand to the operation is the memref being accessed.
    Operation *aliasOp = aliasIface.getOperation();
    Value memref;
    for (Value arg : aliasOp->getOperands()) {
      if (auto argMemref = dyn_cast<MemRefType>(arg.getType())) {
        gpu::AddressSpaceAttr memSpaceAttr =
            dyn_cast_or_null<gpu::AddressSpaceAttr>(argMemref.getMemorySpace());
        if (memSpaceAttr &&
            memSpaceAttr.getValue() == gpu::AddressSpace::Workgroup)
          memref = arg;
      }
    }
    if (!memref)
      return;

    if (Operation *op = traceToNoAliasViewOrSelect(memref, cache)) {
      if (isa<arith::SelectOp>(op)) {
        auto aliasScope = LLVM::AliasScopeAttr::get(
            domain, rewriter.getStringAttr("LDS" + Twine(aliasScopes.size())));
        aliasScopes[op] = rewriter.getArrayAttr(aliasScope);
      }
    }
  });

  // create noalias scopes for alias scopes
  llvm::SmallDenseMap<Operation *, ArrayAttr> noaliasScopes;
  noaliasScopes.reserve(aliasScopes.size());
  {
    SmallVector<Attribute> allButOneScope;
    allButOneScope.reserve(aliasScopes.size());
    for (auto [op, _] : aliasScopes) {
      for (auto [secondOp, aliasInfo] : aliasScopes) {
        if (op != secondOp)
          allButOneScope.push_back(aliasInfo[0]);
      }
      noaliasScopes[op] = rewriter.getArrayAttr(allButOneScope);
      allButOneScope.clear();
    }
  }
  assert(aliasScopes.size() == noaliasScopes.size());

  {
    // The alias analysis interface will pick up all ops that write or load
    func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
      // We will make the simplifying assumption that the last memref-valued
      // operand to the operation is the memref being accessed.
      Operation *aliasOp = aliasIface.getOperation();
      Value memref;
      for (Value arg : aliasOp->getOperands()) {
        if (auto argMemref = dyn_cast<MemRefType>(arg.getType())) {
          gpu::AddressSpaceAttr memSpaceAttr =
              dyn_cast_or_null<gpu::AddressSpaceAttr>(
                  argMemref.getMemorySpace());
          if (memSpaceAttr &&
              memSpaceAttr.getValue() == gpu::AddressSpace::Workgroup)
            memref = arg;
        }
      }
      if (!memref)
        return;

      if (Operation *op = traceToNoAliasViewOrSelect(memref, cache)) {
        if (aliasScopes.contains(op)) {
          assert(aliasScopes.contains(op) && noaliasScopes.contains(op));

          aliasIface.setAliasScopes(aliasScopes[op]);
          aliasIface.setNoAliasScopes(noaliasScopes[op]);
        }
      }
    });
  }

  // finally, rewrite NoAliasViewOp as ViewOp
  for (auto [op, _] : aliasScopes) {
    if (auto view = dyn_cast<rock::NoAliasViewOp>(op)) {
      rewriter.setInsertionPointAfter(view);
      rewriter.replaceOpWithNewOp<memref::ViewOp>(
          view, view.getType(), view.getSource(), view.getByteShift(),
          view.getSizes());
    }
  }

  return success();
}

void RockPropagateAliasPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel"))
    return;

  if (failed(propagateAlias(func))) {
    return signalPassFailure();
  }
}
