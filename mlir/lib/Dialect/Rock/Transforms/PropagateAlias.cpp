//===- PropagateAlias - MLIR Rock ops lowering passes -----===//
//
// Copyright 2024 The MLIR Authors.
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

// Trace a pointer value back to its function argument. Only returns arguments
// of pointer type that are of flat or global type.
static BlockArgument traceToArg(Value memref, func::FuncOp func,
                                DenseMap<Value, BlockArgument> &cache) {
  auto cached = cache.find(memref);
  if (cached != cache.end())
    return cached->second;
  BlockArgument res = nullptr;
  if (auto cast = memref.getDefiningOp<memref::MemorySpaceCastOp>())
    res = traceToArg(cast.getSource(), func, cache);
  else if (auto arg = dyn_cast<BlockArgument>(memref)) {
    if (arg.getOwner() == &func.front())
      res = arg;
  }

  cache.insert({memref, res});
  return res;
}

// Trace a memref value back to its NoAliasViewOp. Only returns arguments
// of pointer type that are of flat or global type.
static NoAliasViewOp traceToNoAliasView(Value memref,
                                        DenseMap<Value, NoAliasViewOp> &cache) {
  auto cached = cache.find(memref);
  if (cached != cache.end())
    return cached->second;
  NoAliasViewOp res = nullptr;
  if (auto view = memref.getDefiningOp<memref::ViewOp>())
    res = traceToNoAliasView(view.getSource(), cache);
  else if (auto subView = memref.getDefiningOp<memref::SubViewOp>())
    res = traceToNoAliasView(subView.getSource(), cache);
  else if (auto view = memref.getDefiningOp<NoAliasViewOp>()) {
    res = view;
  }

  cache.insert({memref, res});
  return res;
}

static LogicalResult propagateAlias(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  SmallVector<NoAliasViewOp> views;
  // see RockPrepareLLVM for similar alias handling for global memory
  llvm::SmallDenseMap<NoAliasViewOp, ArrayAttr> aliasScopes;
  auto domain = rewriter.getAttr<LLVM::AliasScopeDomainAttr>(
      rewriter.getStringAttr(func.getSymName()));

  func.walk([&](NoAliasViewOp viewOp) {
    auto aliasScope = LLVM::AliasScopeAttr::get(
        domain, rewriter.getStringAttr("noAliasView" + Twine(views.size())));
    aliasScopes[viewOp] = rewriter.getArrayAttr(aliasScope);
    views.push_back(viewOp);
  });

  LLVM_DEBUG(llvm::dbgs() << "Found " << func.getNumArguments()
                          << " function arguments\n");
  llvm::SmallDenseMap<size_t, ArrayAttr> argAliasScopes;
  for (size_t i = 0; i < func.getNumArguments(); ++i) {
    if (isa<MemRefType>(func.getArgument(i).getType())) {
      auto aliasScope = LLVM::AliasScopeAttr::get(
          domain, rewriter.getStringAttr("arg" + Twine(i)));
      argAliasScopes[i] = rewriter.getArrayAttr(aliasScope);
    }
  }

  llvm::SmallDenseMap<size_t, ArrayAttr> argNoaliasScopes;
  argNoaliasScopes.reserve(argAliasScopes.size());
  {
    SmallVector<Attribute> allButOneScope;
    allButOneScope.reserve(argAliasScopes.size());
    for (auto [arg, _] : argAliasScopes) {
      for (auto [secondArg, aliasInfo] : argAliasScopes) {
        if (arg != secondArg)
          allButOneScope.push_back(aliasInfo[0]);
      }
      for (auto [_, scope] : aliasScopes) {
        allButOneScope.push_back(scope[0]);
      }
      argNoaliasScopes[arg] = rewriter.getArrayAttr(allButOneScope);
      allButOneScope.clear();
    }
  }

  llvm::SmallDenseMap<NoAliasViewOp, ArrayAttr> noaliasScopes;
  size_t n = aliasScopes.size();
  LLVM_DEBUG(llvm::dbgs() << "Found " << n << " NoAliasViewOp\n");
  noaliasScopes.reserve(n);
  {
    SmallVector<Attribute> allButOneScope;
    allButOneScope.reserve(n);
    for (auto [view, _] : aliasScopes) {
      for (auto [secondView, aliasInfo] : aliasScopes) {
        if (view != secondView)
          allButOneScope.push_back(aliasInfo[0]);
      }
      for (auto [_, argScope] : argAliasScopes) {
        allButOneScope.push_back(argScope[0]);
      }
      noaliasScopes[view] = rewriter.getArrayAttr(allButOneScope);
      allButOneScope.clear();
    }
  }

  {
    llvm::DenseMap<Value, BlockArgument> cache;
    // The alias analysis interface will pick up all ops that write or load
    func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
      // We will make the simplyfying assumption that the first pointer-valued
      // operand to the operation is the pointer being accessed.
      Operation *aliasOp = aliasIface.getOperation();
      Value memref;
      for (Value arg : aliasOp->getOperands()) {
        if (isa<MemRefType>(arg.getType())) {
          memref = arg;
          break;
        }
      }
      if (!memref)
        return;
      BlockArgument funcArg = traceToArg(memref, func, cache);
      if (!funcArg)
        return;
      unsigned argNo = funcArg.getArgNumber();
      aliasIface.setAliasScopes(argAliasScopes[argNo]);
      aliasIface.setNoAliasScopes(argNoaliasScopes[argNo]);
    });
  }

  {
    llvm::DenseMap<Value, NoAliasViewOp> cache;
    // The alias analysis interface will pick up all ops that write or load
    func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
      // We will make the simplyfying assumption that the last pointer-valued
      // operand to the operation is the pointer being accessed.
      Operation *aliasOp = aliasIface.getOperation();
      Value memref;
      for (Value arg : aliasOp->getOperands()) {
        if (isa<MemRefType>(arg.getType())) {
          memref = arg;
        }
      }
      if (!memref)
        return;
      NoAliasViewOp viewOp = traceToNoAliasView(memref, cache);
      if (!viewOp)
        return;
      aliasIface.setAliasScopes(aliasScopes[viewOp]);
      aliasIface.setNoAliasScopes(noaliasScopes[viewOp]);
    });
  }

  // finally, rewrite NoAliasViewOp as ViewOp
  for (NoAliasViewOp view : views) {
    rewriter.setInsertionPointAfter(view);
    rewriter.replaceOpWithNewOp<memref::ViewOp>(
        view, view.getType(), view.getSource(), view.getByteShift(),
        view.getSizes());
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
