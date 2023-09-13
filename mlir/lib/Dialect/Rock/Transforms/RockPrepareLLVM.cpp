//===- RockPrepareLLVM.cpp - prepares the generated code for LLVM       ---===//
//
// Copyright 2022 AMD
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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallBitVector.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKPREPARELLVMPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-prepare-llvm"

using namespace mlir;

namespace {
struct RockPrepareLLVMPass
    : public rock::impl::RockPrepareLLVMPassBase<RockPrepareLLVMPass> {
  void runOnOperation() override;
};
} // end namespace

// Trace a pointer value back to its function argument. Only returns arguments
// of pointer type that are of flat or global type.
static BlockArgument traceToArg(Value pointer, LLVM::LLVMFuncOp func,
                                DenseMap<Value, BlockArgument> &cache) {
  auto cached = cache.find(pointer);
  if (cached != cache.end())
    return cached->second;
  BlockArgument res = nullptr;
  if (auto cast = pointer.getDefiningOp<LLVM::AddrSpaceCastOp>())
    res = traceToArg(cast.getArg(), func, cache);
  else if (auto asBuffer = pointer.getDefiningOp<ROCDL::MakeBufferRsrcOp>())
    res = traceToArg(asBuffer.getBase(), func, cache);
  else if (auto gep = pointer.getDefiningOp<LLVM::GEPOp>())
    res = traceToArg(gep.getBase(), func, cache);
  else if (auto arg = dyn_cast<BlockArgument>(pointer)) {
    auto ptrType = dyn_cast<LLVM::LLVMPointerType>(arg.getType());
    unsigned addrSpace = ~0;
    if (ptrType)
      addrSpace = ptrType.getAddressSpace();
    if (arg.getOwner() == &func.front() && (addrSpace == 0 || addrSpace == 1))
      res = arg;
  }

  cache.insert({pointer, res});
  return res;
}

static int64_t getAlign(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  int64_t byteWidth = elemType.getIntOrFloatBitWidth() / 8;
  if (auto vecType = dyn_cast<VectorType>(type))
    byteWidth *= vecType.getNumElements();
  return std::min(static_cast<int64_t>(16), byteWidth);
}

void RockPrepareLLVMPass::runOnOperation() {
  LLVM::LLVMFuncOp func = getOperation();
  if (!func->hasAttr(ROCDL::ROCDLDialect::getKernelFuncAttrName()))
    return;
  // We're willing to assert that our GEPs are in bounds, unless we're dealing
  // with buffer fat pointers (in which case, the offset is unsigned)
  func.walk([](LLVM::GEPOp gepOp) {
    if (cast<LLVM::LLVMPointerType>(gepOp.getType()).getAddressSpace() != 7)
      gepOp.setInbounds(true);
  });
  OpBuilder b(&getContext());

  // We'd like to do a bunch of annotating on loads and stores.
  // One thing we need to do is fix up alignments: MLIR's `vector.load` lowers
  // to an `llvm.load` whose allignment matches that of the element type,
  // while we know those loads are vector-aligned (and this is similar for
  // stores).
  //
  // We'd also like to reinforce that the loads we're doing from readonly
  // arguments are invariant - concurrent modification of any input we read is
  // undefined behavior.
  //
  // The second set of annotations has to do with a deficiency in the AMDGPU
  // backend. Specifically, the `noalias` attributes on kernel arguments
  // get discarded in the backend as the function is rewritten to include the
  // actual kernel argument loads being performed.
  size_t n = func.getNumArguments();
  llvm::SmallBitVector isReadonly(n);
  // All this alias scope stuff can be removed if the backend fixes things.
  auto domain =
      b.getAttr<LLVM::AliasScopeDomainAttr>(b.getStringAttr(func.getSymName()));
  llvm::SmallVector<ArrayAttr> aliasScopes;
  aliasScopes.reserve(n);
  llvm::SmallVector<ArrayAttr> noaliasScopes;
  noaliasScopes.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (func.getArgAttr(i, LLVM::LLVMDialect::getReadonlyAttrName()))
      isReadonly[i] = true;
    if (!isa<LLVM::LLVMPointerType>(func.getArgument(i).getType())) {
      aliasScopes.push_back(nullptr);
      continue;
    }
    auto aliasScope =
        LLVM::AliasScopeAttr::get(domain, b.getStringAttr("arg" + Twine(i)));
    aliasScopes.push_back(b.getArrayAttr(aliasScope));
  }
  {
    SmallVector<Attribute> allButOneScope;
    allButOneScope.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      for (auto [j, val] : llvm::enumerate(aliasScopes)) {
        if (j != i && val)
          allButOneScope.push_back(val[0]);
      }
      noaliasScopes.push_back(b.getArrayAttr(allButOneScope));
      allButOneScope.clear();
      LLVM_DEBUG(llvm::dbgs() << noaliasScopes.back() << "\n");
    }
  }
  llvm::DenseMap<Value, BlockArgument> toArgCache;

  // The alias analysis interface will pick up ROCDL buffer ops and the native
  // LLVM ops.
  func.walk([&](LLVM::AliasAnalysisOpInterface aliasIface) {
    // We will make the simplyfying assumption that the first pointer-valued
    // operand to the operation is the pointer being accessed.
    Operation *aliasOp = aliasIface.getOperation();
    Value ptrArg;
    for (Value arg : aliasOp->getOperands()) {
      if (isa<LLVM::LLVMPointerType>(arg.getType())) {
        ptrArg = arg;
        break;
      }
    }
    if (!ptrArg)
      return;
    if (auto load = dyn_cast<LLVM::LoadOp>(aliasOp))
      load.setAlignment(getAlign(load.getType()));
    if (auto store = dyn_cast<LLVM::StoreOp>(aliasOp))
      store.setAlignment(getAlign(store.getValue().getType()));
    BlockArgument funcArg = traceToArg(ptrArg, func, toArgCache);
    if (!funcArg)
      return;
    unsigned argNo = funcArg.getArgNumber();
    if (auto load = dyn_cast<LLVM::LoadOp>(aliasOp))
      load.setInvariantLoad(isReadonly[argNo]);
    aliasIface.setAliasScopes(aliasScopes[argNo]);
    aliasIface.setNoAliasScopes(noaliasScopes[argNo]);
  });

  // 3. Relax atomics. We set the atomic order on read-modify-write
  // operations to `monotonic`, which is the extend of the guarantees
  // we need about them, and we set the syncscope to "agent-one-as":
  // per the memory model, this sync scope means we get our atomic guarantees
  // (the monotonicity / lack of data races) above with other atomics executing
  // on the GPU, but not with those executing on, say, the host (which is
  // a situation we won't be in). We also guarantee that a pointer won't be
  // accessed through multiple address spaces.
  func.walk([&](LLVM::AtomicRMWOp op) {
    op.setSyncscope("agent-one-as");
    op.setOrdering(LLVM::AtomicOrdering::monotonic);
  });
  func.walk([&](LLVM::AtomicCmpXchgOp op) {
    op.setSyncscope("agent-one-as");
    op.setSuccessOrdering(LLVM::AtomicOrdering::monotonic);
    op.setFailureOrdering(LLVM::AtomicOrdering::monotonic);
  });

  // 4. TODO: add some invariant.start calls once MLIR's got them.
}
