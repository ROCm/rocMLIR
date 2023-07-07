//===- BufferLoadMerge.cpp - merge identical loads from read-only buffers -===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2022 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This pass will merge together identical amdgpu.raw_buffer_load ops
// when the buffer argument is only passed to buffer loads.
// It assumes that each buffer is accessed through exactly one memref value
// which is true in our generated code but not true in general.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"

#define DEBUG_TYPE "rock-buffer-load-merge"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBUFFERLOADMERGEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

using namespace mlir;
using namespace mlir::rock;

// Ripped out of the CSE code
namespace {
struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhsC), const_cast<Operation *>(rhsC),
        /*mapOperands=*/OperationEquivalence::exactValueMatch,
        /*mapResults=*/nullptr, OperationEquivalence::IgnoreLocations);
  }
};
} // end namespace

namespace {
// This is modelled after the CSE pass upstream, except we get to be a lot
// simpler because we don't have to even try and handle the case of an IR that's
// not in SSA form and because we don't have to be as paranoid about side
// effects

struct RockBufferLoadMergePass final
    : rock::impl::RockBufferLoadMergePassBase<RockBufferLoadMergePass> {
  using MapTy = llvm::DenseMap<Operation *, Operation *, SimpleOperationInfo>;

  bool isMergeable(Value buffer,
                   llvm::DenseMap<Value, bool> &isMergeableCache) {
    auto cached = isMergeableCache.find(buffer);
    if (cached != isMergeableCache.end())
      return cached->second;
    for (Operation *use : buffer.getUsers()) {
      if (!isa<amdgpu::RawBufferLoadOp>(use)) {
        isMergeableCache.insert({buffer, false});
        return false;
      }
    }
    isMergeableCache.insert({buffer, true});
    return true;
  }

  void runOnOperation() override {
    MapTy equivalentOps;
    llvm::DenseMap<Value, bool> isMergeableCache;
    llvm::SmallVector<Operation *, 0> toRemove;

    func::FuncOp op = getOperation();
    op.walk([&](amdgpu::RawBufferLoadOp op) {
      if (!isMergeable(op.getMemref(), isMergeableCache))
        return;
      auto firstLoad = equivalentOps.find(op);
      if (firstLoad != equivalentOps.end()) {
        op->replaceAllUsesWith(firstLoad->second);
        toRemove.push_back(op);
      } else {
        equivalentOps.insert({op, op});
      }
    });
    for (Operation *dead : toRemove) {
      dead->erase();
    }
  }
};
} // end namespace
