//===- BufferDependencyAnalysis.h - a set of useful generic analyses
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <utility>

using namespace mlir;

struct UserWrapper {
  Operation *parentAlias;
  Operation *op;
};

void findAllUsers(Operation *parent, llvm::SmallVector<UserWrapper> &users) {
  for (auto *user : parent->getUsers()) {
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
      findAllUsers(viewOp, users);
    } else {
      users.push_back({parent, user});
    }
  }
}

enum EffectType { read, write, unknown };
EffectType getMemoryEffect(UserWrapper &user) {
  if (auto memoryEffect = dyn_cast<MemoryEffectOpInterface>(user.op)) {
    int64_t index = -1;
    for (auto it : llvm::enumerate(user.op->getOperands())) {
      auto operand = it.value();
      if (operand.getDefiningOp() == user.parentAlias) {
        index = static_cast<int64_t>(it.index());
        break;
      }
    }
    if (index < 0)
      return EffectType::unknown;
    Value value = user.op->getOperand(index);
    if (memoryEffect.getEffectOnValue<MemoryEffects::Write>(value)) {
      return EffectType::write;
    }
    if (memoryEffect.getEffectOnValue<MemoryEffects::Read>(value)) {
      return EffectType::read;
    }
  }
  return EffectType::unknown;
}

BufferDependencyAnalysis::Results
BufferDependencyAnalysis::findReadersAndWriters(
    llvm::SmallVectorImpl<memref::AllocOp> &allocOps) {
  BufferDependencyAnalysis::Results results;

  for (auto &allocOp : allocOps) {
    llvm::SmallVector<Operation *> readers = {};
    llvm::SmallVector<Operation *> writers = {};

    llvm::SmallVector<UserWrapper> users;
    findAllUsers(allocOp, users);
    for (auto &user : users) {
      EffectType memoryEffect = getMemoryEffect(user);
      switch (memoryEffect) {
      case EffectType::read: {
        readers.push_back(user.op);
        break;
      }
      case EffectType::write: {
        writers.push_back(user.op);
        break;
      }
      default:
        break;
      }
    }

    results.readers.insert(std::make_pair(allocOp, readers));
    results.writers.insert(std::make_pair(allocOp, writers));
  }

  return results;
}