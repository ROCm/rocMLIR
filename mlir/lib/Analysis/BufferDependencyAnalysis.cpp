//===- BufferDependencyAnalysis.h - a set of useful generic analyses
//---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferDependencyAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace mlir;

struct UserWrapper {
  Operation *parentAlias;
  Operation *op;
};

// Recursively finds all users fir a given Operation. Note, this function
// is a part of `BufferDependencyAnalysis` which is intended to be performed
// after bufferization
void findAllUsers(Operation *parent, llvm::SmallVector<UserWrapper> &users) {
  for (auto *user : parent->getUsers()) {
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(user)) {
      findAllUsers(viewOp, users);
    } else {
      users.push_back({parent, user});
    }
  }
}

// Finds whether a given op reads or writes to the memory produced by the parent
// operation
enum EffectType { read, write, unknown };
EffectType getMemoryEffectType(UserWrapper user) {
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

struct SearchResults {
  llvm::SmallVector<Operation *> readers;
  llvm::SmallVector<Operation *> writers;
};

// Finds all readers and writers to the memory produced by a given
// `memref::AllocOp`
SearchResults findReadersAndWriters(memref::AllocOp allocOp) {
  llvm::SmallVector<UserWrapper> users;
  SearchResults results;
  findAllUsers(allocOp, users);
  for (auto &user : users) {
    EffectType effectType = getMemoryEffectType(user);
    if (effectType == EffectType::read) {
      results.readers.push_back(user.op);
    }
    if (effectType == EffectType::write) {
      results.writers.push_back(user.op);
    }
  }
  return results;
}

BufferDependencyAnalysis::BufferDependencyAnalysis(Operation *op) : op(op) {
  // find all `memref::AllocOp` in a given Operation (e.g., func::FuncOp)
  llvm::SmallVector<memref::AllocOp, 8> allocOps;
  op->walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });

  // stop the analysis of the enclosing op doesn't have any `memref::AllocOp`
  if (allocOps.empty()) {
    return;
  }

  // find readers and writers for each `memref::AllocOp` and record
  // the obtained results to the corresponding hash tables
  for (auto &allocOp : allocOps) {
    auto searchResults = findReadersAndWriters(allocOp);
    readersTable.insert(std::make_pair(allocOp, searchResults.readers));
    writersTable.insert(std::make_pair(allocOp, searchResults.writers));
  }
}

std::optional<llvm::SmallVector<Operation *>>
BufferDependencyAnalysis::getReaders(memref::AllocOp allocOp) {
  if (readersTable.contains(allocOp)) {
    return readersTable[allocOp];
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<Operation *>>
BufferDependencyAnalysis::getWriters(memref::AllocOp allocOp) {
  if (writersTable.contains(allocOp)) {
    return writersTable[allocOp];
  }
  return std::nullopt;
}
