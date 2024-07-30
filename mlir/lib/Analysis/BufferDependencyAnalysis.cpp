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

// Recursively finds all users fir a given Operation. Note, this function
// is a part of `BufferDependencyAnalysis` which is intended to be performed
// after bufferization.
void findAllUsers(OpOperand *thisUser,
                  llvm::SmallVectorImpl<OpOperand *> &users) {
  if (auto viewOp = dyn_cast<ViewLikeOpInterface>(thisUser->getOwner())) {
    for (OpOperand &user : viewOp->getUses()) {
      findAllUsers(&user, users);
    }
  } else {
    users.push_back(thisUser);
  }
}

void findAllUsers(Operation *root, llvm::SmallVectorImpl<OpOperand *> &users) {
  for (OpOperand &user : root->getUses()) {
    findAllUsers(&user, users);
  }
}

// Finds whether a given op reads or writes to the memory produced by the parent
// operation
enum EffectType { read, write, unknown };
EffectType getMemoryEffectType(OpOperand *use) {
  if (auto memoryEffect = dyn_cast<MemoryEffectOpInterface>(use->getOwner())) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
    memoryEffect.getEffects(effects);
    bool hasWrite = false;
    bool hasRead = false;
    for (auto &effect : effects) {
      if (effect.getEffectValue<OpOperand *>() != use)
        continue;
      hasWrite |= isa<MemoryEffects::Write>(effect.getEffect());
      hasRead |= isa<MemoryEffects::Read>(effect.getEffect());
    }
    if (hasWrite)
      return EffectType::write;
    if (hasRead)
      return EffectType::read;
  }
  return EffectType::unknown;
}

struct SearchResults {
  llvm::SmallVector<OpOperand *> readers;
  llvm::SmallVector<OpOperand *> writers;
};

// Finds all readers and writers to the memory produced by a given
// `memref::AllocOp`
SearchResults findReadersAndWriters(memref::AllocOp allocOp) {
  llvm::SmallVector<OpOperand *> users;
  SearchResults results;
  findAllUsers(allocOp, users);
  for (OpOperand *user : users) {
    EffectType effectType = getMemoryEffectType(user);
    if (effectType == EffectType::read) {
      results.readers.push_back(user);
    }
    if (effectType == EffectType::write) {
      results.writers.push_back(user);
    }
  }
  return results;
}

void BufferDependencyAnalysis::analyze(memref::AllocOp allocOp) {
  SearchResults searchResults = findReadersAndWriters(allocOp);
  readersTable.insert(std::make_pair(allocOp, searchResults.readers));
  writersTable.insert(std::make_pair(allocOp, searchResults.writers));
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
  for (auto &allocOp : allocOps)
    analyze(allocOp);
}

std::optional<llvm::SmallVector<OpOperand *>>
BufferDependencyAnalysis::getReaders(memref::AllocOp allocOp) {
  if (readersTable.contains(allocOp)) {
    return readersTable[allocOp];
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<OpOperand *>>
BufferDependencyAnalysis::getWriters(memref::AllocOp allocOp) {
  if (writersTable.contains(allocOp)) {
    return writersTable[allocOp];
  }
  return std::nullopt;
}
