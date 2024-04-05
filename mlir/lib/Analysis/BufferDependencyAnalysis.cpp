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

template <typename Impl>
void assign(EffectType effect, UserWrapper &user, Impl &impl) {
  if (effect == Impl::type) {
    impl.push_back(user.op);
  }
}

template <typename Impl, typename... Impls>
void assign(EffectType effect, UserWrapper &user, Impl &impl, Impls &...impls) {
  assign(effect, user, impl);
  assign(effect, user, impls...);
}

template <typename Impl, typename... Impls>
void find(memref::AllocOp allocOp, Impl &impl, Impls &...impls) {
  BufferDependencyAnalysis::Pair result;

  llvm::SmallVector<UserWrapper> users;
  findAllUsers(allocOp, users);
  for (auto &user : users) {
    EffectType memoryEffect = getMemoryEffect(user);
    assign(memoryEffect, user, impl, impls...);
  }
}

struct Readers : public llvm::SmallVector<Operation *> {
  static inline constexpr EffectType type = EffectType::read;
};

struct Writers : public llvm::SmallVector<Operation *> {
  static inline constexpr EffectType type = EffectType::write;
};

BufferDependencyAnalysis::BufferDependencyAnalysis(Operation *op) {
  func = dyn_cast<func::FuncOp>(op);
  assert(func &&
         "expected func::FuncOp as the top level operation for the analysis");
}

llvm::SmallVector<Operation *>
BufferDependencyAnalysis::getReaders(memref::AllocOp allocOp) {
  Readers readers;
  find(allocOp, readers);
  return std::move(readers);
}

llvm::SmallVector<Operation *>
BufferDependencyAnalysis::getWriters(memref::AllocOp allocOp) {
  Writers writers;
  find(allocOp, writers);
  return std::move(writers);
}

BufferDependencyAnalysis::Pair
BufferDependencyAnalysis::getReadersAndWriters(memref::AllocOp allocOp) {
  Readers readers;
  Writers writers;
  find(allocOp, readers, writers);

  BufferDependencyAnalysis::Pair result;
  result.readers = std::move(readers);
  result.writers = std::move(writers);

  return result;
}

std::optional<memref::AllocOp>
BufferDependencyAnalysis::getAllocation(Value value) {
  while (auto viewOp = dyn_cast<ViewLikeOpInterface>(value.getDefiningOp())) {
    value = viewOp.getViewSource();
  }
  if (auto allocOp = dyn_cast<memref::AllocOp>(value.getDefiningOp())) {
    return allocOp;
  }
  return std::nullopt;
}

LogicalResult BufferDependencyAnalysis::run() {
  llvm::SmallVector<memref::AllocOp, 8> allocOps;
  func.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });

  if (allocOps.empty()) {
    return failure();
  }

  for (auto &allocOp : allocOps) {
    auto result = BufferDependencyAnalysis::getReadersAndWriters(allocOp);

    readersTable.insert(std::make_pair(allocOp, result.readers));
    writersTable.insert(std::make_pair(allocOp, result.writers));
  }

  return success();
}
