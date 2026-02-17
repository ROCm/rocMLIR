//===- AnnotateLiveness - MLIR Rock ops lowering passes -----===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (c) 2025 Advanced Micro Devices INc.
//===----------------------------------------------------------------------===//
//
// This pass annotates LDS memory with liveness ops (rock.live_in,
// rock.live_out)
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKANNOTATELIVENESSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-annotate-liveness"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockAnnotateLivenessPass
    : public rock::impl::RockAnnotateLivenessPassBase<
          RockAnnotateLivenessPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

// Represents a live range based on write/read patterns
struct LiveRange {
  Operation *firstWrite; // First write in the range
  Operation *lastRead;   // Last read in the range

  LiveRange(Operation *w, Operation *r) : firstWrite(w), lastRead(r) {}
};

// Helper function to check if an operation has a specific memory effect on the
// given alloc
static bool hasEffect(
    Operation *op, GpuAllocOp buffer,
    llvm::function_ref<bool(const MemoryEffects::Effect *)> effectMatcher) {
  auto memEffectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectInterface)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  memEffectInterface.getEffects(effects);

  for (const auto &effect : effects) {
    if (effectMatcher(effect.getEffect())) {
      mlir::Value accessedVal = effect.getValue();
      if (!accessedVal) {
        LLVM_DEBUG(llvm::dbgs() << "[hasEffect] Effect value is null\n");
        continue;
      }
      SmallVector<GpuAllocOp> effectAllocs =
          rock::findAllGpuAllocs(accessedVal);
      if (llvm::is_contained(effectAllocs, buffer)) {
        return true;
      }
    }
  }
  return false;
}

// Check if an operation writes to the given alloc
static bool hasWriteEffect(Operation *op, GpuAllocOp buffer) {
  return hasEffect(op, buffer, [](const MemoryEffects::Effect *e) {
    return isa<MemoryEffects::Write>(e);
  });
}

// Check if an operation reads from the given alloc
static bool hasReadEffect(Operation *op, GpuAllocOp buffer) {
  return hasEffect(op, buffer, [](const MemoryEffects::Effect *e) {
    return isa<MemoryEffects::Read>(e);
  });
}

// Trace value back through views and block args; when we hit
// ExtractMultiBufferOp with non-constant index that includes buffer, return
// true (op may be accessing a sibling buffer).
static bool valueMayBeFromDynamicExtractMultibuffer(Value value,
                                                    GpuAllocOp buffer) {
  SmallVector<Value> worklist = {value};
  SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *curOp = v.getDefiningOp();
    if (BlockArgument blockArg = dyn_cast<BlockArgument>(v)) {
      Block *block = blockArg.getOwner();
      unsigned argNum = blockArg.getArgNumber();
      for (Block *pred : block->getPredecessors()) {
        Operation *branch = pred->getTerminator();
        if (branch && argNum < branch->getNumOperands())
          worklist.push_back(branch->getOperand(argNum));
      }
      continue;
    }
    if (!curOp)
      continue;
    if (auto viewOp = dyn_cast<ViewLikeOpInterface>(curOp)) {
      worklist.push_back(viewOp.getViewSource());
      continue;
    }
    if (auto extractOp = dyn_cast<rock::ExtractMultiBufferOp>(curOp)) {
      if (extractOp.getBuffers().size() <= 1)
        continue;
      if (isa<arith::ConstantIndexOp>(
              extractOp.getSelectIndex().getDefiningOp()))
        continue;
      for (Value b : extractOp.getBuffers()) {
        FailureOr<GpuAllocOp> found = rock::findGpuAlloc(b);
        if (succeeded(found) && *found == buffer)
          return true;
      }
      continue;
    }
  }
  return false;
}

// For an op that has a read effect on buffer, return true if that effect's
// value comes from dynamic extract_multibuffer (so the read might be for a
// sibling buffer).
static bool readMayBeForSiblingBuffer(Operation *op, GpuAllocOp buffer) {
  auto memEffectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectInterface)
    return false;
  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  memEffectInterface.getEffects(effects);
  for (const auto &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()))
      continue;
    Value accessedVal = effect.getValue();
    if (accessedVal &&
        valueMayBeFromDynamicExtractMultibuffer(accessedVal, buffer))
      return true;
  }
  return false;
}

// Collect all operations that access the buffer in program order
static SmallVector<Operation *> getOrderedAccesses(GpuAllocOp buffer,
                                                   func::FuncOp func) {
  SmallVector<Operation *> accesses;

  // Walk the function in program order
  func.walk([&](Operation *op) {
    if (hasWriteEffect(op, buffer) || hasReadEffect(op, buffer))
      accesses.push_back(op);
  });

  return accesses;
}

// Compute live ranges based on write/read pattern
// See the comment in annotateLiveness() to understand the assumptions we make
// here.
static FailureOr<SmallVector<LiveRange>>
computeLiveRanges(GpuAllocOp buffer, func::FuncOp func,
                  const llvm::SmallDenseSet<GpuAllocOp> *multibufferLDSAllocs) {
  SmallVector<LiveRange> ranges;

  // Get all accesses in program order
  SmallVector<Operation *> accesses = getOrderedAccesses(buffer, func);

  if (accesses.empty())
    return ranges;

  // State machine to track write/read patterns
  Operation *currentWrite = nullptr;
  Operation *lastRead = nullptr;

  // Double-buffered LDS (schedule v2) uses extract_multibuffer with dynamic
  // index, so reads may be attributed to both buffers; allow read-before-write
  // for allocs that are part of a multibuffer pair.
  bool allowReadBeforeWrite =
      multibufferLDSAllocs && multibufferLDSAllocs->contains(buffer);

  for (Operation *op : accesses) {
    bool isWrite = hasWriteEffect(op, buffer);
    bool isRead = hasReadEffect(op, buffer);
    assert(!(isWrite && isRead) &&
           "We do not expect to have an op that reads and writes to LDS at "
           "this point (AnnotateLiveness)");

    if (isWrite) {
      // If we have a pending range (write followed by reads), close it
      if (currentWrite && lastRead) {
        LiveRange range{currentWrite, lastRead};
        ranges.emplace_back(range);
        currentWrite = nullptr;
        lastRead = nullptr;
      }

      // Start a new range if we don't have one
      if (!currentWrite) {
        currentWrite = op;
      }
      // If we already have a write, this is another write in the same range
      // (write, write, ... pattern) - keep the first write as range start
    }

    if (isRead) {
      // With double-buffered LDS, ops use extract_multibuffer with dynamic
      // index, so findAllGpuAllocs attributes the access to all buffers. A read
      // may actually target a sibling buffer (e.g. pong); do not error for
      // read-before-write, but extend the live range conservatively.
      if (!currentWrite) {
        if (readMayBeForSiblingBuffer(op, buffer) || allowReadBeforeWrite) {
          currentWrite = op;
          lastRead = op;
          continue;
        }
        return buffer->emitError(
            "Read before write (reading from uninitialized memory)");
      }
      lastRead = op;
    }
  }

  bool hasRead = lastRead != nullptr;
  bool hasWrite = currentWrite != nullptr;
  if (hasRead != hasWrite) {
    // Multibuffer LDS may end with a write (no read after last write)
    if (allowReadBeforeWrite && hasWrite) {
      LiveRange range{currentWrite, currentWrite};
      ranges.emplace_back(range);
      return ranges;
    }
    return buffer->emitError("Found a non closed read-write pattern");
  }

  // Close any remaining range
  if (hasRead && hasWrite) {
    LiveRange range{currentWrite, lastRead};
    ranges.emplace_back(range);
  }

  return ranges;
}

// Annotate LDS buffer usage based on the following assumptions:
// 1. Liveness range is determined by a pattern of one or more write() ops, and
// then one or more read() ops.
// 2. There can be a number of writes and reads belonging to the same pattern,
// example: write(), write(), read(), read()
// 3. No modelling of loop-carried dependencies
// 4. Operations are either write or read. This is true when the pass is run.
// Note that, if we do:
//
// clang-format off
// buff=alloc(3)
// write(buff, [0, 1, 2], [-1, 23, 20])
// read([0, 1])
// write([0], [2])
// read([0, 1, 2]).
// clang-format on
//
// Where write(buffer, indices, data), read(buffer, indices), alloc(size). We
// would be breaking assumption (1). So, we assume that any read() will only
// read data written by the writes() of the same liveness range. Regarding (3),
// if we have:
//
// clang-format off
// buff=alloc(3)
// write(buff, [0, 1, 2], [-1, 23, 20])
// for ... {
//   read([0, 1])
//
//   write([0], [2])
// }
// read([0, 1, 2]).
// clang-format on
//
// The current implementation would create two liveness ranges, one between the
// first write (out of loop) and first read (inside the loop). The other range
// would be between the second write (inside the loop) and the last read
// (outside the loop). This would be incorrect because the buffer is alive for
// the whole loop. However, in practise, this is not a problem because if there
// are any interferences they will also happen in the epilogue and prologue.
// This might need to get improved if changes to pipelining are made.
// Collect LDS allocs that are part of a multibuffer (extract_multibuffer with
// 2+ buffers). For these, read-before-write is allowed due to pipelining.
static llvm::SmallDenseSet<GpuAllocOp>
getMultibufferLDSAllocs(func::FuncOp func) {
  llvm::SmallDenseSet<GpuAllocOp> set;
  func.walk([&](rock::ExtractMultiBufferOp extractOp) {
    if (extractOp.getBuffers().size() <= 1)
      return;
    for (Value b : extractOp.getBuffers()) {
      FailureOr<GpuAllocOp> found = rock::findGpuAlloc(b);
      if (succeeded(found)) {
        auto type = found->getOutput().getType();
        if (getWorkgroupMemorySize(type).has_value())
          set.insert(*found);
      }
    }
  });
  return set;
}

static LogicalResult annotateLiveness(func::FuncOp &func) {
  IRRewriter rewriter(func->getContext());

  // find all LDS rock.allocs
  SmallVector<GpuAllocOp> allocs;
  func.walk([&](GpuAllocOp op) {
    auto type = op.getOutput().getType();

    std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
    if (maybeSize.has_value()) {
      int64_t size = maybeSize.value();
      LLVM_DEBUG(llvm::dbgs() << "Found rock.alloc of " << size << " bytes\n");
      allocs.push_back(op);
    }
  });

  llvm::SmallDenseSet<GpuAllocOp> multibufferLDSAllocs =
      getMultibufferLDSAllocs(func);

  for (auto alloc : allocs) {
    // For each alloc, compute its live ranges based on write/read
    FailureOr<SmallVector<LiveRange>> maybeLiveRanges =
        computeLiveRanges(alloc, func, &multibufferLDSAllocs);
    if (failed(maybeLiveRanges))
      return failure();
    auto liveRanges = maybeLiveRanges.value();

    LLVM_DEBUG(llvm::dbgs()
               << "Found liveness ranges rock.alloc of " << alloc << ":\n");
    // Insert rock.live_in before first write, rock.live_out after last read
    for (const LiveRange &range : liveRanges) {
      LLVM_DEBUG(llvm::dbgs()
                 << "First write = " << *(range.firstWrite) << "\n");
      LLVM_DEBUG(llvm::dbgs() << "Last read = " << *(range.lastRead) << "\n");
      rewriter.setInsertionPoint(range.firstWrite);
      LiveInOp::create(rewriter, range.firstWrite->getLoc(), alloc);

      rewriter.setInsertionPointAfter(range.lastRead);
      LiveOutOp::create(rewriter, range.lastRead->getLoc(), alloc);
    }
  }
  return success();
}

void RockAnnotateLivenessPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Only run this pass on GPU kernel functions.
  if (!func->hasAttr("kernel")) {
    LLVM_DEBUG(llvm::dbgs() << "Skipping RockAnnotateLivenessPass on func with "
                               "no kernel attribute\n");
    return;
  }

  if (failed(annotateLiveness(func))) {
    return signalPassFailure();
  }
}
