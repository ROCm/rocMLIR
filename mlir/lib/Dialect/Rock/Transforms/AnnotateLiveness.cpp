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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

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

// Check if an operation writes to the given alloc
static bool hasWriteEffect(Operation *op, GpuAllocOp buffer) {
  auto memEffectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectInterface)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  memEffectInterface.getEffects(effects);

  for (const auto &effect : effects) {
    // Check if this is a Write effect on our alloc
    if (isa<MemoryEffects::Write>(effect.getEffect())) {
      SmallVector<GpuAllocOp> effectAllocs = rock::findAllGpuAllocs(effect.getValue());
      if (llvm::is_contained(effectAllocs, buffer)) {
        return true;
      }
    }
  }
  return false;
}

// Check if an operation reads from the given alloc
static bool hasReadEffect(Operation *op, GpuAllocOp buffer) {
  auto memEffectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectInterface)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  memEffectInterface.getEffects(effects);

  for (const auto &effect : effects) {
    // Check if this is a Read effect on our alloc
    if (isa<MemoryEffects::Read>(effect.getEffect())) {
      SmallVector<GpuAllocOp> effectAllocs = rock::findAllGpuAllocs(effect.getValue());
      if (llvm::is_contained(effectAllocs, buffer)) {
        return true;
      }
    }
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
static FailureOr<SmallVector<LiveRange>> computeLiveRanges(GpuAllocOp buffer,
                                                           func::FuncOp func) {
  SmallVector<LiveRange> ranges;

  // Get all accesses in program order
  SmallVector<Operation *> accesses = getOrderedAccesses(buffer, func);

  if (accesses.empty())
    return ranges;

  // State machine to track write/read patterns
  Operation *currentWrite = nullptr;
  Operation *lastRead = nullptr;

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
      // Update the last read (could be write, read, read, ... pattern)
      lastRead = op;
      if (!currentWrite) {
        return buffer->emitError(
            "Read before write (reading from uninitialized memory)");
      }
    }
  }

  bool hasRead = lastRead != nullptr;
  bool hasWrite = currentWrite != nullptr;
  if (hasRead != hasWrite) {
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

  for (auto alloc : allocs) {
    // For each alloc, compute its live ranges based on write/read
    FailureOr<SmallVector<LiveRange>> maybeLiveRanges =
        computeLiveRanges(alloc, func);
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
