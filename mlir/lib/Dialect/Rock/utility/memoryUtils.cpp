//===- memoryUtils.cpp - Rock memory utility functions
//---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/memoryUtils.h"

#include "mlir/Dialect/Rock/utility/AmdArchDb.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"

namespace mlir {
namespace rock {

std::optional<int64_t> getWorkgroupMemorySize(MemRefType type) {
  auto memSpaceValue =
      dyn_cast_or_null<gpu::AddressSpaceAttr>(type.getMemorySpace()).getValue();
  if (memSpaceValue == gpu::GPUDialect::getWorkgroupAddressSpace()) {
    return type.getNumElements() * getByteWidth(type.getElementType());
  }
  return std::nullopt;
}

static void assignColors(
    GpuAllocOp alloc, llvm::SetVector<int64_t> &usedColors,
    llvm::MapVector<int64_t, int64_t> &colorMemSize,
    llvm::SmallDenseMap<GpuAllocOp, SetVector<int64_t>> &colorAssignment) {
  const std::optional<int64_t> maybeRequiredSize =
      getWorkgroupMemorySize(alloc.getOutput().getType());
  assert(maybeRequiredSize.has_value());
  const int64_t requiredSize = maybeRequiredSize.value();
  assert(requiredSize > 0);
  int64_t color = 0;
  int64_t allocatedSize = 0;

  while (allocatedSize < requiredSize) {
    if (usedColors.contains(color)) {
      // found a used color, all the assigned colors must be consecutive
      // let's start again
      allocatedSize = 0;
      colorAssignment[alloc].clear();
      // Find the first available color
      while (usedColors.contains(color)) {
        color++;
      }
    }
    colorAssignment[alloc].insert(color);
    // New color
    if (!colorMemSize.contains(color)) {
      // Make this aligned to 128 bits
      colorMemSize[color] = llvm::alignTo(requiredSize - allocatedSize, 16);
    }
    allocatedSize += colorMemSize[color];
    color++;
  }
}

LogicalResult checkLDSSize(Operation *op, int64_t ldsBytes) {
  // Check for arch limitations exceeded
  FailureOr<StringAttr> maybeArch = getArch(op);
  if (succeeded(maybeArch)) {
    StringAttr arch = maybeArch.value();
    const int64_t ldsSize = rock::lookupArchInfo(arch).maxSharedMemPerWG;
    return success(ldsBytes <= ldsSize);
  }
  return success();
}

std::tuple<llvm::MapVector<int64_t, int64_t>,
           SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t, bool>>>
graphColoring(LDSInfo &ldsInfo) {
  llvm::SmallDenseMap<GpuAllocOp, SetVector<int64_t>> colorAssignment;
  llvm::MapVector<int64_t, int64_t> colorMemSize;

  SmallVector<GpuAllocOp> sortedAllocs(ldsInfo.allocs);

  // Sort by alloc size
  llvm::sort(sortedAllocs, [](GpuAllocOp &a, GpuAllocOp &b) {
    auto aSize = getWorkgroupMemorySize(a.getOutput().getType());
    auto bSize = getWorkgroupMemorySize(b.getOutput().getType());
    assert(aSize.has_value() && bSize.has_value());
    return aSize.value() < bSize.value();
  });
  // Assign colors using greedy algorithm
  for (GpuAllocOp alloc : sortedAllocs) {
    llvm::SetVector<int64_t> usedColors;
    for (GpuAllocOp neighbor : ldsInfo.interferenceGraph[alloc]) {
      if (colorAssignment.contains(neighbor)) {
        for (int64_t color : colorAssignment[neighbor]) {
          usedColors.insert(color);
        }
      }
    }

    // Assign a set of colors
    assignColors(alloc, usedColors, colorMemSize, colorAssignment);
  }

  // If we replace all GpuAllocOps with a single one, we run into
  // aliasing issues that cause performance regressions.
  // In order to avoid that, we first merge colors so that each
  // GpuAlloc is assigned a single color and an offset.
  // Then, we can generate more than one GpuAllocOp and improve
  // aliasing issues.
  // This might be removed in the future if aliasing issues are solved.
  llvm::SmallDenseMap<int64_t, SetVector<int64_t>> colorsToMerge;
  llvm::SmallDenseMap<int64_t, int64_t> oldColorToNew;
  for (GpuAllocOp alloc : sortedAllocs) {
    int64_t firstColor = colorAssignment[alloc][0];
    // if the color has been replaced already
    if (oldColorToNew.contains(firstColor)) {
      int64_t newColor = oldColorToNew[firstColor];
      // assign all the colors of the current 'alloc' to newColor
      for (int64_t color : colorAssignment[alloc]) {
        oldColorToNew[color] = newColor;
        colorsToMerge[newColor].insert(color);
      }
    } else {
      // the color has not been replaced yet
      for (auto [i, color] : llvm::enumerate(colorAssignment[alloc])) {
        // merge all non-first colors with the first one
        if (i > 0) {
          oldColorToNew[color] = firstColor;
          colorsToMerge[firstColor].insert(color);
          // if the current 'color' has merged some colors,
          // merge its colors to 'firstColor'
          if (colorsToMerge.contains(color)) {
            for (int64_t otherColor : colorsToMerge[color]) {
              oldColorToNew[otherColor] = firstColor;
              colorsToMerge[firstColor].insert(otherColor);
            }
            colorsToMerge.erase(color);
          }
        }
      }
    }
  }
  // Compute offsets and new sizes (after merging)
  llvm::MapVector<int64_t, int64_t> mergedColorMemSize(colorMemSize);
  llvm::MapVector<int64_t, int64_t> colorOffset;
  for (auto [color, _] : colorMemSize) {
    colorOffset[color] = 0;
  }
  for (auto [color, oldColors] : colorsToMerge) {
    for (int64_t oldColor : oldColors) {
      assert(oldColor > color);
      colorOffset[oldColor] = mergedColorMemSize[color];
      mergedColorMemSize[color] += mergedColorMemSize[oldColor];
      mergedColorMemSize.erase(oldColor);
    }
  }

  // Compute information per GpuAllocOp
  SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t, bool>> gpuAllocInfo;
  llvm::SetVector<int64_t> usedColors;
  for (const GpuAllocOp alloc : ldsInfo.allocs) {
    assert(colorAssignment.contains(alloc));

    // if the color has been used, we are "reusing" memory,
    // we need a LDS barrier
    bool useLDSBarrier = false;
    for (int64_t color : colorAssignment[alloc]) {
      useLDSBarrier |= usedColors.contains(color);
      usedColors.insert(color);
    }

    if (useLDSBarrier) {
      for (GpuAllocOp deadAlloc : ldsInfo.deallocBefore[alloc]) {
        for (int64_t color : colorAssignment[deadAlloc]) {
          if (!colorAssignment[alloc].contains(color)) {
            usedColors.remove(color);
          }
        }
      }
    }

    int64_t oldColor = colorAssignment[alloc][0];
    assert(colorOffset.contains(oldColor));
    int64_t offset = colorOffset[oldColor];
    int64_t newColor =
        (oldColorToNew.contains(oldColor)) ? oldColorToNew[oldColor] : oldColor;
    gpuAllocInfo.push_back(std::tuple(alloc, newColor, offset, useLDSBarrier));
  }

  return std::tuple(mergedColorMemSize, gpuAllocInfo);
}

FailureOr<LDSInfo> createInterferenceGraph(func::FuncOp &func) {
  LDSInfo ldsInfo;
  SetVector<GpuAllocOp> currentAllocs;
  llvm::SmallDenseMap<Value, GpuAllocOp> memrefToAlloc;
  llvm::SetVector<GpuAllocOp> deallocsUpToNow;

  // Create the interference graph and save all allocs and deallocs (LDS)
  WalkResult walkResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto gpuAlloc = dyn_cast<GpuAllocOp>(op)) {
      auto type = gpuAlloc.getOutput().getType();

      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        // save deallocs up to this point
        ldsInfo.deallocBefore[gpuAlloc] = SetVector<GpuAllocOp>(
            deallocsUpToNow.begin(), deallocsUpToNow.end());
        deallocsUpToNow.clear();

        // add vertex and connections
        for (auto alloc : currentAllocs) {
          ldsInfo.interferenceGraph[alloc].insert(gpuAlloc);
          ldsInfo.interferenceGraph[gpuAlloc].insert(alloc);
        }
        // if it has no neighbors, we still want to add a vertex
        if (currentAllocs.empty()) {
          ldsInfo.interferenceGraph[gpuAlloc] = {};
        }
        currentAllocs.insert(gpuAlloc);
        memrefToAlloc[gpuAlloc.getOutput()] = gpuAlloc;
        ldsInfo.allocs.push_back(gpuAlloc);
      }
    } else if (auto gpuDealloc = dyn_cast<GpuDeallocOp>(op)) {
      auto type = gpuDealloc.getMemref().getType();
      std::optional<int64_t> maybeSize = getWorkgroupMemorySize(type);
      if (maybeSize.has_value()) {
        if (memrefToAlloc.find(gpuDealloc.getMemref()) == memrefToAlloc.end()) {
          return WalkResult::interrupt();
        }
        bool erased =
            currentAllocs.remove(memrefToAlloc[gpuDealloc.getMemref()]);
        deallocsUpToNow.insert(memrefToAlloc[gpuDealloc.getMemref()]);
        if (!erased) {
          return WalkResult::interrupt();
        }
        ldsInfo.deallocs.push_back(gpuDealloc);
      }
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    if (ldsInfo.allocs.empty())
      return emitError(UnknownLoc::get(func.getContext()), "Unexpected error");
    return ldsInfo.allocs.front().emitError(
        "Called rock.dealloc multiple times");
  }

  // same number of rock.alloc and rock.dealloc
  if (ldsInfo.deallocs.size() != ldsInfo.allocs.size() ||
      ldsInfo.allocs.size() != ldsInfo.interferenceGraph.size() ||
      !currentAllocs.empty()) {
    return emitError(UnknownLoc::get(func.getContext()),
                     "There should be an equal number of rock.alloc and "
                     "rock.dealloc (for LDS)");
  }

  return ldsInfo;
}

FailureOr<int64_t> getAllocatedLDSAfterReuse(func::FuncOp &func) {
  FailureOr<LDSInfo> maybeLdsInfo = createInterferenceGraph(func);
  if (failed(maybeLdsInfo)) {
    return failure();
  }
  LDSInfo ldsInfo = maybeLdsInfo.value();

  llvm::MapVector<int64_t, int64_t> colorSizes;
  SmallVector<std::tuple<GpuAllocOp, int64_t, int64_t, bool>> allocOffsets;
  std::tie(colorSizes, allocOffsets) = graphColoring(ldsInfo);

  int64_t requiredMemory = 0;
  for (auto [_, size] : colorSizes) {
    requiredMemory += size;
  }

  return requiredMemory;
}

} // namespace rock
} // namespace mlir