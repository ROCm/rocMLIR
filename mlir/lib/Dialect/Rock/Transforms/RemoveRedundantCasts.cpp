//===-------------------- RemoveRedundantCasts.cpp ------------------------===//
//
// Copyright 2026 The MLIR Authors.
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
//
// This pass detects patterns at the LLVM dialect level where wider float values
// are truncated (llvm.fptrunc) to a narrower type, stored to a buffer, then
// loaded and extended (llvm.fpext) back to the original wider type. This pass
// creates a parallel wide buffer (if one doesn't exist) and redirects the loads
// to read the wide values directly, eliminating the fpext and preserving
// precision.
//
// Algorithm:
//   1. Find all fptrunc -> store patterns in the function. For each pattern,
//      record whether there's already a parallel store of the wide value to
//      a separate buffer.
//   2. Find all load -> fpext patterns where the load is from a buffer that
//      has fptrunc stores.
//   3. Verify safety for each load+fpext pattern:
//      - All stores to the narrow buffer must be from tracked fptrunc patterns
//        (i.e., no untracked stores that could write different values)
//      - All tracked stores must dominate the load
//      - The narrow buffer must be an alloca
//   4. For safe patterns, create a wide buffer and the corresponding stores if
//      they don't exist. If a parallel store already exists, reuse it:
//      - Create a wide alloca right after the narrow alloca
//      - For each fptrunc store, insert a store of the wide value to the
//        wide buffer (right after the narrow store, using the same indices)
//   5. Apply the transformation:
//      - Redirect the load to read from the wide buffer instead
//      - Replace uses of the fpext result with the wide load result
//      - Delete the fpext (and the old load/GEP if unused)
//   6. Clean up unused narrow buffer operations:
//      - If the narrow buffer has no remaining uses, erase the fptrunc stores
//        - These can only be erased if they are not used by any other
//          operations
//      - Erase the narrow alloca if it has no remaining uses
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKREMOVEREDUNDANTCASTSPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-remove-redundant-casts"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

// A candidate parallel wide store (stores the wide value to a separate buffer).
struct WideStoreCandidate {
  Value wideBuffer;
  StoreOp wideStore;
};

// Information about a fptrunc -> store pattern
struct FPTruncStoreInfo {
  Value wideValue;     // The input to fptrunc (original wide value)
  StoreOp narrowStore; // The store of the narrow value
  Value narrowBuffer;  // Base alloca of narrow store
  GEPOp narrowGep;     // GEP for narrow store (null if storing directly)

  // All candidate parallel wide stores found during pattern collection.
  SmallVector<WideStoreCandidate> wideStoreCandidates;

  // The chosen wide buffer/store (selected during verification or creation).
  // Default-constructed (null) if no parallel store exists yet.
  WideStoreCandidate chosen;

  bool hasParallelStore() const { return chosen.wideStore != nullptr; }
};

// Information about a load + fpext pattern that can potentially be optimized.
struct LoadFPExtPattern {
  LoadOp loadOp;      // The load from narrow buffer
  FPExtOp fpextOp;    // The fpext that extends the loaded value
  Value narrowBuffer; // Base pointer of the narrow buffer being loaded
  GEPOp gepOp;        // The GEP operation (if any) used for indexing

  // All fptrunc stores that contribute to covering the buffer.
  SmallVector<FPTruncStoreInfo *> matchingStores;
};

// Get the base pointer from a value, tracing through GEP operations.
static Value getBasePointer(Value ptr) {
  while (auto gep = ptr.getDefiningOp<GEPOp>()) {
    ptr = gep.getBase();
  }
  return ptr;
}

// Get the scalar element type, unwrapping vectors if needed.
static Type getScalarType(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type))
    return vecType.getElementType();
  return type;
}

// Compute the alignment guaranteed by a pointer value.
// For allocas, uses explicit alignment or the element type's natural alignment
// (LLVM guarantees allocas are at least ABI-aligned). For GEPs with constant
// indices, accounts for the byte offset to compute the effective alignment.
static unsigned computePointerAlignment(Value ptr, Value baseBuffer,
                                        Type elemType) {
  unsigned elemBytes = std::max(elemType.getIntOrFloatBitWidth() / 8, 1u);

  unsigned baseAlign = elemBytes;
  if (auto alloca = baseBuffer.getDefiningOp<AllocaOp>()) {
    if (auto allocaAlign = alloca.getAlignment())
      baseAlign = *allocaAlign;
  }

  auto gep = ptr.getDefiningOp<GEPOp>();
  if (!gep)
    return baseAlign;

  auto indices = gep.getIndices();
  if (indices.size() == 1) {
    if (auto constIdx = dyn_cast<IntegerAttr>(indices[0])) {
      int64_t idx = constIdx.getInt();
      if (idx == 0)
        return baseAlign;
      unsigned absOffset = static_cast<unsigned>(std::abs(idx)) * elemBytes;
      unsigned effective = baseAlign;
      while (effective > 1 && (absOffset % effective) != 0)
        effective >>= 1;
      return effective;
    }
  }

  return std::min(baseAlign, elemBytes);
}

// Find all fptrunc -> store patterns in the function.
static SmallVector<FPTruncStoreInfo>
findFPTruncStorePatterns(LLVMFuncOp funcOp) {
  SmallVector<FPTruncStoreInfo> results;

  funcOp.walk([&](FPTruncOp fptruncOp) -> WalkResult {
    LLVM_DEBUG(llvm::dbgs() << "Found fptrunc: " << fptruncOp << "\n");
    Value wideValue = fptruncOp.getArg();
    Value narrowValue = fptruncOp.getRes();

    // Pre-collect all stores of the wide value to avoid re-scanning
    // wideValue.getUsers() for each narrow store.
    SmallVector<StoreOp> wideStores;
    for (Operation *wideUser : wideValue.getUsers()) {
      auto wideStore = dyn_cast<StoreOp>(wideUser);
      if (wideStore && wideStore.getValue() == wideValue)
        wideStores.push_back(wideStore);
    }

    // Find direct stores of the narrow value
    for (Operation *user : narrowValue.getUsers()) {
      auto narrowStore = dyn_cast<StoreOp>(user);
      if (!narrowStore)
        continue;

      Value narrowPtr = narrowStore.getAddr();
      Value narrowBuffer = getBasePointer(narrowPtr);
      GEPOp narrowGep = narrowPtr.getDefiningOp<GEPOp>();
      LLVM_DEBUG(llvm::dbgs()
                 << "\tFound narrow store: " << narrowStore << "\n");

      FPTruncStoreInfo info;
      info.wideValue = wideValue;
      info.narrowStore = narrowStore;
      info.narrowBuffer = narrowBuffer;
      info.narrowGep = narrowGep;
      info.chosen = {};

      // Collect all candidate parallel wide stores
      for (StoreOp wideStore : wideStores) {
        Value widePtr = wideStore.getAddr();
        Value wideBuffer = getBasePointer(widePtr);
        if (wideBuffer == narrowBuffer)
          continue;

        LLVM_DEBUG(llvm::dbgs() << "\tFound candidate parallel wide store: "
                                << wideStore << "\n");
        info.wideStoreCandidates.push_back({wideBuffer, wideStore});
      }

      results.push_back(info);
    }

    return WalkResult::advance();
  });

  return results;
}

// Find all load + fpext patterns.
static SmallVector<LoadFPExtPattern> findLoadFPExtPatterns(LLVMFuncOp funcOp) {
  SmallVector<LoadFPExtPattern> results;

  funcOp.walk([&](FPExtOp fpextOp) -> WalkResult {
    Value input = fpextOp.getArg();
    auto loadOp = input.getDefiningOp<LoadOp>();
    if (!loadOp)
      return WalkResult::advance();

    Value loadPtr = loadOp.getAddr();
    Value narrowBuffer = getBasePointer(loadPtr);
    GEPOp gepOp = loadPtr.getDefiningOp<GEPOp>();
    LLVM_DEBUG(llvm::dbgs() << "Found load+fpext pattern:\n");
    LLVM_DEBUG(llvm::dbgs() << "\tLoad: " << loadOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "\tFPExt: " << fpextOp << "\n");

    LoadFPExtPattern pattern;
    pattern.loadOp = loadOp;
    pattern.fpextOp = fpextOp;
    pattern.narrowBuffer = narrowBuffer;
    pattern.gepOp = gepOp;
    results.push_back(pattern);
    return WalkResult::advance();
  });

  return results;
}

// Collect all memory operations of a given type (LoadOp or StoreOp) that
// access a buffer, tracing through GEPs.
template <typename MemOpTy>
static SmallVector<MemOpTy> collectMemOpsOnBuffer(Value buffer) {
  SmallVector<MemOpTy> results;
  SmallVector<Value, 4> worklist;
  worklist.push_back(buffer);

  while (!worklist.empty()) {
    Value ptr = worklist.pop_back_val();
    for (Operation *user : ptr.getUsers()) {
      if (auto memOp = dyn_cast<MemOpTy>(user)) {
        if (memOp.getAddr() == ptr)
          results.push_back(memOp);
      } else if (auto gep = dyn_cast<GEPOp>(user)) {
        if (gep.getBase() == ptr)
          worklist.push_back(gep.getResult());
      }
    }
  }
  return results;
}

// Check if a load is only used by an fpext operation
static bool isLoadOnlyUsedByFPExt(LoadOp load) {
  Value loadResult = load.getRes();
  // The load result should have exactly one use, and it should be fpext
  if (!loadResult.hasOneUse())
    return false;
  Operation *user = *loadResult.getUsers().begin();
  return isa<FPExtOp>(user);
}

// Check if a store is from one of our tracked fptrunc patterns.
static bool
isStoreFromFPTruncPattern(StoreOp store,
                          const SmallVector<FPTruncStoreInfo> &storeInfos) {
  return llvm::any_of(
      storeInfos, [&](const auto &info) { return info.narrowStore == store; });
}

// Represents a range of element indices [start, start + count).
struct IndexRange {
  int64_t start;
  int64_t count;
  bool isValid() const { return count > 0; }
  bool isSubsetOf(const IndexRange &other) const {
    return start >= other.start &&
           (start + count) <= (other.start + other.count);
  }
  bool overlaps(const IndexRange &other) const {
    return start < (other.start + other.count) && other.start < (start + count);
  }
};

// Get the index range for a memory access. Returns an invalid range if we can't
// determine it (dynamic indices, multi-index GEPs, etc.).
static IndexRange getAccessRange(GEPOp gep, Type accessType) {
  int64_t elementCount = 1;
  if (auto vecType = dyn_cast<VectorType>(accessType))
    elementCount = vecType.getNumElements();

  // No GEP means accessing at base (index 0)
  if (!gep)
    return {0, elementCount};

  // GEP indices are in units of the GEP element type. When the access type
  // differs, convert elementCount to GEP-element-type units so that ranges
  // for the same buffer remain comparable.
  Type accessElemType = getScalarType(accessType);
  if (gep.getElemType() != accessElemType) {
    unsigned gepBits = gep.getElemType().getIntOrFloatBitWidth();
    unsigned accessBits = accessElemType.getIntOrFloatBitWidth();
    unsigned totalBits = accessBits * elementCount;
    if (gepBits == 0 || totalBits % gepBits != 0)
      return {-1, 0};
    elementCount = totalBits / gepBits;
  }

  // Check if all indices are constant
  auto indices = gep.getIndices();
  if (indices.empty())
    return {0, elementCount};

  // We only handle single-index GEPs with constant index
  if (indices.size() != 1)
    return {-1, 0}; // Invalid

  auto constIdx = dyn_cast<IntegerAttr>(indices[0]);
  if (!constIdx)
    return {-1, 0}; // Invalid

  return {constIdx.getInt(), elementCount};
}

// Get the total size (in elements) of a buffer from its alloca.
static int64_t getBufferSize(Value buffer) {
  auto alloca = buffer.getDefiningOp<AllocaOp>();
  if (!alloca)
    return -1;

  // Get array size (number of elements allocated)
  Value arraySizeVal = alloca.getArraySize();
  if (auto constOp = arraySizeVal.getDefiningOp<LLVM::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return -1; // Dynamic or unknown size
}

// Find all fptrunc stores that cover the load's location. Only includes stores
// whose wide type matches the fpext target type. Returns all such dominating
// stores if they collectively cover the entire buffer, otherwise returns an
// empty list.
static SmallVector<FPTruncStoreInfo *>
findMatchingFPTruncStores(LoadFPExtPattern &pattern,
                          SmallVector<FPTruncStoreInfo> &storeInfos,
                          DominanceInfo &domInfo) {
  SmallVector<FPTruncStoreInfo *> dominatingStores;
  int64_t bufferSize = getBufferSize(pattern.narrowBuffer);

  if (bufferSize <= 0)
    return dominatingStores;

  Type fpextElemType = getScalarType(pattern.fpextOp.getRes().getType());

  // Track which elements are covered
  std::vector<bool> covered(bufferSize, false);

  for (auto &info : storeInfos) {
    if (info.narrowBuffer != pattern.narrowBuffer)
      continue;
    if (getScalarType(info.wideValue.getType()) != fpextElemType)
      continue;
    if (!domInfo.dominates(info.narrowStore.getOperation(),
                           pattern.loadOp.getOperation()))
      continue;

    IndexRange storeRange =
        getAccessRange(info.narrowGep, info.narrowStore.getValue().getType());
    if (!storeRange.isValid())
      continue;

    dominatingStores.push_back(&info);

    // Mark covered elements
    for (int64_t i = storeRange.start;
         i < storeRange.start + storeRange.count && i < bufferSize; ++i) {
      if (i >= 0)
        covered[i] = true;
    }
  }

  // Check if all elements are covered
  for (bool c : covered) {
    if (!c)
      return {}; // Not fully covered, return empty
  }
  return dominatingStores;
}

// Check that no non-fptrunc stores could intervene between the fptrunc stores
// and the load that would overwrite the fptrunc'd value.
static bool
hasNoInterveningStores(LoadFPExtPattern &pattern,
                       const SmallVector<FPTruncStoreInfo> &storeInfos,
                       DominanceInfo &domInfo) {
  IndexRange loadRange =
      getAccessRange(pattern.gepOp, pattern.loadOp.getRes().getType());
  SmallVector<StoreOp> allStores =
      collectMemOpsOnBuffer<StoreOp>(pattern.narrowBuffer);
  for (auto store : allStores) {
    // Skip fptrunc stores
    if (isStoreFromFPTruncPattern(store, storeInfos))
      continue;

    // If load dominates store, the store happens after the load on all paths
    if (domInfo.dominates(pattern.loadOp.getOperation(), store.getOperation()))
      continue;

    // This non-fptrunc store could execute before the load on some path.
    // Check if it could overwrite what the load is reading.
    GEPOp storeGep = store.getAddr().getDefiningOp<GEPOp>();
    IndexRange storeRange =
        getAccessRange(storeGep, store.getValue().getType());

    // If we can determine ranges and they don't overlap, it's safe
    if (storeRange.isValid() && loadRange.isValid() &&
        !storeRange.overlaps(loadRange))
      continue;

    LLVM_DEBUG(llvm::dbgs()
               << "\tUNSAFE: Non-fptrunc store could overwrite value: " << store
               << "\n");
    return false;
  }
  return true;
}

// Verify that a load -> fpext pattern is safe to optimize.
static FailureOr<SmallVector<FPTruncStoreInfo *>>
verifySafety(LoadFPExtPattern &pattern,
             SmallVector<FPTruncStoreInfo> &storeInfos,
             DominanceInfo &domInfo) {
  // Check that the narrow buffer is an alloca
  if (!pattern.narrowBuffer.getDefiningOp<AllocaOp>()) {
    LLVM_DEBUG(llvm::dbgs() << "\tUNSAFE: Narrow buffer is not an alloca\n");
    return failure();
  }

  // Check whether we can determine the load's access range. This fails for
  // dynamic indices, multi-index GEPs, or any case getAccessRange can't handle.
  IndexRange loadRange =
      getAccessRange(pattern.gepOp, pattern.loadOp.getRes().getType());
  bool hasUnknownRange = !loadRange.isValid();

  // Find all fptrunc stores that cover this load's buffer. For loads with
  // dynamic indices, this checks that the entire buffer is covered by
  // dominating fptrunc stores. If so, any index the load could use will
  // read a value that was originally truncated from a wider type, making
  // the optimization safe. For static indices, full-buffer coverage is
  // also required.
  SmallVector<FPTruncStoreInfo *> matchingStores =
      findMatchingFPTruncStores(pattern, storeInfos, domInfo);
  if (matchingStores.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "\tUNSAFE: No matching fptrunc stores found for load\n");
    return failure();
  }

  if (hasUnknownRange) {
    LLVM_DEBUG(llvm::dbgs()
               << "\tLoad has unresolvable access range (dynamic or "
                  "multi-index GEP), but all "
               << matchingStores.size()
               << " fptrunc store(s) cover the entire buffer and dominate "
                  "the load, so it is safe to optimize\n");
  }

  for (auto *store : matchingStores) {
    // Check that the fptrunc result is only used by the narrow store.
    // If the f16 value has other uses, we can't eliminate the truncation.
    auto fptruncOp = store->narrowStore.getValue().getDefiningOp<FPTruncOp>();
    if (fptruncOp && !fptruncOp.getRes().hasOneUse()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tUNSAFE: fptrunc result has multiple uses - the f16 "
                    "value is used elsewhere\n");
      return failure();
    }
  }

  // Check that all loads from the narrow buffer are used only by fpext.
  // If there are loads that use the f16 values directly,
  // we can't eliminate the narrow buffer.
  SmallVector<LoadOp> allLoads =
      collectMemOpsOnBuffer<LoadOp>(pattern.narrowBuffer);
  for (LoadOp load : allLoads) {
    if (!isLoadOnlyUsedByFPExt(load)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\tUNSAFE: Buffer has loads that don't go through fpext - "
                    "f16 values are used directly: "
                 << load << "\n");
      return failure();
    }
  }

  // Check that no non-fptrunc stores could intervene
  if (!hasNoInterveningStores(pattern, storeInfos, domInfo))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "\tSAFE: All checks passed\n");
  return matchingStores;
}

// Select consistent wide buffers for pre-existing parallel stores.
// All fptrunc stores to a given narrow buffer must use the same wide buffer,
// and all corresponding wide stores must dominate every load from that narrow
// buffer. If no consistent selection exists, all selections are cleared so
// that createWideBuffersAndStores will create a unified wide buffer instead.
static void
selectConsistentWideBuffers(SmallVector<LoadFPExtPattern> &safePatterns,
                            DominanceInfo &domInfo) {
  struct NarrowBufferInfo {
    SmallVector<FPTruncStoreInfo *> stores;
    SmallVector<LoadOp> loads;
  };
  DenseMap<Value, NarrowBufferInfo> bufferInfos;
  DenseSet<FPTruncStoreInfo *> seenStores;
  DenseSet<Operation *> seenLoads;

  for (auto &pattern : safePatterns) {
    auto &info = bufferInfos[pattern.narrowBuffer];
    if (seenLoads.insert(pattern.loadOp.getOperation()).second)
      info.loads.push_back(pattern.loadOp);
    for (auto *store : pattern.matchingStores) {
      if (seenStores.insert(store).second)
        info.stores.push_back(store);
    }
  }

  for (auto &[narrowBuffer, info] : bufferInfos) {
    if (info.stores.empty())
      continue;

    // A valid wide buffer must appear as a candidate in every store, and
    // each candidate's wide store must dominate every load. Use the first
    // store's candidates as starting points (any valid buffer must appear
    // in every store's candidates, including the first).
    bool selected = false;
    for (auto &seed : info.stores[0]->wideStoreCandidates) {
      Value candidateBuffer = seed.wideBuffer;
      bool valid = true;
      SmallVector<WideStoreCandidate *> picks;

      for (auto *store : info.stores) {
        WideStoreCandidate *pick = nullptr;
        for (auto &c : store->wideStoreCandidates) {
          if (c.wideBuffer != candidateBuffer)
            continue;
          if (llvm::all_of(info.loads, [&](LoadOp load) {
                return domInfo.dominates(c.wideStore.getOperation(),
                                         load.getOperation());
              })) {
            pick = &c;
            break;
          }
        }
        if (!pick) {
          valid = false;
          break;
        }
        picks.push_back(pick);
      }

      if (valid) {
        for (size_t i = 0; i < info.stores.size(); ++i) {
          info.stores[i]->chosen = *picks[i];
        }
        selected = true;
        LLVM_DEBUG(llvm::dbgs()
                   << "Selected consistent parallel wide buffer for "
                   << info.stores.size() << " store(s)\n");
        break;
      }
    }

    if (!selected) {
      for (auto *store : info.stores) {
        store->chosen = {};
      }
      LLVM_DEBUG(llvm::dbgs() << "No consistent parallel wide buffer for "
                              << info.stores.size()
                              << " store(s), will create unified buffer\n");
    }
  }
}

// Create a wide store for an fptrunc store info, using the given wide buffer.
static void createWideStore(FPTruncStoreInfo *info, Value wideBuffer,
                            Type wideElemType, OpBuilder &builder) {
  builder.setInsertionPointAfter(info->narrowStore);
  Value widePtr;
  if (info->narrowGep) {
    SmallVector<GEPArg> gepArgs;
    for (auto idx : info->narrowGep.getIndices()) {
      if (auto constIdx = dyn_cast<IntegerAttr>(idx))
        gepArgs.push_back(static_cast<int32_t>(constIdx.getInt()));
      else
        gepArgs.push_back(cast<Value>(idx));
    }
    auto wideGep =
        GEPOp::create(builder, info->narrowGep.getLoc(), wideBuffer.getType(),
                      wideElemType, wideBuffer, gepArgs);
    wideGep.setNoWrapFlags(info->narrowGep.getNoWrapFlags());
    widePtr = wideGep.getResult();
  } else {
    widePtr = wideBuffer;
  }

  unsigned storeAlignment =
      computePointerAlignment(widePtr, wideBuffer, wideElemType);
  auto wideStore = StoreOp::create(
      builder, info->narrowStore.getLoc(), info->wideValue, widePtr,
      storeAlignment, info->narrowStore.getVolatile_(),
      info->narrowStore.getNontemporal(),
      /*isInvariantGroup=*/false, info->narrowStore.getOrdering(),
      info->narrowStore.getSyncscope().value_or(StringRef()));
  info->chosen = {wideBuffer, wideStore};
  LLVM_DEBUG(llvm::dbgs() << "Created wide store: " << wideStore << "\n");
}

// Create wide buffers and stores for safe patterns that don't already have
// them. For patterns with existing parallel wide stores, do nothing. For
// patterns without parallel stores, create a new wide alloca and insert a wide
// store right after each narrow store.
static void
createWideBuffersAndStores(SmallVector<LoadFPExtPattern> &safePatterns,
                           OpBuilder &builder) {
  DenseMap<std::pair<Value, Type>, Value> narrowToWideBuffer;
  DenseSet<FPTruncStoreInfo *> processedStores;
  for (auto &pattern : safePatterns) {
    if (pattern.matchingStores.empty())
      continue;

    Type wideElemType = getScalarType(pattern.fpextOp.getRes().getType());
    for (FPTruncStoreInfo *info : pattern.matchingStores) {
      if (processedStores.contains(info))
        continue;
      processedStores.insert(info);

      if (info->hasParallelStore())
        continue;

      auto key = std::make_pair(info->narrowBuffer, wideElemType);
      auto it = narrowToWideBuffer.find(key);
      if (it != narrowToWideBuffer.end()) {
        createWideStore(info, it->second, wideElemType, builder);
        continue;
      }

      // Need to create new wide buffer
      auto narrowAlloca = info->narrowBuffer.getDefiningOp<AllocaOp>();
      if (!narrowAlloca) {
        LLVM_DEBUG(
            llvm::dbgs()
            << "Cannot create wide buffer: narrow buffer is not alloca\n");
        continue;
      }

      builder.setInsertionPointAfter(narrowAlloca);
      auto wideAlloca = AllocaOp::create(
          builder, narrowAlloca.getLoc(), narrowAlloca.getResult().getType(),
          wideElemType, narrowAlloca.getArraySize());

      int64_t arraySize = getBufferSize(narrowAlloca.getResult());
      if (arraySize > 0) {
        unsigned elemBytes = wideElemType.getIntOrFloatBitWidth() / 8;
        unsigned desiredAlign =
            std::min(static_cast<unsigned>(arraySize) * elemBytes, 16u);
        // Round down to power of 2.
        while (desiredAlign & (desiredAlign - 1))
          desiredAlign &= desiredAlign - 1;
        wideAlloca.setAlignment(desiredAlign);
      }

      LLVM_DEBUG(llvm::dbgs() << "Created wide alloca: " << wideAlloca << "\n");
      narrowToWideBuffer[key] = wideAlloca.getResult();
      createWideStore(info, wideAlloca.getResult(), wideElemType, builder);
    }
  }
}

// Apply the transformation: redirect loads from narrow buffer to wide buffer,
// eliminating the fpext operations.
static void applyTransformation(SmallVector<LoadFPExtPattern> &safePatterns,
                                OpBuilder &builder) {
  for (auto &pattern : safePatterns) {
    // Find the first matching store with a wide buffer
    Value wideBuffer;
    for (auto *store : pattern.matchingStores) {
      if (store->chosen.wideBuffer) {
        wideBuffer = store->chosen.wideBuffer;
        break;
      }
    }
    if (!wideBuffer) {
      LLVM_DEBUG(llvm::dbgs() << "No wide buffer for pattern, skipping\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Transforming pattern:\n");
    LLVM_DEBUG(llvm::dbgs() << "  Load: " << pattern.loadOp << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  FPExt: " << pattern.fpextOp << "\n");

    Type wideType = pattern.fpextOp.getRes().getType();
    Type wideElemType = getScalarType(wideType);
    Value newPtr;
    if (pattern.gepOp) {
      builder.setInsertionPoint(pattern.gepOp);
      SmallVector<GEPArg> gepArgs;
      for (auto idx : pattern.gepOp.getIndices()) {
        if (auto constIdx = dyn_cast<IntegerAttr>(idx)) {
          gepArgs.push_back(static_cast<int32_t>(constIdx.getInt()));
        } else {
          gepArgs.push_back(cast<Value>(idx));
        }
      }

      auto newGep =
          GEPOp::create(builder, pattern.gepOp.getLoc(), wideBuffer.getType(),
                        wideElemType, wideBuffer, gepArgs);
      newGep.setNoWrapFlags(pattern.gepOp.getNoWrapFlags());
      newPtr = newGep.getResult();
    } else {
      newPtr = wideBuffer;
    }

    builder.setInsertionPoint(pattern.loadOp);
    unsigned wideAlignment =
        computePointerAlignment(newPtr, wideBuffer, wideElemType);

    auto newLoad = LoadOp::create(
        builder, pattern.loadOp.getLoc(), wideType, newPtr, wideAlignment,
        pattern.loadOp.getVolatile_(), pattern.loadOp.getNontemporal(),
        /*isInvariant=*/false, /*isInvariantGroup=*/false,
        pattern.loadOp.getOrdering(),
        pattern.loadOp.getSyncscope().value_or(StringRef()));

    // Clean up the load -> fpext
    pattern.fpextOp.getRes().replaceAllUsesWith(newLoad.getRes());
    pattern.fpextOp.erase();
    if (pattern.loadOp.getRes().use_empty()) {
      pattern.loadOp.erase();
    }

    if (pattern.gepOp && pattern.gepOp.getRes().use_empty()) {
      pattern.gepOp.erase();
    }

    LLVM_DEBUG(llvm::dbgs() << "  Transformation complete.\n");
  }
}

// Clean up unused narrow buffer operations after transformation.
// If the narrow buffer has no remaining uses, we can remove the fptrunc stores,
// the fptrunc ops (if only used by the store), and the narrow alloca.
static void
cleanupUnusedNarrowBufferOps(SmallVector<LoadFPExtPattern> &safePatterns) {
  // Collect all narrow buffers and their associated fptrunc stores
  DenseMap<Value, SmallVector<FPTruncStoreInfo *>> bufferToStores;
  for (auto &pattern : safePatterns) {
    for (auto *info : pattern.matchingStores) {
      bufferToStores[info->narrowBuffer].push_back(info);
    }
  }

  DenseSet<Operation *> erased;
  for (auto &[narrowBuffer, stores] : bufferToStores) {
    // Build a whitelist of operations we plan to erase (stores + their GEPs).
    DenseSet<Operation *> trackedOps;
    for (auto *info : stores) {
      trackedOps.insert(info->narrowStore.getOperation());
      if (info->narrowGep)
        trackedOps.insert(info->narrowGep.getOperation());
    }

    bool hasOtherUses =
        llvm::any_of(narrowBuffer.getUsers(), [&](Operation *user) {
          return !trackedOps.contains(user);
        });

    if (hasOtherUses) {
      LLVM_DEBUG(llvm::dbgs() << "Narrow buffer still has other uses, "
                              << "keeping fptrunc stores\n");
      continue;
    }

    // No other uses, so we can clean up the fptrunc stores and related ops
    LLVM_DEBUG(llvm::dbgs() << "Cleaning up unused narrow buffer ops\n");
    for (auto *info : stores) {
      // Capture the fptrunc op before erasing the store
      auto fptruncOp = info->narrowStore.getValue().getDefiningOp<FPTruncOp>();

      // Erase the narrow store
      if (!erased.contains(info->narrowStore.getOperation())) {
        erased.insert(info->narrowStore.getOperation());
        info->narrowStore.erase();
        LLVM_DEBUG(llvm::dbgs() << "\tErased narrow store\n");
      }

      // Erase the GEP if it has no uses
      if (info->narrowGep && info->narrowGep.getRes().use_empty() &&
          !erased.contains(info->narrowGep.getOperation())) {
        erased.insert(info->narrowGep.getOperation());
        info->narrowGep.erase();
        LLVM_DEBUG(llvm::dbgs() << "\tErased narrow GEP\n");
      }

      // Erase the fptrunc if it has no uses
      if (fptruncOp && fptruncOp.getRes().use_empty() &&
          !erased.contains(fptruncOp.getOperation())) {
        erased.insert(fptruncOp.getOperation());
        fptruncOp.erase();
        LLVM_DEBUG(llvm::dbgs() << "\tErased fptrunc\n");
      }
    }

    // Erase the narrow alloca if it has no uses
    if (auto narrowAlloca = narrowBuffer.getDefiningOp<AllocaOp>()) {
      if (narrowAlloca.getResult().use_empty() &&
          !erased.contains(narrowAlloca.getOperation())) {
        erased.insert(narrowAlloca.getOperation());
        narrowAlloca.erase();
        LLVM_DEBUG(llvm::dbgs() << "\tErased narrow alloca\n");
      }
    }
  }
}

struct RockRemoveRedundantCastsPass
    : public rock::impl::RockRemoveRedundantCastsPassBase<
          RockRemoveRedundantCastsPass> {
  void runOnOperation() override;
};

} // end namespace

void RockRemoveRedundantCastsPass::runOnOperation() {
  LLVMFuncOp funcOp = getOperation();
  OpBuilder builder(funcOp.getContext());

  LLVM_DEBUG(llvm::dbgs() << "Running RockRemoveRedundantCastsPass on "
                          << funcOp.getName() << "\n");

  // Step 1: Find all fptrunc -> store patterns
  SmallVector<FPTruncStoreInfo> storeInfo = findFPTruncStorePatterns(funcOp);
  if (storeInfo.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No fptrunc -> store patterns found.\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << storeInfo.size()
                          << " fptrunc -> store patterns.\n");

  // Step 2: Find all load -> fpext patterns
  SmallVector<LoadFPExtPattern> loadFPExtPatterns =
      findLoadFPExtPatterns(funcOp);
  if (loadFPExtPatterns.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No load+fpext patterns found.\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Found " << loadFPExtPatterns.size()
                          << " load+fpext patterns.\n");

  // Step 3: Verify safety (applicability) for each pattern
  DominanceInfo domInfo(funcOp);
  SmallVector<LoadFPExtPattern> safePatterns;
  for (auto &pattern : loadFPExtPatterns) {
    LLVM_DEBUG(llvm::dbgs()
               << "Verifying pattern: load=" << pattern.loadOp << "\n");
    FailureOr<SmallVector<FPTruncStoreInfo *>> result =
        verifySafety(pattern, storeInfo, domInfo);
    if (succeeded(result)) {
      pattern.matchingStores = *result;
      safePatterns.push_back(pattern);
    }
  }

  if (safePatterns.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No safe patterns to optimize.\n");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Found " << safePatterns.size()
                          << " safe pattern combination(s) to optimize.\n");

  // Step 3.5: Select consistent wide buffers for pre-existing parallel stores
  selectConsistentWideBuffers(safePatterns, domInfo);

  // Step 4: Create wide buffers and stores for patterns that need them
  createWideBuffersAndStores(safePatterns, builder);

  // Step 5: Apply transformation (redirect loads to wide buffer)
  applyTransformation(safePatterns, builder);

  // Step 6: Clean up unused narrow buffer operations
  cleanupUnusedNarrowBufferOps(safePatterns);

  LLVM_DEBUG(llvm::dbgs() << "Optimized " << safePatterns.size()
                          << " patterns.\n");
}
