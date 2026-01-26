# Paged Attention: ThreadwiseReadInto Lowering Plan

## Overview

This document outlines the plan for lowering paged attention operations in `mlir/lib/Dialect/Rock/Transforms/ThreadwiseGemmLowering.cpp`. This is the next phase after `BlockwiseLoadTileToThreadwise.cpp`, which creates the page pointer loading infrastructure.

## Background

### Current State After BlockwiseLoadTileToThreadwise

After the `BlockwiseLoadTileToThreadwise` pass, we have:

1. **LoadPagePointers stage**: Loads page pointers from the page table into LDS
   - Thread 0 loads `pageTable[batch, firstPageIdx + 0, 0]` → `ldsPagePtrs[0]`
   - Thread 1 loads `pageTable[batch, firstPageIdx + 1, 0]` → `ldsPagePtrs[1]`
   - Barrier ensures all threads see the loaded pointers

2. **GlobalRead stage**: Contains `ThreadwiseReadIntoOp` with optional paging attributes
   - Has access to `ldsPagePtrs`, `firstPageIndex`, and `pageSize` (when paged)
   - Needs to be lowered to actual memory loads (`PagedGlobalLoadOp` for paged, `GlobalLoadOp` for non-paged)

### Passes That Interact with ThreadwiseReadIntoOp

| Pass | Interaction |
|------|-------------|
| **ThreadwiseGemmLowering.cpp** | Lowers `ThreadwiseReadIntoOp` to actual memory loads |
| **AlignTiling.cpp** | Input fusion - creates/modifies `ThreadwiseReadIntoOp` |
| **AddAsyncWait.cpp** | Analyzes global loads to add async wait ops |
| **OutputSwizzle.cpp** | Creates `ThreadwiseReadIntoOp` operations |
| **BlockwiseGemmToThreadwise.cpp** | Creates `ThreadwiseReadIntoOp` operations |

## Design Decision: Optional Attributes vs. Separate Op

### Recommendation: Add Optional Paging Attributes to ThreadwiseReadIntoOp

Rather than maintaining a separate `ThreadwisePagedReadIntoOp`, we recommend adding optional paging attributes to the existing `ThreadwiseReadIntoOp`:

**Rationale:**
- ✅ Existing passes (`AlignTiling.cpp`, `AddAsyncWait.cpp`) continue to work unchanged
- ✅ Single lowering pattern with conditional paging logic
- ✅ Less code duplication and maintenance burden
- ✅ Paged loads are still "global loads" from the perspective of async waits

**New Optional Attributes:**
```tablegen
// Added to Rock_ThreadwiseReadIntoOp
Optional<MemRefOf<[I64]>>:$ldsPagePtrs,    // Pre-loaded page pointers in LDS
Optional<Index>:$firstPageIndex,           // First page index for this tile  
OptionalAttr<IndexAttr>:$pageSize          // Elements per page
```

## Transform Chain Review

### K Tensor Transform Chain

```
Source shape: memref<2x64x4096xf16>  [G, headDimQK, seqK]

After gridSubTile transforms:
  memref<2x2x128x329x64x16xf16>  [k_loop, g_block, m_block, n_block, tid, iter]

The affine map encodes:
  (k_loop, g_block, m_block, n_block, tid, iter) -> (g, headDim, seqK)
  
Where:
  g = g_block
  headDim = (k_loop * 16 + k_thread) * 2 + k_iter
  seqK = (m_block * 4 + m_thread) * 8 + m_iter
  
And tid decomposes to:
  k_thread = tid % 16
  m_thread = tid / 16
  
And iter decomposes to:
  k_iter = iter % 2
  m_iter = iter / 2
```

### Flat Position Formula

For K tensor coordinates `(g, headDim, seqK)`, the flat position in the virtual paged space is:
```
flat = (g * seqK_size + seqK) * headDim_size + headDim
     = (g * 4096 + seqK) * 64 + headDim
```

From flat position:
```
pageIndex = flat / pageSize
offsetInPage = flat % pageSize
```

## Lowering Plan

### Step 1: Extract Shared Utility Function

We already have `computeTileStartFlat` in `BlockwiseLoadTileToThreadwise.cpp` that walks the transform chain and evaluates affine maps. This should be extracted to a utility file for reuse.

**New file: `mlir/lib/Dialect/Rock/utility/TransformUtils.h` and `.cpp`**

```cpp
// TransformUtils.h
#include "mlir/IR/Value.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace rock {

/// Compute the flat position in the virtual paged space by evaluating
/// the source transforms on the logical coordinates.
/// Stops at DerefOp to stay in the virtual address space.
/// 
/// This is used by:
/// - BlockwiseLoadTileToThreadwise.cpp: to compute firstPageIdx for a tile
/// - ThreadwiseGemmLowering.cpp: to compute per-element flat positions
FailureOr<Value> computeFlatPosition(OpBuilder &b, Location loc,
                                      Value source, ValueRange indices);

} // namespace rock
} // namespace mlir
```

The implementation is essentially the same logic as the existing `computeTileStartFlat`:
1. Walk the transform chain on `source`, stopping at `DerefOp`
2. Compose affine maps from all transforms
3. Evaluate with `affine::expandAffineMap`
4. Return the last coordinate (flat position)

### Step 2: Modify ThreadwiseReadIntoOp Definition

In `mlir/include/mlir/Dialect/Rock/IR/RockOps.td`:

```tablegen
def Rock_ThreadwiseReadIntoOp
    : Rock_ThreadwiseReadOpBase<"threadwise_read_into"> {

  dag additionalArgs = (ins
      Variadic<VectorOfNonZeroRankOf<[I1]>>:$dynamicValidities,
      OptionalAttr<Rock_LDSTransposeConfigAttr>:$ldsTransposeConfig,
      // Paging support (all optional)
      Optional<MemRefOf<[I64]>>:$ldsPagePtrs,
      Optional<Index>:$firstPageIndex,
      OptionalAttr<IndexAttr>:$pageSize);

  let arguments = !con(commonReadArgs, additionalArgs);
  // ... rest unchanged ...
}
```

**Note**: `pageSize` is stored in **elements** (not bytes). Conversion to bytes happens only in `SugarToLoops.cpp`.

### Step 3: Update BlockwiseLoadTileToThreadwise.cpp

1. Replace `ThreadwisePagedReadIntoOp` creation with `ThreadwiseReadIntoOp` with paging attributes
2. Replace inline `computeTileStartFlat` with call to shared utility
3. **FIX BUG**: Ensure validity record type is passed for paged loads (currently missing!)

```cpp
if (isPagedLoad) {
  // Use regular ThreadwiseReadIntoOp with paging attributes
  // IMPORTANT: Pass validityRecordType (was missing in previous implementation!)
  ThreadwiseReadIntoOp::create(b, loc,
      vectorOfBoolShapedLike(loadBuffer),  // validity record type - MUST include!
      wrappedSource, loadBuffer,
      /*dynamicValidities=*/ValueRange{},
      /*extraViews=*/b.getArrayAttr({}),
      /*extraIndices=*/indices,
      forceUnroll, /*useIndexDiffs=*/true,
      /*ldsTransposeConfig=*/nullptr,
      // Paging attributes:
      ldsPagePtrs,
      firstPageIdx,
      b.getIndexAttr(pageSize));
} else {
  // Existing non-paged path (unchanged)
  ThreadwiseReadIntoOp::create(b, loc, vectorOfBoolShapedLike(loadBuffer), ...);
}
```

### Step 4: Add Paging Attributes to GlobalLoadOp (Simpler Approach)

Rather than creating new ops, we add optional paging attributes to the existing `GlobalLoadOp` (and `GlobalLoadToLDSOp`). When these attributes are present, the lowering uses the page pointer directly instead of the memref base.

**Modified `GlobalLoadOp` in `RockOps.td`:**

```tablegen
def Rock_GlobalLoadOp : Rock_Op<"global_load", ...> {
  let arguments = (ins
      Arg<MemRefOf<SupportedMemoryElems>, "source memory">:$source,
      I1:$valid, 
      Variadic<Index>:$sourceCoord, 
      UnitAttr:$needs64BitIdx,
      UnitAttr:$canReadOffEnd,
      // Paging support (optional) - if present, use pagePtr as base
      Optional<I64>:$pagePtr,           // Base address of page (overrides source)
      OptionalAttr<I64Attr>:$pageSize); // Page size in bytes for bounds
  // ...
}
```

**Benefits of this approach:**
- ✅ **No new ops** - reuse existing `GlobalLoadOp` and `GlobalLoadToLDSOp`
- ✅ **All passes work unchanged** - they already handle these ops
- ✅ **Direct-to-LDS works automatically** - same pattern applies
- ✅ **Vectorization preserved** - same `perHardwareOp` logic
- ✅ **Less code to maintain**

When `pagePtr` is present:
- The `source` memref is still used for type information (element type, etc.)
- But the actual load uses `pagePtr` as the base address
- `sourceCoord[0]` is treated as byte offset within the page

### Step 5: Modify ThreadwiseGemmLowering.cpp

#### 5a. Detect Paged Load

```cpp
LogicalResult ThreadwiseReadIntoRewritePattern::matchAndRewrite(
    ThreadwiseReadIntoOp op, ...) {
  // Check if this is a paged load
  Value ldsPagePtrs = op.getLdsPagePtrs();
  Value firstPageIdx = op.getFirstPageIndex();
  std::optional<int64_t> pageSize;
  if (auto pageSizeAttr = op.getPageSizeAttr())
    pageSize = pageSizeAttr.getInt();
  bool isPagedLoad = ldsPagePtrs && firstPageIdx && pageSize.has_value();
  
  // ... existing setup code ...
```

#### 5b. Modify the Load Loop (Keep Multi-Dimensional Coordinates)

Inside the `TransformingForOp` loop body, for paged loads we still need to determine **which page** to use. But we keep multi-dimensional coordinates for the load itself:

```cpp
if (isPagedLoad) {
  ValueRange logicalCoords = loadLoop.getLowerCoords(/*domain=*/0);
  
  // 1. Compute flat position to determine which page
  FailureOr<Value> flatPosOrFail = 
      computeFlatPosition(b, loc, sourceView, logicalCoords);
  if (failed(flatPosOrFail))
    return failure();
  Value flatPos = *flatPosOrFail;
  
  // 2. Compute page index
  Value pageSizeVal = b.createOrFold<arith::ConstantIndexOp>(loc, *pageSize);
  Value pageIdx = arith::DivUIOp::create(b, loc, flatPos, pageSizeVal);
  
  // 3. Get local page index and load page pointer from LDS
  Value localPageIdx = arith::SubIOp::create(b, loc, pageIdx, firstPageIdx);
  
  // Assert: localPageIdx >= 0 and < numPagesForTile
  // This should always hold because:
  // - firstPageIdx is computed from tile origin (tid=0, iter=0)
  // - All threads access positions >= tile origin
  // Add debug assertions for safety:
  // assert(localPageIdx >= 0 && "pageIdx should never be less than firstPageIdx");
  
  Value pagePtr = memref::LoadOp::create(b, loc, ldsPagePtrs, localPageIdx);
  
  // 4. Emit GlobalLoadOp with paging attributes
  //    Keep original multi-dimensional coordinates!
  //    SugarToLoops.cpp will compute flat offset from coords
  Value loaded = GlobalLoadOp::create(b, loc, loadType, buffer, validity,
                                      logicalCoords,  // KEEP multi-dim coords
                                      needs64BitIdx,
                                      /*canReadOffEnd=*/false,
                                      pagePtr,       // page base pointer
                                      b.getI64IntegerAttr(*pageSize));
  
  InBoundsStoreOp::create(b, loc, loaded, dest, destIndex);
} else {
  // Existing non-paged path (unchanged)
  Value loaded = GlobalLoadOp::create(b, loc, loadType, buffer, validity,
                                      loadLoop.getLowerCoords(/*domain=*/0),
                                      needs64BitIdx);
  InBoundsStoreOp::create(b, loc, loaded, dest, destIndex);
}
```

**Note**: We still compute `flatPos` to determine which page, but we pass the original `logicalCoords` to `GlobalLoadOp`. The lowering in `SugarToLoops.cpp` computes the offset-within-page from these coordinates.

### Step 6: Modify GlobalLoadOp Lowering in SugarToLoops.cpp

In the existing `GlobalLoadRewritePattern`, add a check for the optional `pagePtr` attribute:

```cpp
LogicalResult matchAndRewrite(GlobalLoadOp op, PatternRewriter &b) const override {
  Location loc = op.getLoc();
  Value source = op.getSource();
  Value valid = op.getValid();
  SmallVector<Value> coords(op.getSourceCoord());
  MemRefType srcType = cast<MemRefType>(source.getType());
  Type elemType = srcType.getElementType();
  int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
  
  // Check if this is a paged load
  Value pagePtr = op.getPagePtr();
  bool isPagedLoad = pagePtr != nullptr;
  
  if (isPagedLoad) {
    // === Paged load path: use ROCDL intrinsics directly ===
    int64_t pageSize = op.getPageSizeAttr().getInt();  // in elements
    int64_t pageSizeBytes = pageSize * elemBytes;
    
    // 1. Compute flat offset from multi-dimensional coordinates
    //    Use the source memref's layout to linearize coordinates
    Value flatOffset = computeLinearIndex(b, loc, srcType, coords);
    
    // 2. Compute offset within page (in elements, then convert to bytes)
    Value pageSizeVal = b.createOrFold<arith::ConstantIndexOp>(loc, pageSize);
    Value offsetInPage = arith::RemUIOp::create(b, loc, flatOffset, pageSizeVal);
    Value offsetBytes = arith::MulIOp::create(b, loc, offsetInPage, 
        b.createOrFold<arith::ConstantIndexOp>(loc, elemBytes));
    Value offsetI32 = arith::IndexCastOp::create(b, loc, b.getI32Type(), offsetBytes);
    
    // 3. Convert i64 page pointer to LLVM pointer (global address space = 1)
    auto ptrType = LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/1);
    Value ptr = LLVM::IntToPtrOp::create(b, loc, ptrType, pagePtr);
    
    // 4. Create buffer resource (V#)
    Value stride = b.createOrFold<LLVM::ConstantOp>(loc, b.getI16Type(), 
                                                     b.getI16IntegerAttr(0));
    Value numRecords = b.createOrFold<LLVM::ConstantOp>(loc, b.getI64Type(),
        b.getI64IntegerAttr(pageSizeBytes));  // buffer size in bytes
    
    uint32_t flags = (7 << 12) | (4 << 15);
    if (isRDNA) flags |= (1 << 24) | (3 << 28);
    Value flagsVal = b.createOrFold<LLVM::ConstantOp>(loc, b.getI32Type(),
        b.getI32IntegerAttr(flags));
    
    auto rsrcType = LLVM::LLVMPointerType::get(b.getContext(), /*addrSpace=*/8);
    Value rsrc = ROCDL::MakeBufferRsrcOp::create(b, loc, rsrcType, 
                                                  ptr, stride, numRecords, flagsVal);
    
    // 5. Emit vectorized load using perHardwareOp (reuse existing logic!)
    Value soffset = b.createOrFold<LLVM::ConstantOp>(loc, b.getI32Type(), 
                                                      b.getI32IntegerAttr(0));
    Value aux = b.createOrFold<LLVM::ConstantOp>(loc, b.getI32Type(),
                                                  b.getI32IntegerAttr(0));
    
    Value loaded = createZeroConstantOp(b, loc, loadedType);
    perHardwareOp(loadedType, [&](int64_t hwOffset, Type thisOpTy) {
      Value thisOffsetI32 = offsetI32;
      if (hwOffset != 0) {
        Value hwOffsetI32 = b.createOrFold<arith::ConstantIntOp>(loc, 
            hwOffset * elemBytes, 32);
        thisOffsetI32 = arith::AddIOp::create(b, loc, offsetI32, hwOffsetI32);
      }
      Value thisLoad = ROCDL::RawPtrBufferLoadOp::create(b, loc, thisOpTy, 
                                                          rsrc, thisOffsetI32, 
                                                          soffset, aux);
      // ... insert into loaded vector ...
    });
    
    // 6. Handle validity (same as existing)
    // ...
    
    b.replaceOp(op, loaded);
  } else {
    // === Existing non-paged path (unchanged) ===
    // ... existing amdgpu::RawBufferLoadOp logic ...
  }
  
  return success();
}
```

**Key design points**:
- **Multi-dimensional coordinates preserved**: `GlobalLoadOp` still receives the original coordinates
- **Flat offset computed in lowering**: `SugarToLoops.cpp` linearizes coords using source layout
- **Page size in elements**: Stored as elements, converted to bytes only for buffer descriptor
- **Reuses existing infrastructure**: `perHardwareOp` for vectorization, validity handling, etc.

**Cross-page boundary handling (Final Vector Size Filter)**:

The page boundary check is applied as an **additional final filter** after all existing vector size decisions (alignment, memory layout, hardware constraints, etc.). This filter only reduces the vector size further if needed to avoid crossing a page boundary:

```cpp
// In the perHardwareOp loop:
perHardwareOp(loadedType, [&](int64_t hwOffset, Type thisOpTy) {
  // requestedVecLen already reflects all existing filters (alignment, etc.)
  int64_t requestedVecLen = getVectorLength(thisOpTy);
  
  // FINAL FILTER: Cap to remaining space in current page
  Value offsetWithHw = arith::AddIOp::create(b, loc, offsetInPage,
      b.createOrFold<arith::ConstantIndexOp>(loc, hwOffset));
  Value remainingInPage = arith::SubIOp::create(b, loc, pageSizeVal, offsetWithHw);
  
  // Only reduce vector size if it would cross page boundary
  // actualVecLen = min(requestedVecLen, remainingInPage), clamped to power-of-2
  Value actualVecLen = computeLargestFittingVector(b, loc, remainingInPage, requestedVecLen);
  
  // Emit load with potentially smaller vector (only if page boundary would be crossed)
  // ...
});
```

This preserves all existing vector size optimizations and only applies an additional constraint when a load would otherwise cross a page boundary.

## Implementation Order

### Phase 1: Extract Utility Function
1. Create `mlir/lib/Dialect/Rock/utility/TransformUtils.h` and `.cpp`
2. Move `computeTileStartFlat` logic from `BlockwiseLoadTileToThreadwise.cpp` to new utility
3. Generalize as `computeFlatPosition` for reuse in multiple passes
4. Update `BlockwiseLoadTileToThreadwise.cpp` to use the new utility

### Phase 2: Op Definition Changes (Cleanup + Extend)
**Part A - Cleanup previous implementation:**
1. **Remove `Rock_ThreadwiseReadOpBase`** class from `RockOps.td`
2. **Revert `ThreadwiseReadIntoOp`** to its original standalone definition
3. **Remove `ThreadwisePagedReadIntoOp`** definition entirely
4. **Update `RockDialect.cpp`**: Remove all `ThreadwisePagedReadIntoOp` implementations

**Part B - Add paging support:**
5. **Add optional paging attributes** to `ThreadwiseReadIntoOp`:
   - `Optional<MemRefOf<[I64]>>:$ldsPagePtrs`
   - `Optional<Index>:$firstPageIndex`
   - `OptionalAttr<IndexAttr>:$pageSize` (in **elements**, not bytes)
6. **Add optional paging attributes** to `GlobalLoadOp`:
   - `Optional<I64>:$pagePtr`
   - `OptionalAttr<I64Attr>:$pageSize` (in **elements**, not bytes)
7. **Add optional paging attributes** to `GlobalLoadToLDSOp` (same pattern)

**Part C - Update AlignTiling.cpp:**
8. **Preserve paging attributes** when cloning/modifying `ThreadwiseReadIntoOp`
   - Ensure all optional attributes are copied during op replacement
   - Add test coverage for paged loads going through AlignTiling

### Phase 3: Update BlockwiseLoadTileToThreadwise.cpp
1. Replace `ThreadwisePagedReadIntoOp` creation with `ThreadwiseReadIntoOp` with paging attributes
2. Update to use the new shared `computeFlatPosition` utility

### Phase 4: Implement Lowering in ThreadwiseGemmLowering.cpp
1. Add paged load detection (check for optional paging attributes)
2. Use shared `computeFlatPosition` utility
3. Emit `GlobalLoadOp` with paging attributes (reuse existing op!)
4. Add conditional paged load path in the load loop body

### Phase 5: Modify GlobalLoadOp Lowering in SugarToLoops.cpp

**5a. GlobalLoadRewritePattern (loads to registers):**
1. Add `#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"` to SugarToLoops.cpp
2. Add paged load check for `pagePtr` attribute
3. When `pagePtr` is present:
   - Skip `amdgpu::RawBufferLoadOp` (requires memref)
   - Emit ROCDL ops directly: `llvm.inttoptr` + `rocdl.make.buffer.rsrc` + `rocdl.raw.ptr.buffer.load`
4. Reuse existing `perHardwareOp` logic for vectorization
5. Apply page boundary vector size filter as final step

**5b. GlobalLoadToLDSRewritePattern (direct-to-LDS):**
1. Add same paged load check for `pagePtr` attribute
2. When `pagePtr` is present: use `llvm.inttoptr` + `rocdl.make.buffer.rsrc` + `rocdl.raw.ptr.buffer.load.lds`
3. Factor out `createBufferResourceFromPagePtr(b, loc, pagePtr, pageSizeBytes, chipset)` helper

**Investigation confirms this is viable:**
- ✅ `ROCDL::MakeBufferRsrcOp` already used in `RockPrepareLLVM.cpp`
- ✅ `ROCDL::RawPtrBufferLoadOp` signature: `(rsrc: ptr<8>, offset: i32, soffset: i32, aux: i32) → result`
- ✅ `ROCDL::RawPtrBufferLoadLdsOp` signature: `(rsrc: ptr<8>, ldsPtr: ptr<3>, size: i32, voffset: i32, soffset: i32, offset: i32, aux: i32)`
- ✅ Buffer flag setup logic available in `AMDGPUToROCDL.cpp::makeBufferRsrc` (lines 160-184)

### Phase 6: Testing
1. Verify IR after each pass
2. Ensure vectorization is preserved (check for vector loads, not scalar)
3. Run with actual GPU to verify correctness
4. Performance testing and comparison with non-paged path

## Files to Modify

| File | Changes |
|------|---------|
| `mlir/lib/Dialect/Rock/utility/TransformUtils.h` | **NEW**: Shared flat position computation |
| `mlir/lib/Dialect/Rock/utility/TransformUtils.cpp` | **NEW**: Implementation |
| `mlir/include/mlir/Dialect/Rock/IR/RockOps.td` | **REMOVE**: `Rock_ThreadwiseReadOpBase`, `ThreadwisePagedReadIntoOp`; **ADD**: paging attrs to `ThreadwiseReadIntoOp`, `GlobalLoadOp`, `GlobalLoadToLDSOp` |
| `mlir/lib/Dialect/Rock/IR/RockDialect.cpp` | **REMOVE**: All `ThreadwisePagedReadIntoOp` implementations |
| `mlir/lib/Dialect/Rock/Transforms/BlockwiseLoadTileToThreadwise.cpp` | **REPLACE**: `ThreadwisePagedReadIntoOp` → `ThreadwiseReadIntoOp` with paging attrs; **USE**: shared utility |
| `mlir/lib/Dialect/Rock/Transforms/ThreadwiseGemmLowering.cpp` | Add paged load detection, emit `GlobalLoadOp` with paging attrs |
| `mlir/lib/Dialect/Rock/Transforms/SugarToLoops.cpp` | Modify `GlobalLoadRewritePattern` and `GlobalLoadToLDSRewritePattern` to handle paged loads via ROCDL |

## Design Considerations

### Per-Thread Computation

The paged load logic in Step 5b runs **per-thread independently**:

- The `TransformingForOp` creates a loop that each thread executes
- `logicalCoords` are computed from transforms that encode the thread-to-data mapping (based on `tid` and `iter`)
- Each thread computes its own `flatPos`, `pageIdx`, and `offsetInPage`
- Each thread independently loads its page pointer from `ldsPagePtrs[localPageIdx]`

This is the same pattern as non-paged loads, just with additional page index computation.

### Passes That Need Paging Awareness

Since we're adding optional attributes to existing ops (not creating new ops), **no additional passes need modification**.

- `AnalyzeMemoryUse.cpp` - Already checks `GlobalLoadOp`, works unchanged
- `AddAsyncWait.cpp` - Already checks `ThreadwiseReadIntoOp`, works unchanged  
- `AlignTiling.cpp` - Needs update to preserve paging attributes (covered in Phase 2)

## Open Questions

1. **numPagesForTile formula verification**: Current formula assumes a specific memory layout:
   ```cpp
   int64_t span = (dPerBlock - 1) * kGlobal + (kPerBlock - 1);
   numPagesForTile = (pageSize - 1 + span) / pageSize + 1;
   ```
   Need to verify it matches the actual flat position range for all threads in the tile. Add tests with various tile sizes and page sizes.