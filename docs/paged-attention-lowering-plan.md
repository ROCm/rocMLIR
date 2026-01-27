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
- `sourceCoord[0]` is the **offset-in-page in elements** (single flat index, not multi-dim coords)
- `SugarToLoops.cpp` simply converts this element offset to bytes for the buffer load

**Why a single flat offset?**

The offset-in-page must be computed using `computeFlatPosition` which walks the transform chain. This computation happens in `ThreadwiseGemmLowering.cpp` where the transform chain is still available. By the time we reach `SugarToLoops.cpp`, the transform chain has been consumed by `untransform()` and we only have the raw buffer. Passing a single pre-computed offset avoids the need for `SugarToLoops.cpp` to understand the transform semantics.

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

#### 5b. Modify the Load Loop (Compute Offset-in-Page Here)

Inside the `TransformingForOp` loop body, for paged loads we compute **both** the page index and the offset-in-page. We pass the offset as a single flat index to `GlobalLoadOp`:

```cpp
if (isPagedLoad) {
  ValueRange logicalCoords = loadLoop.getLowerCoords(/*domain=*/0);
  
  // 1. Compute flat position using transform chain
  //    This is the ONLY place that understands the transform semantics
  FailureOr<Value> flatPosOrFail = 
      computeFlatPosition(b, loc, sourceView, logicalCoords);
  if (failed(flatPosOrFail))
    return failure();
  Value flatPos = *flatPosOrFail;
  
  // 2. Compute page index AND offset-in-page
  Value pageSizeVal = b.createOrFold<arith::ConstantIndexOp>(loc, *pageSize);
  Value pageIdx = arith::DivUIOp::create(b, loc, flatPos, pageSizeVal);
  Value offsetInPage = arith::RemUIOp::create(b, loc, flatPos, pageSizeVal);
  
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
  //    Pass SINGLE flat offset (offsetInPage in elements)
  //    SugarToLoops.cpp just converts to bytes - no transform knowledge needed
  Value loaded = GlobalLoadOp::create(b, loc, loadType, buffer, validity,
                                      ValueRange{offsetInPage},  // single flat index
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

**Key insight**: The `computeFlatPosition` utility walks the transform chain and evaluates the affine maps. This is only possible at the `ThreadwiseGemmLowering` level where we still have access to `sourceView` with its transforms. By computing `offsetInPage` here and passing it as a single index, `SugarToLoops.cpp` doesn't need any knowledge of the transform chain.

### Step 6: Modify GlobalLoadOp Lowering in SugarToLoops.cpp

In the existing `GlobalLoadRewritePattern`, add a check for the optional `pagePtr` attribute. Since we receive a **single flat offset** (offset-in-page in elements), the lowering is straightforward:

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
    
    // 1. coords[0] is already the offset-in-page (in elements)
    //    No linearization needed - ThreadwiseGemmLowering computed this
    Value offsetInPageElems = coords[0];
    
    // 2. Convert to bytes for buffer load
    Value offsetBytes = arith::MulIOp::create(b, loc, offsetInPageElems, 
        b.createOrFold<arith::ConstantIndexOp>(loc, elemBytes));
    Value offsetI32 = arith::IndexCastOp::create(b, loc, b.getI32Type(), offsetBytes);
    
    // 3. Convert i64 page pointer to LLVM pointer (global address space = 1)
    auto ptrType = LLVM::LLVMPointerType::get(b.getContext(), /*addressSpace=*/1);
    Value ptr = LLVM::IntToPtrOp::create(b, loc, ptrType, pagePtr);
    
    // 4. Create buffer resource (V#)
    //
    // Buffer descriptor flags (from AMDGPUToROCDL.cpp::makeBufferRsrc):
    //   bits 0-11:  dst sel (ignored by loads)
    //   bits 12-14: data format (must be nonzero, 7=float)
    //   bits 15-18: num format (must be nonzero, 4=32bit)
    //   bit 19:     nested heap (0)
    //   bit 20:     behavior on unmap (0 = return 0)
    //   bits 21-22: index stride for swizzles (N/A)
    //   bit 23:     add thread ID (0)
    //   bit 24:     reserved, must be 1 on RDNA, 0 on CDNA
    //   bits 25-26: reserved (0)
    //   bit 27:     non-volatile (CDNA only)
    //   bits 28-29: OOB select (RDNA only: 2=none, 3=bounds check)
    //   bits 30-31: type (must be 0)
    //
    // Detect RDNA vs CDNA from architecture string:
    StringRef arch = rock::getArchValue(op);
    bool isRDNA = arch.starts_with("gfx10") || arch.starts_with("gfx11") || 
                  arch.starts_with("gfx12");
    
    Value stride = b.createOrFold<LLVM::ConstantOp>(loc, b.getI16Type(), 
                                                     b.getI16IntegerAttr(0));
    Value numRecords = b.createOrFold<LLVM::ConstantOp>(loc, b.getI64Type(),
        b.getI64IntegerAttr(pageSizeBytes));  // buffer size in bytes
    
    uint32_t flags = (7 << 12) | (4 << 15);  // base: data format + num format
    if (isRDNA) {
      flags |= (1 << 24);          // RDNA reserved bit
      flags |= (3 << 28);          // OOB select = 3 (bounds check enabled)
    }
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
- **Single flat offset received**: `coords[0]` is the pre-computed offset-in-page in elements
- **No linearization needed**: `ThreadwiseGemmLowering` already evaluated the transform chain
- **Simple byte conversion**: Just multiply by element size
- **Page size in elements**: Stored as elements, converted to bytes only for buffer descriptor
- **Reuses existing infrastructure**: `perHardwareOp` for vectorization, validity handling, etc.

**Cross-page boundary handling**:

Since `offsetInPageElems` is computed from the transform chain and we know `pageSize`, we can statically or dynamically check if a vectorized load would cross the page boundary. However, there are simpler approaches:

1. **Constraint on page size**: Require `pageSize` to be a multiple of the maximum vector length (e.g., 128 bits / elemBitWidth). This ensures vectorized loads never cross page boundaries.

2. **Fallback to scalar**: If a load would cross a page boundary, emit scalar loads instead. This is simpler but slower for boundary cases.

3. **Emit multiple loads**: Split the vector load at the page boundary. The first part loads from the current page, then we'd need to load the next page pointer and continue. This is complex and likely not worth the implementation cost.

**Recommended approach**: Require `pageSize % maxVectorElems == 0` as a constraint. Document this requirement and validate it at the `BlockwiseLoadTileToThreadwise` level.

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
- Each thread computes its own `flatPos`, `pageIdx`, and `offsetInPage` using `computeFlatPosition`
- Each thread independently loads its page pointer from `ldsPagePtrs[localPageIdx]`
- Each thread passes its pre-computed `offsetInPage` (single flat index) to `GlobalLoadOp`

This is the same pattern as non-paged loads, just with additional page index computation. The key insight is that `computeFlatPosition` is called in `ThreadwiseGemmLowering` where we still have access to the transform chain, ensuring consistent flat position calculation across the codebase.

### Multi-Page Access Across Iterations

A single thread may need to load data from **multiple pages** across different loop iterations. The design handles this correctly because the page pointer is computed **inside the loop body**, not outside:

```cpp
// TransformingForOp loop body - executes once per iteration
if (isPagedLoad) {
  // 1. Get THIS iteration's coordinates (varies each iteration)
  ValueRange logicalCoords = loadLoop.getLowerCoords(/*domain=*/0);
  
  // 2. Compute flat position for THIS iteration
  Value flatPos = computeFlatPosition(b, loc, sourceView, logicalCoords);
  
  // 3. Compute which page THIS iteration needs
  Value pageIdx = arith::DivUIOp::create(b, loc, flatPos, pageSizeVal);
  Value offsetInPage = arith::RemUIOp::create(b, loc, flatPos, pageSizeVal);
  
  // 4. Load THIS iteration's page pointer from LDS
  Value localPageIdx = arith::SubIOp::create(b, loc, pageIdx, firstPageIdx);
  Value pagePtr = memref::LoadOp::create(b, loc, ldsPagePtrs, localPageIdx);
  
  // 5. Create GlobalLoadOp with THIS iteration's page pointer
  GlobalLoadOp::create(..., pagePtr, ...);
}
```

**Example**: A thread loading 32 elements with vectorLen=8 and pageSize=256:

| Iteration | flatPos range | pageIdx | localPageIdx | Page Pointer |
|-----------|---------------|---------|--------------|--------------|
| 0 | 248-255 | 0 | 0 | `ldsPagePtrs[0]` → page N |
| 1 | 256-263 | 1 | 1 | `ldsPagePtrs[1]` → page N+1 |
| 2 | 264-271 | 1 | 1 | `ldsPagePtrs[1]` → page N+1 |
| 3 | 272-279 | 1 | 1 | `ldsPagePtrs[1]` → page N+1 |

Each iteration dynamically computes its page index and loads the appropriate pointer from the `ldsPagePtrs` LDS buffer. The `LoadPagePointers` stage (in `BlockwiseLoadTileToThreadwise`) pre-loads all page pointers needed by the tile into this LDS buffer, so the per-iteration LDS loads are fast.

**Key invariant**: The `ldsPagePtrs` buffer must contain pointers for all pages that any thread in the tile might access. This is ensured by the `numPagesForTile` calculation in `BlockwiseLoadTileToThreadwise`.

**Cross-page within a single vector load**: The above handles crossing pages **between iterations**. Crossing pages **within a single vector load** (e.g., elements 252-259 spanning pages N and N+1) is NOT supported and must be prevented via the page size constraint (see Open Questions #2).

### Why Compute Offset-in-Page Early?

The offset-in-page must be computed using `computeFlatPosition` which walks the transform chain and evaluates affine maps. This can only be done in passes that have access to the source view with its transforms:

| Pass | Has Transform Chain? | Can Compute Flat Position? |
|------|---------------------|---------------------------|
| `BlockwiseLoadTileToThreadwise` | ✅ Yes | ✅ Yes (for `firstPageIdx`) |
| `ThreadwiseGemmLowering` | ✅ Yes (via `sourceView`) | ✅ Yes (for `offsetInPage`) |
| `SugarToLoops` | ❌ No (consumed by `untransform()`) | ❌ No |

By computing `offsetInPage` in `ThreadwiseGemmLowering` and passing it as a single index to `GlobalLoadOp`, we avoid the need for `SugarToLoops` to understand transform semantics. This keeps `SugarToLoops` simple and focused on emitting hardware operations.

### Passes That Need Paging Awareness

Since we're adding optional attributes to existing ops (not creating new ops), **no additional passes need modification**.

- `AnalyzeMemoryUse.cpp` - Already checks `GlobalLoadOp`, works unchanged
- `AddAsyncWait.cpp` - Already checks `ThreadwiseReadIntoOp`, works unchanged  
- `AlignTiling.cpp` - Needs update to preserve paging attributes (covered in Phase 2)

## Constraints and Requirements

### Page Size Constraint for Vectorization

To avoid complex cross-page boundary handling, we **require**:

```cpp
pageSize % maxVectorElems == 0
```

where `maxVectorElems = 128 / elemBitWidth` (e.g., 8 for f16, 4 for f32).

**Rationale**: This ensures that if a vector load starts at an aligned offset within a page, it cannot cross into the next page. Since our loop iterations advance by the vector length, maintaining alignment throughout.

**Implementation**: Validate this constraint in `BlockwiseLoadTileToThreadwise` and emit a descriptive error if violated:

```cpp
if (isPagedLoad) {
  int64_t maxVectorElems = 128 / elementType.getIntOrFloatBitWidth();
  if (pageSize % maxVectorElems != 0) {
    return op.emitError("pageSize (") << pageSize 
        << ") must be a multiple of max vector elements (" << maxVectorElems 
        << ") to avoid cross-page vector loads";
  }
}
```

### Monotonic Thread-to-Position Mapping

We assume that the transform chain produces **monotonically increasing flat positions** with increasing `tid` and `iter`. Specifically, `tid=0, iter=0` gives the minimum flat position in the tile, which is used to compute `firstPageIdx`.

**Why this is safe for attention**: The tiling transforms in rocMLIR are designed to:
1. **Match hardware accelerator layouts** (MFMA on CDNA, WMMA on RDNA) which expect specific, positive-stride data orderings
2. **Enable memory coalescing** where adjacent threads access adjacent memory addresses
3. **Maximize LDS data reuse** through standard rectangular tiling patterns

Reversed or negative-stride mappings would provide no computational benefit for attention (the matrix operations are order-agnostic) and would hurt memory performance. All standard attention tiling patterns naturally produce monotonically increasing mappings.

**Runtime validation**: The design includes an assertion that `localPageIdx >= 0` (see Step 5b in ThreadwiseGemmLowering). If this ever triggers, it indicates a bug in the tiling transforms, not in the paging logic.

## Open Questions

1. **numPagesForTile formula verification**: Current formula assumes a specific memory layout:
   ```cpp
   int64_t span = (dPerBlock - 1) * kGlobal + (kPerBlock - 1);
   numPagesForTile = (pageSize - 1 + span) / pageSize + 1;
   ```
   Need to verify it matches the actual flat position range for all threads in the tile. Add tests with various tile sizes and page sizes.

2. **[RESOLVED] Flat position formula**: The flat position formula:
   ```
   flat = (g * seqK_size + seqK) * headDim_size + headDim
   ```
   
   **Is computed by the transform chain's affine maps**, not by assuming row-major order of the source shape. Tracing through the actual IR:
   
   ```mlir
   // DerefOp output: [batch=1, numPages=64, pageSize=8192]
   %4 = rock.deref %1 -> memref<1x64x8192xf16>
   
   // Merge pages: total = numPages * pageSize
   %6 = transform %4 by Merge{64, 8192} -> memref<1x524288xf16>
   
   // Unmerge to logical dims: total = (numHeadsKV * 4096 + seqK) * 64 + headDimQK
   %7 = transform %6 by Unmerge{2, 4096, 64} -> memref<1x2x4096x64xf16>
   
   // Final shape reordering
   %8 = transform %7 -> memref<2x64x4096xf16>  // [G, headDimQK, seqK]
   ```
   
   The affine map in the `Unmerge` transform encodes: `(d1 * 4096 + d2) * 64 + d3`, which gives our formula when composed with the final reordering.
   
   **Conclusion**: `computeFlatPosition` walks the transform chain and evaluates these affine maps, so it produces the correct flat position regardless of what the "logical" source shape appears to be. This is NOT row-major order of `[G, headDimQK, seqK]` (which would be `g * 262144 + headDim * 4096 + seqK`), but that's expected and correct.