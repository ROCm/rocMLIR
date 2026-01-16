# Design: Lowering GridwisePagedAttentionAccelOp to Blockwise Ops

## Overview

This document describes the design for lowering `rock.gridwise_paged_attention_accel` 
to blockwise operations in `GridwiseGemmToBlockwise.cpp`.

## Current Regular Attention Flow (GridwiseAttentionAccelOp)

The existing `GridwiseAttentionAccelRewritePattern` follows this structure:

### 1. Setup Phase
- Extract element types and shapes from Q, K, V, output
- Get tuning parameters (gemm0/gemm1 params)
- Compute derived values (block sizes, number of blocks, etc.)
- Select accelerator emitters for GEMM0 and GEMM1
- Set up grid/workgroup coordinates

### 2. Buffer Allocation
- Allocate LDS buffers for Q, K, V tiles
- Allocate register buffers for:
  - Pre-accelerator intermediate buffers (K, Q, V)
  - Accumulator buffers (gemm0, gemm1)
  - Output buffers (gemm0Out, gemm1Out, attention output)
  - Softmax state (maxRowBuffer, sumRowBuffer, expMaxDiffRowBuffer)

### 3. Optional Q Prefetch (if gemm0K == gemm0KPerBlock)
- Load Q tile outside the M-loop when possible

### 4. M-Loop (over key sequence tiles)
```
for mIter in range(start, end):
    Zero accumulator for GEMM0
    
    // K-Loop (over head dimension tiles)
    for kIter in range(gemm0K / gemm0KPerBlock):
        Load Q tile (if not prefetched)
        Load K tile                           <-- KEY DIFFERENCE FOR PAGED ATTENTION
        LDS barrier
        Blockwise GEMM0 (Q × K^T)
        LDS barrier
    
    Post-process GEMM0 output:
    - Apply fusion ops (preSoftmaxBody)
    - Scale by 1/ln2
    - Handle padding with -inf
    - Apply causal/KV-cache masking
    
    Softmax:
    - Max reduction
    - exp(x - max) normalization
    - Sum reduction
    - Update running row state
    
    // GEMM1 M-Loop (over head dimension tiles)
    for g1MIter in range(gemm1MBlocks):
        Load V tile                           <-- KEY DIFFERENCE FOR PAGED ATTENTION
        LDS barrier
        Blockwise GEMM1 (softmax_output × V)
        LDS barrier
        Post-process: accumulate with row state correction
```

### 5. Final Output
- Scale final output by 1/rowSum
- Compute LSE (if needed)
- Write output to global memory
- Write LSE to global memory (if needed)

## Paged Attention Key Differences

### 1. K/V Loading via `loadFromKVCache` Region

Instead of direct `inK` and `inV` memrefs, paged attention has:
- `keyPagePointers`: memref of pointers to key pages
- `valuePagePointers`: memref of pointers to value pages
- `keyAddressMask` / `valueAddressMask`: validity masks

The `loadFromKVCache` region contains the logic to:
1. Dereference page pointers (`rock.paged_deref`)
2. Apply transforms to reshape pages to K/V format
3. Copy to block arguments (which become the K/V tiles)

### 2. Block Arguments in loadFromKVCache

The region has 4 block arguments:
- `%arg6: memref<...xi64>` - key page pointers (input)
- `%arg7: memref<G x HeadDim x SeqLen xf16>` - keys buffer (output)
- `%arg8: memref<...xi64>` - value page pointers (input) 
- `%arg9: memref<G x SeqLen x HeadDim xf16>` - values buffer (output)

The region ends with `memref.copy` operations that copy transformed data 
to `%arg7` and `%arg9`.

## Design Options

### Option A: "Inline and Replace" the loadFromKVCache Region

**Approach**: Replace the `memref.copy` destinations with tile-sized buffers,
and inline the region logic at the appropriate points in the M-loop.

**Challenges**:
1. The transforms in the region may not tile cleanly
2. Need to adjust block argument types for tile sizes
3. Complex region manipulation

### Option B: Materialize K/V Tiles via Region Execution

**Approach**: At each M-loop iteration, "execute" the loadFromKVCache region
conceptually by:
1. Allocating tile-sized buffers for K and V
2. Computing the slice of K/V needed for this tile
3. Generating code that loads from paged memory into the tile buffers

**Challenges**:
1. Need to extract the paged_deref and transform logic
2. Need to apply appropriate tiling transforms

### Option C: Use Region as Template for Tiled Loading (Recommended)

**Approach**: 
1. Use the `loadFromKVCache` region as a *template* that describes:
   - How to dereference page pointers (`rock.paged_deref`)
   - How to transform page data into K/V format (`rock.transform` chain)
2. At each M-loop iteration:
   - Extract the paged_deref → transform chain from the region
   - Add tiling transforms on top (to slice out the current tile)
   - Generate `rock.load_from_address` ops to load tile into LDS
3. Handle multi-page tiles:
   - If a tile spans multiple pages, generate loads for each page
   - The address mask handles validity for each page access

**Why this approach**:
- Preserves the region abstraction (the template describes the logical K/V layout)
- Allows flexible tiling based on tuning parameters
- Handles multi-page tiles naturally
- Uses new `rock.load_from_address` op for explicit pointer-based loading

## Proposed Implementation Plan

### Phase 1: Create `GridwisePagedAttentionAccelRewritePattern`

Structure similar to `GridwiseAttentionAccelRewritePattern`:

```cpp
struct GridwisePagedAttentionAccelRewritePattern
    : public OpRewritePattern<GridwisePagedAttentionAccelOp> {
  
  LogicalResult matchAndRewrite(GridwisePagedAttentionAccelOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Setup phase (mostly reusable from regular attention)
    // 2. Buffer allocation (mostly reusable)
    // 3. Optional Q prefetch (same as regular attention)
    // 4. M-Loop with paged K/V loading
    // 5. Final output (same as regular attention)
  }
};
```

### Phase 2: Implement Paged K/V Loading

The key new functionality is loading K/V tiles from paged memory.
We need a helper function like:

```cpp
static void loadPagedKVTile(
    PatternRewriter &rewriter, Location loc,
    Region &loadFromKVCacheRegion,
    Value mLoopIV,               // Current M iteration
    Value tid,                   // Thread ID
    layout::GridCoordinates gridCoords,
    Value destLDSBufferK,        // Destination LDS for K tile
    Value destLDSBufferV,        // Destination LDS for V tile
    /* tuning params, etc. */
) {
    // 1. Compute which pages are needed for this tile
    //    based on mLoopIV and tuning params
    
    // 2. For each thread, determine its portion of the tile
    
    // 3. Use rock.load_from_address to load from page pointers
    //    to load the K portion into LDS
    
    // 4. Similarly for V
}
```

### Phase 3: Extracting and Tiling the paged_deref + transform chain

The `loadFromKVCache` region contains:
```mlir
// Dereference key pages
%28 = rock.paged_deref %arg6, %16 : memref<...xi64>, memref<...xi1> -> memref<...xf16>
// Transform chain to reshape pages to K format
%29 = rock.transform %28 by ... : memref<...> to memref<...>
%30 = rock.transform %29 by ... : memref<...> to memref<...>
...
%36 = rock.transform %35 by ... : memref<...> to memref<K_shape>
// Copy to output block argument
memref.copy %36, %arg7 : memref<K_shape> to memref<K_shape>
```

**Strategy**:
1. Walk the region to find the final transformed value before each `memref.copy`
2. Extract the SSA value that represents the "logical" K/V view
3. Add tiling transforms on top to slice for the current M-iteration
4. Use `rock.load_from_address` to load into LDS tile buffer

**Key insight**: The transform chain ending at `%36` already describes how to
view the paged memory as a logical K tensor of shape `[G, HeadDim, SeqLen]`.
We just need to add a tiling transform that selects:
- The current `mLoopIV * mPerBlock` to `(mLoopIV + 1) * mPerBlock` slice of SeqLen

**Multi-page handling**: If the tile spans multiple pages:
- The `paged_deref` with address mask will handle validity
- We generate loads that may access multiple page boundaries
- Invalid accesses (beyond address mask) return zeros or are masked out

## Clarified Design Decisions

Based on discussion, the following decisions have been made:

1. **Tile-level Loading**: ✅ Confirmed
   - K/V tiles are loaded one per M-loop iteration (same as regular attention)
   - Each M-iteration loads a `mPerBlock`-sized slice of the key sequence

2. **Memory Stability**: ✅ Confirmed
   - Pages are stable during attention computation
   - Address masks ensure validity; we don't need to handle page invalidation

3. **Multi-page Tiles**: ✅ Confirmed
   - If a tile spans multiple pages, we load from multiple pages
   - Page size/block size is determined before attention computation
   - The paged_deref + mask handles boundary cases

4. **K-loop vs M-loop Loading**: ✅ Confirmed
   - K-loop (inner loop) iterates over head dimension, loading Q tiles
   - M-loop (outer loop) iterates over key sequence, loading K tiles from paged cache
   - This is the same structure as regular attention

5. **V Loading Pattern**: ✅ Confirmed
   - V loading in GEMM1 M-loop follows the same pattern as K loading in GEMM0 M-loop
   - Both use the paged_deref + transform chain from `loadFromKVCache` region

6. **Transform Chain Handling**:
   - The `loadFromKVCache` region contains the full transform chain from pages → logical K/V
   - We need to add **tiling transforms on top** to slice out individual tiles
   - The existing `applyTransformsInLoadFromKVCacheRegion` in GemmToGridwise handles
     normalization/padding; here we add the M-iteration slicing

## Assumptions

1. **Transforms include normalization/padding**: The `loadFromKVCache` region 
   already has normalization (transpose) and padding transforms applied by 
   `applyTransformsInLoadFromKVCacheRegion` in GemmToGridwise.

2. **Tiling transforms needed**: We must add tiling transforms on top of the
   existing chain to slice out individual M-iteration tiles.

3. **paged_deref produces valid memrefs**: The `rock.paged_deref` op handles
   page pointer dereferencing, and the address mask handles validity.

4. **Same tuning applies**: The same GEMM0/GEMM1 tuning parameters work for
   both regular and paged attention.

5. **Element types match**: K/V element types from paged cache match what
   regular attention expects.

6. **Pages are contiguous within each page**: While the KV cache is paged
   (non-contiguous across pages), each page itself is a contiguous memory region.

## Implementation Milestones

1. **Milestone 1**: Stub pattern that mirrors regular attention structure
2. **Milestone 2**: Implement K tile loading from paged cache (M-loop)
3. **Milestone 3**: Implement V tile loading from paged cache (GEMM1 M-loop)
4. **Milestone 4**: Handle softmax and output correctly
5. **Milestone 5**: Test with real paged attention workloads

## Code Reuse Strategy

Many components can be shared between regular and paged attention:

| Component | Reusable? | Notes |
|-----------|-----------|-------|
| Setup (types, shapes, params) | Mostly | K/V types from region block args |
| Buffer allocation | Yes | Same buffers needed |
| Q loading | Yes | Same logic |
| K loading | **No** | Paged version needed |
| V loading | **No** | Paged version needed |
| Softmax ops | Yes | Identical |
| GEMM0 compute | Yes | Same `blockwiseGemmAccel` |
| GEMM1 compute | Yes | Same `blockwiseGemmAccel` |
| Output write | Yes | Same logic |
| LSE write | Yes | Same logic |

Consider extracting shared logic into helper functions or a base class.

## Detailed Implementation Plan

### Step 1: Create Pattern Skeleton

Create `GridwisePagedAttentionAccelRewritePattern` that mirrors the structure
of `GridwiseAttentionAccelRewritePattern`:

```cpp
struct GridwisePagedAttentionAccelRewritePattern
    : public OpRewritePattern<GridwisePagedAttentionAccelOp> {
  using OpRewritePattern<GridwisePagedAttentionAccelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GridwisePagedAttentionAccelOp op,
                                PatternRewriter &rewriter) const override;
};
```

### Step 2: Extract K/V Types from Region

Instead of `op.getKeys()` and `op.getValues()`, we use:
```cpp
Type keysType = op.getKeysType();    // From region block arg %arg7
Type valuesType = op.getValuesType(); // From region block arg %arg9
```

### Step 3: Implement Paged K Loading Helper

```cpp
// Load K tile from paged cache for current M-iteration
static void loadPagedKeyTile(
    PatternRewriter &rewriter, Location loc,
    GridwisePagedAttentionAccelOp op,
    Value mLoopIV,                    // Current M iteration index
    Value tid,                        // Thread ID
    layout::GridCoordinates gridCoords,
    Value destLDSBuffer,              // Destination: LDS tile buffer
    Value destRegBuffer,              // Destination: register buffer
    GemmLoadTileType loadType,
    int64_t blockSize,
    RockAccelTuningParamAttrInterface tuningParams,
    GemmFeaturesAttr features,
    BlockwiseMatrixParamsAttr matrixParams
) {
    // 1. Walk loadFromKVCache region to find key's paged_deref + transforms
    Region &region = op.getLoadFromKVCache();
    Value keyTransformedView = extractKeyTransformChain(region);
    
    // 2. Add tiling transform for current M-iteration
    //    Slice: [mLoopIV * mPerBlock, (mLoopIV+1) * mPerBlock]
    Value tiledKeyView = addMIterationSlice(rewriter, loc, 
        keyTransformedView, mLoopIV, tuningParams.getMPerBlock());
    
    // 3. Use loadAndStoreGemmInputTile or equivalent
    loadAndStoreGemmInputTile(
        rewriter, loc, tiledKeyView, /*kiter=*/zero, tid, gridCoords,
        destLDSBuffer, destRegBuffer, loadType, "m", blockSize,
        /* element types */, tuningParams, features, matrixParams);
}
```

### Step 4: Implement Paged V Loading Helper

Similar to Step 3, but for V:
```cpp
static void loadPagedValueTile(
    PatternRewriter &rewriter, Location loc,
    GridwisePagedAttentionAccelOp op,
    Value mLoopIV,                    // From GEMM0's M-loop (key sequence position)
    Value g1MLoopIV,                  // From GEMM1's M-loop (head dim tiles)
    Value tid,
    layout::GridCoordinates gridCoords,
    Value destLDSBuffer,
    Value destRegBuffer,
    /* ... */
) {
    // Similar structure to loadPagedKeyTile
}
```

### Step 5: Integrate into Main Pattern

Replace the regular K/V loading calls in the M-loop and GEMM1 M-loop:

**Before (regular attention)**:
```cpp
loadAndStoreGemmInputTile(rewriter, loc, inK, kLoopIV, ...);
```

**After (paged attention)**:
```cpp
loadPagedKeyTile(rewriter, loc, op, mLoopIV, tid, gridCoords, 
                 ldsByteBufferK, preAccelRegBufferK, ...);
```

### Step 6: Register Pattern

Add to `RockGridwiseGemmToBlockwisePass::runOnOperation()`:
```cpp
target.addIllegalOp<..., GridwisePagedAttentionAccelOp>();
patterns.add<..., GridwisePagedAttentionAccelRewritePattern>(ctx);
```

## Open Questions for Implementation

1. **How to extract the transform chain from the region?**
   - Walk backwards from `memref.copy` to find the source value
   - That source value has the full paged_deref + transform chain attached

2. **How to add tiling transforms dynamically?**
   - Use `TopDownTMBuilder` or `BottomUpTMBuilder` to add a Slice transform
   - The slice bounds depend on `mLoopIV` (dynamic) and `mPerBlock` (static)

3. ~~**How does ThreadwiseReadIntoOp handle paged_deref?**~~ **RESOLVED**
   - Use `rock.load_from_address` op instead of relying on paged_deref lowering

4. **What about the preSoftmaxBody region?**
   - Handle the same way as regular attention (already done in GemmToGridwise)

## New Op: `rock.load_from_address`

### Problem Statement

The Rock dialect operates on memrefs, not raw pointers. For paged attention:
- `pagePointers = memref<1x64x1xi64>` contains 64 pointers to pages
- Each i64 is a memory address pointing to 8192 contiguous f16 values
- We need to read the i64 address and load data from it

Currently, `rock.paged_deref` has **no lowering pattern** - it stays in the IR.

### Proposed Op Definition

```tablegen
def Rock_LoadFromAddressOp
    : Rock_Op<"load_from_address", 
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      Arguments<(ins 
          I64:$address,         // Base address (i64 pointer value)
          Index:$offset,        // Byte offset from base
          I1:$valid,            // Validity flag (for masking)
          IndexAttr:$length     // Number of elements to load
      )>,
      Results<(outs AnyType:$result)> {
  let summary = "Load data from a raw memory address";
  let description = [{
    Loads elements from a raw memory address. If valid is false, returns zeros.
    
    Example:
    %addr = memref.load %pagePointers[%pageIdx] : memref<64xi64>
    %data = rock.load_from_address %addr, %offset if %valid {length = 8}
            : i64, index -> vector<8xf16>
  }];
}
```

### Lowering to LLVM

```mlir
// 1. Convert i64 address to pointer
%ptr = llvm.inttoptr %address : i64 to !llvm.ptr

// 2. Compute byte offset  
%gep = llvm.getelementptr %ptr[%offset] : (!llvm.ptr, index) -> !llvm.ptr

// 3. Conditional load
%result = scf.if %valid -> vector<8xf16> {
  %loaded = llvm.load %gep : !llvm.ptr -> vector<8xf16>
  scf.yield %loaded
} else {
  %zeros = arith.constant dense<0.0> : vector<8xf16>
  scf.yield %zeros
}
```

### Usage in Paged Attention

```cpp
// 1. Compute page index for current M-iteration
Value pageIdx = computePageIndex(mLoopIV, mPerBlock, pageSize);

// 2. Load the page pointer
Value pageAddr = memref::LoadOp::create(rewriter, loc, 
    op.getKeyPagePointers(), ValueRange{c0, pageIdx, c0});

// 3. Compute offset within page for this thread
Value offsetInPage = computeThreadOffset(tid, ...);

// 4. Load validity from address mask
Value valid = memref::LoadOp::create(rewriter, loc,
    op.getKeyAddressMask(), ValueRange{...});

// 5. Load data using the new op
Value kData = rock::LoadFromAddressOp::create(rewriter, loc,
    vectorType, pageAddr, offsetInPage, valid, lengthAttr);

// 6. Store to LDS
rock::InBoundsStoreOp::create(rewriter, loc, kData, ldsByteBufferK, ...);
```

### Multi-Page Tile Handling

If a tile spans multiple pages:

```cpp
for (int p = 0; p < pagesPerTile; p++) {
  Value pageIdx = arith.AddIOp::create(..., basePageIdx, p);
  Value pageAddr = memref::LoadOp::create(..., pageIdx, ...);
  Value valid = memref::LoadOp::create(...);
  Value data = rock::LoadFromAddressOp::create(...);
  rock::InBoundsStoreOp::create(..., p * pageSize, ...);
}
```

## Updated Implementation Milestones

0. **Milestone 0**: Define and implement `rock.load_from_address` op
   - Add op definition to RockOps.td
   - Add verification logic  
   - Add lowering to LLVM dialect

1. **Milestone 1**: Stub pattern mirroring regular attention
2. **Milestone 2**: K tile loading using `load_from_address`
3. **Milestone 3**: V tile loading
4. **Milestone 4**: Softmax and output
5. **Milestone 5**: Testing
