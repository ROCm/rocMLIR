# Design: Unified Paged Attention via Expanded rock.deref

## Overview

This document describes a unified approach to paged attention that avoids 
duplicating attention ops and lowering patterns. Instead of separate 
`PagedAttentionOp` and `GridwisePagedAttentionAccelOp`, we:

1. Expand `rock.deref` to encapsulate address computation and masking
2. Add optional deref operands to the existing attention ops
3. Reuse all existing attention lowering infrastructure

## Motivation

The current approach (separate paged attention ops) has drawbacks:
- Code duplication in GemmToGridwise patterns
- Code duplication in GridwiseGemmToBlockwise patterns
- Two paths to maintain for any attention optimization
- Separate op definitions that mirror each other

The unified approach addresses these by keeping paging logic in `rock.deref`
and treating paged attention as "regular attention with paged K/V sources".

## Current State (TOSA level)

Looking at the IR before Rock lowering:

```mlir
// Address computation + masking (lines 730-743 of before-all.mlir)
%33 = tosa.mul %expanded_3, %20 ...     // Compute page addresses
%35 = tosa.add %33, %34 ...             // Combine offsets
%43 = ... -> tensor<1x64x8192xi1>       // Validity mask
%44 = tosa.select %43, %36, %35 ...     // Select real or fallback addresses

// The deref - loads data from computed addresses
%45 = tosa.custom %44 {operator_name = "deref"} 
      : (tensor<1x64x8192xi64>) -> tensor<1x64x8192xf16>

// External transforms (reshape, GQA broadcast)
%collapsed = tensor.collapse_shape %45 ...
%expanded = tensor.expand_shape %collapsed ...
%keys = tosa.mul %expanded, %broadcast ...   // GQA

// Used in attention
%result = tosa.matmul %queries, %keys ...
```

## Proposed Design

### 1. Expanded `rock.deref` Op

The deref op contains a region with the address computation logic:

```tablegen
def Rock_DerefOp : Rock_Op<"deref", [SingleBlock, IsolatedFromAbove]>,
    Arguments<(ins
        TensorOrMemRefOf<[I64]>:$pagePointers   // Array of page pointers
    )>,
    Results<(outs AnyTensorOrMemRef:$output)> {
    
  let regions = (region SizedRegion<1>:$addressComputation);
  
  let summary = "Load data from paged memory with computed addresses";
  let description = [{
    The `rock.deref` op loads data from non-contiguous paged memory.
    
    The `addressComputation` region computes the final addresses to load from,
    including any masking logic. The region:
    - Receives `pagePointers` as block argument
    - Contains address computation ops (arithmetic, masking, select)
    - Yields a tensor/memref of final i64 addresses (after masking/fallback)
    
    The op then loads data from these addresses to produce the output.
    
    Example:
    ```mlir
    %keys = rock.deref %pagePointers {
      ^bb0(%ptrs: tensor<1x64x1xi64>):
        // Broadcast page pointers to full address space
        %expanded = ... expand %ptrs to <1x64x8192xi64> ...
        // Add per-element offsets
        %addrs = arith.addi %expanded, %offsets
        // Compute validity mask from sequence length
        %mask = arith.cmpi slt, %indices, %seqLen
        // Select real vs fallback addresses
        %final = arith.select %mask, %addrs, %fallback
        rock.yield %final : tensor<1x64x8192xi64>
    } : tensor<1x64x1xi64> -> tensor<1x64x8192xf16>
    ```
    
    The pagePointers shape is [batch, numPages, 1] where each element is
    an i64 memory address pointing to a contiguous page of data.
  }];
}
```

**Key points:**
- Only `pagePointers` as input (no separate addressInputs)
- Uses `TensorOrMemRefOf` to work pre and post bufferization
- Region captures all address computation logic from TOSA level
- Region yields the computed addresses; the op handles the actual loading

### 2. Attention Op with Optional Address Operands

Add optional operands to reference the deref sources:

```tablegen
// In Rock_AttentionOpBase or Rock_AttentionOp
let arguments = (ins
    // Existing operands
    TensorOrMemRefOf<...>:$queries,
    TensorOrMemRefOf<...>:$keys,
    TensorOrMemRefOf<...>:$values,
    // ... other existing operands ...
    
    // NEW: Optional address sources (outputs from deref ops)
    // Uses TensorOrMemRef to work pre and post bufferization
    Optional<TensorOrMemRefOf<[F16, F32, BF16]>>:$keyAddresses,
    Optional<TensorOrMemRefOf<[F16, F32, BF16]>>:$valueAddresses
);
```

**Naming**: Using `keyAddresses` and `valueAddresses` instead of 
`keyDerefSource` / `valueDerefSource` for clarity.

Usage:
```mlir
// Deref ops produce the raw K/V data
%keys_raw = rock.deref %keyPtrs { /* address computation region */ }
%values_raw = rock.deref %valuePtrs { /* address computation region */ }

// External transforms (reshape, GQA, etc.)
%keys = rock.transform %keys_raw by ...
%values = rock.transform %values_raw by ...

// Attention with address source references
rock.attention(%queries, %keys, %values)
    keyAddresses = %keys_raw
    valueAddresses = %values_raw
    -> %output
```

### 3. GridwiseAttentionAccelOp

Similarly add optional address operands (post-bufferization, so memref):

```tablegen
// In Rock_GridwiseAttentionAccelOp or base class
let arguments = (ins
    // ... existing operands ...
    
    // Optional address sources - carried through from high-level attention
    Optional<MemRefOf<[F16, F32, BF16]>>:$keyAddresses,
    Optional<MemRefOf<[F16, F32, BF16]>>:$valueAddresses
);
```

## Lowering Flow

### TosaToRock

This requires expanding the existing TosaToRock lowering. Currently, the
`tosa.custom` deref op is converted to `rock.paged_deref`. We need to:

1. **Detect the address computation pattern** (lines 730-743 in TOSA IR)
   - Find the ops that compute addresses from page pointers
   - Find the masking/select logic
   
2. **Create `rock.deref` with populated region**
   - Clone the address computation ops into the deref region
   - Set pagePointers from the original source
   
3. **Handle external transforms**
   - The reshape/GQA broadcast ops after deref become external transforms
   - These are applied as normal Rock transforms

4. **Create attention op with address references**
   - Set keyAddresses/valueAddresses to the deref outputs

```cpp
// Pseudocode for TosaToRock
LogicalResult matchAndRewrite(AttentionPattern op, ...) {
  // Check if K/V come from deref pattern
  auto keyDerefPattern = matchDerefPattern(op.getKeys());
  auto valueDerefPattern = matchDerefPattern(op.getValues());
  
  if (keyDerefPattern && valueDerefPattern) {
    // Create rock.deref ops with address computation regions
    auto keyDeref = createDerefWithAddressRegion(
        rewriter, loc, keyDerefPattern);
    auto valueDeref = createDerefWithAddressRegion(
        rewriter, loc, valueDerefPattern);
    
    // Apply external transforms to deref outputs
    Value keys = applyExternalTransforms(keyDeref.getOutput(), 
                                         keyDerefPattern.transforms);
    Value values = applyExternalTransforms(valueDeref.getOutput(),
                                           valueDerefPattern.transforms);
    
    // Create attention with address references
    rock::AttentionOp::create(rewriter, loc,
        queries, keys, values,
        /*keyAddresses=*/keyDeref.getOutput(),
        /*valueAddresses=*/valueDeref.getOutput(),
        /* ... other operands ... */);
  } else {
    // Regular (non-paged) attention
    // ... existing logic ...
  }
}
```

### Bufferization

The `rock.deref` op requires a bufferizable interface to convert from tensor
to memref semantics. We already have `PagedDerefOpInterface` in
`BufferizableOpInterfaceImpl.cpp` that handles this.

**Updates needed for expanded deref:**

1. **Region handling**: The `addressComputation` region operates on tensors
   internally. During bufferization, we need to:
   - Bufferize the `pagePointers` input 
   - The region contents can remain as tensor operations (they describe the
     address computation pattern, not actual memory operations)
   - Bufferize the output to a memref

2. **Interface implementation**:
```cpp
struct DerefOpInterface
    : public BufferizableOpInterface::ExternalModel<DerefOpInterface,
                                                    DerefOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // pagePointers are read
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Doesn't write to inputs - produces new result
    return false;
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    auto derefOp = mlir::cast<DerefOp>(op);

    // Get buffer for pagePointers
    FailureOr<Value> pagePointersBuffer =
        getBuffer(rewriter, derefOp.getPagePointers(), options, state);
    if (failed(pagePointersBuffer))
      return failure();

    // Determine the result memref type
    auto resultType = derefOp.getOutput().getType();
    MemRefType resultMemRefType;
    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
      resultMemRefType = MemRefType::get(tensorType.getShape(),
                                         tensorType.getElementType());
    } else {
      resultMemRefType = cast<MemRefType>(resultType);
    }

    // Create new op with memref types
    // Note: The region is cloned as-is - it describes address computation
    replaceOpWithNewBufferizedOp<DerefOp>(
        rewriter, op, resultMemRefType, *pagePointersBuffer);
    return success();
  }
};
```

### GemmToGridwise

Mostly unchanged! The pattern:
1. Creates `GridwiseAttentionAccelOp` as before
2. Passes through `keyAddresses` / `valueAddresses` if present
3. All existing transform logic (normalization, padding, GQA) works unchanged

The K/V values used by attention have already been transformed; the address
operands are just carried through for later use in GridwiseGemmToBlockwise.

**Note on Tuning Parameters**: Tile sizes (`mPerBlock`, `nPerBlock`, etc.) are 
NOT set in GemmToGridwise. They are set in `AffixTuningParameters.cpp`:
- Default perf_config: `"attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1"`
- Parameters: `mPerBlockG0, mPerBlockG1, nPerBlockG0, kpackPerBlock, ...`

**No changes needed in AffixTuningParameters** - our ToBlockwise code must 
handle any tile/page size combination. The tuning infrastructure will 
naturally discover the best tile sizes through performance testing.

```cpp
// In AttentionRewritePattern (GemmToGridwise)
LogicalResult matchAndRewrite(AttentionOp op, ...) {
  // ... existing setup code (unchanged) ...
  
  // Pass through the address sources (may be null for non-paged)
  Value keyAddresses = op.getKeyAddresses();
  Value valueAddresses = op.getValueAddresses();
  
  auto gridwiseOp = GridwiseAttentionAccelOp::create(rewriter, loc,
      /* ... existing operands (unchanged) ... */
      keyAddresses,
      valueAddresses);
  
  // ... existing region handling (unchanged) ...
}
```

### GridwiseGemmToBlockwise

This is where paged loading happens. Key insight:

**The K/V arguments have transforms applied (reshape, GQA, normalization, 
padding). But we don't need to "invert" these transforms to load from pages.**

Instead:
1. The `keyAddresses`/`valueAddresses` point to the **raw deref output**
2. The deref op's region tells us how to compute addresses from page pointers
3. We know the **page layout**: `[batch, numPages, pageSize]` 
4. We know the **tile size** from tuning parameters
5. We can compute: for tile at `mIter`, which page(s) do we need?

### Critical Design Point: From Tile Position to Page Pointers

**The Problem**: The attention op's `keys` operand has transforms applied (GQA 
broadcast, normalization, etc.). The `mLoopIV` indexes into this transformed 
space. But we need to load from the **original paged structure**.

**Key Insight**: The transforms are shape manipulations (broadcast, reshape, 
transpose), NOT actual data movement. The underlying data layout in memory 
hasn't changed - we just have different indexing into it.

**Solution**: The `rock.deref` op's output represents the raw paged data 
BEFORE any transforms. We track this separately via `keyAddresses`. The 
transforms on `keys` tell us how to map from the transformed M-loop index 
back to the original page structure.

```
Original (from deref):  [batch, numPages, pageSize, headDim]
                                    |
                         External transforms (GQA, reshape, normalize, pad)
                                    |
                                    v
Transformed (keys):     [G, seqLen, headDim]  <- mLoopIV indexes here
```

**How to compute the mapping**:

1. **At ToBlockwise time**, we have access to:
   - The transform chain from `deref.output` -> `attention.keys`
   - The `mLoopIV` in transformed space
   - The original page structure from deref

2. **The transforms are invertible** for index computation:
   - Padding: Just clamp to original bounds (handled by masking)
   - GQA broadcast: The sequence dimension is unchanged
   - Transpose/reshape: Reorder indices but same underlying data

3. **For the sequence dimension (M)**, the mapping is typically 1:1:
   - `seqLen = numPages * pageSize` (before any padding)
   - Position `mLoopIV * mPerBlock` in the M-loop maps to the same position 
     in the original paged structure
   - Padding just adds positions at the end (handled by masking)

```cpp
// In GridwiseAttentionAccelRewritePattern
// Given mLoopIV in transformed space:
Value seqPos = mLoopIV * mPerBlock;  // Position in sequence

// This maps directly to the paged structure:
// seqPos / pageSize = which page
// seqPos % pageSize = offset within page

// The transforms don't change this because:
// - GQA broadcast: only affects heads, not sequence position
// - Normalize (transpose): reorders dims but not the sequence indexing
// - Padding: adds to the end, masked out
```

**Why this works**: The sequence dimension flows through transforms unchanged. 
GQA broadcasts across heads, not sequence. Transposition reorders dimensions 
but the actual sequence index is preserved. Padding extends the sequence but 
we mask those positions anyway.

**What if transforms DO affect sequence indexing?**
If a transform actually remaps sequence positions (not just extends via 
padding), we would need to invert that transform. However, in the current 
attention pattern, this doesn't happen - sequence position is preserved.

```cpp
LogicalResult matchAndRewrite(GridwiseAttentionAccelOp op, ...) {
  // ... existing setup (types, shapes, tuning params) ...
  
  // Check if this is paged attention
  Value keyAddresses = op.getKeyAddresses();
  Value valueAddresses = op.getValueAddresses();
  bool isPaged = keyAddresses != nullptr;
  
  // Get the deref ops if paged
  rock::DerefOp keyDeref = nullptr;
  rock::DerefOp valueDeref = nullptr;
  if (isPaged) {
    keyDeref = keyAddresses.getDefiningOp<rock::DerefOp>();
    valueDeref = valueAddresses.getDefiningOp<rock::DerefOp>();
    assert(keyDeref && valueDeref && "Expected deref ops");
  }
  
  // ... buffer allocation (same for both paged and non-paged) ...
  // ... Q loading (same for both) ...
  
  // M-Loop (over key sequence tiles)
  for (mIter = ...) {
    // K loading
    if (keyDeref) {
      // PAGED PATH
      loadPagedKeyTile(keyDeref, mLoopIV, tid, ldsByteBufferK,
                       pageSize, mPerBlock, ...);
    } else {
      // REGULAR PATH (unchanged)
      loadAndStoreGemmInputTile(inK, kLoopIV, tid, ldsByteBufferK, ...);
    }
    
    // ... GEMM0, softmax (same for both) ...
    
    // V loading  
    if (valueDeref) {
      // PAGED PATH
      loadPagedValueTile(valueDeref, g1MLoopIV, tid, ldsByteBufferV,
                         pageSize, mPerBlock, ...);
    } else {
      // REGULAR PATH (unchanged)
      loadAndStoreGemmInputTile(inV, g1MLoopIV, tid, ldsByteBufferV, ...);
    }
    
    // ... GEMM1, output (same for both) ...
  }
}
```

### Paged Tile Loading - Detailed Design

The key challenge: given a tile position, which page(s) and offsets do we need?

#### What We Know and When

**Known at compile time (static):**
- `pageSize`: Number of elements per page (e.g., 8192)
  - Source: Encoded in the deref op's input shape (from TosaToRock)
  - Example: `pagePointers: memref<64xi64>` with output `memref<1x64x8192xf16>`
    means pageSize = 8192
- `mPerBlock`: Elements per tile in M dimension (from tuning params)
  - Source: `tuningParams.getMPerBlock()` in GridwiseGemmToBlockwise
  - Set in AffixTuningParameters from perf_config string
- `numPages`: Total number of pages
  - Source: Deref output shape's second dimension (64 in example above)
- `headDim`: Number of channels per head
  - Source: From the attention op's Q/K/V shapes (separate from page structure)

**Known at runtime (dynamic):**
- `mLoopIV`: Current M-loop iteration (index into key sequence)
- `tid`: Thread ID within workgroup
- `validSeqLen`: Actual sequence length for masking (from `currentSeqLen` operand)
- `batchIdx`/`headIdx`: Current batch and head from grid coordinates

#### Masking Strategy: Skip Entire Tiles When Possible

**Key optimization**: If an entire tile is beyond `validSeqLen`, skip the load entirely.

There are two levels of masking:
1. **Tile-level masking**: If `mLoopIV * mPerBlock >= validSeqLen`, skip the entire tile
2. **Element-level masking**: For partial tiles at the boundary, load zeros for 
   positions beyond `validSeqLen`

#### Page Caching: Avoid Redundant Loads When Page > Tile

**The scenario**: If `pageSize = 8192` and `mPerBlock = 32`, then each page 
contains 256 tiles. We should NOT load from the page table for every tile - 
once we have a page loaded, we can serve 256 consecutive tiles from it.

**Optimization approach**:

```cpp
// Track current loaded page info across M-loop iterations
Value currentPageIdx = allocate register/LDS for page index
Value currentPageAddr = allocate register/LDS for page base address
Value currentPageStartPos = allocate register/LDS for start position in sequence

// In M-loop:
Value tileStartPos = mLoopIV * mPerBlock;
Value neededPageIdx = tileStartPos / pageSize;

// Check if we need to load a new page
Value needNewPage = arith.CmpIOp(arith.CmpIPredicate::ne, 
                                  neededPageIdx, currentPageIdx);

scf.if (needNewPage) {
  // Load new page pointer from page table
  Value newPageAddr = memref.LoadOp(pagePointers, {batch, neededPageIdx});
  // Update tracking
  currentPageIdx = neededPageIdx;
  currentPageAddr = newPageAddr;
  currentPageStartPos = neededPageIdx * pageSize;
}

// Now use currentPageAddr for tile loading
// Offset within page = tileStartPos - currentPageStartPos
Value pageOffset = tileStartPos - currentPageStartPos;
loadTileFromPage(currentPageAddr, pageOffset, ...);
```

**When this matters**:
- pageSize=8192, mPerBlock=32: 256 tiles per page → 255 loads avoided per page
- pageSize=8192, mPerBlock=64: 128 tiles per page → 127 loads avoided per page

**Implementation note**: This optimization adds some state tracking but 
significantly reduces page table accesses. The check `needNewPage` is cheap 
(integer compare), and the branch is predictable (rarely taken).

```cpp
// In the M-loop, before loading:
Value tileStartPos = mLoopIV * mPerBlock;
Value tileFullyMasked = arith.CmpIOp(arith.CmpIPredicate::uge, 
                                      tileStartPos, validSeqLen);

scf.if (tileFullyMasked) {
  // Skip this tile entirely - fill LDS with zeros or use previous values
  // No memory load issued!
} else {
  // Load tile (may have partial masking at the end)
  loadPagedKeyTile(...);
}
```

This is critical for paged attention because:
1. **Avoids unnecessary memory traffic** - no loads for masked positions
2. **Avoids accessing invalid page pointers** - pages beyond validSeqLen may 
   not be allocated
3. **Reduces latency** - memory loads are expensive, skipping them is free

#### How to Compute Which Pages to Load

Given an M-loop iteration, we need to determine:
1. Which page(s) contain the data for this tile
2. The offset within each page
3. Which elements are valid (within actual sequence length)

**Step-by-step computation:**

```
For tile at mLoopIV:
  
  1. Compute global position in K sequence:
     startPos = mLoopIV * mPerBlock
     endPos = startPos + mPerBlock - 1
  
  2. Compute which pages this tile spans:
     startPage = startPos / pageSize
     endPage = endPos / pageSize
     numPagesForTile = endPage - startPage + 1
  
  3. For each page in [startPage, endPage]:
     - pageOffset = (page == startPage) ? (startPos % pageSize) : 0
     - elementsInPage = min(pageSize - pageOffset, remaining elements)
     - Load elements from this page at pageOffset
     - Store to LDS at appropriate offset
  
  4. Apply masking:
     For each element position:
       globalPos = page * pageSize + offset
       valid = globalPos < validSeqLen
```

**Example:**
- pageSize = 8192, mPerBlock = 32, mLoopIV = 255
- startPos = 255 * 32 = 8160
- endPos = 8160 + 31 = 8191
- startPage = 8160 / 8192 = 0
- endPage = 8191 / 8192 = 0
- Tile fits in page 0! Load from offset 8160.

**Example (tile spans pages):**
- pageSize = 8192, mPerBlock = 64, mLoopIV = 127
- startPos = 127 * 64 = 8128
- endPos = 8128 + 63 = 8191
- startPage = 8128 / 8192 = 0
- endPage = 8191 / 8192 = 0
- Still fits in one page!

**Example (truly spanning):**
- pageSize = 8192, mPerBlock = 128, mLoopIV = 63
- startPos = 63 * 128 = 8064
- endPos = 8064 + 127 = 8191
- startPage = 8064 / 8192 = 0
- endPage = 8191 / 8192 = 0
- Still fits! (barely)

**Example (actually spans):**
- pageSize = 8192, mPerBlock = 128, mLoopIV = 64
- startPos = 64 * 128 = 8192
- endPos = 8192 + 127 = 8319
- startPage = 8192 / 8192 = 1
- endPage = 8319 / 8192 = 1
- Fits in page 1!

In practice, with pageSize = 8192 and typical tile sizes (32-128), tiles will
almost always fit within a single page, except at page boundaries.

**Computation (pseudocode):**
```cpp
static void loadPagedKeyTile(
    rock::DerefOp derefOp,
    Value mLoopIV,          // Current M iteration
    Value tid,              // Thread ID
    Value destLDS,          // Destination LDS buffer
    Value validSeqLen,      // Runtime sequence length for masking
    int64_t pageSize,       // Elements per page (static, e.g., 8192)
    int64_t mPerBlock,      // Tile size in M dimension
    int64_t numThreads,     // Threads per workgroup
    ...
) {
  // Get page pointers from deref op
  Value pagePointers = derefOp.getPagePointers();
  
  // Compute starting position in the full K sequence
  // startPos = mLoopIV * mPerBlock
  Value startPos = arith.MulIOp::create(rewriter, loc,
      mLoopIV, 
      arith.ConstantIndexOp::create(rewriter, loc, mPerBlock));
  
  // TILE-LEVEL MASKING: Skip entire tile if fully beyond validSeqLen
  Value tileFullyMasked = arith.CmpIOp::create(rewriter, loc,
      arith.CmpIPredicate::uge, startPos, validSeqLen);
  
  scf.IfOp::create(rewriter, loc, tileFullyMasked, /*withElse=*/true,
    /*then=*/[&](OpBuilder &b, Location loc) {
      // Tile is fully masked - fill LDS with zeros, no memory access!
      fillLDSWithZeros(b, loc, destLDS, mPerBlock);
      b.create<scf::YieldOp>(loc);
    },
    /*else=*/[&](OpBuilder &b, Location loc) {
      // Tile has valid data - proceed with paged loading
      loadPagedKeyTileImpl(b, loc, derefOp, startPos, tid, destLDS, 
                           validSeqLen, pageSize, mPerBlock, numThreads);
      b.create<scf::YieldOp>(loc);
    });
}

// The actual loading logic (only called when tile has valid data)
static void loadPagedKeyTileImpl(
    OpBuilder &rewriter, Location loc,
    rock::DerefOp derefOp,
    Value startPos,         // Starting position in K sequence
    Value tid,              // Thread ID
    Value destLDS,          // Destination LDS buffer
    Value validSeqLen,      // For element-level masking
    int64_t pageSize,
    int64_t mPerBlock,
    int64_t numThreads,
    ...
) {
  Value pagePointers = derefOp.getPagePointers();
  
  // Compute which page this tile starts in
  // startPage = startPos / pageSize
  Value startPage = arith.DivUIOp::create(rewriter, loc,
      startPos,
      arith.ConstantIndexOp::create(rewriter, loc, pageSize));
  
  // Compute offset within starting page
  // pageOffset = startPos % pageSize
  Value pageOffset = arith.RemUIOp::create(rewriter, loc,
      startPos,
      arith.ConstantIndexOp::create(rewriter, loc, pageSize));
  
  // Determine how many pages this tile spans
  int64_t pagesPerTile = (mPerBlock + pageSize - 1) / pageSize;
  
  if (pagesPerTile == 1) {
    // Simple case: tile fits in one page
    loadFromSinglePage(derefOp, startPage, pageOffset, tid, destLDS, 
                       validSeqLen, mPerBlock, /*ldsOffset=*/0, ...);
  } else {
    // Complex case: tile spans multiple pages
    for (int64_t p = 0; p < pagesPerTile; p++) {
      Value currentPage = arith.AddIOp::create(rewriter, loc,
          startPage,
          arith.ConstantIndexOp::create(rewriter, loc, p));
      
      int64_t elementsFromPage = (p == 0) ? 
          (pageSize - pageOffset) : 
          std::min(pageSize, mPerBlock - p * pageSize);
      
      loadFromSinglePage(derefOp, currentPage, 
                         (p == 0) ? pageOffset : 0,
                         tid, destLDS, validSeqLen, 
                         elementsFromPage, /*ldsOffset=*/p * pageSize, ...);
    }
  }
}

static void loadFromSinglePage(
    rock::DerefOp derefOp,
    Value pageIdx,          // Which page to load from
    Value pageOffset,       // Offset within the page
    Value tid,              // Thread ID
    Value destLDS,          // Destination
    Value validSeqLen,      // For element-level masking
    int64_t numElements,    // Elements to load
    int64_t ldsOffset,      // Offset in destination LDS
    int64_t pageSize,       // Static page size
    ...
) {
  Value pagePointers = derefOp.getPagePointers();
  
  // Load the page's base address
  // Note: Only do this if we haven't already (pageIdx might be invalid for
  // tiles that are fully masked, but we already handled that in the caller)
  Value pageAddr = memref::LoadOp::create(rewriter, loc,
      pagePointers, ValueRange{c0, pageIdx, c0});
  
  // Compute per-thread offset
  // Each thread loads a portion of the tile
  Value threadOffset = computeThreadOffset(tid, numElements, numThreads);
  
  // Total byte offset = (pageOffset + threadOffset) * sizeof(element)
  Value totalOffset = arith.AddIOp::create(rewriter, loc, 
      pageOffset, threadOffset);
  Value byteOffset = arith.MulIOp::create(rewriter, loc,
      totalOffset,
      arith.ConstantIndexOp::create(rewriter, loc, sizeof_element));
  
  // ELEMENT-LEVEL MASKING: Check each position against validSeqLen
  // This handles partial tiles at the sequence boundary
  // Note: Masking is handled HERE in ToBlockwise, not in rock.load_from_address
  Value globalPos = arith.AddIOp::create(rewriter, loc,
      arith.MulIOp::create(rewriter, loc, pageIdx, 
          arith.ConstantIndexOp::create(rewriter, loc, pageSize)),
      arith.AddIOp::create(rewriter, loc, pageOffset, threadOffset));
  Value valid = arith.CmpIOp::create(rewriter, loc, 
      arith.CmpIPredicate::ult, globalPos, validSeqLen);
  
  // Option A: Use scf.if for masking (simple, but has divergence overhead)
  Value data = scf.IfOp::create(rewriter, loc, vectorType, valid,
      /*then=*/[&](OpBuilder &b, Location l) {
        Value loaded = rock::LoadFromAddressOp::create(b, l,
            vectorType, pageAddr, byteOffset, vectorLengthAttr);
        b.create<scf::YieldOp>(l, loaded);
      },
      /*else=*/[&](OpBuilder &b, Location l) {
        Value zeros = arith.ConstantOp::create(b, l, 
            DenseElementsAttr::get(vectorType, 0.0f));
        b.create<scf::YieldOp>(l, zeros);
      });
  
  // Option B: When using amdgpu.raw_buffer_load, the hardware handles 
  // out-of-bounds automatically - we can skip the scf.if entirely!
  // This is a lowering optimization, not a ToBlockwise change.
  
  // Store to LDS at appropriate offset
  Value ldsPos = arith.AddIOp::create(rewriter, loc,
      threadOffset,
      arith.ConstantIndexOp::create(rewriter, loc, ldsOffset));
  rock::InBoundsStoreOp::create(rewriter, loc, data, destLDS, ldsPos);
}
```

## New Op: `rock.load_from_address`

### Definition

```tablegen
def Rock_LoadFromAddressOp
    : Rock_Op<"load_from_address", 
              [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>,
      Arguments<(ins 
          I64:$address,         // Base address (i64 pointer value)
          Index:$byteOffset,    // Byte offset from base address
          IndexAttr:$numElements // Number of elements to load
      )>,
      Results<(outs VectorOf<[F16, F32, BF16]>:$result)> {
  let summary = "Load data from a raw memory address";
  let description = [{
    Loads a vector of elements from a raw memory address.
    
    The `address` operand is an i64 value representing a memory pointer.
    The `byteOffset` is added to the address before loading.
    
    This op does NOT handle validity/masking - that is done at the ToBlockwise 
    level using scf.if or by relying on amdgpu.raw_buffer_load's OOB handling.
    
    This is used for paged attention where page pointers are stored as i64 
    values, and we need to dereference them to load actual data.
    
    Example:
    ```mlir
    // Load the page pointer (an i64 address)
    %addr = memref.load %pagePointers[%pageIdx] : memref<64xi64>
    // Load 8 f16 elements starting at addr + byteOffset
    %data = rock.load_from_address %addr, %byteOffset 
            {numElements = 8} : i64, index -> vector<8xf16>
    ```
  }];
}
```

### Lowering to LLVM

The `rock.load_from_address` op lowers through these steps:

**Step 1: RockToGPU or new pass**

Convert to LLVM dialect operations:

```mlir
// Input:
%data = rock.load_from_address %addr, %byteOffset 
        {numElements = 8} : i64, index -> vector<8xf16>

// Output:
// 1. Convert i64 address to pointer
%ptr = llvm.inttoptr %addr : i64 to !llvm.ptr<1>  // address space 1 = global

// 2. Add byte offset using GEP on i8 pointer
%ptr_i8 = llvm.bitcast %ptr : !llvm.ptr<1> to !llvm.ptr<1, i8>
%offset_i64 = llvm.sext %byteOffset : index to i64
%gep = llvm.getelementptr %ptr_i8[%offset_i64] 
       : (!llvm.ptr<1, i8>, i64) -> !llvm.ptr<1, i8>
%typed_ptr = llvm.bitcast %gep : !llvm.ptr<1, i8> to !llvm.ptr<1, vector<8xf16>>

// 3. Load (unconditionally - masking handled at ToBlockwise level)
%loaded = llvm.load %typed_ptr {alignment = 16} : !llvm.ptr<1> -> vector<8xf16>

// 3. Conditional load
%result = scf.if %valid -> vector<8xf16> {
  %loaded = llvm.load %typed_ptr {alignment = 16} : !llvm.ptr<1> -> vector<8xf16>
  scf.yield %loaded : vector<8xf16>
} else {
  %zeros = arith.constant dense<0.0> : vector<8xf16>
  scf.yield %zeros : vector<8xf16>
}
```

**Alternative: Use amdgpu.raw_buffer_load**

For better performance on AMD GPUs, we could lower to buffer loads:

```mlir
// Create buffer resource from pointer
%rsrc = rocdl.make.buffer.rsrc %ptr, ... 

// Use raw_buffer_load with bounds checking via sgpr_offset
%loaded = amdgpu.raw_buffer_load %rsrc[%offset] if %valid 
          : memref<...> -> vector<8xf16>
```

**Why raw_buffer_load is faster:**

1. **Hardware bounds checking**: Buffer loads have built-in out-of-bounds 
   handling in hardware. If `valid` is false or offset is out of range, the 
   hardware returns zeros without executing a real memory operation. With 
   `llvm.load` + `scf.if`, we have:
   - A conditional branch (divergent if threads have different validity)
   - Potential warp divergence penalty
   - Software overhead for the condition check

2. **No warp divergence**: With `scf.if`, if some threads in a warp need to 
   load and others don't, the warp must execute both paths (masked). With 
   buffer loads, all threads issue the same instruction - invalid ones just 
   get zeros back from hardware.

3. **Better instruction scheduling**: Buffer load instructions can be issued 
   unconditionally, allowing the compiler to schedule them earlier and hide 
   memory latency better. Conditional loads create dependencies that limit 
   scheduling freedom.

4. **Single instruction vs multiple**: 
   - `scf.if` approach: CMP + BRANCH + LOAD (conditional) + PHI 
   - Buffer load: Single BUFFER_LOAD instruction with OOB=zero semantics

5. **Cache behavior**: Buffer loads use the texture cache path on AMD GPUs, 
   which can be more efficient for scattered reads (common in paged attention).

**Trade-off**: Using `amdgpu.raw_buffer_load` requires creating buffer 
descriptors (`rocdl.make.buffer.rsrc`) from our raw i64 pointers, which adds 
some infrastructure complexity. However, for paged attention where we're doing 
many scattered loads, the performance benefit is likely significant.

**Key point: This is ONLY a lowering change**. The `rock.load_from_address` op 
definition and all ToBlockwise code remain unchanged. We just change how 
`rock.load_from_address` lowers:
- Current: `rock.load_from_address` → `llvm.inttoptr` + `llvm.load`
- Optimized: `rock.load_from_address` → `rocdl.make.buffer.rsrc` + `amdgpu.raw_buffer_load`

No upstream changes needed - ToBlockwise still generates `rock.load_from_address`,
and the lowering pass decides how to implement it.

**With raw_buffer_load, address masks become unnecessary**:
Since the hardware handles out-of-bounds accesses (returning zeros), we can 
potentially simplify ToBlockwise to not generate the `scf.if` masking. Instead,
we'd set up the buffer bounds to match `validSeqLen` and let the hardware handle it.
This is a future optimization, not required for initial implementation.

**Recommendation**: Start with `llvm.load` + `scf.if` (in ToBlockwise) for 
correctness, then optimize to buffer loads once the basic flow is working.

## Tile Size and Page Size Alignment

### The Problem

If tile size doesn't align well with page size, a single tile may span 
multiple pages, requiring multiple load operations with address arithmetic.

### Design Considerations

**Page size is static** - determined before attention computation (e.g., 8192).

**Tile size comes from tuning parameters** - `mPerBlock` in the tuning config.

**Question**: Can we constrain tile sizes based on page size?

### Options

1. **Require tile size ≤ page size**: 
   - Pro: Each tile loads from exactly one page (simple)
   - Con: May limit performance if optimal tile is larger

2. **Require tile size divides page size evenly**:
   - Pro: Predictable page boundaries
   - Con: May not be optimal for all cases

3. **Handle arbitrary tile sizes** ← **CHOSEN APPROACH**:
   - Pro: Maximum flexibility, tuning can find optimal performance
   - Con: More complex loading code for multi-page tiles
   - This allows the tuning infrastructure to discover the best tile size
     without artificial constraints

### Where Tile Size is Set

**The tuning parameter flow:**

1. **AffixTuningParameters.cpp** (lines 287-361):
   - Entry point for setting attention tuning parameters
   - Uses `RockGemmGemmWrapperInterface` to access attention-like ops
   - Default perf_config: `"attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1"`
   - Parses config via `GemmGemmParamsAttr::get(perfConfigStrAttr, isWmma)`

2. **GemmGemmParamsAttr::get** (RockDialect.cpp lines 3891-3936):
   - Parses the perf_config string
   - Extracts: `mPerBlockG0`, `mPerBlockG1`, `nPerBlockG0`, `kpackPerBlock`,
     `mPerWave`, `nPerWave`, `mnPerXdl`, `kpack`, etc.

3. **PopulateParamsGemmGemm::getAccelGemmParams** (GridwiseGemmGemmParams.cpp):
   - Generates the actual `RockAccelTuningParamAttrInterface` objects
   - Returns a pair: (params0 for GEMM0, params1 for GEMM1)

4. **The tile sizes**:
   - `mPerBlock` (M dimension tile): `tuningParams.getMPerBlock()` (typically 32)
   - `nPerBlock` (N dimension tile): `tuningParams.getNPerBlock()` (typically 32)
   - `kpackPerBlock` (K dimension): for K-loop iteration

**For paged attention**, the M dimension corresponds to the K sequence 
(transposed), so `mPerBlock` determines how many sequence positions we load 
per tile. With pageSize = 8192 and mPerBlock = 32, we have 256 tiles per page.

### Practical Impact

With typical parameters:
- pageSize = 8192
- mPerBlock = 32 (default)
- Tiles per page = 8192 / 32 = 256

A tile will span pages only when:
- `(mLoopIV * mPerBlock) / pageSize != ((mLoopIV + 1) * mPerBlock - 1) / pageSize`
- This happens at mLoopIV = 256, 512, 768, ... (every 256 iterations)

So for most iterations, we load from a single page. Multi-page handling is 
only needed at page boundaries (every 256th tile with these parameters).

## What Gets Removed

With this approach, we remove:
- `Rock_PagedAttentionOp` 
- `Rock_GridwisePagedAttentionAccelOp`
- `PagedAttentionRewritePattern` in GemmToGridwise
- All code specific to paged attention ops

## What Gets Added

- Expanded `rock.deref` op with address computation region
- Optional `keyAddresses` / `valueAddresses` operands on attention ops
- `rock.load_from_address` op with LLVM lowering
- `loadPagedKeyTile` / `loadPagedValueTile` helper functions in ToBlockwise
- Page caching logic to avoid redundant page table accesses

## Implementation Plan

### Phase 0: Remove Existing Paged Attention Code
1. Delete `Rock_PagedAttentionOp` from RockOps.td
2. Delete `Rock_GridwisePagedAttentionAccelOp` from RockOps.td
3. Remove `PagedAttentionRewritePattern` from GemmToGridwise.cpp
4. Remove any paged attention code from other passes
5. Clean up tests

### Phase 1: Expand rock.deref
1. Update `Rock_DerefOp` definition to use `TensorOrMemRefOf<[I64]>` input
2. Add the `addressComputation` region
3. Update bufferization interface in `BufferizableOpInterfaceImpl.cpp`
   - Rename `PagedDerefOpInterface` to `DerefOpInterface`
   - Handle region cloning during bufferization
   - Ensure pagePointers input is bufferized
4. Update TosaToRock to populate the address computation region

### Phase 2: Add address operands to attention ops  
1. Add optional `keyAddresses` / `valueAddresses` to AttentionOp
2. Add same to GridwiseAttentionAccelOp
3. **Update ALL creation sites** in the codebase:
   - `GemmToGridwise.cpp`: AttentionRewritePattern, GridwiseAttentionAccelOp::create
   - `TosaToRock.cpp`: AttentionOp::create calls
   - `rocmlir-gen/`: Any test generation code
   - Tests: Update any tests that create attention ops
4. Update GemmToGridwise to pass through address operands
5. Update TosaToRock to set address operands

### Phase 3: Implement rock.load_from_address
1. Define op in RockOps.td
2. Implement verification
3. Implement lowering to LLVM dialect
4. Test with simple standalone cases

### Phase 4: Implement paged tile loading
1. Add `loadPagedKeyTile` / `loadPagedValueTile` helpers
2. Add conditional branch in GridwiseAttentionAccelRewritePattern
3. Implement page caching optimization (avoid reloading same page for multiple tiles)
4. Handle all tile/page size combinations (tile < page, tile = page, tile > page)
5. Test end-to-end with paged attention

### Phase 5 (Future): Performance Optimization
1. Replace `llvm.load` + `scf.if` with `amdgpu.raw_buffer_load` for hardware masking
2. Profile and optimize page table access patterns
3. Let tuning infrastructure discover optimal tile sizes for paged attention

## Resolved Design Decisions

1. **Deref region usage**: Generate loading code in GridwiseGemmToBlockwise
   using the deref's page pointers directly. The region serves as documentation
   of the address computation logic, but we generate tiled loads directly.

2. **Multi-page tiles**: Handle any tile/page size combination:
   - Tile < Page: Most common case, page caching avoids redundant page table loads
   - Tile = Page: Simple 1:1 mapping
   - Tile > Page: Generate separate loads for each page in the tile

3. **Tile/page size tuning**: No special validation in AffixTuningParameters.
   ToBlockwise must handle any combination, and the tuning infrastructure will
   discover optimal tile sizes through performance testing.

4. **Backwards compatibility**: Remove PagedAttentionOp completely. Start
   fresh on a new branch to implement the unified approach.

5. **Transform handling**: The sequence dimension flows through transforms 
   unchanged (GQA affects heads, not sequence; padding extends at the end).
   Therefore, the M-loop index maps directly to page positions.

6. **Masking responsibility**: Validity/masking is handled at ToBlockwise level,
   NOT in `rock.load_from_address`. ToBlockwise generates scf.if for masking
   (or relies on buffer load OOB semantics as a future optimization).

7. **amdgpu.raw_buffer_load**: This is purely a lowering optimization for
   `rock.load_from_address`. No changes needed to ToBlockwise or op definitions.
   It eliminates scf.if overhead by using hardware OOB handling.
