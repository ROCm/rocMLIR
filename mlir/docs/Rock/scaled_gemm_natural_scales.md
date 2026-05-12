# Scaled GEMM: Natural-Form Scales (Issue #2127)

Status: implemented on the `users/umayadav/scaled-gemm-no-broadcast`
branch ("Lower scaled GEMM scales without broadcasting to data shape"),
rebased onto `origin/develop` past PR #2210 ("Add FP8/BF8 support for
LDS transpose load").

This document explains how the rocMLIR lowering pipeline propagates the
*natural* `(G, K / kQuantBlockSize, D)` scale shape for scaled GEMM
operations end-to-end, instead of expanding scales to one value per K
element ("broadcasted form"). It covers the background, the new IR
contract, every transform in the lowering chain, the LDS/register
layouts that result, and the fallback path used when a workgroup tile
cannot evenly distribute the natural-form scale tile.

---

## 1. Background

### 1.1 Block-quantized GEMM

Scaled GEMM is the building block for narrow-precision matmuls such as
the OCP MX (microscaling) FP4 and FP8 formats. The data tensors `A`,
`B` are quantized in groups of `kQuantBlockSize` consecutive K elements;
each group shares one `f8E8M0FNU` scale value:

```
A[g, k, m]      ∈ {f4E2M1FN, f8E*FN*, ...}      // shape (G, K, M)
scaleA[g, q, m] ∈ f8E8M0FNU,  q = k / 32        // shape (G, K/32, M)
```

The hardware instruction `amdgpu.scaled_mfma` (gfx950 / MI350X) consumes
**one** scale per operand per MFMA: when an MFMA tile spans 32 K
elements (`kQuantBlockSize`), it reads exactly one scale per workitem
per operand. There is no architectural reason to materialise a separate
scale value for each of the 32 K positions.

Throughout this document `kQuantBlockSize == 32` (the OCP MX block
size). The constant is exposed in
`mlir/include/mlir/Dialect/Rock/utility/loweringUtils.h`.

### 1.2 Why scales used to be broadcast

The original lowering pipeline pre-dated scaled MFMA and reused the
existing data-tile machinery (`loadAndStoreGemmInputTile`,
`wrapLDSBufferForStore`, `wrapLDSBufferForLoad`,
`createRegInterrimBufferForAccel`, `ThreadwiseGemmLowering`) for
scales. To make scales fit in those code paths, the front-end
broadcast each scale value along K so that the scale tensor had the
same K extent as the data tensor:

```
broadcasted scaleA[g, k, m] = naturalScaleA[g, k / 32, m]
```

Per-thread register tiles for scales then had element type
`vector<kQuantBlockSize x f8E8M0FNU>`, of which only lane 0 was actually
used by `amdgpu.scaled_mfma` — the other 31 lanes were redundant
copies. `ThreadwiseGemmLowering` extracted that lane with a
`vector.extract %scale[0]` right before the MFMA call.

This had three concrete costs:

| Resource              | Cost factor (relative to data tile) |
| --------------------- | ----------------------------------- |
| Global → LDS traffic  | 32x more bytes than needed          |
| LDS allocation        | 32x more bytes than needed          |
| Per-thread VGPR tile  | 32x more bytes than needed          |

For a typical FP4 GEMM workgroup (`kPerBlock=128`, `mPerBlock=128`,
`blockSize=256`), the scale-A LDS tile alone went from a needed 512 B
(= 128 × 128 / 32) to 16 384 B (= 128 × 128). On gfx950 LDS is shared
across all waves on a CU, so this directly capped occupancy.

### 1.3 Issue #2127 in two sentences

> The rock dialect already accepts un-broadcasted scales at the
> front-end after the prior part of #2127 ("Make scaled GEMM accept
> and emit un-broadcasted scales"), but the
> lowering pipeline still expanded them to per-K-element form before
> allocating LDS and registers. Propagate the natural `(G, K/32, D)`
> scale shape all the way down so LDS, global loads, and per-thread
> registers are 32x smaller, and so `amdgpu.scaled_mfma` consumes
> scalars directly instead of `vector.extract %v[0]`.

---

## 2. Design

### 2.1 The natural scale shape as a first-class IR concept

Three IR surfaces now know about block quantization explicitly:

1. `Rock_BlockwiseLoadTileOp` carries a `quantBlockSize : I64Attr =
   1` attribute (`mlir/include/mlir/Dialect/Rock/IR/RockOps.td`).
   `quantBlockSize > 1` declares "this op loads a scale tile whose K
   extent is `kPerBlock / quantBlockSize`".
2. `accel::AccelEmitter::wrapLDSBufferForLoad` takes
   `int64_t quantBlockSize = 1`
   (`mlir/include/mlir/Dialect/Rock/IR/AccelEmitter.h`).
3. `rock::wrapLDSBufferForStore` takes `bool ldsLayoutDxK = false`
   (`mlir/include/mlir/Dialect/Rock/utility/loweringUtils.h`).

`(1)` lets a rewriter mark a `BlockwiseLoadTileOp` as a "scale" load
without creating a parallel op. `(2)` and `(3)` give the LDS write and
read sides a way to agree on a DxK layout for natural-form scales (see
§4.4).

### 2.2 Where scales transition through the pipeline

```
rock.gemm                                           (G, K/32, D) or (G, K, D)
   │  GemmToGridwise          : compactBroadcastedScale → natural
   ▼
rock.gridwise_gemm_accel                            (G, K/32, D)
   │  GridwiseGemmToBlockwise : decide useNaturalScale; allocate LDS
   ▼
rock.blockwise_load_tile  + rock.blockwise_gemm_accel  (LDS = (D, K/32))
   │  BlockwiseLoadTileToThreadwise (double-buffer schedule)
   │  BlockwiseGemmToThreadwise     (single-buffer schedule)
   ▼
rock.threadwise_gemm_accel                          (per-thread regs = K/32 scalars)
   │  ThreadwiseGemmLowering       : amdgpu.scaled_mfma reads scalars
   ▼
amdgpu.scaled_mfma
```

The natural `(G, K/32, D)` shape is preserved at every level. There is
**no broadcast-to-K** anywhere on the optimised path; the per-thread
register element type is the scalar scale type
(`f8E8M0FNU` for FP4 / FP8 inputs).

### 2.3 Compatibility with hand-written IR

Hand-written `rock.gemm` and `rock.gridwise_gemm_accel` MLIR (used in
many unit tests under `mlir/test/Dialect/Rock/`) still passes scales in
the legacy broadcasted form `(G, K, D)`. Both `GemmToGridwise` and
`GridwiseGemmToBlockwise` therefore implement a `compactBroadcastedScale`
helper that view-rewrites a broadcasted scale to its natural shape. No
data is moved: the helper builds a transform chain that splits the K
dim into `(K/32, 32)` and drops the inner index 0 (since each
32-element block is a constant in the broadcasted form).

`rocmlir-gen --scaledGemm` defaults to emitting scales in their natural
form (`-broadcastScales=false`). Pass `-broadcastScales=true` to
generate the legacy broadcasted IR (e.g. for fuzzing the
`compactBroadcastedScale` path or for the rare `-quantBlockSize` value
that the lowering does not yet accept in natural form).

### 2.4 Fallback when the natural tile is too small

`loadAndStoreGemmInputTile` distributes the elements of one
`(K_tile, D_tile)` tile across `blockSize` workitems and requires that
each workitem load **at least one element**:

```
copyPerThread = (K_tile_natural * D_tile) / blockSize  must be ≥ 1
              = (kPerBlock / 32 * dPerBlock) / blockSize
```

For small `dPerBlock` tunings (e.g. `mPerBlock = 16`, `kPerBlock = 32`,
`blockSize = 256`) this becomes 0 and lowering would fail. To keep the
existing `loadAndStoreGemmInputTile` path usable we fall back to the
legacy broadcasted layout *for that one operand*, while still using a
scalar per-thread register tile (so the only cost is the LDS bytes).
The `useNaturalScaleA` / `useNaturalScaleB` flags in
`GridwiseGemmToBlockwise` choose between the two paths per operand
based on the tuning parameters.

### 2.5 LDS layout: KxD vs. DxK

`wrapLDSBufferForStore` and `wrapLDSBufferForLoad` historically picked
between two raw layouts of the flat LDS byte buffer:

* **KxD** (`raw = unmerge(k_outer, d, kpack_idx)`) — default for data
  tiles, optimised for the kpack-vectorised store side.
* **DxK** (`raw = unmerge(d, k_outer, kpack_idx)`) — used selectively
  to keep K contiguous in LDS for some tunings.

For natural-form scales we **must** use DxK because:

* `kPack` is forced to 1 for scales (one scalar per quantization block,
  no benefit to packing).
* The per-thread K iterations (`kIter = kpackPerThread`) are
  consecutive in K, and `ThreadwiseReadIntoOp` vectorises reads along
  the contiguous trailing dimension. With KxD the K-stride in LDS
  would be `dPerBlock`, which kills vectorisation and gave numerically
  wrong answers in the `mixr-gemm-fp4` E2E tests.

DxK is therefore forced unconditionally for natural-form scales on
both the write side (`isNaturalFormScale = quantBlockSize > 1` in
`BlockwiseLoadTileToThreadwise`) and the read side
(`if (quantBlockSize > 1) ldsLayoutDxK = true; rotateDWithK = false;`
in `MfmaEmitter::wrapLDSBufferForLoad`). The same store/load layout
must agree, which is why `quantBlockSize` must be threaded through
both schedules — see §4.5 for the double-buffer fix that closed this.

### 2.6 Scale element-type asymmetry across verifier surfaces

`rock.gemm` and the post-canonicalisation
`rock.gridwise_gemm_accel` / `rock.threadwise_gemm_accel` ops have
*intentionally* different scale element-type policies, documented in
both `mlir/include/mlir/Dialect/Rock/IR/RockOps.td` (ODS constraint)
and `mlir/lib/Dialect/Rock/IR/RockDialect.cpp` (verifier comments):

| Op                              | Allowed scale element types |
| ------------------------------- | --------------------------- |
| `rock.gemm`                     | `f8E8M0FNU` or `f32`        |
| `rock.gridwise_gemm_accel`      | `f8E8M0FNU` only            |
| `rock.threadwise_gemm_accel`    | `f8E8M0FNU` only            |

The asymmetry exists because `GemmToGridwise` inserts a `linalg.generic`
truncf cast (`createTypeConversionLaGeneric`) when scales arrive as
`f32` — by the time the IR reaches the gridwise / threadwise ops the
scales have already been canonicalised to `f8E8M0FNU` (the only format
the MFMA scaled-GEMM hardware path consumes). This lets user IR write
either dtype while keeping the post-lowering invariant tight.

`rocmlir-gen --scaledGemm` defaults to emitting `f8E8M0FNU` scales but
also accepts `-scale_a_dtype f32 -scale_b_dtype f32` (see
`mlir/test/rocmlir-gen/gemm-kernel-scaled.mlir` for both axes).

---

## 3. The two scale forms

### 3.1 Natural form (the optimised path)

```
scale layout:           (G, K/32, D)
LDS element type:       f8E8M0FNU (scalar)
LDS bytes per tile:     kPerBlock/32 * dPerBlock
LDS K-major layout:     DxK  (forced)
kpack:                  1    (forced for scales)
Per-thread reg type:    f8E8M0FNU (scalar)
Per-thread reg tile:    kBasePerThread elements
ExtractOp before MFMA:  none
```

### 3.2 Broadcasted form (the fallback path, also legacy IR)

```
scale layout:           (G, K, D)         (= broadcast of natural along K)
LDS element type:       vector<kpack x f8E8M0FNU>
LDS bytes per tile:     kPerBlock * dPerBlock
LDS K-major layout:     KxD  (data-tile default)
kpack:                  same as data tile
Per-thread reg type:    vector<argTypeWidth x f8E8M0FNU>   (argTypeWidth ≈ kBase)
Per-thread reg tile:    kBasePerThread vectors
ExtractOp before MFMA:  vector.extract %arg[0]  (kept for compat)
```

Both forms still produce identical results from `amdgpu.scaled_mfma`;
only the resource footprint differs.

---

## 4. Implementation walkthrough

This section follows the data flow from `rock.gemm` down to
`amdgpu.scaled_mfma`. Each subsection cites the file that owns the
change and explains the relevant transform chain.

### 4.1 IR surface changes

`mlir/include/mlir/Dialect/Rock/IR/RockOps.td`:

* `Rock_BlockwiseLoadTileOp` gains
  `DefaultValuedAttr<I64Attr, "1">:$quantBlockSize` and a verifier
  rule that `quantBlockSize` must be `1` (data tile) or
  `kQuantBlockSize` (32) — implemented in
  `BlockwiseLoadTileOp::verify` so that hand-written IR is rejected at
  parse time rather than crashing at lowering. The op description
  documents the contract: when > 1, the source memref's K dim is
  `K_data / quantBlockSize` and the LDS / register destinations must
  be sized accordingly.
* `Rock_BlockwiseGemmAccelOp` description was updated to lift the
  per-thread iteration shape match invariant out of the verifier (the
  invariant is "the per-thread iteration shape implied by `kPack` and
  `kQuantBlockSize` must agree on the data and scale operands"). Users
  see the rule directly in the op description.
* `Rock_GemmOp` keeps the looser `[F8E8M0FNU, F32]` scale dtype
  constraint to preserve the asymmetry described in §2.6.

`mlir/include/mlir/Dialect/Rock/IR/AccelEmitter.h`:

* The base virtual `wrapLDSBufferForLoad` and both overrides
  (`MfmaEmitter`, `WmmaEmitter`) take `int64_t quantBlockSize = 1`.
  `WmmaEmitter` asserts `quantBlockSize == 1` defensively;
  `GridwiseGemmAccelRewritePattern` already emits a proper
  `op.emitOpError(...)` diagnostic if WMMA is selected for a scaled
  GEMM, so the assert is a backstop, not the user-visible failure.

`mlir/include/mlir/Dialect/Rock/utility/loweringUtils.h`:

* `wrapLDSBufferForStore` takes `bool ldsLayoutDxK = false`. Defaults
  preserve the old behaviour for data tiles.
* New free helpers (made `inline` to be safe across translation units):
  * `compactBroadcastedScale(b, loc, scale, matK)` and
    `broadcastScaleAlongK(b, loc, scale, matK)` — pure view chains
    extracted from `GemmToGridwise.cpp` and `GridwiseGemmToBlockwise.cpp`
    so the two callers stay in sync.
  * `isValidScaleK(scaleK, matK)`,
    `isNaturalFormScaleK(scaleK, matK)`,
    `isBroadcastedScaleK(scaleK, matK)` — single source of truth for
    the scale-K shape rule (used by both `rock.gemm`'s verifier and
    the shared `verifyScales` helper).
  * `inferQuantBlockSize(scaleLdsType)` — encapsulates the
    `isa<VectorType>(...) ? 1 : kQuantBlockSize` rule used in
    `BlockwiseGemmToThreadwise.cpp` to recover the quant block size
    from the LDS view's element type.
  * `scaleArgType(elementTypeScale, dataArgType, useNatural)` — builds
    the per-thread scale arg type; replaces a hard-to-read nested
    ternary in `GridwiseGemmToBlockwise.cpp`.
  * `scaleLdsElemCount(useNaturalScale, kpacksPerBlock, kpack, dPerBlock)`
    — single helper for the LDS element count for a scale tile; used
    by both `ldsBlockScaleA/BSize` math and the `createLDSByteBuffer`
    calls.
  * `rescaleScaleKExtents(quantBlockSize, kpackPerThread, kPack, kPerBlock)`
    — the "force `kPack=1` and rescale K extents" trick, factored out
    of `MfmaEmitter::wrapLDSBufferForLoad` and
    `BlockwiseLoadTileToThreadwise::matchAndRewrite` so the two sites
    cannot drift.

### 4.2 Front-end compaction (`GemmToGridwise.cpp`)

`mlir/lib/Dialect/Rock/Transforms/GemmToGridwise.cpp` is the first pass
that sees user IR. After normalising scales into `(G, K, D)` layout
(transposing `(G, D, K)` if necessary) it invokes
`compactBroadcastedScale`. The helper builds two `BottomUpTMBuilder`
transforms:

```text
gemmK  ──unmerge──▶  gemmKNat × gemmKIntra        (split K into 32 blocks)
gemmKIntra ──dropDimAtIndex(0)──▶  ⌀              (each block is constant)
```

Result:

```
(G, K, D)  ── view ──▶  (G, K/32, D)              // natural form
```

If `matK % kQuantBlockSize != 0` (some unit tests use `K = 1` or
`K = 72`), compaction is skipped at this stage; the matrix gets padded
out to a multiple of 32 in `arrangeSplitKTransform` and
`GridwiseGemmToBlockwise` will compact what's left.

`compactBroadcastedScale` also propagates into the split-K alignment
logic. Because `compactBroadcastedScale` runs **before**
`arrangeSplitKTransform`, scales arrive at the split-K logic in their
natural form whenever `matK % kQuantBlockSize == 0`. The two scale
forms then need different `kAlign` values:

| Scale form         | `kAlign`                                | Why                                                                                                                                      |
| ------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Natural**        | `splitKFactor * kQuantBlockSize`        | Per-split scale K = `matK_padded / (splitKFactor * kQuantBlockSize)` must be an integer for `Unmerge{splitKFactor, _}` to be exact.      |
| **Broadcasted**    | `lcm(splitKFactor, kQuantBlockSize)`    | Scale K equals matrix K, so the looser `lcm` alignment is sufficient at the matrix granularity (the broadcast handles per-position correctness). |

Without this distinction, a natural-form input with e.g. K=96 and
splitKFactor=2 would not get padded (96 is already a multiple of 32),
leaving scale K = 3 — which cannot be `Unmerge`d into 2 splits. The
regression test
`gemm_scaled_fp4_natural_splitk_lcm_unsafe` in
`mlir/test/Dialect/Rock/gemm_to_gridwise.mlir` exercises exactly this
case (matK 96 → 128, scaleK 3 → 4, splits cleanly).

### 4.3 Workgroup tile decisions (`GridwiseGemmToBlockwise.cpp`)

`GridwiseGemmAccelRewritePattern` is where most of the new policy lives:

1. **Re-compaction of any leftover broadcasted scales.** Same
   `compactBroadcastedScale` helper as above, applied
   defensively. After this point we assert that `scaleA / scaleB` are
   in the natural form.
2. **`useNaturalScale{A,B}` decision.** The natural-form tile size per
   workgroup is
   `naturalKPerBlock * dPerBlock = (kPerBlock / 32) * dPerBlock`. If
   that is `< blockSize` for an operand, set `useNaturalScale{A,B} =
   false` for it and call `broadcastScaleAlongK` to expand the scale
   memref back to `(G, K, D)`.
3. **LDS sizing** — uses the
   `scaleLdsElemCount(useNaturalScale, kpacksPerBlock, kpack, dPerBlock)`
   helper from `loweringUtils.h` so `ldsBlockScaleA/BSize` and the two
   `createLDSByteBuffer` calls cannot drift:
   ```
   natural  : (kpacksPerBlock * kpack / kQuantBlockSize) * dPerBlock scalars
   broadcast:  kpacksPerBlock * kpack                    * dPerBlock packed slots
   ```
   The natural path is `kQuantBlockSize`-fold smaller.
4. **LDS view typing.**
   `viewBufferAs(b, ldsByteBufferScaleA, useNaturalScaleA
                  ? Type(elementTypeScaleA)
                  : vectorTypeOrSelf(elementTypeScaleA, kpack));`
   The LDS view's element type encodes which form is in use; the
   downstream `BlockwiseGemmToThreadwise` lowering uses the
   `inferQuantBlockSize(scaleLdsType)` helper (single source of truth)
   to recover `quantBlockSize` from this typing.
5. **Per-thread register sizing** — uses the `scaleArgType` helper:
   ```cpp
   Type argTypeScaleA = scaleArgType(elementTypeScaleA,
                                     params.argTypeA,
                                     useNaturalScaleA);
   createRegInterrimBufferForAccel(b, loc, argTypeScaleA,
                                   params.kBasePerThread, ...);
   ```
   In the natural case the per-thread tile is `kBasePerThread` scalars
   (one per quantization block). In the broadcast-fallback case it is
   `kBasePerThread` vectors whose width matches the data operand's
   `argType` width (typically `kBase`, the per-MFMA K-element count) so
   that the existing `loadAndStoreGemmInputTile` and
   `ThreadwiseGemmLowering` paths can consume the buffer unchanged.
6. **Quant block plumbing through `loadAndStoreGemmInputTile`.**
   Each scale `BlockwiseLoadTileOp` is created with
   `quantBlockSize = useNaturalScaleX ? kQuantBlockSize : 1`. Data
   tiles always use the default `1`. The `setQuantBlockSize` call is
   unconditional (the default is `1`, so a no-op for data tiles), to
   keep the code path uniform between data and scale operands.
7. **Direct-to-LDS rejection.** When the workgroup is configured for
   direct-to-LDS *and* the GEMM is scaled, the rewriter emits a clean
   `op.emitOpError(...)` diagnostic; the two paths are not yet
   compatible.

The `compactBroadcastedScale` and `broadcastScaleAlongK` helpers in
this file build pure view chains (no data movement). The broadcast
helper is non-trivial: it adds a length-1 K-block dim, broadcasts it to
`kQuantBlockSize`, then merges the two K dims back into the data tile's
K extent.

### 4.4 Threadwise lowering of `BlockwiseLoadTile`

There are two schedules through which a `BlockwiseLoadTileOp` reaches
the threadwise level:

* **Default / single-buffer schedule** (`scheduleVersion == 1`): goes
  through `BlockwiseGemmToThreadwise.cpp`, where
  `BlockwiseGemmAccelOp::lower` calls `wrapLDSBufferForLoad` at MMA
  time.
* **DoubleBuffer schedule** (`scheduleVersion == 2`): goes through
  `BlockwiseLoadTileToThreadwise.cpp`, where `generateReadLoop` calls
  `wrapLDSBufferForLoad` at LDS-read time.

Both must pass the same `quantBlockSize` (and therefore agree on the
DxK LDS layout) for the read view to align with the write view from
`wrapLDSBufferForStore`.

`mlir/lib/Dialect/Rock/Transforms/BlockwiseGemmToThreadwise.cpp`:

```cpp
// Recover quantBlockSize from the LDS element type, since
// BlockwiseGemmAccelOp does not see the quantBlockSize attr directly.
// `inferQuantBlockSize` is the single source of truth for this rule.
auto scaleATypeForLDS =
    cast<MemRefType>(op.getScaleA().getType()).getElementType();
int64_t scaleAQuantBlockSize = inferQuantBlockSize(scaleATypeForLDS);
wrappedLDSBufferForScaleA = accelEmitterPtr->wrapLDSBufferForLoad(
    b, loc, op.getScaleA(), matrixParamsA, blockSize, "m",
    /*useLdsTransposeLoad=*/false,
    /*quantBlockSize=*/scaleAQuantBlockSize);
```

`inferQuantBlockSize` returns `1` for vector LDS elements (broadcasted
scale) and `kQuantBlockSize` for scalar LDS elements (natural scale).
This is the inverse of the typing choice in step 4 of §4.3.

`mlir/lib/Dialect/Rock/Transforms/BlockwiseLoadTileToThreadwise.cpp`:

* The pattern reads `op.getQuantBlockSize()`. If > 1, it calls the
  shared `rescaleScaleKExtents(quantBlockSize, kpackPerThread, kPack,
  kPerBlock)` helper to rescale the blockwise tuning so that
  downstream code sees the K extent in scale-elements
  (`kpackPerThread *= kPack / quantBlockSize; kPack = 1; kPerBlock /=
  quantBlockSize`). The same helper is used by
  `MfmaEmitter::wrapLDSBufferForLoad` (§4.5) so the two sites cannot
  drift.
* For the LDS write side, it forces `ldsLayoutDxK = true` and disables
  rotation when `isNaturalFormScale = (quantBlockSize > 1)`.
* `generateReadLoop` accepts a `quantBlockSize` parameter and forwards
  it to `wrapLDSBufferForLoad`. Without this fix, the
  DoubleBuffer schedule would write the LDS in DxK (because the write
  side does check `isNaturalFormScale`) but read it in KxD (because
  `wrapLDSBufferForLoad` would receive the default `quantBlockSize=1`),
  and produce garbage. This was the root cause of the `GemmScaled`
  E2E regressions seen mid-implementation.

### 4.5 Accel emitter (`AccelEmitter.cpp`)

`MfmaEmitter::wrapLDSBufferForLoad` is the single place that builds the
read-side transform chain for `MFMA` accelerators. The new behaviour
when `quantBlockSize > 1`:

```cpp
rescaleScaleKExtents(quantBlockSize, kpackPerThread, kPack, kPerBlock);
kIter          = kpackPerThread;
ldsLayoutDxK   = true;   // force K-contiguous (see §2.5)
rotateDWithK   = false;
```

`rescaleScaleKExtents` (defined in `loweringUtils.h`) is the same
helper used by the DoubleBuffer schedule in
`BlockwiseLoadTileToThreadwise`, so the read and write sides cannot
disagree on the rescaled extents. The rest of the function is
unchanged — it builds the same
`splitTid → splitWaveId → toLDSRowCol → offset` transform chain — but
operates on the rescaled K extent and uses DxK at the
`unmerge("source_offset", ...)` step.

### 4.6 LDS write helper (`loweringUtils.cpp`)

`mlir::rock::wrapLDSBufferForStore` previously had the layout choice
hard-coded to KxD. Now it takes `bool ldsLayoutDxK`:

```cpp
if (ldsLayoutDxK) {
  reshapeBuf.unmerge("raw", 0, {dName, "k_outer", "kpack_idx"},
                     {d, kOuter, threadsPerKpack});
} else {
  reshapeBuf.unmerge("raw", 0, {"k_outer", dName, "kpack_idx"},
                     {kOuter, d, threadsPerKpack});
}
```

This is the *write* side counterpart to the
`wrapLDSBufferForLoad` change. The two have to agree on the chosen
layout, which is enforced by sourcing `ldsLayoutDxK` from the same
`isNaturalFormScale` condition in
`BlockwiseLoadTileToThreadwise::matchAndRewrite`.

### 4.7 Threadwise GEMM lowering (`ThreadwiseGemmLowering.cpp`)

`ThreadwiseGemmLowering` consumes the per-thread scale buffers and
emits `amdgpu.scaled_mfma`. Two changes:

1. **Dynamic `argTypeScale`.** Instead of always treating the scale
   element as a vector, derive it from the buffer:
   ```cpp
   Type argTypeScaleA = dataTypeScaleA, argTypeScaleB = dataTypeScaleB;
   if (auto vecA = dyn_cast<VectorType>(bufScaleA.getType().getElementType()))
     argTypeScaleA = vecA;       // legacy / broadcasted
   ```
   Natural-form scales pass through unchanged as scalars; broadcasted
   scales are typed as `vector<kBase x f8E8M0FNU>` to match the
   per-thread tile.
2. **Conditional `vector::ExtractOp`.** Keep the historical
   `vector.extract %scale[0]` only when the loaded value is actually a
   vector. For scalars it is omitted, so `amdgpu.scaled_mfma` consumes
   the loaded value directly.

This is the only place where the *consumer* of the per-thread scale
buffer is rewritten; everything upstream just stops creating the
vector wrapping.

---

## 5. Worked example: a typical FP4 tile

For an FP4 scaled GEMM with `M=N=4096, K=4096, G=1`,
`-transA=false -transB=true`, on gfx950 (MI350X). Numbers below come
from `rocmlir-driver --kernel-pipeline=gpu` output (the
`rock.shared_buffer_size` attribute on `gpu.func` and the workgroup /
private memref types).

### LDS allocation (per workgroup, double-buffered)

| Source                   | Baseline (broadcasted) | HEAD (natural)        |
| ------------------------ | ---------------------- | --------------------- |
| matA tile (x2 buffers)   | 32 768 B               | 32 768 B              |
| matB tile (x2 buffers)   | unchanged              | unchanged             |
| scaleA tile (x2 buffers) | 16 384 B               | 1 024 B (16x smaller) |
| scaleB tile (x2 buffers) | 16 384 B               | 1 024 B (16x smaller) |
| **`rock.shared_buffer_size`** | **98 304 B**       | **34 816 B**          |

A 63 488 B reduction (~2.8x). On gfx950 (160 KB LDS per CU) this lifts
the occupancy ceiling from 1 workgroup per CU to 4 workgroups per CU.

### Per-thread scale register tile

| Quantity                | Baseline                  | HEAD             |
| ----------------------- | ------------------------- | ---------------- |
| Per-thread scaleA buf   | `8 x vector<32xf8E8M0FNU>` | `4 x f8E8M0FNU` |
| Bytes / thread (scaleA) | 256 B                     | 4 B (64x smaller)|
| Per-thread scaleB buf   | `8 x vector<32xf8E8M0FNU>` | `4 x f8E8M0FNU` |
| Bytes / thread (scaleB) | 256 B                     | 4 B (64x smaller)|

(The 64x rather than 32x reduction also folds in halving the total
elements: HEAD only stores `kBasePerThread / kpack` quantization
blocks instead of `kBasePerThread` packed K positions.)

### Operations dropped

| Op                          | Baseline        | HEAD |
| --------------------------- | --------------- | ---- |
| `vector.extract %scale[0]`  | 1 per MFMA call | 0    |

---

## 6. Transform chains in detail

### 6.1 `compactBroadcastedScale` (broadcasted → natural)

Used in both `GemmToGridwise.cpp` and `GridwiseGemmToBlockwise.cpp`:

```text
input shape :  (G, K_data,            D)
              ┌──────────────────────────────────────┐
step 1 unmerge:│ K_data ↦ (K_data/32, 32) (K_nat × K_intra) │
              └──────────────────────────────────────┘
              ↓
              (G, K_nat, K_intra, D)
              ┌──────────────────────────────────────┐
step 2 drop  : │ K_intra: take index 0 (broadcast → take canonical lane) │
              └──────────────────────────────────────┘
              ↓
output shape:  (G, K_nat,            D)              // K_nat = K_data / 32
```

This is purely a `rock.transform` view chain; no data is copied. The
`drop.dropDimAtIndex("gemmKIntra", 0)` step relies on the invariant
that all 32 lanes of a broadcasted block hold the same value, so
keeping lane 0 loses no information.

### 6.2 `broadcastScaleAlongK` (natural → broadcasted, fallback)

Used in `GridwiseGemmToBlockwise.cpp` when a natural-form scale tile
would be too small to give every workitem at least one element to
load:

```text
input shape :  (G, K_nat, D)
              ┌──────────────────────────────────────┐
step 1 addDim:│ insert length-1 K_block dim           │
              └──────────────────────────────────────┘
              ↓
              (G, K_nat, 1, D)
              ┌──────────────────────────────────────┐
step 2 bcast :│ broadcast K_block: 1 → kQuantBlockSize │
              └──────────────────────────────────────┘
              ↓
              (G, K_nat, 32, D)
              ┌──────────────────────────────────────┐
step 3 merge :│ K_data ← merge(K_nat, K_block)         │
              └──────────────────────────────────────┘
              ↓
output shape:  (G, K_data, D)                         // K_data = K_nat * 32
```

Again, no data is moved — `rock.transform` with a `Broadcast`
sub-transform represents the duplication purely in the index map.

### 6.3 Per-thread / LDS layout for natural-form scales

The DxK LDS layout for natural-form scales is built by the
combination of these transforms:

* `wrapLDSBufferForStore` (write side):
  ```text
  input :  (k_outer, d) thread-iter coords
  step 1:  merge k_outer × kpack_idx × kpack_vec → k     (kpack=1, no-op)
  step 2:  rotate(d, k_outer)                            (skipped, rotateDWithK=false)
  step 3:  unmerge(d, k_outer, kpack_idx) → raw          (DxK because ldsLayoutDxK=true)
  ```
* `wrapLDSBufferForLoad` (read side):
  ```text
  input :  (tid, d_iter, k_iter)
  step 1:  splitTid                                      (waves vs lanes)
  step 2:  toLDSRowCol                                   (build d, k)
  step 3:  unmerge(d, k) → source_offset                 (DxK because ldsLayoutDxK=true)
  ```

The agreement between the two `unmerge` invocations is what guarantees
that vector loads on the per-thread buffer hit a contiguous K span in
LDS, which in turn lets `ThreadwiseReadIntoOp` keep the vectorised
load it picks for data tiles.

---

## 7. Tooling & workflow integration

### `rocmlir-gen`

* `--broadcastScales` defaults to `false` (natural form).
  Pass `-broadcastScales=true` to generate the legacy broadcasted IR.
* `--quantBlockSize` is hard-rejected for any value other than
  `kQuantBlockSize` (32) for both natural and broadcast paths,
  because the lowering would silently re-group broadcasted scales to
  32-element blocks otherwise.
* The non-accel CPU verifier path (`-pv`) re-broadcasts scales on the
  fly when `broadcastScales=false`, so numerical comparison still
  works against the pre-existing elementwise reference.

### `perfRunner.py` & `tuningRunner.py`

* `--data-type` accepts `bf16` and `f4E2M1FN`. The latter maps to
  `out_dtype f32` for scaled-GEMM output.
* `perfRunner.py` skips any `-scaledGemm` config line whose data type
  is not `f4E2M1FN` (the `rock.gemm` verifier only accepts FP4 for
  scaled GEMM). Without this guard, the cross-product expansion of
  `DATA_TYPES_GEMM` × config lines would generate failing
  configurations at run time.
* The tuning runner uses the default natural-form scales path and
  forwards `gemm_scale_type` through `GemmConfiguration` unchanged.

### External benchmark drivers (`benchmarkUtils.cpp`)

* `parseCommandLine` accepts both space-separated (`-flag value`) and
  equals-sign (`-flag=value`) formats for scaled-GEMM flags
  (`-scale_a_dtype`, `-scale_b_dtype`, `-transScaleA`,
  `-transScaleB`), and both `-scaledGemm` / `--scaledGemm`. This
  matches what `perfRunner.py` emits today and lets external CK /
  hipBLASLt comparison paths consume the same command lines.
* `--broadcastScales` and `--quantBlockSize` are intentionally
  generation-time flags (consumed by `rocmlir-gen` only) and are not
  accepted by the benchmark binary.

### Jenkins / CI configurations

* Three representative scaled-GEMM entries were added to
  `mlir/utils/jenkins/ci-configs/selected-gemm-configs` so automated
  perf and tuning workflows exercise the natural-form path. They use
  `f4E2M1FN` data with `f8E8M0FNU` scales, matching the only
  combination the verifier accepts for scaled GEMM.

---

## 8. Validation

### 8.1 Unit tests (`mlir/test/Dialect/Rock/`)

Three FileCheck tests were rebaselined against the new IR shapes:

* `gemm_to_gridwise.mlir` — checks that `compactBroadcastedScale`
  inserts the right transform chain and that the resulting
  `rock.gridwise_gemm_accel` carries the natural-form scale memrefs.
* `gridwise_gemm_accel_lowering.mlir` — checks LDS allocation sizes
  for both the natural and broadcasted (fallback) paths, and the
  `quantBlockSize` attribute on `rock.blockwise_load_tile`.
* `lowering_to_threadwise_accel.mlir` — checks the per-thread register
  type, the absence of `vector.extract` for natural scales, and the
  presence of `vector.extract` for the broadcast fallback.

All `Dialect/Rock` tests pass (122 tests, 100% of supported cases).

### 8.2 End-to-end tests (gfx950 / MI350X)

| Suite                              | Result                 |
| ---------------------------------- | ---------------------- |
| `GemmScaled` + `GemmScaledF32` + `GemmScaledF4Regression` + `gemm_scaled_split_k_f4` + `GemmFp8NeutralScales` (combined) | 187 / 187 pass |
| `Fusion :: mixr-gemm-fp4`          | 9 / 9 pass             |
| `Dialect/Rock`                     | all supported tests pass |
| `Conversion` + `IR`                | 83 / 83 pass           |
| Full `e2e`                         | 11 596 pass, 0 failed, 96 unsupported |

The unsupported cases in the full e2e run are shapes that pre-date
scaled MFMA support (e.g. WMMA on gfx1250) and are unrelated to this
change. No regressions in any data type or schedule.

### 8.3 IR-level checks

For a representative FP4 kernel
(`-g 1 -m 4096 -n 4096 -k 4096 -transB true`), dumped IR after
`rocmlir-driver --kernel-pipeline=gpu` confirms:

| Attribute / op                            | Baseline                              | HEAD                          |
| ----------------------------------------- | ------------------------------------- | ----------------------------- |
| `gpu.func` `rock.shared_buffer_size`      | `98304`                               | `34816`                       |
| Per-thread scale buffer element type      | `vector<32xf8E8M0FNU>`                | `f8E8M0FNU` (scalar)          |
| `vector.extract %scale[0]` before MFMA    | yes                                   | absent                        |

### 8.4 End-to-end performance (gfx950 / MI350X)

Measured with `mlir/utils/performance/perfRunner.py --op gemm -b`
(MLIR-only batch mode, 100 repeats, rocprofv3) on a single
MI350X (CDNA4, 256 CUs, 8 chiplets) at 04:18 UTC, with FP4
(`f4E2M1FN`) data and `f8E8M0FNU` scales for both operands. Each row
runs the same shape on the same machine, swapping only the rocMLIR
binaries:

* **Baseline** = `origin/develop` at PR #2210, i.e. the same state the
  branch is rebased onto, **without** the natural-scale changes.
* **HEAD** = `users/umayadav/scaled-gemm-no-broadcast` after rebase.

| G | M    | N    | K     | Baseline TFlops | HEAD TFlops | Speedup |
| - | ---- | ---- | ----- | --------------- | ----------- | ------- |
| 1 | 4096 | 4096 |  4096 |          172.28 |     1487.13 |  8.63x  |
| 1 | 8192 | 1024 |  4096 |          425.91 |     1065.50 |  2.50x  |
| 1 | 1024 | 8192 |  4096 |          431.05 |     1130.72 |  2.62x  |
| 1 | 2048 | 2048 |  8192 |          436.99 |     1102.23 |  2.52x  |
| 1 | 1024 | 1024 | 16384 |          439.75 |      803.79 |  1.83x  |
| 3 | 1024 | 1024 |   768 |          392.83 |      545.60 |  1.39x  |
| **Geomean** |  |  |       |  |  | **2.67x** |

The 4096³ case sees the largest improvement (8.6x) because the
baseline fits only **1 workgroup per CU** (its 96 KB LDS footprint
blows past the 64 KB per-CU limit before doubling on the 160 KB
gfx950 LDS), while HEAD's 34 KB footprint admits 4 workgroups per CU.
Smaller / longer-K shapes show diminishing returns because they were
not LDS-bound at baseline; the 1.4-2.6x range there comes from the
combined VGPR-pressure and `vector.extract` removals.

The full e2e run (e.g. all `GemmScaled*` configs at
`schedule_version 1` and `schedule_version 2`) reports no numerical
or correctness regressions, so all of the speedup translates to
realised application throughput.

---

## 9. Files changed (summary)

Generated from `git diff --stat origin/develop` against the rebased
branch (excluding the `docs/` entry itself):

```
 mlir/include/mlir/Dialect/Rock/IR/AccelEmitter.h                  |  12 +-
 mlir/include/mlir/Dialect/Rock/IR/RockOps.td                      |  20 +-
 mlir/include/mlir/Dialect/Rock/utility/loweringUtils.h            | 126 +-
 mlir/lib/Conversion/TosaToRock/TosaToRock.cpp                     |  91 +-
 mlir/lib/Dialect/Rock/IR/RockDialect.cpp                          | 113 +-
 mlir/lib/Dialect/Rock/Transforms/BlockwiseGemmToThreadwise.cpp    |  18 +-
 mlir/lib/Dialect/Rock/Transforms/BlockwiseLoadTileToThreadwise.cpp|  28 +-
 mlir/lib/Dialect/Rock/Transforms/GemmToGridwise.cpp               | 133 +-
 mlir/lib/Dialect/Rock/Transforms/GridwiseGemmToBlockwise.cpp      | 240 +-
 mlir/lib/Dialect/Rock/Transforms/ThreadwiseGemmLowering.cpp       |  39 +-
 mlir/lib/Dialect/Rock/utility/AccelEmitter.cpp                    |  29 +-
 mlir/lib/Dialect/Rock/utility/loweringUtils.cpp                   |  92 +-
 mlir/test/Conversion/TosaToRock/tosa-to-rock-matmul-t-block-scaled.mlir |  26 +-
 mlir/test/Dialect/Rock/gemm_to_gridwise.mlir                      |  86 +-
 mlir/test/Dialect/Rock/gridwise_gemm_accel_lowering.mlir          | 105 +-
 mlir/test/Dialect/Rock/lowering_to_threadwise_accel.mlir          | 147 +-
 mlir/test/Dialect/Rock/ops_error.mlir                             |   6 +-
 mlir/test/rocmlir-gen/gemm-kernel-scaled.mlir                     | 256 +-
 mlir/tools/rocmlir-gen/rocmlir-gen.cpp                            |  60 +-
 mlir/utils/jenkins/ci-configs/selected-gemm-configs               |   8 +
 mlir/utils/performance/common/benchmarkUtils.cpp                  |  76 +-
 mlir/utils/performance/perfRunner.py                              |  19 +-
```

Total: 22 files changed, ~1157 insertions, ~573 deletions (excluding
the design document itself).

---

## 10. Future work

1. **Remove the broadcast fallback.** The fallback exists only because
   `loadAndStoreGemmInputTile` requires `copyPerThread ≥ 1`. A future
   change could allow workitems to participate in scale loads at sub-K
   granularity (or have idle workitems) and drop the broadcast path
   entirely.
2. **WMMA scaled GEMM.** `WmmaEmitter::wrapLDSBufferForLoad` asserts
   `quantBlockSize == 1` defensively, and
   `GridwiseGemmAccelRewritePattern` produces a clean op-level
   `emitOpError(...)` diagnostic when the combination is requested
   (so users see a proper error, not a release-build crash). Adding
   scaled WMMA support would lift both the assert and the diagnostic
   and reuse the same transform chain as the MFMA path.
3. **Direct-to-LDS scale loads.** The current implementation rejects
   scaled GEMM with `directToLDS = true`
   (`GridwiseGemmToBlockwise.cpp`). Re-enabling that path would let
   scales bypass the per-thread regs entirely, which is plausible now
   that the LDS view is K-contiguous.
4. **Other quant block sizes.** `kQuantBlockSize` is the OCP MX
   constant 32 today. The plumbing already takes the value as a
   parameter, so supporting other block sizes would only require
   exposing it on the `rock.gemm` op. Until then, `rocmlir-gen` and
   the lowering reject `quantBlockSize != 32` so that broadcasted
   scales can never be silently re-grouped to the wrong block size.
