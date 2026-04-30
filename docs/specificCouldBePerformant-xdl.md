# `PopulateParamsXDL::specificCouldBePerformant` — what it does and why it exists

This note explains **`specificCouldBePerformant`** in `mlir/lib/Dialect/Rock/Tuning/GridwiseGemmParams.cpp` **without assuming** prior knowledge of Rock tuning or GPU GEMMs.

---

## The problem it solves

Rock picks **tuning parameters** for accelerated (MFMA / “XDL”) GEMMs: block sizes, waves per block, instruction tile shape, and so on. Many combinations are **legal** for the compiler but **not worth exploring** in practice.

The tuner therefore uses a **cheap filter**: for some tuning modes (notably the **“Full”** brute-force space), it drops candidates that are **unlikely to be competitive** before running heavier checks or search. That keeps iteration counts and CI/runtime closer to what the project historically tuned against.

This function is the **MFMA-specific** piece of that filter. The **WMMA** path uses a stub that accepts everything (`success()`), so MFMA is where the interesting pruning lives.

---

## Where this hook sits

Rough call chain (accel / MFMA path, no fused reduction):

1. **`PopulateParamsAccel::couldBePerformant`**  
   - If the GEMM has a **fused reduction**, a *different* heuristic runs (`couldFusedReductionBePerformant`).  
   - Otherwise it calls **`specificCouldBePerformant`** (MFMA vs WMMA implementation).

2. **`PopulateParamsXDL::specificCouldBePerformant`**  
   - Implements the MFMA-only rule described below.

3. **When it matters**  
   In `RockTuningImpl.cpp`, the brute-force builder **`createGemmTuningRangeBF`** only applies **`couldBePerformant`** when the tuning set kind is **`Full`**. For other kinds, the same structural search may skip this “unlikely to be good” layer.

So: this is **not** a correctness check. Validity is handled elsewhere (e.g. `paramsProbablyValid` / `isValidBlockwiseGemm`). This is **search-space shaping** for performance tuning.

---

## Vocabulary (minimal)

From `AccelGemmParamsAttr` (see `RockAttrDefs.td`):

| Field | Meaning (intuitive) |
|--------|---------------------|
| **`mPerBlock` / `nPerBlock`** | How much of the M and N dimensions one workgroup covers. |
| **`mPerWave` / `nPerWave`** | How much M and N one **wavefront** owns within that tile. |
| **`mnPerXdl`** | The M×N “shape” of the **MFMA/XDL instruction** used in this config (instruction-level tile). |

**Waves in the 2D grid:**  
If the block is divided into wave tiles of size `mPerWave × nPerWave`, then:

- **`mWaves`** = `mPerBlock / mPerWave` — number of waves along M.  
- **`nWaves`** = `nPerBlock / nPerWave` — number of waves along N.  
- **`numWaves`** = `mWaves * nWaves` — total wavefronts in that workgroup for this config.

(The code assumes these divisions are exact in contexts where the attr is already consistent; the filter is applied to candidate attrs produced by the tuner.)

---

## What the function computes (step by step)

```text
nPerWave   ← params.getNPerWave()
mWaves     ← mPerBlock / mPerWave
nWaves     ← nPerBlock / nPerWave
mnPerXdl   ← params.getMnPerXdl()
numWaves   ← mWaves * nWaves
```

Then it **accepts** the config (`success()`) only if **one** of these holds:

1. **`numWaves == 4`** and **`mnPerXdl <= nPerWave`**
2. **`numWaves == 2`** and **`mnPerXdl == nPerWave`**
3. **`numWaves == 1`** and **`mnPerXdl == nPerWave`**

Otherwise it **rejects** (`failure()`).

So the filter only looks at **wave layout** (`numWaves` ∈ {1, 2, 4}) and a **relationship between instruction tile width** (`mnPerXdl`) and **N per wave** (`nPerWave`). It does **not** look at K, split-K, datatype, or architecture strings (the `dataTypeA` / `dataTypeB` parameters are unused here).

---

## Why this “works” (what that really means)

The in-code comment says the intent is to **keep full tuning aligned with how it behaved before** similar logic was refactored — i.e. it is a **preserved empirical / historical envelope**, not a proof that every rejected config is slow or every accepted one is fast.

In practice:

- **Accepted configs** form a **small family** of (wave count, instruction vs wave N geometry) combinations that matched the **legacy tuning set** well enough that the team kept them as the default “interesting” region for **Full** search.
- **Rejected configs** are not necessarily invalid; they are **deprioritized** so the **Full** space does not explode (e.g. toward thousands of combinations) when a port omits this filter.

So “works” here means: **reduces the Full tuning set to a manageable, historically consistent subset**, not “optimal for all future hardware.”

---

## Can we apply the same technique elsewhere?

Yes — as a **pattern**, not necessarily by copying these exact inequalities.

### What the technique *is*

1. **Separate concerns**  
   - **Legality** (can we compile and run this config?) vs **likelihood** (should we spend budget exploring it?).

2. **Cheap predicate before expensive work**  
   A few integer checks on parameters already in hand, before benchmarking or large nested search.

3. **Domain-specific heuristic**  
   The actual rule is tied to **MFMA block/wave/instruction geometry** in Rock. It is **not** universal.

### Where it applies well

- Any **autotuning** or **grid search** with a **huge Cartesian product** (tiling, fusion, backends).
- Any pipeline with **`paramsProbablyValid`-style checks** plus an optional **“soft” filter** for exploration order or set size.
- **Ports of Rock tuning** (e.g. another runtime listing the same attr fields): reusing **this exact function** keeps Full-set sizes comparable to rocMLIR; omitting it often inflates counts sharply.

### Risks if you copy blindly

- **New architectures** or **new MFMA shapes** may need different geometry rules; a frozen heuristic can **hide** good configs or **keep** bad ones.
- **`dataTypeA` / `dataTypeB`** are unused here; if you need dtype-aware pruning, this function would need extending, not duplicating as-is.
- **WMMA** already uses “accept all” in rocMLIR; unifying MFMA/WMMA might need **different** predicates per backend.

### Summary

**Same technique:** optional **`couldBePerformant`** (or similar) layer, applied only where you want a **smaller exploratory set**, documented as **heuristic / historical**, and kept separate from **validity**.

**Same formula:** only where the **parameter meanings and codegen** match Rock’s MFMA path; otherwise, derive a **new** small predicate from your own baselines and measurements.

---

## References in-tree

| Item | Location |
|------|----------|
| MFMA implementation | `mlir/lib/Dialect/Rock/Tuning/GridwiseGemmParams.cpp` (`PopulateParamsXDL::specificCouldBePerformant`) |
| Dispatch + fused-reduction branch | `PopulateParamsAccel::couldBePerformant` in the same file |
| Full-space insertion guard | `createGemmTuningRangeBF` in `mlir/lib/Dialect/Rock/Tuning/RockTuningImpl.cpp` |
| Attr fields | `Rock_AccelGemmParamsAttr` in `mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td` |
