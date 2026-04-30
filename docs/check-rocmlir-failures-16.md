# Analysis: 16 `ninja check-rocmlir` failures

Captured from `llvm-lit -a` on each failing test (same machine / `gfx942` as your run). Failures fall into **four** patterns; several tie directly to **clone-harness Option C** (`func.call` + `return` vs `mhal.launch` + `mhal.await`) and the shared flag in `MIGraphXExperimentalFlags.h`.

---

## Summary table

| Category | Count | Symptom |
|----------|------|---------|
| A. Harness contract / FileCheck | 1 | First `RUN` expects `mhal.launch` in generated IR; C1 path emits **`func.call`** |
| B. Verifier output shape | 6 | FileCheck expects `RMS = …`; runner prints only **`[1 1 1]`** (no histogram) |
| C. Compile-time lowering (`-ph`) | 5 | **`failed to legalize unresolved materialization`** (`tensor` ↔ `!migraphx.shaped<…>`), then empty stdin for FileCheck |
| D. Numerical / NaN / histogram | 4 | **`RMS = nan`**, **`-nan`/`-inf`**, or histogram vs CHECK mismatch |

---

## A. Explicit `mhal.launch` expectation (clone-harness IR change)

**Test:** `fusion/pr-e2e/mixr-dot-int4-f16.mlir` (first `RUN` only)

- **Expected (HASINT4):** `mhal.launch` and `tensor<64xi4>` in harness output.
- **Actual:** With default `gMigraphxExperimentalCloneHarnessCallPath == true`, `populateCloneHarnessLogic` emits **`func.call`** to the FUT, so **`mhal.launch` never appears** and FileCheck fails (later lines in the log show module IR scrolling past unrelated checks).

**Fix directions**

- Update the `HASINT4` checks to accept **either** legacy launch **or** call-based harness (e.g. `CHECK-DAG: func.call` vs `mhal.launch`, or a `SAME` line that matches both shapes), **or** split into two `RUN` lines with `REQUIRES` / separate prefixes for “legacy launch” vs “C1 call” if you keep both modes testable.
- Alternatively, gate that `RUN` on a feature only when the flag is false (if you add a lit feature for rebuilds with the flag off).

---

## B. Verifier prints `[1 1 1]` instead of full RMS block

**Tests**

- `fusion/pr-e2e/tosa-to-rock-exp.e2e.mlir`
- `xmir/pr-e2e/resnet18_blk1/resnet18_blk_part0.mlir`
- `xmir/pr-e2e/resnet18_blk1/resnet18_blk_part2.mlir`
- `xmir/pr-e2e/resnet18_blk2/resnet18_blk2_part0.mlir`
- `xmir/pr-e2e/resnet18_blk3/resnet18_blk3_part1.mlir`
- `xmir/pr-e2e/resnet18_blk3/resnet18_blk3_part2.mlir`

**Symptom:** FileCheck looks for `// CHECK: RMS = {{.*}}e-0X` as the **first** meaningful line; **stdin is only** `[1 1 1]` (three pass bits), so no `RMS` line at all.

**Interpretation:** The clone verifier path is taking a **short output** path (three comparisons all pass → `[1 1 1]`) while tests were written for the **full** RMS / histogram dump. This is likely **rocmlir-gen `-ph` / verifier formatting**, not random GPU noise.

**Root cause (verified for `tosa-to-rock-exp.e2e.mlir`):** `mcpuVerify` in `mlir/lib/ExecutionEngine/conv-validation-wrappers.cpp` only calls `printDebugVerifyResults` (RMS + histogram) when `-print-verify-results=always`, or when verification **fails** (`summary` / `failure` modes). Default is **`summary`**, so on **success** the only stdout is the final `printf("[%d %d %d]\n", …)` → **`[1 1 1]`**. This is unrelated to clone-harness vs `mhal.launch`.

**Fix applied (example):** add `-print-verify-results=always` to `rocmlir-gen -ph` and match verbose output in FileCheck (`Number of elements:`, `RMS = …`, `[1 1 1]`). Alternatively keep `summary` and change checks to expect only `[1 1 1]` (weaker).

**Fix directions**

- Prefer **`-print-verify-results=always`** on tests that need RMS/histogram in stdout, or accept **`[1 1 1]`**-only checks under default summary.

---

## C. Unresolved materialization during `rocmlir-gen -ph`

**Tests**

- `fusion/pr-e2e/mixr-conv-bias-clipped-relu.mlir` — `tensor<108xf32>` vs `!migraphx.shaped<4x3x3x3xf32, …>`
- `fusion/pr-e2e/mixr-gemm/mixr-gemm-tr-folding.mlir` — shaped ↔ flat `tensor<256xf32>` / similar
- `fusion/pr-e2e/mixr-gemm/mixr-gemm-tr-folding2.mlir`
- `fusion/pr-e2e/mixr-sd-explicit-broadcasting.mlir` — `tensor<256xf32>` vs large `!migraphx.shaped<…>`

**Symptom:** Failure in the **second** `rocmlir-gen -ph` step after `rocmlir-driver … mhal -kernel-pipeline full`; then assertion `does -fut point to the wrong function?`, “Architecture not specified…”, empty FileCheck input.

**Interpretation:** **Boundary / shaped lowering** does not fully bridge tensor constants or block arguments when the IR produced for the wrapper + populate-harness step changed (e.g. missing `MHALLaunchConverter` rewrites that used to insert casts). This is **compiler-side**, not GPU numerical.

**Fix directions**

- Inspect IR **before** `-ph` for these tests with C1 on vs flag off; add **casts / rewrite patterns** in MIGraphX→* passes or in `rocmlir-gen` populate logic so no live `tensor`/`shaped` materializations remain.
- Confirm `-fut …_wrapper` still names the function the pass expects after clone-harness renames the entry.

---

## D. Numerical: NaN, inf, or histogram mismatch

**Tests**

- `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache.mlir`
- `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache-f16.mlir`
- `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache-prefix-causal.mlir`  
  - Outputs include **`-nan`**, **`-inf`**, `[0 1 1]` / `[1 1 1]` lines; CHECK lines expecting finite head values fail.
- `fusion/pr-e2e/mixr-expand-strides-non-multiple.mlir` — **`RMS = nan`**
- `fusion/pr-e2e/mixr-non-contiguous-strides.mlir` — large **relDiff** buckets vs CHECK

**Interpretation:** These use **full** pipelines (`mhal`, `full`, `xmir-runner` / GPU). NaNs often indicate **wrong buffers, sync, or dispatch order** when the **host** side no longer uses async `mhal.launch`/`await`. Stride tests may be **environment/arch flakiness** or the same root cause.

**Fix directions**

- Treat as **Type B (MHAL + full)** issues: compare traces with **`gMigraphxExperimentalCloneHarnessCallPath = false`** (rebuild `rocmlir-gen` + driver-linked passes). If failures disappear, the fix belongs in **runtime dispatch / buffer lifetime / sync** for the call-based wrapper, not FileCheck tweaks.
- For attention, re-read the in-file comment about masked positions and LSE; still, pervasive **NaN/inf** suggests execution bug, not tolerance.

---

## Suggested priority

1. **Quick test-suite unblock:** Fix **A** (`mhal.launch` → dual pattern) and **B** (RMS vs `[1 1 1]` contract) — mostly test + verifier output alignment.
2. **Correctness:** **C** (materializations) — real lowering gaps for `-ph` on C1 IR.
3. **Deep:** **D** — may require MHAL/runtime work for async semantics parity with **func.call** host.

---

## Reference: failing test list (from `check-rocmlir`)

1. `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache-f16.mlir` — D  
2. `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache-prefix-causal.mlir` — D  
3. `fusion/pr-e2e/attention/mixr-attention-flash-decoding-kvcache.mlir` — D  
4. `fusion/pr-e2e/mixr-conv-bias-clipped-relu.mlir` — C  
5. `fusion/pr-e2e/mixr-dot-int4-f16.mlir` — A (and possibly second `RUN` if still failing)  
6. `fusion/pr-e2e/mixr-expand-strides-non-multiple.mlir` — D  
7. `fusion/pr-e2e/mixr-gemm/mixr-gemm-tr-folding.mlir` — C  
8. `fusion/pr-e2e/mixr-gemm/mixr-gemm-tr-folding2.mlir` — C  
9. `fusion/pr-e2e/mixr-non-contiguous-strides.mlir` — D  
10. `fusion/pr-e2e/mixr-sd-explicit-broadcasting.mlir` — C  
11. `fusion/pr-e2e/tosa-to-rock-exp.e2e.mlir` — B  
12–16. `xmir/pr-e2e/resnet18_*/resnet18_*.mlir` — B  

---

*Generated for debugging `ninja check-rocmlir` failures; re-run individual tests with `llvm-lit -a <path>` after fixes.*
