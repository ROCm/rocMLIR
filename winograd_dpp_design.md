# DPP-Based Winograd Transforms: Design Document

## 1. Problem Statement

The Winograd F(2,3) convolution requires three value-domain transforms:
- **Input transform**: V = B^T * d * B (4x4 -> 4x4, additions only)
- **Filter transform**: U = G * g * G^T (3x3 -> 4x4, additions + 0.5 multiplies)
- **Output transform**: Y = A^T * M * A (4x4 -> 2x2, additions only)

Each transform is a matrix multiplication on a small tile. The current scalar implementation computes these per-thread with all 16 values in one thread's registers. DPP (Data Parallel Primitives) can accelerate these by distributing tile values across GPU lanes and using cross-lane operations.

## 2. What Was Tried

### Attempt 1: 16-Thread-Per-Tile (files: `GridwiseWinogradGemmLowering_DPP.cpp.bak`, `_DPP_v2.cpp.bak`, `_DPP_LATEST.cpp.bak`)

**Design**: 16 threads per tile, each lane holds ONE value of the 4x4 Winograd domain. Lane `L` holds position `(L/4, L%4)`.

**DPP operations used**:
- `row_shr(N)`: shift data N lanes right within 16-lane row (for row transforms)
- `row_shl(N)`: shift data N lanes left (for row transforms)
- `quad_perm`: shuffle within 4-lane quads (for column transforms)

**Row transform (B^T * d) via row_shr/row_shl**:
```
Row 0: d[0] - d[2] = self - row_shr(d,8)     // lane 0 reads lane 8
Row 1: d[1] + d[2] = self + row_shr(d,4)     // lane 4 reads lane 8
Row 2: -d[1] + d[2] = self - row_shl(d,4)    // lane 8 reads lane 4
Row 3: d[1] - d[3] = row_shl(d,8) - self     // lane 12 reads lane 4
```

**Result**: ALL values wrong (0% correct). The error was initially masked by a missing `amdgpu::AMDGPUDialect` dependency that caused the DPP ops to crash silently, falling through to a broken code path.

**Root causes identified**:
1. Missing `amdgpu::AMDGPUDialect` in `dependentDialects` of the pass registration in `Passes.td`. Fixed by adding the dialect dependency.
2. Missing `#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"` in `Passes.h`. Fixed by adding the include.
3. After fixing both, DPP ops were correctly emitted (verified: 20 `amdgpu.dpp` ops in IR) but still produced wrong results.
4. `row_shr` and `row_shl` do NOT wrap within the 16-lane DPP row. Out-of-bounds lanes receive the `old` parameter value (zero). This was accounted for by using per-row `arith.select` to choose the correct formula, but the final result was still wrong.
5. The exact mathematical error was never pinpointed. Possible causes: incorrect interaction between `row_shr`/`row_shl` and the `select`-based per-row dispatch, or unexpected DPP behavior at row boundaries.

### Attempt 2: 4-Thread-Per-Tile (file: `GridwiseWinogradGemmLowering_DPP4.cpp.bak`)

**Design**: 4 threads per tile, each thread = 1 column. Each thread holds all 4 row values of its column in separate registers. `quad_perm` used for column transforms, scalar for row transforms.

**DPP operations used**:
- `quad_perm:[c,c,c,c]`: broadcast column `c` to all 4 lanes in quad (for column transforms)
- No `row_shr`/`row_shl` (row transforms done with scalar add/sub within each thread)

**Column transform (temp * B) via quad_perm**:
```cpp
Value c0 = qpBroadcast(val, 0);  // all lanes get thread 0's value
Value c1 = qpBroadcast(val, 1);  // all lanes get thread 1's value
Value c2 = qpBroadcast(val, 2);  // all lanes get thread 2's value
Value c3 = qpBroadcast(val, 3);  // all lanes get thread 3's value

// B columns: V[i][0]=c0-c2, V[i][1]=c1+c2, V[i][2]=c2-c1, V[i][3]=c1-c3
// Select based on myCol
```

**Row transform (B^T * d) via scalar**:
```cpp
// Each thread has d[0..3][myCol] in registers
temp[0] = d[0] - d[2];    // BT row 0
temp[1] = d[1] + d[2];    // BT row 1
temp[2] = d[2] - d[1];    // BT row 2
temp[3] = d[1] - d[3];    // BT row 3
```

**Result**: ~50% of values exactly correct (relDiff < 1e-6), ~50% exactly NEGATED (relDiff = 2.0). Example: expected 164.84, got -164.84.

**Critical debugging finding**: The tests that appeared to "pass" (K=32, K=48, K=63) were actually bypassing Winograd entirely and going through the direct convolution path! The eligibility check in `ConvToWinograd.cpp` had a profitability threshold `totalOutput < 256` that rejected small output sizes. Only K=64 (totalOutput=256) actually went through Winograd and FAILED.

**Root cause analysis for the negation**:
- The sign flip is CONSISTENT and EXACT (values are precisely negated, not approximately wrong)
- This suggests a single sign error in one of the DPP transform formulas
- The B column transform formulas were verified correct against the B matrix
- The GT column transform formulas were verified correct against the GT matrix
- The A column transform formulas were verified correct against the A matrix
- The G row transform (scalar) was verified correct
- The BT row transform (scalar) was verified correct
- The AT row transform (scalar) was verified correct

**Remaining hypotheses for the negation bug**:
1. The `quad_perm` DPP op might have different semantics than expected on gfx942. The `old` parameter and `bound_ctrl` might interact unexpectedly.
2. The thread-to-quad mapping might not align as expected. With `blockSize=256`, threads 0-3 should be in quad 0 of wavefront 0, but there might be a platform-specific thread assignment.
3. The `arith.select` based column selection might have a bug in the select chain ordering (though it was verified correct in the code).
4. The filter transform (G row + GT column) might produce values with the wrong sign for specific column positions, causing the element-wise multiply to negate the accumulator.

## 3. How MIOpen Implements DPP Winograd

### Source Reference
- [Conv_Winograd_v21_1_3_gfx9_fp32_f2x3_stride1.inc](https://github.com/ROCm/rocm-libraries/blob/develop/projects/miopen/src/kernels/Conv_Winograd_v21_1_3_gfx9_fp32_f2x3_stride1.inc) (2784 lines of GCN assembly)
- [Conv_Winograd_v40_6_0_gfx12_fp16_dot2_f2x3_stride1.inc](https://github.com/ROCm/rocm-libraries/blob/develop/projects/miopen/src/kernels/Conv_Winograd_v40_6_0_gfx12_fp16_dot2_f2x3_stride1.inc) (11692 lines, "Fury" kernel)

### MIOpen's Data Layout

MIOpen does NOT use 16-threads-per-tile or 4-threads-per-tile. Instead:

**Multiple registers per lane**: Each lane (thread) holds MULTIPLE Winograd values in SEPARATE registers. For example, registers v2-v5 in each lane hold 4 different values related to the SAME tile position but for different transform coefficients or accumulator positions.

**Multiple tiles processed per DPP row**: The 16-lane DPP row processes MULTIPLE tiles simultaneously. Each set of 4 registers (e.g., v2,v3,v4,v5) holds values for DIFFERENT tiles at the same Winograd position.

### MIOpen's Butterfly Pattern (Output Transform A^T * M * A)

From the v21 fp32 kernel (lines 2417-2427):

```asm
// Phase 1: quad_perm pair swap [2,3,0,1] with sign coefficient, bank_mask:0xc
v_mac_f32_dpp v4, v4, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v5, v5, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v2, v2, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc
v_mac_f32_dpp v3, v3, v101 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xc

// Phase 2: row_mirror + add
v_add_f32_dpp v3, v4, v3 row_mirror row_mask:0xf bank_mask:0xf
v_add_f32_dpp v2, v5, v2 row_mirror row_mask:0xf bank_mask:0xf

// Phase 3: quad_perm adjacent swap [1,0,3,2] with sign coefficient, bank_mask:0x6
v_mac_f32_dpp v3, v3, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6
v_mac_f32_dpp v2, v2, v102 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0x6

// Phase 4: row_half_mirror + add → final result
v_add_f32_dpp v2, v3, v2 row_half_mirror row_mask:0xf bank_mask:0xf
```

### Sign Coefficient Setup

MIOpen pre-computes sign coefficients into registers v101 and v102:

```asm
v_mov_b32 v101, 1
v_xor_b32_dpp v101, v0, v0 quad_perm:[2,3,2,3] bank_mask:0x4  // modify bank 2
v_xor_b32_dpp v101, v0, v0 quad_perm:[0,1,0,1] bank_mask:0x8  // modify bank 3
v_subrev_co_u32 v101, vcc, 1, v101   // 1 - result
v_cvt_f32_i32 v101, v101             // convert to float {+1.0, -1.0, ...}
```

The `bank_mask` selectively applies XOR patterns to specific quads (banks) within the 16-lane row. This creates {+1, -1} patterns that encode the A^T/A matrix signs.

### Key Differences from Our Attempts

| Aspect | Our 16-thread attempt | Our 4-thread attempt | MIOpen |
|--------|----------------------|---------------------|--------|
| Threads per tile | 16 | 4 | Not fixed to tile |
| Values per lane | 1 | 4 (one per row) | Multiple (multiple tiles) |
| Row operations | `row_shr`/`row_shl` | Scalar per-thread | `row_mirror` + `row_half_mirror` |
| Column operations | `quad_perm` broadcast | `quad_perm` broadcast | `quad_perm` pair/adjacent swap |
| Sign handling | `arith.select` per row | `arith.select` per col | Pre-computed sign registers + `v_fmac_dpp` |
| bank_mask usage | Always 0xf (all) | Always 0xf (all) | Selective (0xc, 0x6) per phase |
| DPP instructions | `amdgpu.dpp` | `amdgpu.dpp` | `v_mac_f32_dpp`, `v_add_f32_dpp` |

### MIOpen's Butterfly Is a Radix-2 Reduction

The 4-phase butterfly is mathematically a radix-2 parallel reduction:
1. **Phase 1**: Combine values at distance 2 within each quad (pair swap)
2. **Phase 2**: Combine values across 8-lane half-rows (row_mirror)
3. **Phase 3**: Combine values at distance 1 within each quad (adjacent swap)
4. **Phase 4**: Combine values across 4-lane quarter-rows (row_half_mirror)

This reduces 16 values to the 4 output values in O(log n) steps. The sign coefficients ensure the correct {+1, -1} weights from the A^T and A matrices are applied at each stage.

## 4. Suggested Approaches for Future Implementation

### Approach A: Match MIOpen's Butterfly (Hardest, Best Performance)

1. **Study the exact lane-to-tile mapping** in MIOpen's kernel by tracing the data flow from input load through the GEMM loop to the output transform
2. **Implement sign coefficient setup** using `v_xor_b32_dpp` + `v_subrev` + `v_cvt_f32_i32` to create the correct {+1, -1} patterns in dedicated registers
3. **Use `v_mac_f32_dpp`** (multiply-accumulate with DPP source) instead of separate broadcast + multiply + add
4. **Use selective `bank_mask`** to apply different operations to different quads within the same instruction
5. **Map multiple tiles to the same DPP row** for better utilization

This requires deep understanding of MIOpen's register allocation and may need ISA-level verification.

### Approach B: 4-Thread with Debugging (Easiest to Debug)

The 4-thread approach is closest to working. To fix the negation bug:

1. **Write a standalone unit test** that ONLY tests `quad_perm` broadcast between 4 threads:
   - Thread 0 writes value `A`, threads 1-3 write different values
   - Use `quad_perm:[0,0,0,0]` to broadcast
   - ALL threads should see `A`
   - Run on GPU and verify

2. **Test each transform independently**: Create a kernel that ONLY does the B column transform (no filter, no MAC, no output transform) and compares against expected values

3. **Check for platform-specific quad_perm behavior**: The gfx942 (CDNA3) has wave64. A 4-lane quad within wave64 might behave differently than in wave32

4. **Try LDS instead of DPP**: Use shared memory to exchange values between threads instead of DPP. This is slower but correct, and would confirm whether the issue is in DPP semantics or the transform math

### Approach C: Hybrid Scalar+DPP (Incremental)

1. Keep the working scalar kernel as baseline
2. Add a **post-GEMM DPP optimization** that ONLY applies to the output transform (A^T * M * A), which reduces 16 values to 4
3. The output transform is the simplest to DPP-ify because it's a pure reduction (no data loading issues)
4. If this works, extend to the input transform, then the filter transform

### Approach D: Use the Rock GEMM Pipeline (Best Architecture)

Instead of a standalone DPP kernel, route through the existing `rock.gemm` -> `GridwiseGemmToBlockwise` pipeline:

1. Pre-transform input and filter in separate kernels (simple per-thread loops)
2. Use standard `rock.gemm` with `gemmG = alphaSq * groups` for the batched GEMM
3. Post-transform the output in a separate kernel

This gets MFMA/WMMA acceleration for the GEMM (the compute-bound part) for FREE. The transforms are add/sub only and don't need DPP -- scalar is sufficient. This avoids the DPP complexity entirely while achieving the main performance benefit.

Requires either:
- Multi-function module support in the test harness
- Or changes to `rocmlir-gen` to allocate intermediate buffers

## 5. Files and Infrastructure

### Working Files (in repo)
- `GridwiseWinogradGemmLowering.cpp` -- working scalar 1-thread-per-tile kernel
- `WinogradConsts.h` -- transform matrices B, BT, G, GT, A, AT for F(2,3)
- `WinogradDPP.h` -- DPP helper functions (unused by scalar kernel, available for DPP work)
- `ConvToWinograd.cpp` -- eligibility check with MIOpen-style selection
- `WinogradToGemm.cpp` -- emits `gridwise_winograd_gemm` op

### DPP Backup Files
- `GridwiseWinogradGemmLowering_DPP4.cpp.bak` -- 4-thread DPP (closest to working, ~50% correct)
- `GridwiseWinogradGemmLowering_DPP.cpp.bak` -- 16-thread DPP (first attempt)
- `GridwiseWinogradGemmLowering_DPP_v2.cpp.bak` -- 16-thread with AMDGPU dialect fix
- `GridwiseWinogradGemmLowering_DPP_LATEST.cpp.bak` -- 16-thread final version
- `GridwiseWinogradGemmLowering_SCALAR.cpp.bak` -- backup of working scalar version

### Key Infrastructure Requirements for DPP
1. `amdgpu::AMDGPUDialect` must be in `dependentDialects` of the lowering pass (in `Passes.td`)
2. `#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"` must be in `Passes.h`
3. Grid size must match: `totalThreads = totalTiles * threadsPerTile`
4. The `blockSize` attribute on the function must be set by `WinogradToGemm` (not left to `affix-params`)
5. The profitability threshold in `ConvToWinograd.cpp` must be low enough that the test cases actually go through Winograd (not fallback to direct conv)

## 6. Transform Matrices (from WinogradConsts.h)

```
BT (input row transform, 4x4):
[[ 1,  0, -1,  0],
 [ 0,  1,  1,  0],
 [ 0, -1,  1,  0],
 [ 0,  1,  0, -1]]

B (input column transform, 4x4):
[[ 1,  0,  0,  0],
 [ 0,  1, -1,  1],
 [-1,  1,  1,  0],
 [ 0,  0,  0, -1]]

G (filter row transform, 4x3):
[[ 1,    0,    0  ],
 [ 0.5,  0.5,  0.5],
 [ 0.5, -0.5,  0.5],
 [ 0,    0,    1  ]]

GT (filter column transform, 3x4):
[[ 1,    0.5,  0.5,  0  ],
 [ 0,    0.5, -0.5,  0  ],
 [ 0,    0.5,  0.5,  1  ]]

AT (output row transform, 2x4):
[[ 1,  1,  1,  0],
 [ 0, -1,  1,  1]]

A (output column transform, 4x2):
[[ 1,  0],
 [ 1, -1],
 [ 1,  1],
 [ 0,  1]]
```

## 7. DPP Quick Reference

| Mode | What it does | Wraps? | Use case |
|------|-------------|--------|----------|
| `quad_perm:[a,b,c,d]` | Lane i in quad gets lane perm[i%4]'s value | N/A (within 4) | Column transforms |
| `row_shr(N)` | Lane i gets lane (i+N) if < 16 | NO | Cross-row data (broken for us) |
| `row_shl(N)` | Lane i gets lane (i-N) if >= 0 | NO | Cross-row data (broken for us) |
| `row_mirror` | Lane i gets lane (15-i) within 16-row | Yes | MIOpen butterfly phase 2 |
| `row_half_mirror` | Lane i gets lane (7-i%8) within 8-half | Yes | MIOpen butterfly phase 4 |
| `bank_mask:0xN` | Only apply DPP to selected banks (quads) | - | Selective per-row operations |
| `bound_ctrl` | true: OOB gets 0; false: OOB gets `old` | - | Edge handling |

**Critical note**: `row_shr` and `row_shl` do NOT wrap. This is the primary reason the 16-thread approach failed -- cross-row data movement via shifts produces zeros at boundary lanes.
