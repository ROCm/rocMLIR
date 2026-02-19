# Triton Block Ping-Pong Scheduling: Complete Analysis

## Executive Summary

This document provides a comprehensive analysis of Triton's Block Ping-Pong scheduling
implementation for AMD GPUs. Triton achieves **true compute/memory overlap** through a
**phase-shifted, barrier-rendezvous scheduling scheme** that interleaves **Dot (MFMA)
clusters** with **Memory clusters** across two wave groups.

**Key Insight:** The fundamental difference between Triton's approach and rocMLIR's
current implementation is that Triton **interleaves DIFFERENT types of operations**
(compute vs memory) at cluster boundaries, allowing wave groups to use **different
hardware simultaneously** (MFMA units vs memory controllers).

---

## Table of Contents

1. [Glossary of Terms](#glossary-of-terms)
2. [Core Concepts](#core-concepts)
3. [Triton's Ping-Pong Variants](#tritons-ping-pong-variants)
4. [Phase Shift Mechanism](#phase-shift-mechanism)
5. [Cluster Structure and Barriers](#cluster-structure-and-barriers)
6. [The Critical Difference: Compute/Memory Interleaving](#the-critical-difference-computememory-interleaving)
7. [Code References](#code-references)
8. [Comparison with rocMLIR](#comparison-with-rocmlir)
9. [Lessons for rocMLIR](#lessons-for-rocmlir)

---

## Glossary of Terms

### Triton-Specific Terms

| Term | Definition |
|------|------------|
| **Dot Cluster** | A cluster containing MFMA/WMMA compute operations (`tt.dot`). Uses MFMA/VALU execution units. |
| **Memory Cluster** | A cluster containing memory operations: global loads, local stores, LDS reads/writes. Uses memory controllers and LDS. |
| **Cluster** | A group of operations that execute together as a unit, separated by barriers. |
| **Cluster Barrier** | A synchronization point (`ttg.barrier` + `sched.barrier`) between clusters. |
| **Phase Shift** | The initial offset created between wave groups so they execute different clusters simultaneously. |
| **cond_barrier** | Conditional barrier (`amdg.cond_barrier`): only threads with condition=true execute the barrier. Used to create phase shift. |
| **sched.barrier** | Scheduler barrier (`rocdl.sched.barrier`): prevents the LLVM backend from reordering instructions across this boundary. |
| **s_setprio** | Priority instruction: sets wave priority (0-3) to control which wave wins when competing for instruction slots. |
| **Dot Slicing** | Splitting a single `tt.dot` operation into multiple smaller slices along the K-dimension. |
| **numStages** | Number of pipeline stages (buffers) for software pipelining. |
| **numWarps** | Number of warps per workgroup (typically 4 or 8). |
| **Asymmetric Sync** | Triton's term for phase-shifted synchronization where wave groups are at different loop iterations. |

### AMD GPU ISA Terms

| Term | Definition |
|------|------------|
| **MFMA** | Matrix Fused Multiply-Add instruction. Executes matrix operations on tensor cores. |
| **VALU** | Vector ALU instructions (`v_xxx`). Shared between compute and memory address calculations. |
| **SALU** | Scalar ALU instructions (`s_xxx`). Used for control flow, barriers, waits. |
| **s_barrier** | Workgroup barrier instruction. Waits until all waves in workgroup reach barrier. |
| **s_waitcnt** | Wait instruction. Waits for memory operations to complete (`vmcnt`, `lgkmcnt`). |
| **lgkmcnt** | LDS/GDS/Konstant memory counter. Counts outstanding LDS operations. |
| **vmcnt** | Vector memory counter. Counts outstanding global memory operations. |
| **ds_read** | LDS (shared memory) read instruction. |
| **ds_write** | LDS (shared memory) write instruction. |

### Pipeline Terms

| Term | Definition |
|------|------------|
| **Prologue** | Initial iterations that fill the pipeline (prefetch data). |
| **Main Loop** | Steady-state iterations with full pipeline overlap. |
| **Epilogue** | Final iterations that drain the pipeline. |
| **Double Buffering** | Using 2 buffers to overlap iteration N's compute with iteration N+1's memory. |
| **Triple Buffering** | Using 3 buffers for deeper pipeline overlap. |

---

## Core Concepts

### What is Ping-Pong Scheduling?

Ping-pong scheduling is a technique to maximize hardware utilization by having two
wave groups execute **different types of work simultaneously**:

```
Traditional Execution (All waves do same work):
  Time →
  Wave Group 0:  [Memory][Barrier][Compute][Barrier][Memory][Barrier][Compute]...
  Wave Group 1:  [Memory][Barrier][Compute][Barrier][Memory][Barrier][Compute]...
                 ↑ All waves idle during barriers, then all compete for same HW

Ping-Pong Execution (Wave groups do different work):
  Time →
  Wave Group 0:  [Compute][Barrier][Memory][Barrier][Compute][Barrier][Memory]...
  Wave Group 1:       [Memory][Barrier][Compute][Barrier][Memory][Barrier][Compute]...
                      ↑ Phase shift: Group 1 starts one cluster behind
                 
  At any barrier release:
    - Group 0 uses MFMA units (Compute cluster)
    - Group 1 uses Memory controllers (Memory cluster)
    - DIFFERENT HARDWARE → TRUE PARALLEL EXECUTION
```

### Why This Works

1. **Different Hardware Units**: Compute uses MFMA units; Memory uses memory controllers/LDS
2. **Barrier Pairing**: Both groups hit `s_barrier` (4+4=8 waves), counter reaches 8, all release
3. **Phase Offset**: Groups are always at different cluster types after barriers release
4. **Priority Control**: `s_setprio` prevents one group from monopolizing shared VALU

---

## Triton's Ping-Pong Variants

Triton implements **6 different ping-pong scheduling variants**, each optimized for
specific tile sizes and kernel characteristics:

### Variant 1: One-Cluster Ping-Pong (`transformOnePPClusters`)

**Target:** Small tiles (e.g., 128×128×64 FP16) with `numWarps=4`

**Structure:**
```
Memory Cluster #0:
  setprio(1)
  global_load A
  sched.barrier
  local_load B
  setprio(0)
  global_load B

Dot Cluster #0:
  setprio(1)
  dot(A, B)
  setprio(0)
```

**Key Feature:** Used when numWarps=4 and waves from different blocks share a SIMD.
No asymmetric sync needed because waves are from different workgroups.

### Variant 2: Two-Cluster Ping-Pong (`transformTwoPPClusters`)

**Target:** Medium tiles (e.g., 256×128×64 FP16) with `numWarps=8`, `numStages=2`

**Structure:**
```
Memory Cluster #0:
  local_load A(slice 0), B(slice 0)
  sched.barrier
  global_load A
  sched.barrier
  local_load A(slice 1), B(slice 1)
  sched.barrier
  global_load B
  s_barrier              ← Control-only, no LDS wait
  sched.barrier

Dot Cluster #0:
  setprio(1)
  dot(slice 0)
  setprio(0)
  ttg.barrier(local)     ← Full barrier with LDS sync
  sched.barrier

Memory Cluster #1:
  local_store A, B

Dot Cluster #1:
  setprio(1)
  dot(slice 1)
  setprio(0)
  ttg.barrier(local)
  sched.barrier
```

**Key Feature:** Slices dot into 2 pieces, interleaves memory operations with compute.
Uses `s_barrier` (not `ttg.barrier`) at first cluster boundary to avoid pulling in LDS waits.

### Variant 3: Four-Cluster Ping-Pong (`transformFourPPClusters`)

**Target:** Large tiles (e.g., 256×256×64 FP16) with `numWarps=8`, `numStages=2`

**Structure:**
```
mem0: global_load A, local_load A(1/4), B(1/4)
      ttg.barrier + sched.barrier
dot0: setprio(1), dot(1/4), setprio(0)
      ttg.barrier + sched.barrier

mem1: global_load B, local_load A(2/4), B(2/4)
      ttg.barrier + sched.barrier
dot1: setprio(1), dot(2/4), setprio(0)
      ttg.barrier + sched.barrier

mem2: local_load A(3/4, 4/4), B(3/4, 4/4)
      ttg.barrier + sched.barrier
dot2: setprio(1), dot(3/4), setprio(0)
      ttg.barrier + sched.barrier

mem3: local_store A, B
      ttg.barrier + sched.barrier
dot3: setprio(1), dot(4/4), setprio(0)
```

**Key Feature:** Slices dot into 4 pieces to reduce register pressure.
More cluster boundaries = more overlap opportunities but higher overhead.

### Variant 4: Async Two-Cluster (`transformTwoClusterWithAsyncAndAll`)

**Target:** Scaled-dot operations (FP8) with async copy, `numWarps=8`, tile 256×256

**Structure:**
```
Cluster #0: (Async copy only)
  async_commit
  global_load (LDS-bypassed)
  sched.barrier
  s_barrier
  sched.barrier

Cluster #1: (All other ops)
  dot with "pingpong_2step" attribute
  (triggers special MFMA lowering to keep ds_read with first MFMA group)
```

**Key Feature:** Isolates async copy from compute. Requires special two-step lowering
where dot is split into MFMA groups during LLVM lowering.

### Variant 5: Chained-Dot Schedule (`transformChainedDotSchedule`)

**Target:** FlashAttention-style kernels with 2 chained dots, `numStages=4`

**Structure:**
```
// Memory gets HIGHER priority than compute!
ComputeCluster1:
  s_barrier (at loop start)
  sched.barrier
  dot1

MemoryCluster1:
  setprio(1)            ← Higher priority for memory!
  sched.barrier
  ttg.barrier(local) OR async_wait
  s_waitcnt lgkmcnt(0)  ← At END of memory cluster

ComputeCluster2:
  sched.barrier
  setprio(0)
  s_barrier
  s_waitcnt lgkmcnt(0)
  dot2

MemoryCluster2:
  setprio(1)
  sched.barrier
  ttg.barrier(local) OR async_wait
  s_waitcnt lgkmcnt(0)
```

**Key Feature:** Memory cluster gets HIGHER priority than compute cluster.
This is because both clusters contain `v_xxx` (VALU) instructions, and giving
memory higher priority ensures it can issue address-update instructions even
when compute is busy. This enables true overlap.

### Variant 6: Two-Cluster with Local Load (`transformTwoClusterWithLocalLoadAndAll`)

**Target:** Large LDS usage with async copy, `numStages=3`

**Structure:**
```
local_load A, B
sched.barrier
async_copy A
async_commit
sched.barrier
async_wait
sched.barrier
sched_group_barrier (interleave SALU with MFMA)
async_copy B
async_commit
dot
sched.barrier
s_barrier
sched.barrier
```

**Key Feature:** Uses `sched_group_barrier` to explicitly interleave 3 SALU
instructions per MFMA for better instruction-level parallelism.

---

## Phase Shift Mechanism

### How Phase Shift is Created (`addAsymmetricSyncToLoop`)

```cpp
// Before the loop:
ttg::barrier(local)              // Full barrier to sync all waves
cond_barrier(warpHigh)           // Only waves 4-7 execute s_barrier
                                 // → Waves 4-7 wait while 0-3 enter loop

// After the loop:
cond_barrier(warpLow)            // Only waves 0-3 execute s_barrier
                                 // → Waves 0-3 wait for 4-7 to finish
```

### Why This Doesn't Deadlock

The key insight is that **inside the loop, ALL 8 waves hit the same `s_barrier`**:

```
Pre-loop:
  All 8 waves sync at ttg::barrier → counter = 8 → all pass
  Waves 4-7 hit cond_barrier(warpHigh) → counter = 4
  Waves 0-3 enter loop, hit first cluster barrier → counter = 4+4 = 8 → all release!
  Waves 4-7 exit cond_barrier, enter loop

Inside loop:
  Waves 0-3 at cluster N:   hit s_barrier → counter = 4
  Waves 4-7 at cluster N-1: hit s_barrier → counter = 4+4 = 8 → all release
  Both groups proceed, maintaining 1-cluster offset

Post-loop:
  Waves 0-3 finish last iteration, hit cond_barrier(warpLow) → wait
  Waves 4-7 finish their last iteration
  cond_barrier releases → all 8 waves converge
```

---

## Cluster Structure and Barriers

### Barrier Types Used

| Barrier | When Used | Effect |
|---------|-----------|--------|
| `ttg.barrier(local)` | Between memory and compute clusters | Syncs all waves + ensures LDS visibility |
| `s_barrier` | Inside loop when LDS wait not needed | Syncs waves without waiting for LDS |
| `sched.barrier(0)` | At every cluster boundary | Prevents LLVM from reordering instructions |
| `cond_barrier` | Before/after loop | Creates phase shift (partial barrier) |

### Why `s_barrier` vs `ttg.barrier` Matters

From `BlockPingpong.cpp:L607-L610`:
> "The first cluster just fits into the two cluster pingpong and cannot include wait
> of the local_load inserted by the ttg.barrier, using s.barrier instead. Backend
> will schedule the local memory fences later in the dot0 cluster."

`ttg.barrier(local)` = `s_barrier` + `s_waitcnt lgkmcnt(0)`

The `lgkmcnt(0)` wait can disrupt the ping-pong pattern if placed at the wrong boundary.
By using raw `s_barrier`, Triton has more control over where waits are inserted.

---

## The Critical Difference: Compute/Memory Interleaving

### Triton's Cluster Structure (Four-Cluster Example)

```
Iteration N:
  [mem0: gl A, ll A/4, ll B/4]  ← Memory cluster (uses memory controllers)
  barrier
  [dot0: mfma(A/4, B/4)]        ← Compute cluster (uses MFMA units)
  barrier
  [mem1: gl B, ll A/4, ll B/4]  ← Memory cluster
  barrier
  [dot1: mfma(A/4, B/4)]        ← Compute cluster
  barrier
  ...
```

### What Happens at Barrier Release

```
At barrier between mem0 and dot0:
  Wave Group 0:  Enters dot0 (MFMA operations)
  Wave Group 1:  Enters mem0 (Memory operations, was 1 cluster behind)

Hardware utilization:
  - Wave Group 0 uses MFMA units
  - Wave Group 1 uses memory controllers, LDS ports
  - DIFFERENT HARDWARE → TRUE PARALLEL EXECUTION
```

### Why rocMLIR's M-Loop Split Doesn't Work

rocMLIR's M-loop split creates:
```
  [MFMA cluster 0]  ← Uses MFMA units
  barrier
  [MFMA cluster 1]  ← Uses MFMA units (SAME HARDWARE!)
  [Memory ops]
```

Both clusters use MFMA units → waves compete → serialization → waterfall.

---

## Code References

### Key Files in Triton

| File | Purpose |
|------|---------|
| `BlockPingpong.cpp` | Main ping-pong scheduling pass with 6 variants |
| `ScheduleLoops.cpp` | Creates coarse schedule for software pipelining |
| `LowerLoops.cpp` | Lowers schedule to LDS allocs/loads/stores |
| `Pipeline.cpp` | Expands loops according to schedule |
| `WarpPipeliner.cpp` | Stage-based warp pipelining (Gluon) |
| `ConvertWarpPipeline.cpp` | Lowers warp pipeline to LLVM IR |

### Key Functions

| Function | Purpose |
|----------|---------|
| `transformOnePPClusters` | One cluster for small tiles |
| `transformTwoPPClusters` | Two clusters with dot slicing |
| `transformFourPPClusters` | Four clusters for large tiles |
| `transformChainedDotSchedule` | Two chained dots (FlashAttention) |
| `addAsymmetricSyncToLoop` | Adds cond_barrier for phase shift |
| `sliceDot` | Slices dot along K-dimension |
| `genClusterBarrier` | Creates `ttg.barrier + sched.barrier` |

---

## Comparison with rocMLIR

| Aspect | Triton | rocMLIR (current) |
|--------|--------|-------------------|
| **Cluster Types** | Memory + Compute interleaved | Compute + Compute (M-loop split) |
| **Hardware Overlap** | Yes (different units) | No (same MFMA units) |
| **Dot Slicing** | Along K-dimension | Along M-dimension |
| **Phase Shift** | Creates compute/memory offset | Creates iteration offset |
| **Barrier Inside Loop** | Full barrier at each cluster | s_barrier (control-only) |
| **Loop Structure** | Multiple memory+compute pairs | MFMA pairs + memory at end |
| **numWarps Support** | 4 (inter-block) or 8 (intra-block) | 8 only |

### Why Triton's K-Slicing Works Better Than M-Slicing

**K-dimension slicing** (Triton):
```
K = total reduction dimension
slice0: K[0:K/2]
slice1: K[K/2:K]

dot(slice0) → partial result
barrier
memory ops (global load, LDS write for next iteration)
barrier
dot(slice1) → accumulate into same output
```

**M-dimension slicing** (rocMLIR):
```
M = output rows
slice0: M[0:M/2]
slice1: M[M/2:M]

dot(slice0) → output rows 0-M/2
barrier
dot(slice1) → output rows M/2-M (STILL MFMA, SAME HARDWARE!)
```

K-slicing keeps the same output accumulator but loads different input slices.
Memory operations can be interleaved because they're loading DIFFERENT data.
M-slicing produces different outputs but both halves still use MFMA units.

---

## Lessons for rocMLIR

### What Would Be Required for True Ping-Pong

1. **Restructure loop body to interleave compute and memory**:
   ```
   MFMA(i)              ← Compute cluster
   s_barrier            ← Cluster boundary
   DSR(i+1), DSW(i+2)   ← Memory cluster
   GL(i+3)
   ```

2. **Single barrier per iteration** (or carefully placed multiple barriers):
   - Each barrier should separate DIFFERENT types of work
   - Memory cluster after barrier, compute cluster before (or vice versa)

3. **K-dimension dot slicing** instead of M-dimension:
   - Split the reduction loop into slices
   - Interleave memory operations between compute slices

4. **Priority management** (from chained-dot variant):
   - Give memory cluster HIGHER priority when both contain VALU instructions
   - This ensures memory can make progress even when compute is busy

### Specific Changes Needed

**GridwiseGemmToBlockwise.cpp** or **RockPipeline.cpp**:
- Restructure the main loop to: `[MFMA] → [barrier] → [Memory ops]`
- Single barrier per iteration (or barrier between different cluster types)

**RockBlockPingpong.cpp**:
- Phase shift mechanism is correct (cond_barrier before/after loop)
- But needs proper compute/memory cluster interleaving to leverage it

---

## References

- [Triton Warp-Pipeline Documentation](https://github.com/jungpark-mlir/triton/blob/wp-document/third_party/amd/docs/warpPipeline.md)
- [Triton BlockPingpong.cpp](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp)
- [Triton ScheduleLoops.cpp](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUTransforms/ScheduleLoops.cpp)
- [Triton LowerLoops.cpp](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUTransforms/LowerLoops.cpp)
- AMD CDNA3 ISA Manual: `s_barrier`, `s_setprio`, `s_waitcnt` instruction semantics

---

## Appendix: Triton Code Excerpts

### Dot Slicing (`sliceDot`)

```cpp
// Split dot into 'numSlices' pieces along K-dimension
LogicalResult Pingponger::sliceDot(OpBuilder &builder, Location loc,
                                   tt::DotOp op, unsigned numSlices) {
  auto typeB = op.getB().getType();
  auto shapeB = typeB.getShape();
  int64_t sliceWidth = shapeB[0] / numSlices;  // K-dimension is shapeB[0]
  
  // Generate local load slices for A and B
  genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth);
  genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth);
  
  // Clone dots to consume all slices
  for (int i = 0; i < numSlices; i++) {
    auto newOp = builder.clone(*op, mapping);
    dotSliceOps.push_back(newOp);
  }
}
```

### Cluster Barrier Generation

```cpp
SmallVector<Operation *> Pingponger::genClusterBarrier(OpBuilder &builder,
                                                        Location loc) {
  // ttg.barrier with local memory sync
  auto barrierOp = triton::gpu::BarrierOp::create(
      builder, loc, triton::gpu::AddrSpace::Local);
  // Scheduler barrier to prevent instruction reordering
  auto schedBarrierOp = ROCDL::SchedBarrier::create(builder, loc, 0);
  return {barrierOp, schedBarrierOp};
}
```

### Phase Shift (`addAsymmetricSyncToLoop`)

```cpp
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
  // Full barrier before loop
  auto preBarrier = triton::gpu::BarrierOp::create(
      builder, loc, triton::gpu::AddrSpace::Local);
  preBarrier->moveBefore(forOp);
  
  // Calculate wave group predicates
  auto warpIDX = arith::DivSIOp::create(builder, loc, workIDX, constWarpSize);
  auto warpLow = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                       warpIDX, constZero);  // waves 0-3
  auto warpHigh = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                        warpIDX, constZero); // waves 4-7
  
  // Pre-loop: hold waves 4-7
  auto condBarrierHigh = tt::amdgpu::CondBarrierOp::create(builder, loc, warpHigh);
  
  // Post-loop: hold waves 0-3
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = tt::amdgpu::CondBarrierOp::create(builder, loc, warpLow);
}
```

### Priority Control in Chained-Dot

```cpp
// Memory cluster gets HIGHER priority
prependOp(ROCDL::SetPrioOp::create(builder, loc, highPriority), false);

// Compute cluster gets LOWER priority  
prependOp(ROCDL::SetPrioOp::create(builder, loc, lowPriority), false);

// Why: Both clusters contain v_xxx (VALU) instructions.
// If compute has higher priority, it monopolizes VALU issue slots.
// Memory cluster can't make progress on address updates.
// By giving memory higher priority, both can proceed in parallel.
```

---

## Appendix B: Variant Selection Logic

### Tile Size Thresholds

Triton calculates tile size as: `tileSize = M × N × K × elemWidth`

```cpp
const int64_t minTile    = 262144;      // 32×128×64×16bit
const int64_t smallTile  = 16777216;    // 128×128×64×16bit
const int64_t mediumTile = 33554432;    // smallTile × 2 (e.g., 256×128×64)
const int64_t largeTile  = 67108864;    // 256×256×64×16bit
```

### Variant Dispatch Decision Tree

```
getDotPingponged()
├── numStages <= 1? → return (no pipelining)
├── numOfDotLikeOps == 2?
│   └── numStages == 4? → transformChainedDotSchedule + asymmetricSync
├── numOfDotLikeOps != 1? → return
├── scaledDotOps.size() == 1 && numWarps == 8 && numStages == 2 && asyncCopy?
│   └── M×N == 256×256 && elemWidth == 8? → transformTwoClusterWithAsyncAndAll
├── dotOps.size() == 1 && useAsyncCopy && numWarps == 8 && numStages == 3?
│   └── M > 64 && N > 64? → transformTwoClusterWithLocalLoadAndAll
├── numWarps == 4?
│   └── minTile <= tileSize <= smallTile? → transformOnePPClusters (NO asymmetricSync)
├── numWarps == 8 && numStages == 2?
│   ├── tileSize == mediumTile? → transformTwoPPClusters + asymmetricSync
│   └── tileSize >= largeTile?  → transformFourPPClusters + asymmetricSync
└── else → return (no transformation)
```

### Why numWarps == 4 Doesn't Need Asymmetric Sync

With `numWarps=4`, each SIMD can hold waves from **different blocks** (different
workgroups). These waves don't share LDS and don't need to synchronize with each
other. The ping-pong effect comes from waves of different blocks naturally being
at different execution points.

With `numWarps=8`, all waves are from the **same block** and share LDS. They must
be explicitly phase-shifted using conditional barriers to ensure they reach
cluster boundaries at different times.

---

## Appendix C: Pipeline Flow in Triton

### Overall Compilation Flow

```
Triton Python Code
       ↓
   TritonGPU IR
       ↓
┌──────────────────────────────────────────────────────────┐
│ ScheduleLoops Pass (ScheduleLoops.cpp)                   │
│ - Creates coarse schedule for software pipelining        │
│ - Determines load distances and compute stages           │
│ - Serializes schedule into IR attributes                 │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│ LowerLoops Pass (LowerLoops.cpp)                         │
│ - Creates LDS allocations                                │
│ - Creates local_load/local_store or async_copy ops       │
│ - Updates schedule with new ops                          │
│ - Handles bypassLDS optimization                         │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│ Pipeline Pass (Pipeline.cpp)                             │
│ - Calls PipelineExpander to rewrite loop                 │
│ - Creates prologue, main loop, epilogue                  │
│ - Applies predication for boundary handling              │
└──────────────────────────────────────────────────────────┘
       ↓
┌──────────────────────────────────────────────────────────┐
│ BlockPingpong Pass (BlockPingpong.cpp)                   │
│ - Collects dot/load ops from loop                        │
│ - Selects variant based on tile size and numWarps        │
│ - Reorders ops into Memory/Dot clusters                  │
│ - Inserts barriers and priority instructions             │
│ - Adds asymmetric sync around loop                       │
└──────────────────────────────────────────────────────────┘
       ↓
   LLVM IR → AMD ISA
```

### Schedule Serialization Format

Triton uses IR attributes to pass schedule information between passes:

```mlir
scf.for %arg0 = %lb to %ub step %step
    iter_args(%buf = %init)
    {tt.loop.stage = [0, 0, 1, 1]}  // Stage assignment for each op
    {tt.loop.cluster = [0, 1, 0, 1]} // Cluster assignment
```

---

## Appendix D: Example IR Transformations

### Before BlockPingpong (after Pipeline pass)

```mlir
scf.for %i = %c0 to %c16 step %c1 iter_args(%C = %C_init) {
  // All operations in sequential order
  %A_tile = tt.load %ptrA : tensor<128x64xf16>
  %B_tile = tt.load %ptrB : tensor<64x128xf16>
  %A_lds = ttg.local_load %A_buf : tensor<128x64xf16>
  %B_lds = ttg.local_load %B_buf : tensor<64x128xf16>
  %D = tt.dot %A_lds, %B_lds, %C : tensor<128x128xf32>
  ttg.local_store %A_tile, %A_buf_next
  ttg.local_store %B_tile, %B_buf_next
  scf.yield %D
}
```

### After BlockPingpong (Two-Cluster variant)

```mlir
// Phase shift: hold waves 4-7
ttg.barrier addr=local
%warpHigh = arith.cmpi ne, %warpIdx, %c0
amdg.cond_barrier %warpHigh

scf.for %i = %c0 to %c16 step %c1 iter_args(%C = %C_init) {
  // Memory Cluster #0
  %A_slice0 = ttg.local_load %A_buf[0:K/2]
  %B_slice0 = ttg.local_load %B_buf[0:K/2]
  rocdl.sched.barrier 0
  %A_tile = tt.load %ptrA
  rocdl.sched.barrier 0
  %A_slice1 = ttg.local_load %A_buf[K/2:K]
  %B_slice1 = ttg.local_load %B_buf[K/2:K]
  rocdl.sched.barrier 0
  %B_tile = tt.load %ptrB
  rocdl.s.barrier              // Control-only barrier
  rocdl.sched.barrier 0
  
  // Dot Cluster #0
  rocdl.s.setprio 1
  %D0 = tt.dot %A_slice0, %B_slice0, %C
  rocdl.s.setprio 0
  ttg.barrier addr=local       // Full barrier with LDS sync
  rocdl.sched.barrier 0
  
  // Memory Cluster #1
  ttg.local_store %A_tile, %A_buf_next
  ttg.local_store %B_tile, %B_buf_next
  ttg.barrier addr=local
  rocdl.sched.barrier 0
  
  // Dot Cluster #1
  rocdl.s.setprio 1
  %D1 = tt.dot %A_slice1, %B_slice1, %D0
  rocdl.s.setprio 0
  
  ttg.barrier addr=local       // End of iteration barrier
  rocdl.sched.barrier 0
  
  scf.yield %D1
}

// Phase shift: hold waves 0-3 to reconverge
%warpLow = arith.cmpi eq, %warpIdx, %c0
amdg.cond_barrier %warpLow
```

### ISA-Level Structure (Conceptual)

```asm
; Pre-loop phase shift
s_barrier                    ; Sync all 8 waves
s_cmp_lg_u32 warpIdx, 0
s_cbranch_scc0 skip_barrier
s_barrier                    ; Only waves 4-7 hit this
skip_barrier:

loop_start:
; Memory Cluster #0
ds_read_b128 v[0:3], v_addr_A    ; local_load A slice 0
ds_read_b128 v[4:7], v_addr_B    ; local_load B slice 0
; (sched.barrier prevents reordering)
global_load_dwordx4 v[8:11], v[ptrA], off
; (sched.barrier)
ds_read_b128 v[12:15], v_addr_A  ; local_load A slice 1
ds_read_b128 v[16:19], v_addr_B  ; local_load B slice 1
; (sched.barrier)
global_load_dwordx4 v[20:23], v[ptrB], off
s_barrier                        ; Control-only sync
; (sched.barrier)

; Dot Cluster #0
s_setprio 1                      ; High priority for compute
v_mfma_f32_32x32x8f16 ...        ; dot slice 0
v_mfma_f32_32x32x8f16 ...
s_setprio 0                      ; Back to low priority
s_waitcnt lgkmcnt(0)             ; Wait for LDS
s_barrier                        ; Full sync
; (sched.barrier)

; Memory Cluster #1
ds_write_b128 v_next_A, v[8:11]  ; local_store A
ds_write_b128 v_next_B, v[20:23] ; local_store B
s_waitcnt lgkmcnt(0)
s_barrier

; Dot Cluster #1
s_setprio 1
v_mfma_f32_32x32x8f16 ...        ; dot slice 1
v_mfma_f32_32x32x8f16 ...
s_setprio 0

s_waitcnt lgkmcnt(0)
s_barrier
; (sched.barrier)

s_cbranch_scc1 loop_start

; Post-loop reconvergence
s_cmp_eq_u32 warpIdx, 0
s_cbranch_scc0 skip_barrier2
s_barrier                    ; Only waves 0-3 hit this
skip_barrier2:
```

---

## Appendix E: Hardware Execution Model

### CDNA3 Execution Units

```
┌─────────────────────────────────────────────────────────────┐
│                    Compute Unit (CU)                         │
├───────────────┬───────────────┬───────────────┬─────────────┤
│    SIMD 0     │    SIMD 1     │    SIMD 2     │   SIMD 3    │
│  Wave 0,4     │  Wave 1,5     │  Wave 2,6     │  Wave 3,7   │
├───────────────┴───────────────┴───────────────┴─────────────┤
│                      MFMA Units (shared)                     │
├─────────────────────────────────────────────────────────────┤
│                   LDS (64KB shared)                          │
├─────────────────────────────────────────────────────────────┤
│              Memory Controller (global access)               │
└─────────────────────────────────────────────────────────────┘
```

### Why Ping-Pong Works: Resource Independence

```
Wave Group 0 (waves 0-3) at Dot Cluster:
  - Uses: MFMA units, VALU for accumulation
  - Blocked on: MFMA latency (accumulator writes)

Wave Group 1 (waves 4-7) at Memory Cluster:
  - Uses: Memory controller, LDS ports, VALU for address math
  - Blocked on: Memory latency (global load, LDS access)

When barrier releases:
  - Group 0 starts MFMA → MFMA units busy
  - Group 1 issues global_load, ds_read → Memory ports busy
  - DIFFERENT RESOURCES → TRUE PARALLEL EXECUTION

After next barrier:
  - Groups swap roles
  - Group 0 does memory work
  - Group 1 does MFMA work
  - Still parallel because different resources
```

### Priority and Issue Slot Contention

The challenge is that both Memory and Compute clusters contain VALU instructions:
- Compute: `v_mfma_*`, `v_add_*` (accumulation)
- Memory: `v_add_*` (address calculation), `v_mov_*`

If both groups try to issue VALU simultaneously, they compete for slots.
`s_setprio` helps:

```
Group at Memory: s_setprio 1 (high)
Group at Compute: s_setprio 0 (low) by default

When VALU slot available:
  - Memory group wins (higher priority)
  - Issues address calculation VALU
  - Compute group waits briefly
  - But MFMA doesn't need VALU slot
  - So compute continues with MFMA

Result: Both groups make progress
```
