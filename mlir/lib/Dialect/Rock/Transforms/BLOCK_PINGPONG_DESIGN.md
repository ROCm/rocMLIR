# Block Ping-Pong Scheduling in rocMLIR

## Executive Summary

This document describes the Block Ping-Pong scheduling implementation in rocMLIR and provides
a comprehensive analysis of why **true ping-pong overlap is NOT achieved** despite implementing
the structural elements (triple buffering, M-loop slicing, phase shift, s_barrier).

### Key Findings

| Finding | Status |
|---------|--------|
| Triple buffering (3 LDS buffers) | ✅ Implemented, prevents buffer conflicts |
| M-loop slicing (2 compute clusters) | ✅ Implemented, creates cluster boundaries |
| Phase shift via cond_barrier | ✅ Implemented, staggers wave groups |
| s_barrier inside loop | ✅ Implemented, provides control-only sync |
| **Compute-first schedule** | ✅ **Implemented** - MMA first, then barrier, then memory |
| E2E correctness | ✅ Verified (`[1 1 1]`) |

### Compute-First Schedule (IMPLEMENTED)

The compute-first schedule restructures the loop body to **interleave compute and memory operations**:

```
MFMA(i)              ← Compute cluster (uses MFMA units)
s_barrier            ← SINGLE barrier
DSR(i+1), DSW(i+2)   ← Memory cluster (uses memory controllers)
GL(i+3)              ← Global load (prefetch)
```

**True ping-pong is now possible because:**
- One wave group executing COMPUTE (MFMA units)
- Other wave group executing MEMORY (memory controllers)
- These use DIFFERENT hardware → true parallel execution

**Implementation:** See `createComputeFirstSchedule()` in `RockPipeline.cpp`.

---

## Table of Contents

1. [What is Ping-Pong Scheduling?](#what-is-ping-pong-scheduling)
2. [What We Implemented](#what-we-implemented)
3. [Why True Ping-Pong Cannot Be Achieved](#why-true-ping-pong-cannot-be-achieved)
4. [The Waterfall Pattern Explained](#the-waterfall-pattern-explained)
5. [Detailed Technical Analysis](#detailed-technical-analysis)
6. [Comparison with Triton](#comparison-with-triton)
7. [What Would Be Needed](#what-would-be-needed)
8. [Implementation Attempts Summary](#implementation-attempts-summary)
9. [Conclusion and Recommendations](#conclusion-and-recommendations)

---

## What is Ping-Pong Scheduling?

Ping-Pong scheduling is a technique to overlap compute and memory operations by having two groups of waves execute in a **phase-shifted** manner:

```
Traditional (Waterfall) Execution:
  Time ->
  Wave Group 0:  [GL0][DSW0][Barrier][DSR0][MFMA0][GL1][DSW1][Barrier][DSR1][MFMA1]...
  Wave Group 1:  [GL0][DSW0][Barrier][DSR0][MFMA0][GL1][DSW1][Barrier][DSR1][MFMA1]...
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 All waves do the same thing at the same time = no overlap

Ping-Pong (Phase-Shifted) Execution:
  Time ->
  Wave Group 0:  [GL0][DSW0][Barrier]...[DSR0][MFMA0][GL1][DSW1][Barrier]...[DSR1][MFMA1]...
  Wave Group 1:       [GL0][DSW0][Barrier]...[DSR0][MFMA0][GL1][DSW1][Barrier]...[DSR1][MFMA1]
                 ^^^^^                  ^^^^^
                 While Group 0 computes, Group 1 does memory
                 = TRUE OVERLAP of compute and memory
```

**Legend:**
- `GL` = Global Load (prefetch from global memory)
- `DSW` = DS Write (write to LDS)
- `DSR` = DS Read (read from LDS)
- `MFMA` = Matrix Fused Multiply-Add (compute)
- `Barrier` = Synchronization point

The key insight is that with **phase shift**, while one group of waves is doing compute (MFMA), the other group is doing memory operations (GL/DSW/DSR), achieving true overlap.

---

## What We Implemented

### Current Implementation (Two Modes)

The `--use-block-pingpong` flag enables one of two modes depending on buffering:

#### Mode 1: Full Ping-Pong (with triple buffering)

When `rock.triple_buffered` attribute is present:

1. **Triple Buffering** - 3 LDS buffers per allocation
   - Prevents read-write conflicts when waves are at different iterations
   - `buffer[N%3] != buffer[(N-1)%3]`

2. **M-Loop Slicing** - Split the M-dimension loop into 2 halves
   - Creates 2 compute clusters per iteration
   - `s_barrier` inserted between the halves

3. **Phase Shift** - Conditional barriers before/after the main loop
   - Pre-loop: `cond_barrier(warpHigh)` - waves 4-7 wait
   - Post-loop: `cond_barrier(warpLow)` - waves 0-3 wait to reconverge

4. **Scheduling Hints** - Cluster boundaries and priority hints
   - `sched_barrier` around existing barriers
   - `setprio(1)` before MFMA, `setprio(0)` after

#### Mode 2: Scheduling Hints Only (with double buffering)

When only `rock.double_buffered` is present:

1. **No phase shift** (would cause buffer conflicts with only 2 buffers)
2. **Scheduling hints only** - `sched_barrier` and `setprio` instructions

### Generated IR Structure (Full Ping-Pong Mode)

```mlir
func.func @rock_gemm(...) attributes {rock.triple_buffered, rock.use_block_pingpong} {
  // Prologue: 2 iterations prefetch
  rock.lds_barrier
  rock.lds_barrier
  
  // Phase shift entry
  rock.cond_barrier %warpHigh  // Waves 4-7 wait here initially
  
  scf.for %i = 0 to N {
    rock.s_barrier             // Cluster boundary (no LDS wait)
    
    // Memory operations: DSW, GL
    rock.threadwise_write_all  // Write to LDS
    rock.threadwise_read_into  // Global load (prefetch)
    
    rocdl.s.setprio 1          // Prioritize compute
    affine.for %m = 0 to M/2:  // MFMA Cluster 0
      rock.threadwise_gemm_accel
    rocdl.s.setprio 0
    
    rock.s_barrier             // Cluster boundary
    
    rocdl.s.setprio 1
    affine.for %m = M/2 to M:  // MFMA Cluster 1
      rock.threadwise_gemm_accel
    rocdl.s.setprio 0
    
    // LDS read operations
    rock.threadwise_read_into  // Read from LDS
  }
  
  // Phase shift exit
  rock.cond_barrier %warpLow   // Waves 0-3 wait to reconverge
  
  // Epilogue
  rock.lds_barrier
  // Final MFMA operations
}
```

### Code Locations

- **RockBlockPingpong.cpp** - Main pass for ping-pong transformations
  - `applyDotSlicing()` - Splits M-loop into 2 halves
  - `applyTritonStylePhaseShift()` - Inserts cond_barriers
  - `insertClusterBoundariesAtExistingBarriers()` - Adds sched_barrier
  - `insertSetPrioAroundMFMA()` - Adds setprio hints

- **RockPipeline.cpp** - Pipeline pass modifications
  - Upgrades LDS allocations from 2 to 3 buffers when `use_block_pingpong` is set
  - Adds `rock.triple_buffered` attribute to function

- **RockToGPU.cpp** - Lowering patterns
  - `CondBarrierRewritePattern` - Lowers `rock.cond_barrier` to CFG with `amdgpu.s_barrier`

---

## The Waterfall Pattern Explained

### What the Trace Shows

When profiling the kernel with `rocprofv3 --hip-trace`, the trace shows:

```
Wave 0: [----][mem][bar][====compute====][bar][----][mem][bar][====compute====]
Wave 1:      [----][mem][bar][====compute====][bar][----][mem][bar][====compute====]
Wave 2:           [----][mem][bar][====compute====][bar][----][mem][bar][====]
Wave 3:                [----][mem][bar][====compute====][bar][----][mem][bar]
Wave 4:                     [----][mem][bar][====compute====][bar][----][mem]
Wave 5:                          [----][mem][bar][====compute====][bar][----]
Wave 6:                               [----][mem][bar][====compute====][bar]
Wave 7:                                    [----][mem][bar][====compute====]
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        STAGGERED but NOT OVERLAPPING - this is a WATERFALL pattern
```

**Legend:**
- `[mem]` = Memory operations (orange in trace)
- `[compute]` = MFMA operations (green in trace)
- `[bar]` = Barrier (yellow in trace)
- `[----]` = Idle/waiting

### Why This Happens

1. **Phase shift creates stagger**: The `cond_barrier` at loop entry causes waves to enter
   the loop at different times. Wave 0-3 enter first, then waves 4-7.

2. **But both clusters are COMPUTE**: Inside the loop, we have:
   ```
   s_barrier
   [MFMA cluster 0]  ← All waves do compute
   s_barrier
   [MFMA cluster 1]  ← All waves do compute
   [Memory ops]
   ```

3. **MFMA units are shared**: When the barrier releases, ALL 8 waves proceed to execute
   MFMA instructions. Since MFMA units are a **shared resource**, waves compete and
   must take turns → **serialization** → waterfall.

### What True Ping-Pong Should Look Like

```
Wave 0-3: [====compute====][bar][----memory----][bar][====compute====][bar]
Wave 4-7:      [----memory----][bar][====compute====][bar][----memory----]
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          OVERLAPPING - compute and memory use DIFFERENT hardware
```

When one group does compute (uses MFMA units) and the other does memory (uses memory
controllers), they can execute **truly in parallel** because they use different hardware.

---

## Why True Ping-Pong Cannot Be Achieved

### The Core Problem: AMD's s_barrier Semantics

AMD's `s_barrier` instruction is a **single counting barrier per workgroup**:

```
s_barrier behavior:
  - Each workgroup has ONE barrier counter
  - When a wave executes s_barrier, it increments the counter
  - When counter == wavesPerBlock, ALL waves are released
  - The counter then resets to 0
```

**Critical Insight:** There is no way to have "4 waves sync together separately from the other 4 waves" using `s_barrier`. ALL waves in the workgroup are always counted together.

### The Phase Shift Deadlock

Consider attempting phase shift with conditional barriers:

```mlir
// Attempted phase shift implementation:
// Pre-loop: Create phase shift
rock.lds_barrier              // All 8 waves sync (counter hits 8, all pass)
rock.cond_barrier(%warpHigh)  // Only waves 4-7 execute s_barrier

scf.for ... {
  rock.lds_barrier            // Loop barrier
  // compute & memory ops
}

rock.cond_barrier(%warpLow)   // Post-loop reconvergence
```

**Why this deadlocks:**

```
Step 1: Pre-loop lds_barrier
  - All 8 waves hit s_barrier → counter = 8 → all pass ✓

Step 2: cond_barrier(%warpHigh) - only waves 4-7 execute
  - Waves 4-7 hit s_barrier → counter = 4
  - Waves 0-3 skip (don't execute s_barrier)
  - Counter stuck at 4, never reaches 8
  - Waves 4-7 BLOCKED FOREVER ✗

DEADLOCK!
```

### Why Triton's Approach Works (And Ours Doesn't)

Triton's ping-pong relies on a **very specific loop structure**:

```
Triton's loop structure (simplified):
  barrier()              // ONE barrier per iteration
  for i in range(N):
    compute(i)
    barrier()            // THE ONLY barrier in the loop
    prefetch(i+1)
```

With **exactly ONE barrier per loop iteration**, Triton's phase shift works because:

```
Iteration 0:              Iteration 1:
  Waves 4-7: barrier(A)     Waves 4-7: barrier(B)
  Waves 0-3: barrier(A)     Waves 0-3: barrier(B) [at same time as 4-7's barrier(B)]
             ↑                         ↑
             These "pair up"           These "pair up"
             (4 + 4 = 8)               (4 + 4 = 8)
```

### rocMLIR's Problem: Multiple Barriers Per Iteration

rocMLIR's GEMM pipeline has **MULTIPLE barriers per loop iteration**:

```mlir
// rocMLIR's actual loop structure:
scf.for %i = ... {
  rock.lds_barrier          // Barrier 1: before DS read
  rock.threadwise_read_into // DS read (from LDS)
  rock.threadwise_gemm_accel // MFMA compute
  rock.lds_barrier          // Barrier 2: before DS write  
  rock.threadwise_write_all  // DS write (to LDS)
  rock.async_wait           // Wait for global loads
  rock.lds_barrier          // Barrier 3: after async wait
}
```

With **3+ barriers per iteration**, phase shift breaks:

```
Attempted phase shift with multiple barriers:

Wave Group 0 (iteration N):    Wave Group 1 (iteration N+1):
  barrier(1) ────────────────────── barrier(1)  = 8 waves ✓
  DS_READ                           DS_READ
  MFMA                              MFMA
  barrier(2) ────────────────────── ??? 
      ↑
      Wave Group 0 at barrier(2) of iteration N
      Wave Group 1 at barrier(1) of iteration N+1
      These are DIFFERENT s_barrier instructions!
      Counter never reaches 8 → DEADLOCK ✗
```

---

## Detailed Technical Analysis

### Experiment: Conditional Barriers

We implemented and tested conditional barriers:

```cpp
// rock.cond_barrier operation
// Lowers to:
//   if (condition) {
//     s_barrier
//   }
```

**Result:** Produces `s_cbranch_execz` + `s_barrier` in ISA, but causes deadlock because the skipping waves don't contribute to the barrier count.

### Experiment: Triple Buffering

We tried 3 LDS buffers to avoid read-write conflicts:

```
With 3 buffers:
  - Iteration N reads from buffer (N % 3)
  - Iteration N writes to buffer ((N+2) % 3)
  - All three buffers distinct → no conflict
```

**Result:** Triple buffering allocates correctly, but doesn't help because the fundamental barrier problem remains. The issue is not buffer conflicts - it's the synchronization mechanism.

### ISA Analysis

Generated ISA shows the problem clearly:

```asm
; Without phase shift (current implementation):
s_barrier                    ; All 8 waves sync
s_waitcnt vmcnt(0)
ds_read_b128 ...
v_mfma_f32_16x16x16f16 ...
s_barrier
ds_write_b128 ...

; With attempted phase shift:
s_cbranch_execz skip_barrier ; Half the waves skip
s_barrier                    ; Only 4 waves hit this
skip_barrier:
; ... DEADLOCK - barrier never completes
```

---

## Comparison with Triton

| Aspect | Triton | rocMLIR (current) |
|--------|--------|-------------------|
| Cluster Types | Memory + Compute interleaved | Compute + Compute (M-loop split) |
| Overlap Achieved | Yes (different hardware) | No (same hardware contention) |
| Phase Shift | Works (compute vs memory overlap) | Creates stagger but not overlap |
| Barrier Structure | Multiple per iteration | Multiple per iteration |
| Buffering | Double (2 buffers) | Triple (3 buffers) |

### The Critical Difference: Cluster Types

**Triton interleaves MEMORY and COMPUTE clusters:**
```
Triton's loop:
  [Memory Cluster]  ← Uses memory controllers
  barrier
  [Compute Cluster] ← Uses MFMA units
  barrier
  [Memory Cluster]  ← Uses memory controllers
  barrier
  [Compute Cluster] ← Uses MFMA units
```

**rocMLIR has COMPUTE-only clusters:**
```
rocMLIR's loop (with M-loop split):
  [Compute Cluster 0] ← Uses MFMA units
  barrier
  [Compute Cluster 1] ← Uses MFMA units (SAME HARDWARE!)
  [Memory ops]
```

When both clusters use the same hardware (MFMA units), waves must take turns → waterfall.

### Triton's Key Design Choice

Triton explicitly structures its loops to have **interleaved compute and memory clusters**:

```python
# Triton's approach (pseudocode):
for i in range(num_iterations):
    # Load tile (from LDS, already populated)
    a = tl.load(A_ptr)
    b = tl.load(B_ptr)
    
    # Compute
    acc = tl.dot(a, b, acc)
    
    # Single barrier - THE KEY TO PING-PONG
    tl.barrier()
    
    # Prefetch next tile (populate LDS for next iteration)
    next_a = tl.load(A_global_ptr)
    next_b = tl.load(B_global_ptr)
    tl.store(A_ptr, next_a)
    tl.store(B_ptr, next_b)
```

rocMLIR's pipeline has more synchronization points due to its different design:
- Separate barriers for DS read safety
- Separate barriers for DS write safety
- Separate barriers for async load completion

---

## How Triton Avoids Read-Write Conflicts

### Key Insight: Triton Uses Double Buffering (numStages=2), NOT Triple Buffering

Triton's ping-pong implementation (`BlockPingpong.cpp`) has these characteristics:

1. **Double buffering (2 LDS buffers)** - `numStages=2` means 2 buffers
2. **FULL barriers inside the loop** - `triton::gpu::BarrierOp` syncs ALL 8 waves
3. **Phase shift ONLY via pre/post-loop cond_barriers** - NOT by replacing in-loop barriers

```cpp
// Triton's addAsymmetricSyncToLoop (lines 914-945):
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
  // Pre-loop: full barrier + cond_barrier(warpHigh)
  auto preBarrier = triton::gpu::BarrierOp::create(...);  // FULL barrier
  preBarrier->moveBefore(forOp);
  auto condBarrierHigh = tt::amdgpu::CondBarrierOp::create(builder, loc, warpHigh);
  
  // Post-loop: cond_barrier(warpLow)  
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = tt::amdgpu::CondBarrierOp::create(builder, loc, warpLow);
}
```

### Why Triton's Approach Works with 2 Buffers

The key is that **inside the loop, ALL waves sync together at cluster barriers**:

```
Triton's loop structure:
  [Memory Cluster 0]
    local_load A(slice 0), B(slice 0)
    global_load A, B
  s_barrier           ← FULL BARRIER (all 8 waves sync)
  sched_barrier
  [Dot Cluster 0]
    dot(slice 0)
  gpu::BarrierOp      ← FULL BARRIER (all 8 waves sync)  
  sched_barrier
  [Memory Cluster 1]
    local_store A, B
  gpu::BarrierOp      ← FULL BARRIER (all 8 waves sync)
  sched_barrier
  [Dot Cluster 1]
    dot(slice 1)
  gpu::BarrierOp      ← END OF ITERATION
```

**Inside each iteration, waves execute in lockstep** (all 8 waves at same point).

**The phase shift happens at LOOP BOUNDARIES only:**
- Pre-loop: `cond_barrier(warpHigh)` makes waves 4-7 enter loop first
- Post-loop: `cond_barrier(warpLow)` reconverges

### No Read-Write Conflict Because:

1. **Within each iteration**: All 8 waves execute the SAME barrier instructions
2. **At iteration boundary**: The full barrier at end of iteration ensures all waves complete iteration N before any wave starts iteration N+1
3. **Phase shift offset**: One group of waves is ~1 iteration ahead, but they sync at every cluster barrier

```
Timeline:
  Wave 4-7:  [iter 0: mem0→dot0→mem1→dot1→barrier] → [iter 1: mem0→...
  Wave 0-3:       [wait]  → [iter 0: mem0→dot0→mem1→dot1→barrier] → [iter 1: ...
                  ↑
                  cond_barrier(warpHigh) delays these waves

At each cluster barrier, both groups sync:
  - Wave 4-7 at dot0 of iter 1, Wave 0-3 at dot0 of iter 0
  - Both hit gpu::BarrierOp → 4+4=8 waves → barrier releases
  - Then Wave 4-7 proceeds to mem1 of iter 1, Wave 0-3 to mem1 of iter 0
```

**Result:** Both groups access DIFFERENT buffers because they're at different iterations, but they use the SAME buffer index calculation (`iter % 2`). Since one group is always 1 iteration behind, their buffer accesses don't conflict!

---

## What Would Be Needed

To achieve true ping-pong scheduling in rocMLIR, a **single-barrier-per-iteration** schedule is required.

### Option 1: Compute-First Schedule (Most Promising)

A "compute-first" schedule that puts MFMA before the barrier:

```
Prologue:
  GL(0)                    // Global Load iteration 0
  ---
  DSW(0) → LDS_Ping        // DS Write to Ping buffer
  GL(1)                    // Global Load iteration 1
  ---
  LDSBarrier               // Sync: DSW(0) complete, safe to read
  DSR(0) ← LDS_Ping        // DS Read from Ping buffer
  DSW(1) → LDS_Pong        // DS Write to Pong buffer
  GL(2)                    // Global Load iteration 2
  ---

Main Loop (scf.for i = 0 to N-3):
  MFMA(i)                  // Compute FIRST (data already in registers from DSR)
  LDSBarrier               // THE ONLY BARRIER - enables ping-pong!
  DSR(i+1) ← LDS[(i+1)%2]  // DS Read from alternating buffer
  DSW(i+2) → LDS[i%2]      // DS Write to OTHER buffer
  GL(i+3)                  // Global Load (prefetch)

Epilogue:
  ... (drain remaining MFMA operations)
```

**Why this works for phase shift:**

With exactly ONE barrier per iteration, barriers "pair up" across iterations:

```
Wave Group 0 (iteration N):        Wave Group 1 (iteration N+1):
  MFMA(N)                            MFMA(N+1)
  LDSBarrier ─────────────────────── LDSBarrier   ← These "pair up"!
  DSR(N+1)                           DSR(N+2)       4 + 4 = 8 waves
  DSW(N+2)                           DSW(N+3)       Counter hits 8
  GL(N+3)                            GL(N+4)        All pass! ✓
```

**WAIT: Triton uses 2 buffers and it works! Why?**

The analysis above assumes waves at different iterations access buffers **simultaneously
without synchronization**. But Triton's approach is different:

```
Triton's approach (cluster barriers sync ALL waves at each cluster):
  
  At cluster barrier, BOTH groups sync:
    Wave 4-7 at iter N+1, Wave 0-3 at iter N
    Both hit full barrier → 4+4=8 waves → releases
    
  After barrier:
    Wave 4-7 continues iter N+1
    Wave 0-3 continues iter N
    
  Buffer accesses are TEMPORALLY separated by barriers!
```

**Key insight:** With full barriers inside the loop (syncing all 8 waves), the
buffer accesses are **sequenced** even though waves are at different iterations.
The barrier ensures one group finishes its buffer access before the other starts.

**This is fundamentally different from replacing barriers with conditional barriers!**

### Two Possible Approaches for rocMLIR

**Approach A: Triton-style (Full barriers inside loop)**
- Use FULL barriers inside loop (all 8 waves sync at each cluster)
- Phase shift via cond_barrier ONLY before/after loop
- Works with 2 buffers (double buffering)
- Requires restructuring loop into Memory→Dot→Memory→Dot clusters

**Approach B: True Asymmetric (Conditional barriers inside loop)**  
- Replace full barriers with conditional barriers inside loop
- Each wave group syncs independently
- Requires 3 buffers to avoid conflicts
- Requires single-barrier-per-iteration structure

We tried Approach B with triple buffering but failed because we didn't restructure
the loop first. The multiple barriers per iteration caused the deadlock.

**Implementation requirements for Approach A (Triton-style):**

1. Restructure loop into distinct clusters: [Memory] → [Dot] → [Memory] → [Dot]
2. Keep FULL barriers between clusters (all 8 waves sync)
3. Add cond_barrier ONLY before/after loop for phase shift
4. Use existing 2-buffer (double buffering) infrastructure

**Implementation requirements for Approach B (True Asymmetric):**

1. Restructure loop to single-barrier-per-iteration
2. Replace that single barrier with two cond_barriers
3. Use 3-buffer infrastructure
4. Add cond_barrier before/after loop for phase shift

### Option 2: Named Barriers (Hardware Support)

Use named/indexed barriers that allow partial synchronization:

```asm
; Hypothetical (not available on current AMD GPUs):
s_barrier_named 0   ; Only waves that call this sync together
s_barrier_named 1   ; Different barrier, different sync group
```

**Status:** Not available on current AMD GPU architectures (gfx9xx).

### Option 3: Wave-Level Synchronization

Use wave-level primitives instead of workgroup barriers:

```cpp
// Hypothetical:
__builtin_amdgcn_wave_barrier();  // Sync within single wave only
```

**Limitation:** Doesn't help for cross-wave synchronization needed in ping-pong.

---

## Summary: Why Current Implementation Doesn't Achieve Ping-Pong

### Two Independent Problems

**Problem 1: Multiple Barriers Per Iteration (Current State)**

The current rocMLIR pipeline has 2-3 barriers per loop iteration:
```
scf.for {
  LDSBarrier    ← Barrier 1 (before DSR)
  DSR
  MFMA
  LDSBarrier    ← Barrier 2 (before DSW) 
  DSW
  async_wait
  LDSBarrier    ← Barrier 3 (after async)
  GL
}
```

With multiple barriers, phase shift causes deadlock (as explained above).

**Problem 2: Buffer Conflicts with 2 Buffers + Phase Shift**

Even if we fix Problem 1 (single barrier), using only 2 buffers with 1-iteration
phase shift causes read-write conflicts:

```
Iteration:     N              N+1 (phase shifted)
DSR reads:     buffer[(N+1)%2] buffer[(N+2)%2]
DSW writes:    buffer[(N+2)%2] buffer[(N+3)%2]

When N=0:
  Group 0: DSR from buf[1], DSW to buf[0]
  Group 1: DSR from buf[0], DSW to buf[1]
           ↑ CONFLICT: Group 1 reads buf[0] while Group 0 writes buf[0]!
```

**Both problems must be solved TOGETHER for true ping-pong:**

| Problem | Solution | Status |
|---------|----------|--------|
| Multiple barriers | Compute-first schedule (single barrier) | **NOT implemented** |
| 2-buffer conflict | Triple buffering (3 buffers) | Was implemented, removed |

### What We Actually Tried (And Why It Failed)

#### Attempt 1: Triple Buffering + Phase Shift

We implemented triple buffering + phase shift on TOP of the existing multi-barrier pipeline:

```
What we tried:
  scf.for {
    LDSBarrier           ← Still multiple barriers!
    DSR
    MFMA
    LDSBarrier           ← Problem: these don't pair up
    DSW
    LDSBarrier
    GL
  }
  + Triple buffering (3 LDS allocations)
  + Phase shift (cond_barrier before/after loop)
```

**Result:** E2E tests failed with `[0 0 0]` (complete accuracy regression).

**Root cause:** Triple buffering solves the buffer conflict problem, but does NOT solve
the multiple-barrier problem. The phase shift caused waves at different iterations to
hit DIFFERENT `s_barrier` instructions, leading to deadlock or incorrect data.

```
The failure scenario:
  Wave Group 0 at iteration N:     Wave Group 1 at iteration N+1:
    LDSBarrier (1st) ─────────────── LDSBarrier (1st)   ← Pairs? Maybe...
    DSR                               DSR
    MFMA                              MFMA
    LDSBarrier (2nd) ─────────────── ???
         ↑
    Group 0 at 2nd barrier of iter N
    Group 1 at 1st barrier of iter N+1
    These are DIFFERENT s_barrier instructions!
    Counter never reaches 8 → DEADLOCK or wrong sync
```

**Conclusion:** Triple buffering alone is NOT sufficient. We MUST restructure the
pipeline to have exactly ONE barrier per iteration FIRST, then add triple buffering.

#### Attempt 2: Triton-Style Option A (cond_barrier at loop boundaries)

Based on Triton's implementation, we tried inserting cond_barriers ONLY at loop
boundaries while keeping full barriers inside the loop:

```cpp
// What we tried (Option A - Triton-style):
applyTritonStylePhaseShift() {
  // Before scf.for:
  rock::LDSBarrierOp::create();           // Full barrier to sync all
  rock::CondBarrierOp::create(warpHigh);  // Waves 4-7 wait here

  // scf.for { ... existing barriers ... }

  // After scf.for:
  rock::CondBarrierOp::create(warpLow);   // Waves 0-3 wait here
}
```

**Result:** E2E tests failed with `[0 0 1]` (partial accuracy regression).

**Root cause:** rocMLIR's pipelining structure is fundamentally different from Triton's:

```
rocMLIR structure:                    Triton structure:
┌─────────────────────────────┐       ┌─────────────────────────────┐
│  [PROLOGUE - peeled stages] │       │  scf.for (includes all) {   │
│    GL(0), DSW(0), DSR(0)... │       │    // Stages for iter N     │
│    LDSBarrier               │       │    // Stages for iter N+1   │
│    ...                      │       │    // etc.                  │
├─────────────────────────────┤       │  }                          │
│  scf.for (steady state) {   │       └─────────────────────────────┘
│    // Only iteration body   │
│  }                          │
├─────────────────────────────┤
│  [EPILOGUE - drain stages]  │
│    LDSBarrier               │
│    ...                      │
└─────────────────────────────┘
```

When we add cond_barrier BEFORE scf.for:
1. Waves 0-3 skip cond_barrier and execute the ENTIRE prologue
2. Waves 4-7 wait at cond_barrier
3. When waves 0-3 reach the first barrier INSIDE scf.for, waves 4-7 release
4. BUT: waves 4-7 now start from the PROLOGUE, not from the loop!
5. Result: Complete desynchronization - waves are at totally different stages

**Why Triton works:** Triton's software pipelining keeps ALL stages inside the loop.
The prologue iterations are handled by adjusting loop bounds, not by peeling.
When cond_barrier releases waves 4-7, they enter the SAME scf.for at iteration 0,
while waves 0-3 are at iteration 1 of the SAME loop.

**Why rocMLIR fails:** With explicit prologue/epilogue, waves desync completely.

---

## Attempt 3: Control-Only Barriers Inside Loop (s_barrier)

After understanding the issue with LDS wait (`s_waitcnt lgkmcnt(0)`), we tried a more
targeted approach:

1. Added `rock.s_barrier` op that lowers to pure `s_barrier` (no LDS wait)
2. Replaced `rock.lds_barrier` inside the loop with `rock.s_barrier`
3. Kept `rock.lds_barrier` at prologue/epilogue boundaries

**Result:** E2E tests still failed with `[0 0 1]`.

**Root Cause Analysis:** The problem is more fundamental than barrier types.

### The Real Issue: rocMLIR Has ONE Barrier Per Iteration

Looking at Triton's ping-pong implementation in detail, the key difference is:

```
Triton loop structure (2 barriers per iteration):
┌──────────────────────────────────────────────────────┐
│  scf.for {                                           │
│    ComputeCluster1: dot[0] ← s_barrier               │
│    MemoryCluster1:  memory ops                       │
│    ComputeCluster2: dot[1] ← s_barrier               │
│    MemoryCluster2:  memory ops                       │
│  }                                                   │
└──────────────────────────────────────────────────────┘

rocMLIR loop structure (1 barrier per iteration):
┌──────────────────────────────────────────────────────┐
│  scf.for {                                           │
│    s_barrier                                         │
│    DSW, GL, DSR, MFMA (all operations)              │
│  }                                                   │
└──────────────────────────────────────────────────────┘
```

**Why 2 barriers per iteration matter for ping-pong:**

With the pre-loop cond_barrier creating a phase shift:
- Group 0 is at barrier B1 (first barrier in iteration N)
- Group 1 is at barrier B2 (second barrier in iteration N-1)
- Total 8 waves → barrier releases
- Now Group 0 proceeds to memory, Group 1 proceeds to compute
- **THIS IS TRUE PING-PONG**: different groups do different work

With only 1 barrier per iteration:
- Group 0 is at barrier in iteration N  
- Group 1 is at cond_barrier (BEFORE the loop, not inside!)
- After release, Group 1 **enters iteration 0**, same as where Group 0 started
- **NO PHASE SHIFT**: both groups end up doing the same thing

### Why Phase Shift Doesn't Create True Iteration Offset

The cond_barrier before the loop only **DELAYS** when waves enter the loop.
Once inside, ALL waves use the SAME loop induction variable `%arg3`.

```
After cond_barrier releases:
  Group 0: already at s_barrier inside iteration 0 (%arg3 = 0)
  Group 1: just entering loop, starts at iteration 0 (%arg3 = 0)
  
  RESULT: Both groups are at iteration 0!
```

The loop counter is **shared across all threads** - there's no per-wave-group counter.

### What Triton Does Differently

Triton's 2-barriers-per-iteration structure means:
1. When Group 0 hits barrier B1 in iter N, Group 1 is at barrier B2 in iter N-1
2. Both groups are **inside the loop** at the same iteration
3. The phase shift is **within iteration**, not across iterations

This is achieved by Triton's ping-pong pass that:
1. Splits each dot operation into 2 slices
2. Inserts barriers between the slices
3. Reorders operations to create 2 compute clusters per iteration

---

## Attempt 4: Loop Unrolling (Failed)

### Hypothesis

If we unroll the loop by factor 2, each unrolled iteration will have 2 barriers
(one from each original iteration), creating the 2-barriers-per-iteration structure.

### Implementation

```cpp
// In RockBlockPingpong.cpp
static scf::ForOp unrollLoopForPingPong(scf::ForOp forOp) {
  auto result = loopUnrollByFactor(forOp, /*unrollFactor=*/2);
  if (failed(result)) return nullptr;
  return forOp;
}
```

### Result: E2E Failed with `[0 0 1]`

**Root Cause:** Loop unrolling doesn't create true ping-pong.

Even with 2 barriers per unrolled iteration:
- Both wave groups still execute the **same unrolled iteration**
- The phase shift only delays entry into the loop
- Once inside, both groups use the same iteration variable `%arg3`
- `extract_multibuffer(%arg3)` returns the same buffer for both groups
- Result: Both groups access same data = corruption

**Diagram:**
```
Without unrolling:              With unrolling:
┌───────────────────┐           ┌───────────────────────────────┐
│ iter 0:           │           │ unrolled iter 0:              │
│   Barrier         │           │   Barrier (orig iter 0)       │
│   ops(buffer 0)   │           │   ops(buffer 0)               │
├───────────────────┤           │   Barrier (orig iter 1)       │
│ iter 1:           │           │   ops(buffer 1)               │
│   Barrier         │           ├───────────────────────────────┤
│   ops(buffer 1)   │           │ unrolled iter 2:              │
└───────────────────┘           │   ...                         │
                                └───────────────────────────────┘

Problem: Both groups 0 and 1 execute "unrolled iter 0" together!
They both access buffer 0, then both access buffer 1.
No overlap = no ping-pong.
```

---

## Triton's Actual Approach: Dot Slicing

Studying Triton's `BlockPingpong.cpp` reveals a fundamentally different approach.

### Triton's Key Functions

1. **`sliceDot()`** - Slices the dot operation into multiple pieces:
   ```cpp
   // Split A matrix: [M, K] → slices of [M, K/numSlices]
   // Split B matrix: [K, N] → slices of [K/numSlices, N]
   // Split dot: one dot per slice
   ```

2. **`transformTwoPPClusters()`** - Creates 2 compute clusters per iteration:
   ```cpp
   // mem0: global load, local load slice0
   // dot0: dot(slice0)
   // barrier
   // mem1: local store, local load slice1
   // dot1: dot(slice1)
   // barrier
   ```

### Why Dot Slicing Works

**The key insight:** Each slice operates on DIFFERENT data within the SAME iteration.

```
Triton's structure (one iteration):
┌─────────────────────────────────┐
│ ComputeCluster0:                │
│   local_load A[0:K/2]           │
│   local_load B[0:K/2]           │
│   dot(A_slice0, B_slice0)       │
│   Barrier ←────────────────────── Group 0 here
├─────────────────────────────────┤
│ ComputeCluster1:                │
│   global_load (prefetch)        │
│   local_load A[K/2:K]           │
│   local_load B[K/2:K]           │
│   dot(A_slice1, B_slice1)       │
│   Barrier ←────────────────────── Group 1 here
└─────────────────────────────────┘

With phase shift:
- Group 0 at Barrier0 (iter N), Group 1 at Barrier1 (iter N-1)
- Group 0 computes slice0, Group 1 computes slice1
- DIFFERENT SLICES = no conflict!
```

### rocMLIR vs Triton Pipeline Comparison

| Aspect | rocMLIR | Triton |
|--------|---------|--------|
| Dot granularity | Full M×N×K tile | Sliced to M×N×(K/2) |
| Barriers per iteration | 1 | 2+ |
| Compute clusters | 1 | 2+ |
| Phase shift creates | Same work delayed | Different work |

---

## Attempt 5: M-Loop Slicing (Failed)

### Hypothesis

Split the mRepeats affine.for loop into 2 halves with s_barrier between,
creating 2 compute clusters per iteration like Triton's dot slicing.

### Implementation

```cpp
// In RockBlockPingpong.cpp
static bool applyDotSlicing(scf::ForOp forOp) {
  // Find outermost affine.for loops containing threadwise_gemm_accel
  // Split: affine.for m = 0 to 4 → two loops [0,2) and [2,4)
  // Insert s_barrier between the halves
}
```

### Result: E2E Failed with `[0 0 1]`

**Root Cause:** M-slicing is NOT equivalent to Triton's K-slicing.

1. **Triton slices along K (reduction dimension):**
   - Each slice computes partial sums on the SAME output elements
   - K/2 slice0 + K/2 slice1 = full K reduction
   - Both slices access DIFFERENT parts of LDS (K dimension split)

2. **rocMLIR M-slicing splits output dimension:**
   - Each slice computes DIFFERENT output elements
   - Both slices access the SAME LDS data (full K)
   - With phase shift, waves at different main iterations conflict

3. **The fundamental issue is MAIN LOOP iteration offset:**
   ```
   With phase shift (double buffering):
   - Group 0 at K-iter N: reads LDS[N%2], writes LDS[(N+1)%2]
   - Group 1 at K-iter N-1: reads LDS[(N-1)%2], writes LDS[N%2]
   
   If N=2 (even):
   - Group 0 reads LDS[0], writes LDS[1]
   - Group 1 reads LDS[1], writes LDS[0]
   → Group 0 reads what Group 1 wrote LAST iteration = stale data
   → Group 1 writes what Group 0 is reading = race condition
   ```

4. **Why Triton's approach avoids this:**
   - In Triton, phase shift is WITHIN iteration, not across iterations
   - Both groups are at the SAME main iteration, accessing SAME LDS
   - They just compute DIFFERENT K-slices of the same data
   - No iteration offset = no buffer conflict

### Conclusion

The M-slicing approach cannot work because:
- Phase shift creates iteration OFFSET (N vs N-1)
- Double buffering can't handle waves at different iterations
- Would need TRIPLE buffering to allow 1-iteration offset

---

## Implementation Attempts Summary

### Complete History of All Attempts

| # | Attempt | Result | Root Cause of Failure |
|---|---------|--------|----------------------|
| 1 | Scheduling hints only | ✅ Works (no overlap) | Baseline, no overlap attempted |
| 2 | Phase shift (cond_barrier) | ❌ Deadlock | Only 4 waves hit s_barrier, counter stuck |
| 3 | Loop unrolling × 2 | ❌ Wrong results | Both groups process same logical iteration |
| 4 | s_barrier inside loop | ❌ Deadlock | Multiple barriers, waves at different positions |
| 5 | Triple buffering alone | ❌ No overlap | Buffer OK, but no phase shift mechanism |
| 6 | M-loop slicing + double buffer | ❌ Wrong results | Buffer conflict from iteration offset |
| 7 | M-loop slicing + triple buffer + phase shift | ✅ Correct, ❌ No overlap | **Both clusters are compute, not compute+memory** |

### Attempt 7: Current Implementation (Detailed Analysis)

**What was implemented:**
- Triple buffering (3 LDS buffers)
- M-loop split into 2 halves with s_barrier between
- Phase shift via cond_barrier before/after loop
- Scheduling hints (sched_barrier, setprio)

**Why it achieves correctness:**
1. Triple buffering prevents buffer conflicts when waves are at different iterations
2. Phase shift via cond_barrier doesn't cause deadlock (full barrier before it ensures 8 waves sync)
3. s_barrier inside loop pairs up correctly (4 waves at iter N + 4 waves at iter N-1 = 8)

**Why it does NOT achieve true overlap:**
```
Generated structure:
  rock.s_barrier
  affine.for %m = 0 to 1:        ← MFMA cluster 0 (COMPUTE)
    threadwise_gemm_accel
  rock.s_barrier
  affine.for %m = 1 to 2:        ← MFMA cluster 1 (COMPUTE)
    threadwise_gemm_accel
  // Memory operations at end
```

Both clusters are **COMPUTE-only** (MFMA operations). When the barrier releases:
- Wave group 0 → proceeds to MFMA cluster 0
- Wave group 1 → proceeds to MFMA cluster 1
- **Both groups compete for MFMA units** → serialization → waterfall

**The trace shows:**
- Staggered start (phase shift working)
- But sequential compute blocks (no overlap)
- This is the **waterfall pattern**, not ping-pong

### What Would Actually Work

**Compute-first schedule with interleaved clusters:**
```
scf.for %i = 0 to N {
  [MFMA operations]     ← Compute cluster (uses MFMA units)
  s_barrier             ← SINGLE barrier
  [DSR, DSW, GL ops]    ← Memory cluster (uses memory controllers)
}
```

With phase shift:
- Wave group 0 at compute cluster (iter N) → uses MFMA units
- Wave group 1 at memory cluster (iter N-1) → uses memory controllers
- **Different hardware** → true parallel execution → ping-pong overlap

---

## Conclusion

### What We Achieved

| Aspect | Status |
|--------|--------|
| Triple buffering (3 LDS buffers) | ✅ Implemented |
| M-loop slicing (2 clusters) | ✅ Implemented |
| Phase shift (cond_barrier) | ✅ Implemented |
| Scheduling hints (sched_barrier, setprio) | ✅ Implemented |
| E2E correctness | ✅ Verified (`[1 1 1]`) |
| **True compute/memory overlap** | ❌ **NOT achieved** |

### Root Cause Analysis

**Why the waterfall pattern persists:**

1. M-loop slicing creates **2 MFMA clusters** (both compute)
2. Triton's dot slicing creates **MFMA + Memory clusters** (different hardware)
3. When waves compete for same hardware (MFMA units), they serialize
4. When waves use different hardware (MFMA vs memory), they overlap

### Code Changes Made

**RockPipeline.cpp:**
```cpp
// Upgrade LDS buffers to triple buffering for pingpong mode
bool useTripleBuffering =
    func->hasAttr("rock.use_block_pingpong") &&
    func->hasAttr("rock.double_buffered");

if (useTripleBuffering && factor >= 2 &&
    getAddressSpace(alloc) == AddressSpace::Workgroup) {
  effectiveFactor = std::max(factor, 3);
}

if (useTripleBuffering) {
  func->setAttr("rock.triple_buffered", rewriter.getUnitAttr());
}
```

**RockBlockPingpong.cpp:**
```cpp
// Full ping-pong mode with triple buffering
if (isTripleBuffered) {
  applyDotSlicing(forOp);           // Split M-loop
  applyTritonStylePhaseShift(...);  // Add cond_barriers
  insertClusterBoundariesAtExistingBarriers(forOp);
  insertSetPrioAroundMFMA(forOp);
}
```

### Debug Output

```
[rock-block-pingpong]: triple-buffered mode, applying full ping-pong with dot slicing + phase shift
[rock-block-pingpong]: slicing mLoop: [0, 2) step 1 at 1
[rock-block-pingpong]: applied dot slicing to 1 M loops
[rock-block-pingpong]: replaced 1 lds_barrier with s_barrier inside loop
[rock-block-pingpong]: applied Triton-style phase shift
[rock-block-pingpong]: applied full ping-pong mode
```

### The Path Forward (COMPUTE-FIRST IMPLEMENTED)

**UPDATE:** The compute-first schedule has been implemented in `RockPipeline.cpp`.

The implementation now has:

1. ✅ **Compute-first schedule** - MMA executes FIRST in the main loop
2. ✅ **Single barrier per iteration** - After MMA, before memory operations
3. ✅ **Double buffering** - 2 LDS buffers (scheduleVersion=2)
4. ✅ **Inverted stage offsets** - MMA=3, LDSRead=2, LDSWrite=1, GlobalRead=0
5. ✅ **E2E correctness** - All tests pass `[1 1 1]`

**Main loop structure:**
```
scf.for i = 0 to N-3:
  MMA(i)                  // Compute FIRST
  LDSBarrier              // Single barrier
  DSR(i+1) ← LDS[(i+1)%2] // LDS Read from alternating buffer
  DSW(i+2) → LDS[i%2]     // LDS Write to OTHER buffer
  GL(i+3)                 // Global Load (prefetch)
```

**Prologue structure:**
```
Iteration 0: GlobalRead(0)
Iteration 1: LDSWrite(0), GlobalRead(1)
Iteration 2: Barrier, LDSRead(0), LDSWrite(1), GlobalRead(2)
```

This enables ping-pong scheduling because:
- MMA uses MFMA units (compute hardware)
- DSR/DSW/GL use memory controllers (memory hardware)
- With phase shift, one wave group does compute while the other does memory

### Why Waterfall Still Occurs

**Root Cause:** Both clusters are COMPUTE clusters (MFMA operations). When the barrier
releases, both wave groups proceed to execute MFMA instructions simultaneously. Since
they're competing for the same hardware resources (MFMA units), they serialize.

**What we have:**
```
scf.for {
  [MFMA cluster 0]  ← Both groups do compute
  s_barrier
  [MFMA cluster 1]  ← Both groups do compute
  [Memory ops]      ← Memory happens after compute
}
```

**What Triton achieves:**
```
scf.for {
  [Memory cluster 0]  ← Group 0 does memory, Group 1 does compute
  barrier
  [Dot cluster 0]     ← Group 0 does compute, Group 1 does memory
  barrier
  [Memory cluster 1]  ← Interleaved operations
  barrier
  [Dot cluster 1]     ← Interleaved operations
}
```

The key difference: Triton interleaves MEMORY and COMPUTE clusters, so when barriers
release, wave groups execute DIFFERENT types of work using DIFFERENT hardware:
- Compute waves use MFMA units
- Memory waves use memory controllers/LDS

This allows true parallel execution and overlap.

### What Would Be Required for True Ping-Pong

1. **Restructure the loop to interleave compute and memory operations**:
   ```
   Compute-first schedule:
     MFMA(i)            ← Compute cluster (uses MFMA units)
     s_barrier          ← Cluster boundary
     DSR(i+1)           ← Memory cluster (uses memory controllers)
     DSW(i+2)
     GL(i+3)
   ```

2. **Single barrier per iteration** - The current 2+ barriers cause all waves to sync
   at multiple points, reducing overlap opportunities.

3. **Phase shift must create compute/memory offset**, not just iteration offset:
   - Group 0 at iteration N's compute cluster
   - Group 1 at iteration N-1's memory cluster
   - These use different hardware → true parallel execution

**Why Previous Attempts Failed:**

| Attempt | Why Failed |
|---------|------------|
| Loop unrolling | Both groups execute same unrolled iteration |
| Phase shift only | Groups at same iteration access same buffers |
| Triple buffering | Still one barrier per iteration, groups sync together |
| s_barrier in loop | Iteration structure unchanged, same data accessed |

### Why Current rocMLIR Doesn't Work

```
Current rocMLIR pipeline:           Required for ping-pong:
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  Barrier                    │     │  MFMA(i)      ← compute     │
│  DSR(i)                     │     │  Barrier      ← SINGLE!     │
│  MFMA(i)                    │     │  DSR(i+1)     ← memory      │
│  Barrier                    │     │  DSW(i+2)                   │
│  DSW(i+1)                   │     │  GL(i+3)                    │
│  async_wait                 │     └─────────────────────────────┘
│  Barrier                    │
│  GL(i+2)                    │
└─────────────────────────────┘

Multiple barriers → deadlock      Single barrier → works!
```

### Recommendations

**Status Update:** The current implementation achieves correctness and provides scheduling
hints, but does NOT achieve true compute/memory overlap (ping-pong). The trace shows a
waterfall pattern because both clusters are COMPUTE operations.

1. **Short term (current):** Scheduling hints only
   - `sched_barrier` and `setprio` provide cluster boundaries
   - Verified correct with E2E tests (`[1 1 1]`)
   - **Does NOT achieve true ping-pong** - waterfall pattern observed

2. **THE REAL SOLUTION:** Interleave compute and memory clusters

   **Why current approach fails:**
   ```
   Current (both clusters = compute):    Required (interleaved):
   [MFMA cluster 0] ─┐                   [MFMA cluster]  ← Group 0
   s_barrier        ─┼→ All compete      s_barrier
   [MFMA cluster 1] ─┘  for MFMA units   [Memory cluster] ← Group 1 (different HW)
   [Memory ops]                          
   ```
   
   When wave groups execute DIFFERENT hardware (MFMA vs memory controllers),
   they can run truly in parallel. When both execute MFMA, they serialize.

   **Implementation approach:**
   a. **Compute-first schedule** - restructure loop body:
      ```
      MFMA(i)            ← Compute cluster (uses MFMA units)
      s_barrier          ← SINGLE barrier
      DSR(i+1)           ← Memory cluster (uses memory controllers)
      DSW(i+2)
      GL(i+3)
      ```
   b. Phase shift creates compute/memory offset:
      - Group 0: compute of iteration N
      - Group 1: memory of iteration N-1
      - Different hardware → true overlap

3. **Why splitting M-loop doesn't help:**
   - M-loop contains MFMA operations only
   - Splitting it creates 2 MFMA clusters
   - Both clusters use same hardware → still serialize
   
4. **What Triton does differently:**
   - Triton's dot slicing is along K-dimension (reduction)
   - Creates multiple dot+memory clusters per iteration
   - Each cluster alternates between dot (MFMA) and memory (LDS/global)
   - Phase shift creates dot/memory interleaving between wave groups

5. **Long term:** Named barriers in future AMD architectures
   - Would allow "barrier per cluster" instead of "barrier per workgroup"
   - Would enable true independent cluster execution

---

## Appendix A: Test Commands

### E2E Verification

```bash
# Run E2E verification with block ping-pong (should show [1 1 1])
cd /home/umayadav/repo/rocMLIR/build
./bin/rocmlir-gen -operation gemm -t f16 --arch gfx950 -m 512 -k 512 -n 4096 \
  --use-block-pingpong --perf_config=v4:128,64,4,32,32,16,4,1,2,2,2,0,1,1 \
  -pv_with_gpu | ./bin/rocmlir-driver -c | $HOME/repo/rocMLIR/mlir/utils/widgets/rocm-run
```

### IR Inspection

```bash
# Check generated IR for triple buffering and phase shift
./bin/rocmlir-gen -operation gemm -t f16 --arch gfx950 -m 512 -k 512 -n 4096 \
  --use-block-pingpong --perf_config=v4:128,64,4,32,32,16,4,1,2,2,2,0,1,1 \
  | ./bin/rocmlir-driver --kernel-pipeline=applicability \
  | grep -E "rock\.(triple_buffered|cond_barrier|s_barrier|lds_barrier)"

# Check loop structure
./bin/rocmlir-gen ... | ./bin/rocmlir-driver --kernel-pipeline=applicability \
  | grep -E "scf.for|affine.for|threadwise_gemm|setprio" | head -30

# Full IR dump
./bin/rocmlir-gen ... | ./bin/rocmlir-driver --kernel-pipeline=applicability > /tmp/ir.mlir
```

### Profiling (if rocprofv3 works)

```bash
# Profile kernel execution
rocprofv3 --hip-trace -- ./my_gemm_binary

# Output will be in rocprof-* directory, open in Perfetto UI
```

### Debug Output

```bash
# Enable debug output for RockBlockPingpong pass
./bin/rocmlir-gen ... | ./bin/rocmlir-driver --kernel-pipeline=applicability \
  --mlir-print-debuginfo 2>&1 | grep "rock-block-pingpong"
```

---

## Appendix B: Key Code Locations

| File | Purpose |
|------|---------|
| `RockBlockPingpong.cpp` | Main ping-pong pass (M-loop slicing, phase shift, hints) |
| `RockPipeline.cpp` | Pipeline scheduling, triple buffering upgrade |
| `RockToGPU.cpp` | Lowering `rock.cond_barrier` to CFG + `amdgpu.s_barrier` |
| `GridwiseGemmToBlockwise.cpp` | Creates main scf.for K-loop |
| `BlockwiseGemmToThreadwise.cpp` | Creates affine.for M/N/K loops with MFMA |

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **Ping-Pong** | Scheduling where wave groups alternate between compute and memory |
| **Waterfall** | Sequential execution where waves take turns (no overlap) |
| **Phase Shift** | Staggering wave groups so they're at different points in the loop |
| **s_barrier** | AMD GPU workgroup barrier (counting, all waves must participate) |
| **cond_barrier** | Conditional barrier (only threads with pred=true execute s_barrier) |
| **MFMA** | Matrix Fused Multiply-Add instruction (compute) |
| **DSR/DSW** | Data Share Read/Write (LDS operations) |
| **GL** | Global Load (memory operation) |
| **M-loop** | Loop over M-dimension repeats in threadwise GEMM |
| **K-loop** | Main reduction loop in GEMM |

---

## References

- AMD CDNA3 ISA Manual: `s_barrier` instruction semantics
- Triton Block Ping-Pong: `third_party/amd/lib/TritonAMDGPUTransforms/BlockPingpong.cpp`
- rocMLIR Pipeline: `mlir/lib/Dialect/Rock/Transforms/RockPipeline.cpp`
- rocMLIR Block Ping-Pong: `mlir/lib/Dialect/Rock/Transforms/RockBlockPingpong.cpp`

---

## Document History

| Date | Change |
|------|--------|
| Initial | Documented scheduling hints implementation |
| Update 1 | Added phase shift attempts and failure analysis |
| Update 2 | Added loop unrolling attempt (failed) |
| Update 3 | Added M-loop slicing attempt with double buffering (failed) |
| Update 4 | Added triple buffering + M-loop slicing (correctness achieved) |
| **Final** | **Documented waterfall pattern analysis - both clusters are compute, not compute+memory** |
