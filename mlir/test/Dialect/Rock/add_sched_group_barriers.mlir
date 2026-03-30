// RUN: rocmlir-driver --kernel-pipeline=gpu %s --mlir-print-ir-after=rock-add-sched-group-barriers \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// All GEMM configs use f16, g=1, m=384, n=768, k=768 (except WMMA which targets gfx1100).
// Attention config uses f16, seq_q=4096, seq_k=4096, head_dim=64 on gfx942.
//
// The pass requires: double-buffered loop, no direct-to-LDS, no conditional code,
// <=25 MFMA/WMMA ops per iteration, <=1 rock.lds_barrier, no existing barriers.
// amdgpu.async_load_to_lds is also treated as direct-to-LDS (gfx1250+).

// Negative test: Direct-to-LDS single-buffered (scheduleVersion=3) -- skipped.
// CHECK-LABEL: func @gemm_dtlds_sb
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
func.func @gemm_dtlds_sb(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<3>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

// Negative test: Non-double-buffered (default scheduleVersion=1) -- skipped.
// CHECK-LABEL: func @gemm_default
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
func.func @gemm_default(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

// Negative test: Direct-to-LDS double-buffered (scheduleVersion=4) -- skipped.
// CHECK-LABEL: func @gemm_dtlds_db
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
func.func @gemm_dtlds_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<4>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

// Positive test: Double-buffered f32 GEMM on gfx942 (scheduleVersion=2).
// f32 uses 16x16x4 MFMA instructions, giving 8 MFMAs per iteration (<=25).
// CHECK-LABEL: func @gemm_f32_db
// CHECK: amdgpu.sched_barrier allow = <none>
// CHECK: rocdl.sched.group.barrier 8, 1, 0
// CHECK: rocdl.sched.group.barrier
// CHECK: amdgpu.sched_barrier allow = <none>
func.func @gemm_f32_db(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>, %arg2: memref<262144xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<2>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x512x512xf32> = memref<1x512x512xf32> * memref<1x512x512xf32>
  return
}

// Positive test: Double-buffered f16 GEMM on gfx942 (scheduleVersion=2).
// CHECK-LABEL: func @gemm_db
// CHECK: amdgpu.sched_barrier allow = <none>
// CHECK: rocdl.sched.group.barrier 8, 1, 0
// CHECK: rocdl.sched.group.barrier 512, 1, 0
// CHECK: rocdl.sched.group.barrier 32, 2, 0
// CHECK: rocdl.sched.group.barrier 256, 1, 0
// CHECK: amdgpu.sched_barrier allow = <none>
func.func @gemm_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<2>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

// Negative test: Attention kernel on gfx942 with schedule_version=2. Even
// though it's double-buffered, it has multiple rock.lds_barrier ops inside
// the loop (softmax, rescaling phases), so the pass skips it.
// CHECK-LABEL: func @attention
// CHECK-NOT: amdgpu.sched_barrier
// CHECK-NOT: rocdl.sched.group.barrier
func.func @attention(%arg0: memref<262144xf16>, %arg1: memref<262144xf16>, %arg2: memref<262144xf16>, %arg3: memref<262144xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<2>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{4096, 64} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 64] -> [262144]> : memref<262144xf16> to memref<1x4096x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{64, 4096} ["head_qk", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 4096] -> [262144]> : memref<262144xf16> to memref<1x64x4096xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{4096, 64} ["seq_k", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 64] -> [262144]> : memref<262144xf16> to memref<1x4096x64xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{4096, 64} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 64] -> [262144]> : memref<262144xf16> to memref<1x4096x64xf16>
  rock.attention{
   qk = %0 * %1 : memref<1x4096x64xf16>, memref<1x64x4096xf16>
   qk = elementwise {
  ^bb0(%arg4: memref<1x4096x4096xf16>, %arg5: memref<1x4096x4096xf16>):
    memref.copy %arg4, %arg5 : memref<1x4096x4096xf16> to memref<1x4096x4096xf16>
    rock.yield
  }
   %3 = softmax(qk) * %2 : memref<1x4096x64xf16> -> memref<1x4096x64xf16>
  } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

// Positive test: Double-buffered f16 GEMM on gfx1100 using WMMA (scheduleVersion=2).
// CHECK-LABEL: func @gemm_wmma_db
// CHECK: amdgpu.sched_barrier allow = <none>
// CHECK: rocdl.sched.group.barrier
// CHECK: amdgpu.sched_barrier allow = <none>
func.func @gemm_wmma_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", schedule_version = #rock.schedule_version<2>, num_cu = 96 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = dot|atomic_add|wmma storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}
