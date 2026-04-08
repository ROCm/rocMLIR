// RUN: rocmlir-driver --kernel-pipeline=gpu %s --mlir-print-ir-after=rock-add-sched-group-barriers \
// RUN:   -o /dev/null 2>&1 | FileCheck %s

// The pass inserts rocdl.iglp.opt for eligible double-buffered GEMM loops.
// It skips direct-to-LDS, non-double-buffered, and nested-loop (attention) kernels.

// Pipeline output order: gemm_dtlds_sb, gemm_dtlds_db, gemm_f32_db, gemm_db, attention, gemm_wmma_db.

// CHECK-LABEL: func @gemm_dtlds_sb(
// CHECK-NOT: rocdl.iglp.opt

// CHECK-LABEL: func @gemm_dtlds_db(
// CHECK-NOT: rocdl.iglp.opt

// CHECK-LABEL: func @gemm_f32_db(
// CHECK: rocdl.iglp.opt

// CHECK-LABEL: func @gemm_db(
// CHECK: rocdl.iglp.opt

// CHECK-LABEL: func @attention(
// CHECK-NOT: rocdl.iglp.opt

// CHECK-LABEL: func @gemm_wmma_db(
// CHECK: rocdl.iglp.opt

func.func @gemm_dtlds_sb(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<3>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

func.func @gemm_dtlds_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<4>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

func.func @gemm_f32_db(%arg0: memref<262144xf32>, %arg1: memref<262144xf32>, %arg2: memref<262144xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<2>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 512 + d2)> by [<Unmerge{512, 512} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 512, 512] -> [262144]> : memref<262144xf32> to memref<1x512x512xf32>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x512x512xf32> = memref<1x512x512xf32> * memref<1x512x512xf32>
  return
}

func.func @gemm_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942", schedule_version = #rock.schedule_version<2>, num_cu = 304 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}

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

func.func @gemm_wmma_db(%arg0: memref<294912xf16>, %arg1: memref<589824xf16>, %arg2: memref<294912xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", schedule_version = #rock.schedule_version<2>, num_cu = 96 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{768, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 768, 768] -> [589824]> : memref<589824xf16> to memref<1x768x768xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 768 + d2)> by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]> : memref<294912xf16> to memref<1x384x768xf16>
  rock.gemm %2 = %0 * %1 features = dot|atomic_add|wmma storeMethod = set : memref<1x384x768xf16> = memref<1x384x768xf16> * memref<1x768x768xf16>
  return
}
