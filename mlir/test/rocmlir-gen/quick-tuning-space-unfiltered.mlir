// The quick tuning space is no longer pruned by `couldBePerformant`; it now
// keeps every config that is `paramsProbablyValid`. For a GEMM with a fused 
// reduction the old filter dropped any tile whose M/N-per-block did
// not evenly divide the (16-aligned) 64x32 gemm, so those tiles used to be
// missing from the quick space and must now appear.

// RUN: sed -e 's/##TOKEN_ARCH##/amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-/g' -e 's/##TOKEN_FEATURES##/mfma|dot|atomic_add|atomic_add_f16/g' %s \
// RUN:   | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --check-prefix=MFMA
// RUN: sed -e 's/##TOKEN_ARCH##/amdgcn-amd-amdhsa:gfx1100/g' -e 's/##TOKEN_FEATURES##/wmma|dot|atomic_add|atomic_add_f16/g' %s \
// RUN:   | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --check-prefix=WMMA
// RUN: sed -e 's/##TOKEN_ARCH##/amdgcn-amd-amdhsa:gfx906/g' -e 's/##TOKEN_FEATURES##/dot|atomic_add/g' %s \
// RUN:   | rocmlir-gen --emit-tuning-space=quick - | FileCheck %s --check-prefix=NONACCEL

// MFMA branch: tiles kept before and after.
// MFMA-DAG: v4:32,32,8,16,16,16,4,1,2,2,0,0,1,1
// MFMA-DAG: v4:64,32,4,16,32,16,8,1,2,2,0,0,1,1
// MFMA branch: tiles the fused-reduction filter used to drop, now kept.
// MFMA-DAG: v4:32,64,8,16,32,16,4,1,1,2,0,0,1,1
// MFMA-DAG: v4:64,64,4,32,32,32,8,1,1,2,0,0,1,1
// MFMA-DAG: v4:128,64,4,128,16,16,4,1,1,2,0,0,1,1

// WMMA branch: tiles kept before and after.
// WMMA-DAG: v3:64,32,32,4,2,2,1,1,2
// WMMA-DAG: v3:64,64,32,16,2,2,1,1,2
// WMMA branch: tiles the fused-reduction filter used to drop, now kept.
// WMMA-DAG: v3:64,64,64,4,2,2,1,1,2
// WMMA-DAG: v3:128,64,64,4,2,2,1,1,2
// WMMA-DAG: v3:128,128,128,16,2,2,1,1,2

// Non-accelerated branch: tiles kept before and after.
// NONACCEL-DAG: v3:64,32,32,8,2,2,1,1,2
// NONACCEL-DAG: v3:64,64,32,16,2,2,1,1,2
// Non-accelerated branch: tiles the fused-reduction filter used to drop, now kept.
// NONACCEL-DAG: v3:64,64,64,8,2,4,1,1,2
// NONACCEL-DAG: v3:64,128,64,16,2,2,1,1,2
// NONACCEL-DAG: v3:128,128,128,4,4,2,1,1,2

module {
  func.func @mlir_dot_reduce(%arg0: memref<1x64x64xf32>, %arg1: memref<1x64x32xf32>, %arg2: memref<1x64x1xf32>) attributes {rock.kernel, mhal.arch = "##TOKEN_ARCH##", features = #rock<GemmFeatures ##TOKEN_FEATURES##>} {
    %alloc = memref.alloc() alignment = 64 : memref<1x64x32xf32>
    rock.gemm %alloc = %arg0 * %arg1 storeMethod = set : memref<1x64x32xf32> = memref<1x64x64xf32> * memref<1x64x32xf32>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 32] -> [1, 64, 32]> : memref<1x64x32xf32> to memref<64x32xf32>
    %alloc_1 = memref.alloc() alignment = 64 : memref<64x1xf32>
    rock.reduce sum %0 into %alloc_1 {axis = 1 : index, blockSize = 256 : i32, gridSize = 64 : i32} : memref<64x32xf32> into memref<64x1xf32>
    %2 = rock.transform %alloc_1 by <affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)> by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 64, 1] -> [64, 1]> : memref<64x1xf32> to memref<1x64x1xf32>
    memref.copy %2, %arg2 : memref<1x64x1xf32> to memref<1x64x1xf32>
    return
  }
}
