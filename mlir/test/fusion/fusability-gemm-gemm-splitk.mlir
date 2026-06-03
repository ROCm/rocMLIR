// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v2:32,128,32,32,32,16,8,4,1,2,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v2:32,128,32,32,32,16,8,1,1,2,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @rock_gemm_gemm(%arg0: memref<1474560xf16>, %arg1: memref<1474560xf16>, %arg2: memref<1474560xf16>, %arg3: memref<4096xf16>) attributes {rock.enable_splitk_for_tuning, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
    %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : memref<1474560xf16> to memref<1x360x4096xf16>
    %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
    %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 1 + d2)> by [<Unmerge{4096, 1} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 1] -> [4096]> : memref<4096xf16> to memref<1x4096x1xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4096x360xf16>
    rock.gemm_elementwise_gemm{
     ab = %0 * %1 : memref<1x4096x360xf16>, memref<1x360x4096xf16>
     ab = elementwise {
    ^bb0(%arg4: memref<1x4096x4096xf16>, %arg5: memref<1x4096x4096xf16>):
      memref.copy %arg4, %arg5 : memref<1x4096x4096xf16> to memref<1x4096x4096xf16>
      rock.yield
    }
     %alloc = ab * %2 : memref<1x4096x360xf16> -> memref<1x4096x360xf16>
    } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>}
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x4096x1xf16>
    rock.reduce  sum %alloc into %alloc_1 {axis = 2 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<1x4096x360xf16> into memref<1x4096x1xf16>
    memref.copy %alloc_1, %3 : memref<1x4096x1xf16> to memref<1x4096x1xf16>
    return
  }
}
