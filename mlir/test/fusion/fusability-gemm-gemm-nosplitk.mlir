// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v2:32,128,32,32,32,16,8,4,1,2,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=attn:v2:32,128,32,32,32,16,8,1,1,2,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_gemm_gemm(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>, %arg2: memref<4096xf32>, %arg3: memref<4096xf32>, %arg4: memref<4096xf32>) attributes {rock.enable_splitk_for_tuning, rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %0 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 32 + d2) * 32 + d3)> by [<Unmerge{4, 32, 32} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 32, 32] -> [4096]> : memref<4096xf32> to memref<1x4x32x32xf32>
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)> by [<PassThrough ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 3, 2]>] bounds = [1, 4, 32, 32] -> [1, 4, 32, 32]> : memref<1x4x32x32xf32> to memref<1x4x32x32xf32>
    %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)> by [<Unmerge{4, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 32, 32] -> [4096]> : memref<4096xf32> to memref<4x32x32xf32>
    %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [4, 32, 32] -> [1, 4, 32, 32]> : memref<1x4x32x32xf32> to memref<4x32x32xf32>
    %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)> by [<Unmerge{4, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [4, 32, 32] -> [4096]> : memref<4096xf32> to memref<4x32x32xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x32x32xf32>
    %5 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [4, 32, 32] -> [4, 32, 32]> : memref<4x32x32xf32> to memref<4x32x32xf32>
    rock.gemm_elementwise_gemm{
      ab = %2 * tr %5 : memref<4x32x32xf32>, memref<4x32x32xf32>
      ab = elementwise otherIns(%arg1 : memref<4096xf32>) {
    ^bb0(%arg5: memref<4x32x32xf32>, %arg6: memref<4096xf32>, %arg7: memref<4x32x32xf32>):
      %7 = rock.transform %arg5 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{4} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 32, 32] -> [4, 32, 32]> : memref<4x32x32xf32> to memref<1x4x32x32xf32>
      %8 = rock.transform %arg6 by <affine_map<(d0, d1, d2, d3) -> ((d1 * 32 + d2) * 32 + d3)> by [<Unmerge{4, 32, 32} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 32, 32] -> [4096]> : memref<4096xf32> to memref<1x4x32x32xf32>
      %9 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [4, 32, 32] -> [1, 4, 32, 32]> : memref<1x4x32x32xf32> to memref<4x32x32xf32>
      %10 = rock.transform %8 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [4, 32, 32] -> [1, 4, 32, 32]> : memref<1x4x32x32xf32> to memref<4x32x32xf32>
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4x32x32xf32>
      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%9, %10 : memref<4x32x32xf32>, memref<4x32x32xf32>) outs(%alloc_0 : memref<4x32x32xf32>) {
      ^bb0(%in: f32, %in_1: f32, %out: f32):
        %13 = arith.mulf %in, %in_1 : f32
        linalg.yield %13 : f32
      }
      %11 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)> by [<Unmerge{4} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 32, 32] -> [4, 32, 32]> : memref<4x32x32xf32> to memref<1x4x32x32xf32>
      %12 = rock.transform %11 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [4, 32, 32] -> [1, 4, 32, 32]> : memref<1x4x32x32xf32> to memref<4x32x32xf32>
      memref.copy %12, %arg7 : memref<4x32x32xf32> to memref<4x32x32xf32>
      rock.yield
    }
      %alloc = ab * %4 : memref<4x32x32xf32> -> memref<4x32x32xf32>
    } {firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>}
    %6 = rock.transform %alloc by <affine_map<(d0) -> (d0 floordiv 1024, (d0 mod 1024) floordiv 32, d0 mod 32)> by [<Merge{4, 32, 32} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [4096] -> [4, 32, 32]> : memref<4x32x32xf32> to memref<4096xf32>
    memref.copy %6, %arg4 : memref<4096xf32> to memref<4096xf32>
    return
  }
}
