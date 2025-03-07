// RUN: rocmlir-opt -rock-gemm-linalg-splitk-normalization %s | FileCheck %s

#wg = #gpu.address_space<workgroup>
#priv = #gpu.address_space<private>

// CHECK-LABEL: func.func @matmul_splitk_addf_constant
func.func @matmul_splitk_addf_constant(%arg0: memref<32768xf32>, %arg1: memref<16384xf32>, %arg2: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 1.000000e+01 : f32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 2.500000e+00 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.addf %in, %cst : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg2 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_subf_constant
func.func @matmul_splitk_subf_constant(%arg0: memref<32768xf32>, %arg1: memref<16384xf32>, %arg2: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 1.000000e+01 : f32
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 2.500000e+00 : f32
    // CHECK: %[[OUT:.*]] = arith.subf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.subf %in, %cst : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg2 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_addf_same
func.func @matmul_splitk_addf_same(%arg0: memref<32768xf32>, %arg1: memref<16384xf32>, %arg2: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[OUT:.*]] = arith.addf %in, %in : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.addf %in, %in : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg2 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_subf_same
func.func @matmul_splitk_subf_same(%arg0: memref<32768xf32>, %arg1: memref<16384xf32>, %arg2: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[OUT:.*]] = arith.subf %in, %in : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.subf %in, %in : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg2 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_addf
func.func @matmul_splitk_addf(%arg0: memref<2x4x4xf32>, %arg1: memref<32768xf32>, %arg2: memref<16384xf32>, %arg3: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 1.000000e+01 : f32
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %arg0 : memref<2x4x4xf32>, memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 4.000000e+00 : f32
    // CHECK: %[[NEW:.*]] = arith.divf %in_1, %[[CST]] : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[NEW]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.addf %in, %in_1 : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg3 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_subf
func.func @matmul_splitk_subf(%arg0: memref<2x4x4xf32>, %arg1: memref<32768xf32>, %arg2: memref<16384xf32>, %arg3: memref<32xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 1.000000e+01 : f32
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %arg0 : memref<2x4x4xf32>, memref<2x4x4xf32>) outs(%alloc_0 : memref<2x4x4xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 4.000000e+00 : f32
    // CHECK: %[[NEW:.*]] = arith.divf %in_1, %[[CST]] : f32
    // CHECK: %[[OUT:.*]] = arith.subf %in, %[[NEW]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %7 = arith.subf %in, %in_1 : f32
    linalg.yield %7 : f32
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<32xf32>
  memref.copy %6, %arg3 : memref<32xf32> to memref<32xf32>
  return
}

// CHECK-LABEL: func.func @matmul_splitk_multiple
func.func @matmul_splitk_multiple(%arg0: memref<2x4x4xf32>, %arg1: memref<2x4x4xf16>, %arg2: memref<32768xf32>, %arg3: memref<16384xf32>, %arg4: memref<32xf16>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 1.000000e+01 : f32
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 4 + d1) * 4096 + d2)> by [<Unmerge{2, 4, 4096} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4, 4096] -> [32768]> : memref<32768xf32> to memref<2x4x4096xf32>
  %1 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 4 + d2)> by [<Unmerge{1, 4096, 4} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 4096, 4] -> [16384]> : memref<16384xf32> to memref<1x4096x4xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 4096, 4] -> [1, 4096, 4]> : memref<1x4096x4xf32> to memref<2x4096x4xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4096] -> [2, 4, 4096]> : memref<2x4x4096xf32> to memref<8x4096xf32>
  %4 = rock.transform %2 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<ConstDim{0, 2} [] at [] -> ["g"] at [0]>, <PassThrough ["d0", "d1"] at [0, 1] -> ["d0", "d1"] at [1, 2]>] bounds = [4096, 4] -> [2, 4096, 4]> : memref<2x4096x4xf32> to memref<4096x4xf32>
  %5 = rock.transform %alloc by <affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, d1)> by [<Merge{2, 4} ["gd0"] at [0] -> ["g", "d0"] at [0, 1]>, <PassThrough ["d1"] at [1] -> ["d1"] at [2]>] bounds = [8, 4] -> [2, 4, 4]> : memref<2x4x4xf32> to memref<8x4xf32>
  rock.gemm %5 = %3 * %4 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : memref<8x4xf32> = memref<8x4096xf32> * memref<4096x4xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x4xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %arg0, %arg1 : memref<2x4x4xf32>, memref<2x4x4xf32>, memref<2x4x4xf16>) outs(%alloc_0 : memref<2x4x4xf16>) {
  ^bb0(%in: f32, %in_1: f32, %in_2: f16, %out: f16):
    // CHECK: %[[CST:.*]] = arith.constant 4.000000e+00 : f32
    // CHECK: %[[NEW:.*]] = arith.divf %in_1, %[[CST]] : f32
    // CHECK: %[[ADD:.*]] = arith.addf %in, %[[NEW]] : f32
    // CHECK: %[[IN2:.*]] = arith.extf %in_2 : f16 to f32
    // CHECK: %[[MUL:.*]] = arith.mulf %in, %[[IN2]] : f32
    // CHECK: %[[LINEAR:.*]] = arith.addf %[[ADD]], %[[MUL]] : f32
    // CHECK: %[[OUT:.*]] = arith.truncf %[[LINEAR]] : f32 to f16
    // CHECK: linalg.yield %[[OUT]] : f16
    %7 = arith.addf %in, %in_1 : f32
    %8 = arith.extf %in_2 : f16 to f32
    %9 = arith.mulf %in, %8 : f32
    %10 = arith.addf %7, %9 : f32
    %11 = arith.truncf %10 : f32 to f16
    linalg.yield %11 : f16
  }
  %6 = rock.transform %alloc_0 by <affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 4, d0 mod 4)> by [<Merge{2, 4, 4} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [2, 4, 4]> : memref<2x4x4xf16> to memref<32xf16>
  memref.copy %6, %arg4 : memref<32xf16> to memref<32xf16>
  return
}
