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

// CHECK-LABEL: func.func @convolution_multi_output_add
func.func @convolution_multi_output_add(%arg0: memref<32768xf32>, %arg1: memref<11520xf32>, %arg2: memref<2621440xf32>, %arg3: memref<2621440xf32>) attributes {arch = "", block_size = 128 : i32, enable_splitk_for_tuning, kernel} {
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.44140629E-5 : f32
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)> by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]> : memref<11520xf32> to memref<320x4x3x3xf32>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)> by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]> : memref<32768xf32> to memref<2x4x64x64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]> : memref<2x4x64x64xf32> to memref<2x1x4x64x64xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]> : memref<320x4x3x3xf32> to memref<1x320x4x3x3xf32>
  %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x1x320x64x64xf32>
  %5 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 36, 320] -> [1, 320, 4, 3, 3]> : memref<1x320x4x3x3xf32> to memref<1x36x320xf32>
  %6 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [2, 1, 4, 66, 66] -> [2, 1, 4, 64, 64]> : memref<2x1x4x64x64xf32> to memref<2x1x4x66x66xf32>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [2, 1, 4, 3, 64, 3, 64] -> [2, 1, 4, 66, 66]> : memref<2x1x4x66x66xf32> to memref<2x1x4x3x64x3x64xf32>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 4096) floordiv 64, d1 mod 3, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 36, 8192] -> [2, 1, 4, 3, 64, 3, 64]> : memref<2x1x4x3x64x3x64xf32> to memref<1x36x8192xf32>
  %9 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 3, 4]>] bounds = [1, 320, 8192] -> [2, 1, 320, 64, 64]> : memref<2x1x320x64x64xf32> to memref<1x320x8192xf32>
  rock.gemm %9 = tr %5 * %8 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x320x8192xf32> = memref<1x36x320xf32> * memref<1x36x8192xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc : memref<2x320x64x64xf32>) outs(%alloc_2 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.mulf %in, %cst_1 : f32
    linalg.yield %12 : f32
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_2 : memref<2x320x64x64xf32>) outs(%alloc_3 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 2.500000e-01 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %12 = arith.addf %in, %cst_0 : f32
    linalg.yield %12 : f32
  }
  %10 = rock.transform %alloc_3 by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 320, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [2621440] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2621440xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_2 : memref<2x320x64x64xf32>) outs(%alloc_4 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %12 = arith.addf %in, %cst : f32
    linalg.yield %12 : f32
  }
  %11 = rock.transform %alloc_4 by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 320, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [2621440] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2621440xf32>
  memref.copy %10, %arg2 : memref<2621440xf32> to memref<2621440xf32>
  memref.copy %11, %arg3 : memref<2621440xf32> to memref<2621440xf32>
  return
}

// CHECK-LABEL: func.func @convolution_multi_output_add_twice
func.func @convolution_multi_output_add_twice(%arg0: memref<32768xf32> {mhal.read_access}, %arg1: memref<11520xf32> {mhal.read_access}, %arg2: memref<2621440xf32> {mhal.write_access}, %arg3: memref<2621440xf32> {mhal.write_access}) attributes {arch = "gfx942:sramecc+:xnack-", block_size = 128 : i32, enable_splitk_for_tuning, kernel, original_func = @mlir_convolution_multi_output_add} {
  %cst = arith.constant 2.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.44140629E-5 : f32
  %cst_2 = arith.constant 3.000000e+00 : f32
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)> by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]> : memref<11520xf32> to memref<320x4x3x3xf32>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)> by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]> : memref<32768xf32> to memref<2x4x64x64xf32>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]> : memref<2x4x64x64xf32> to memref<2x1x4x64x64xf32>
  %3 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]> : memref<320x4x3x3xf32> to memref<1x320x4x3x3xf32>
  %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 64, 64] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2x1x320x64x64xf32>
  %5 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 36, 320] -> [1, 320, 4, 3, 3]> : memref<1x320x4x3x3xf32> to memref<1x36x320xf32>
  %6 = rock.transform %2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{1, 1, 1, 1} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [2, 1, 4, 66, 66] -> [2, 1, 4, 64, 64]> : memref<2x1x4x64x64xf32> to memref<2x1x4x66x66xf32>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["0ipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["1ipad"] at [4]>] bounds = [2, 1, 4, 3, 64, 3, 64] -> [2, 1, 4, 66, 66]> : memref<2x1x4x66x66xf32> to memref<2x1x4x3x64x3x64xf32>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 4096) floordiv 64, d1 mod 3, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{4, 3, 3} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 36, 8192] -> [2, 1, 4, 3, 64, 3, 64]> : memref<2x1x4x3x64x3x64xf32> to memref<1x36x8192xf32>
  %9 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d2 floordiv 4096, d0, d1, (d2 mod 4096) floordiv 64, d2 mod 64)> by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{2, 64, 64} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 3, 4]>] bounds = [1, 320, 8192] -> [2, 1, 320, 64, 64]> : memref<2x1x320x64x64xf32> to memref<1x320x8192xf32>
  rock.gemm %9 = tr %5 * %8 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942:sramecc+:xnack-", derivedBlockSize = 128 : i32, params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 4, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>} : memref<1x320x8192xf32> = memref<1x36x320xf32> * memref<1x36x8192xf32>
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc : memref<2x320x64x64xf32>) outs(%alloc_2 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %12 = arith.mulf %in, %cst_1 : f32
    linalg.yield %12 : f32
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_2 : memref<2x320x64x64xf32>) outs(%alloc_3 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 2.500000e-01 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %12 = arith.addf %in, %cst_0 : f32
    linalg.yield %12 : f32
  }
  %10 = rock.transform %alloc_3 by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 320, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [2621440] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2621440xf32>
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_2 : memref<2x320x64x64xf32>) outs(%alloc_4 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 5.000000e-01 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %12 = arith.addf %in, %cst : f32
    linalg.yield %12 : f32
  }

  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_4 : memref<2x320x64x64xf32>) outs(%alloc_5 : memref<2x320x64x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    // CHECK: %[[CST:.*]] = arith.constant 7.500000e-01 : f32
    // CHECK: %[[OUT:.*]] = arith.addf %in, %[[CST]] : f32
    // CHECK: linalg.yield %[[OUT]] : f32
    %12 = arith.addf %in, %cst_2 : f32
    linalg.yield %12 : f32
  }
  %11 = rock.transform %alloc_5 by <affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)> by [<Merge{2, 320, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [2621440] -> [2, 320, 64, 64]> : memref<2x320x64x64xf32> to memref<2621440xf32>
  memref.copy %10, %arg2 : memref<2621440xf32> to memref<2621440xf32>
  memref.copy %11, %arg3 : memref<2621440xf32> to memref<2621440xf32>
  return
}
