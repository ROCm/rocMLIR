// RUN: rocmlir-opt -verify-diagnostics %s

func.func @gridwise_attn_atomic_add_fail(%arg0: memref<1x384x64xf32>, %arg1: memref<1x64x384xf32>, %arg2: memref<1x384x64xf32>, %arg3: memref<1x384x64xf32>) attributes {block_size = 64 : i32, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  
  // expected-error @below {{Only set store method is supported for attention.}}
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    storeMethod = #rock<StoreMethod atomic_add>,
    splitKV = 1 : i32,
    enableSoftmax = true,
    numHeadsKV = 1 : i32, 
    numHeadsQ = 1 : i32,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1, 0>
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

func.func @attention_nonset(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @below {{Only set store method is supported for attention.}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod atomic_add>}
  return
}

func.func @attention_numheadskv_negative(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @below {{numHeadsKV must be positive}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = -1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @attention_numheadsq_negative(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @below {{numHeadsQ must be positive}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = -1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @attention_numheadsq_not_divisible(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @below {{numHeadsQ is not divisible by numHeadsKV}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 3 : i32, numHeadsQ = 4 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @attention_numheadsq_smaller_than_numheadskv(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @below {{numHeadsQ is not divisible by numHeadsKV}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 4 : i32, numHeadsQ = 2 : i32, storeMethod = #rock<StoreMethod set>}
  return
}
