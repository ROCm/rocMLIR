// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.attention{
    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @rock_gemm_gemm_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.gemm_elementwise_gemm{
    ab = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = ab * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @rock_conv_gemm_invalid_perf_config(%arg0: memref<1x128x256x1x1xf16>, %arg1: memref<2x1x256x32x32xf16>, %arg2: memref<1x128x128xf16>, %arg3: memref<1x2048x128xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.conv_elementwise_gemm{
    ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
    %arg3 = ab * %arg2 : memref<1x128x128xf16> -> memref<1x2048x128xf16>
  } {dilations = [1 : index, 1 : index], perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], storeMethod = #rock<StoreMethod set>, strides = [1 : index, 1 : index]}
  return
}

func.func @rock_conv_schedulev2(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) attributes {schedule_version =  #rock.schedule_version<2>, arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+1 {{kernel has both perf_config and schedule_version attribute set. Please modify schedule version directly inside perf_config and remove schedule_version}}
  rock.conv(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "v3:64,128,8,64,32,1,1,1,2,1,1"
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// expected-error @below {{Multiple Fusion Roots detected in a single function. This is not supported.}}
func.func @two_gemms(
    %a0: memref<1x72x128xf8E4M3FN>, %b0: memref<1x72x115200xf8E5M2>, %c0: memref<1x128x115200xf32>,
    %a1: memref<1x72x128xf8E4M3FN>, %b1: memref<1x72x115200xf8E5M2>, %c1: memref<1x128x115200xf32>)
    attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
  // First GEMM
  rock.gemm %c0 = tr %a0 * %b0 features = mfma|dot|atomic_add|atomic_add_f16|atomic_add_bf16 storeMethod = set
    : memref<1x128x115200xf32> = memref<1x72x128xf8E4M3FN> * memref<1x72x115200xf8E5M2>
  rock.gemm %c1 = tr %a1 * %b1 features = mfma|dot|atomic_add|atomic_add_f16|atomic_add_bf16 storeMethod = set
    : memref<1x128x115200xf32> = memref<1x72x128xf8E4M3FN> * memref<1x72x115200xf8E5M2>
      return
}

func.func @rock_attn_schedulev2(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<2>, arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+1 {{kernel has both perf_config and schedule_version attribute set. Please modify schedule version directly inside perf_config and remove schedule_version}}
  rock.attention{
    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {perf_config = "attn:v2:128,128,16,8,32,64,8,1,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @rock_attn_perfconfig_schedulev3_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx1200"} {
  // expected-error @+1 {{schedule version not supported}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>, perf_config = "attn:v2:32,32,32,32,32,32,1,1,3,2,1"}
  return
}

func.func @rock_attn_perfconfig_schedulev4_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx1200"} {
  // expected-error @+1 {{schedule version not supported}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>, perf_config = "attn:v2:32,32,32,32,32,32,1,1,4,2,1"}
  return
}

func.func @rock_attn_schedulev3_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<3>, arch = "amdgcn-amd-amdhsa:gfx1200"} {
  // expected-error @+1 {{schedule version not supported}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @rock_attn_schedulev4_navi(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {schedule_version =  #rock.schedule_version<4>, arch = "amdgcn-amd-amdhsa:gfx1200"} {
  // expected-error @+1 {{schedule version not supported}}
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

func.func @rock_gemm_gemm_splitk(%arg0: memref<1474560xf16>, %arg1: memref<1474560xf16>, %arg2: memref<1474560xf16>, %arg3: memref<1474560xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
    %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 4096 + d2)> by [<Unmerge{360, 4096} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 360, 4096] -> [1474560]> : memref<1474560xf16> to memref<1x360x4096xf16>
    %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 360 + d2)> by [<Unmerge{4096, 360} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
    %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 1 + d2)> by [<Unmerge{4096, 360} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 4096, 360] -> [1474560]> : memref<1474560xf16> to memref<1x4096x360xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4096x360xf16>
    // expected-error @+1 {{Fusion with SplitK perfConfig is not legal}}
    rock.gemm_elementwise_gemm{
     ab = %0 * %1 : memref<1x4096x360xf16>, memref<1x360x4096xf16>
     ab = elementwise {
    ^bb0(%arg4: memref<1x4096x4096xf16>, %arg5: memref<1x4096x4096xf16>):
      memref.copy %arg4, %arg5 : memref<1x4096x4096xf16> to memref<1x4096x4096xf16>
      rock.yield
    }
     %alloc = ab * %2 : memref<1x4096x360xf16> -> memref<1x4096x360xf16>
    } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>, perf_config="attn:v3:32,32,32,32,32,32,16,1,2,1,2,1"}
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<1x4096x360xf16>

    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc : memref<1x4096x360xf16>) outs(%alloc_1 : memref<1x4096x360xf16>) {
    ^bb0(%in: f16, %out: f16):
      %5 = arith.fptoui %in : f16 to i8
      %6 = arith.sitofp %5 : i8 to f16
      linalg.yield %6 : f16
    }
    memref.copy %alloc_1, %3 : memref<1x4096x360xf16> to memref<1x4096x360xf16>
    return
  }

 func.func @mlir_dot_max_splitk(%arg1: memref<1x2x1280xf32>, %arg2: memref<1x1280x320xf32>, %arg3: memref<1x2x320xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x320xf32>
    // expected-error @+1 {{Fusion with SplitK perfConfig is not legal}}
    rock.gemm %alloc = %arg1 * %arg2 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", perf_config = "v4:16,16,4,16,16,16,1,5,1,2,1,1"} : memref<1x2x320xf32> = memref<1x2x1280xf32> * memref<1x1280x320xf32>
    %0 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [2, 320] -> [1, 2, 320]> : memref<1x2x320xf32> to memref<2x320xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x320xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0: memref<2x320xf32>) outs(%alloc_0 : memref<2x320xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %cst : f32
      linalg.yield %1 : f32
    }
    %2 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2) -> (d0 * 2 + d1, d2)> by [<Unmerge{1, 2} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 2, 320] -> [2, 320]> : memref<2x320xf32> to memref<1x2x320xf32>
    memref.copy %2, %arg3 : memref<1x2x320xf32> to memref<1x2x320xf32>
    return
  }