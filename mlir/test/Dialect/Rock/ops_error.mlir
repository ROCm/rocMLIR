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

// -----------------------------------------------------------------------------
// gemm tests 
// -----------------------------------------------------------------------------

func.func @gemm_scale_presence_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA: memref<2x64x128xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gemm %c = %a scaled by %scaleA * %b features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN>
  func.return
}

func.func @gemm_scaleA_type_invalid(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA_bad: memref<2x64x128xf8E4M3FN>, %scaleB: memref<2x128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{rock.gemm' op operand #3 must be Constraints the type to be either a Tensor or MemRef of certain types of elements., but got 'memref<2x64x128xf8E4M3FN>}}
  rock.gemm %c = %a scaled by %scaleA_bad * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E4M3FN> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleA_k_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA_kbad: memref<2x64x127xf8E8M0FNU>, %scaleB: memref<2x128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %c = %a scaled by %scaleA_kbad * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x127xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleA_m_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA_mbad: memref<2x63x128xf8E8M0FNU>, %scaleB: memref<2x128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's M dimension must match matrix A's M dimension}}
  rock.gemm %c = %a scaled by %scaleA_mbad * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x63x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleA_g_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA_gbad: memref<3x64x128xf8E8M0FNU>, %scaleB: memref<2x128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's G dimension must match matrix A's G dimension}}
  rock.gemm %c = %a scaled by %scaleA_gbad * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<3x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleB_k_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA: memref<2x64x128xf8E8M0FNU>, %scaleB_kbad: memref<2x127x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by %scaleB_kbad features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x127x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleB_n_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA: memref<2x64x128xf8E8M0FNU>, %scaleB_nbad: memref<2x128x31xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's N dimension must match matrix B's N dimension}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by %scaleB_nbad features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x31xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleB_g_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA: memref<2x64x128xf8E8M0FNU>, %scaleB_gbad: memref<3x128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's G dimension must match matrix B's G dimension}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by %scaleB_gbad features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<3x128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleB_type_invalid(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA : memref<2x64x128xf8E8M0FNU>, %scaleB_bad : memref<2x128x32xf8E4M3FN>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{rock.gemm' op operand #4 must be Constraints the type to be either a Tensor or MemRef of certain types of elements., but got 'memref<2x128x32xf8E4M3FN>}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by %scaleB_bad  features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x128x32xf8E4M3FN>
  func.return
}

func.func @gemm_scaleA_transposed_k_mismatch(%a: memref<64x128xf4E2M1FN>, %b: memref<128x32xf4E2M1FN>,
  %c: memref<64x32xf32>, %scaleA_tbad: memref<127x64xf8E8M0FNU>, %scaleB: memref<128x32xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA's K dimension must match matrix A's K dimension}}
  rock.gemm %c = %a scaled by tr %scaleA_tbad * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<64x32xf32> = memref<64x128xf4E2M1FN> scaled by memref<127x64xf8E8M0FNU> * memref<128x32xf4E2M1FN> scaled by memref<128x32xf8E8M0FNU>
  func.return
}

func.func @gemm_scaleB_transposed_k_mismatch(%a: memref<2x64x128xf4E2M1FN>, %b: memref<2x128x32xf4E2M1FN>,
  %c: memref<2x64x32xf32>, %scaleA: memref<2x64x128xf8E8M0FNU>, %scaleB_kbad: memref<2x32x127xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB's K dimension must match matrix B's K dimension}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by tr %scaleB_kbad features = mfma storeMethod = set
  : memref<2x64x32xf32> = memref<2x64x128xf4E2M1FN> scaled by memref<2x64x128xf8E8M0FNU> * memref<2x128x32xf4E2M1FN> scaled by memref<2x32x127xf8E8M0FNU>
  func.return
}

func.func @rock_scaled_gemm_invalid_arch(%a : memref<32x64xf4E2M1FN>, %b : memref<1x32x128xf4E2M1FN>, %c : memref<64x128xf32>, %scaleA : memref<32x64xf8E8M0FNU>, %scaleB : memref<1x32x128xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx942"} {
  // expected-error @+1 {{'rock.gemm' op Mfma does not support Float4E2M1FN data type}}
  rock.gemm %c = tr %a scaled by tr %scaleA * %b scaled by %scaleB features = mfma storeMethod = set
  : memref<64x128xf32> = memref<32x64xf4E2M1FN> scaled by memref<32x64xf8E8M0FNU> * memref<1x32x128xf4E2M1FN> scaled by memref<1x32x128xf8E8M0FNU>
  func.return
}

func.func @gemm_scaled_inputs_not_float4e2m1(%a: memref<2x64x128xf16>,
                                            %b: memref<2x128x32xf16>,
                                            %c: memref<2x64x32xf32>,
                                            %scaleA: memref<2x64x128xf8E8M0FNU>,
                                            %scaleB: memref<2x128x32xf8E8M0FNU>)
    attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{Scaled GEMMs are only supported for Float4E2M1FN input type}}
  rock.gemm %c = %a scaled by %scaleA * %b scaled by %scaleB features = mfma storeMethod = set
    : memref<2x64x32xf32> =
      memref<2x64x128xf16> scaled by memref<2x64x128xf8E8M0FNU> *
      memref<2x128x32xf16> scaled by memref<2x128x32xf8E8M0FNU>
  func.return
}

// -----------------------------------------------------------------------------
// Gridwise gemm accel tests 
// -----------------------------------------------------------------------------

#common_params = #rock.xdlops_gemm_derived_params<
  kpackPerBlock = 4,
  kpack = 4,
  mPerBlock = 64,
  mPerWave = 32,
  nPerBlock = 64,
  nPerWave = 32,
  mnPerXdl = 32,
  splitKFactor = 1,
  scheduleVersion = 1,
  outputSwizzle = 2,
  forceUnroll = true>

func.func @gridwise_gemm_accel_scale_presence_a_only(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA: memref<1x4x8xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x8xf8E8M0FNU>
  func.return
}

// Scale presence B only
func.func @gridwise_gemm_accel_scale_presence_b_only(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleB: memref<1x4x16xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleB) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x16xf8E8M0FNU>
  func.return
}

// scaleA type invalid
func.func @gridwise_gemm_accel_scaleA_type_invalid(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA_bad: memref<1x4x8xf8E4M3FN>, %scaleB: memref<1x4x16xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm_accel' op operand #3 must be 3D memref of f8E8M0FNU type values, but got 'memref<1x4x8xf8E4M3FN>'}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA_bad, %scaleB) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x8xf8E4M3FN>, memref<1x4x16xf8E8M0FNU>
  func.return
}

// scaleA dims mismatch
func.func @gridwise_gemm_accel_scaleA_dims_mismatch(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA_bad_dims: memref<1x4x7xf8E8M0FNU>, %scaleB: memref<1x4x16xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleA dimensions must match matrix A dimensions}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA_bad_dims, %scaleB) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x7xf8E8M0FNU>, memref<1x4x16xf8E8M0FNU>
  func.return
}

// scaleA input type invalid
func.func @gridwise_gemm_accel_scaleA_input_type_invalid(%A: memref<1x4x8xf16>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA: memref<1x4x8xf8E8M0FNU>, %scaleB: memref<1x4x16xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{Matrix A must be of type Float4E2M1FNType when scaleA is provided.}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf16>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x8xf8E8M0FNU>, memref<1x4x16xf8E8M0FNU>
  func.return
}

// scaleB type invalid
func.func @gridwise_gemm_accel_scaleB_type_invalid(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA: memref<1x4x8xf8E8M0FNU>, %scaleB_bad: memref<1x4x16xf8E4M3FN>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{'rock.gridwise_gemm_accel' op operand #4 must be 3D memref of f8E8M0FNU type values, but got 'memref<1x4x16xf8E4M3FN>'}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB_bad) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x8xf8E8M0FNU>, memref<1x4x16xf8E4M3FN>
  func.return
}

// scaleB dims mismatch
func.func @gridwise_gemm_accel_scaleB_dims_mismatch(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf4E2M1FN>, %C: memref<1x8x16xf32>, %scaleA: memref<1x4x8xf8E8M0FNU>, %scaleB_bad_dims: memref<1x4x15xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{scaleB dimensions must match matrix B dimensions}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB_bad_dims) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf4E2M1FN>, memref<1x8x16xf32>, memref<1x4x8xf8E8M0FNU>, memref<1x4x15xf8E8M0FNU>
  func.return
}

// scaleB input type invalid
func.func @gridwise_gemm_accel_scaleB_input_type_invalid(%A: memref<1x4x8xf4E2M1FN>, %B: memref<1x4x16xf16>, %C: memref<1x8x16xf32>, %scaleA: memref<1x4x8xf8E8M0FNU>, %scaleB: memref<1x4x16xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950"} {
  // expected-error @+1 {{Matrix B must be of type Float4E2M1FNType when scaleB is provided.}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB) storeMethod(set) features = mfma {
    blockSize = 64 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<1x4x8xf4E2M1FN>, memref<1x4x16xf16>, memref<1x8x16xf32>, memref<1x4x8xf8E8M0FNU>, memref<1x4x16xf8E8M0FNU>
  func.return
}

// Invalid arch 
func.func @rock_gridwise_gemm_accel_invalid_arch(%A: memref<2x1024x1024xf4E2M1FN>, %B: memref<2x1024x2048xf4E2M1FN>, %C: memref<2x1024x2048xf32>, %scaleA : memref<2x1024x1024xf8E8M0FNU>, %scaleB : memref<2x1024x2048xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", numCU = 304 : i32} {
  // expected-error @+1 {{'rock.gridwise_gemm_accel' op Mfma does not support Float4E2M1FN data type}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB) storeMethod(set) features = mfma {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<2x1024x1024xf4E2M1FN>, memref<2x1024x2048xf4E2M1FN>, memref<2x1024x2048xf32>, memref<2x1024x1024xf8E8M0FNU>, memref<2x1024x2048xf8E8M0FNU>
  return
}

// out data type invalid
func.func @rock_gridwise_gemm_accel_invalid_out_dtype(%A: memref<2x1024x1024xf4E2M1FN>, %B: memref<2x1024x2048xf4E2M1FN>, %C: memref<2x1024x2048xf16>, %scaleA : memref<2x1024x1024xf8E8M0FNU>, %scaleB : memref<2x1024x2048xf8E8M0FNU>) attributes {arch = "amdgcn-amd-amdhsa:gfx950", numCU = 256 : i32} {
  // expected-error @+1 {{'rock.gridwise_gemm_accel' op 4-bit or 8-bit float input requires f32 output}}
  rock.gridwise_gemm_accel(%A, %B, %C, %scaleA, %scaleB) storeMethod(set) features = mfma {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #common_params
  } : memref<2x1024x1024xf4E2M1FN>, memref<2x1024x2048xf4E2M1FN>, memref<2x1024x2048xf16>, memref<2x1024x1024xf8E8M0FNU>, memref<2x1024x2048xf8E8M0FNU>
  return
}

// -----------------------------------------------------------------------------
// Blockwise gemm accel tests 
// -----------------------------------------------------------------------------
#blockwise_params = #rock.xdlops_gemm_derived_params<
  kpackPerBlock = 2,
  kpack = 2,
  mPerBlock = 128,
  mPerWave = 64,
  nPerBlock = 128,
  nPerWave = 64,
  mnPerXdl = 32,
  splitKFactor = 1,
  scheduleVersion = 1,
  outputSwizzle = 2,
  forceUnroll = true>

func.func @blockwise_gemm_accel_loadA_no_matrixA(
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If loadAFromLDS is enabled, matrixA must be non-null.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA
    * %bufferB from %matrixB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>>
        from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scale_buffer_presence_a_only(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>
) {
  // expected-error @+1 {{scaleA and scaleB buffers must both be present or both be null.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA
    * %bufferB from %matrixB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>>
        from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>>
        from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_loadA_scaleA_missing_lds(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If loadAFromLDS is enabled, scaleA must be loaded from LDS.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>> 
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>> 
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_loadA_scaleA_lds_only(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If scaleA is loaded from LDS, scaleA buffer must be non-null.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleA_lds_shape_mismatch(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA_bad: memref<128xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If scaleA is loaded from LDS, its shape must match matrixA's shape.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA_bad
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<128xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleA_lds_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA_bad: memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %bufferScaleB: memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{ScaleA must be of type Float8E8M0FNU.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA_bad
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<vector<4xf8E8M0FNU>, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_matrixA_type_bad(
  %matrixA_bad: memref<256xvector<2xf16>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{For the scaled GEMMs, matrixA must be of type Float4E2M1FNType.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA_bad
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf16>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleA_buffer_shape_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA_bad: memref<5xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If scaleA buffer is non-null, its shape must match bufferA's shape.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA_bad from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<5xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleA_buffer_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA_bad: memref<4xf8E4M3FN, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{ScaleA buffer must be of type Float8E8M0FNU.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA_bad from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      blockSize = 256 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E4M3FN, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_bufferA_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA_bad: memref<4xf16, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{For the scaled GEMMs, bufferA must be of type Float4E2M1FNType.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA_bad from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf16, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scale_buffer_presence_b_only(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>
) {
  // expected-error @+1 {{scaleA and scaleB buffers must both be present or both be null.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    * %bufferB from %matrixB
    scaled by %bufferScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      blockSize = 256 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>>
        from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>>
        from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>>
  return
}

func.func @blockwise_gemm_accel_scaleB_lds_shape_mismatch(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB_bad: memref<255xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If scaleB is loaded from LDS, its shape must match matrixB's shape.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB_bad
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      blockSize = 256 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<255xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleB_lds_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB_bad: memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{ScaleB must be of type Float8E8M0FNU.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB_bad
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      blockSize = 256 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E4M3FN>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_matrixB_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB_bad: memref<256xvector<2xf16>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{For the scaled GEMMs, matrixB must be of type Float4E2M1FNType.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB_bad
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      loadAfromLDS,
      loadBfromLDS,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      blockSize = 256 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf16>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleB_buffer_shape_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB_bad: memref<5xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{If scaleB buffer is non-null, its shape must match bufferB's shape.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB_bad from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<5xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_scaleB_buffer_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB_bad: memref<4xf8E4M3FN, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{ScaleB buffer must be of type Float8E8M0FNU.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB_bad from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E4M3FN, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_bufferB_type_bad(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB_bad: memref<4xf16, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{For the scaled GEMMs, bufferB must be of type Float4E2M1FNType.}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB_bad from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx950",
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params
    } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>>  from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf16, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

func.func @blockwise_gemm_accel_invalid_arch(
  %matrixA: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixB: memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>,
  %matrixScaleA: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %matrixScaleB: memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>,
  %bufferA: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferB: memref<4xf4E2M1FN, #gpu.address_space<private>>,
  %bufferScaleA: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %bufferScaleB: memref<4xf8E8M0FNU, #gpu.address_space<private>>,
  %matrixC: memref<4xvector<16xf32>, #gpu.address_space<private>>
) {
  // expected-error @+1 {{'rock.blockwise_gemm_accel' op Mfma does not support Float4E2M1FN data type}}
  rock.blockwise_gemm_accel
    %matrixC
    += %bufferA from %matrixA
    scaled by %bufferScaleA from %matrixScaleA
    * %bufferB from %matrixB
    scaled by %bufferScaleB from %matrixScaleB
    features = mfma {
      arch = "amdgcn-amd-amdhsa:gfx942",
      loadAfromLDS,
      loadBfromLDS,
      blockSize = 256 : i32,
      inMPerThread = 2 : i32,
      inNPerThread = 2 : i32,
      params = #blockwise_params    
      } : memref<4xvector<16xf32>, #gpu.address_space<private>>
        += memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
        * memref<4xf4E2M1FN, #gpu.address_space<private>> from memref<256xvector<2xf4E2M1FN>, #gpu.address_space<workgroup>>
        scaled by memref<4xf8E8M0FNU, #gpu.address_space<private>> from memref<256xvector<2xf8E8M0FNU>, #gpu.address_space<workgroup>>
  return
}

//===----------------------------------------------------------------------===//
// Test cases for rock.threadwise_accel_gemm
//===----------------------------------------------------------------------===//

#params = #rock.xdlops_gemm_derived_params<
  mPerBlock = 256,
  nPerBlock = 256,
  kpackPerBlock = 16,
  mPerWave = 128,
  nPerWave = 64,
  mnPerXdl = 32,
  kpack = 1,
  splitKFactor = 1, 
  scheduleVersion = 1, 
  outputSwizzle = 2,
  forceUnroll = true>

// Error case: Only scaleA provided
func.func @threadwise_gemm_accel_scale_mismatch1(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>      // matches matrixA
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleA and ScaleB must both be present or both be null.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA * %matrixB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5>
  return
}

// Error case: Only scaleB provided
func.func @threadwise_gemm_accel_scale_mismatch2(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleA and ScaleB must both be present or both be null.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}

// Error case: Wrong scale type for scaleA
func.func @threadwise_gemm_accel_wrong_scale_type_A(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA_wrong: memref<2x4xf8E4M3FN, 5>,  // Wrong type
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleA must be of type Float8E8M0FNU.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA_wrong * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E4M3FN, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}

// Error case: Wrong scale type for scaleB
func.func @threadwise_gemm_accel_wrong_scale_type_B(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
  %scaleB_wrong: memref<3x4xf8E4M3FN, 5>  // Wrong type
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleB must be of type Float8E8M0FNU.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB_wrong at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E4M3FN, 5>
  return
}

// Error case: Wrong input type for matrixA with scaling
func.func @threadwise_gemm_accel_wrong_matrix_type_A(
  %matrixA_wrong: memref<2x4xf16, 5>,    // Not f4E2M1FN
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA dimensions
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{For the scaled GEMMs, matrixA must be of type Float4E2M1FNType.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA_wrong scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf16, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}

// Error case: Wrong input type for matrixB with scaling
func.func @threadwise_gemm_accel_wrong_matrix_type_B(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB_wrong: memref<3x4xf16, 5>,    // Not f4E2M1FN
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB dimensions
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{For the scaled GEMMs, matrixB must be of type Float4E2M1FNType.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA * %matrixB_wrong scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf16, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}

// Error case: Scale A shape doesn't match matrixA shape
func.func @threadwise_gemm_accel_scale_shape_mismatch_A(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA_wrong: memref<3x4xf8E8M0FNU, 5>,  // Wrong shape (m dimension mismatch)
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleA shape must match matrixA shape.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA_wrong * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}

// Error case: Scale B shape doesn't match matrixB shape
func.func @threadwise_gemm_accel_scale_shape_mismatch_B(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
  %scaleB_wrong: memref<4x4xf8E8M0FNU, 5>  // Wrong shape (n dimension mismatch)
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{ScaleB shape must match matrixB shape.}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB_wrong at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx950",
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<4x4xf8E8M0FNU, 5>
  return
}

// Error case: Architecture not supporting Float4E2M1FN
func.func @threadwise_gemm_accel_unsupported_arch(
  %matrixA: memref<2x4xf4E2M1FN, 5>,     // m=2, k=4
  %matrixB: memref<3x4xf4E2M1FN, 5>,     // n=3, k=4
  %matrixC: memref<2x3xf32, 5>,          // m=2, n=3
  %scaleA: memref<2x4xf8E8M0FNU, 5>,     // matches matrixA
  %scaleB: memref<3x4xf8E8M0FNU, 5>      // matches matrixB
) {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Mfma does not support Float4E2M1FN data type}}
  rock.threadwise_accel_gemm 
    %matrixC += %matrixA scaled by %scaleA * %matrixB scaled by %scaleB at [%c0, %c0, %c0] 
    features = mfma{
      arch = "amdgcn-amd-amdhsa:gfx942", // Unsupported architecture for Float4E2M1FN
      params = #params
    } : memref<2x3xf32, 5> += memref<2x4xf4E2M1FN, 5> scaled by memref<2x4xf8E8M0FNU, 5> * memref<3x4xf4E2M1FN, 5> scaled by memref<3x4xf8E8M0FNU, 5>
  return
}
