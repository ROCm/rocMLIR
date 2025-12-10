// This test verifies that the applicability pipeline correctly handles splitK perf configs
// for GEMM, Attention, GemmGemm, and ConvGemm operations

//===----------------------------------------------------------------------===//
// GEMM Tests
//===----------------------------------------------------------------------===//

// Test valid splitK config passes applicability pipeline
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx90a -g 1 -m 128 -k 1024 -n 256 --perf_config "v4:64,64,16,32,32,16,4,4,1,2,1,1" | rocmlir-driver --kernel-pipeline=applicability 2>&1 | FileCheck %s --check-prefix=VALID-SPLITK
// VALID-SPLITK: rock.threadwise_gemm_accel
// VALID-SPLITK-NOT: error

// Test splitK=1 (no split) config passes applicability pipeline
// RUN: rocmlir-gen --operation gemm -t f32 --arch gfx90a -g 1 -m 128 -k 1024 -n 256 --perf_config "v4:64,64,16,32,32,16,4,1,1,2,1,1" | rocmlir-driver --kernel-pipeline=applicability 2>&1 | FileCheck %s --check-prefix=VALID-NOSPLIT
// VALID-NOSPLIT: rock.threadwise_gemm_accel
// VALID-NOSPLIT-NOT: error

//===----------------------------------------------------------------------===//
// Attention Tests 
//===----------------------------------------------------------------------===//

// Test attention with splitK=1 passes applicability pipeline
// RUN: rocmlir-gen --arch gfx90a --operation attention -t f16 -seq_len_q 256 -seq_len_k 256 -head_dim_qk 64 -head_dim_v 64 -g 1 -perf_config "attn:v3:32,32,32,32,32,32,16,8,1,1,2,1" | rocmlir-driver --kernel-pipeline=applicability 2>&1 | FileCheck %s --check-prefix=ATTN-VALID
// ATTN-VALID: rock.threadwise_gemm_accel
// ATTN-VALID-NOT: error

//===----------------------------------------------------------------------===//
// GemmGemm (gemm_elementwise_gemm) Tests 
//===----------------------------------------------------------------------===//

// Test gemm_elementwise_gemm with splitK=2 in second gemm passes applicability
// RUN: rocmlir-driver --kernel-pipeline=applicability %s 2>&1 | FileCheck %s --check-prefix=GEMMGEMM-SPLITK

// GEMMGEMM-SPLITK: rock.threadwise_gemm_accel
// GEMMGEMM-SPLITK-NOT: error
func.func @gemm_gemm_splitk_valid(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  rock.gemm_elementwise_gemm{
    ab = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = ab * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v3:32,32,32,32,32,32,16,8,2,1,2,1", firstGemmIndices = array<i64: 0>, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}

//===----------------------------------------------------------------------===//
// ConvGemm (conv_elementwise_gemm) Tests
//===----------------------------------------------------------------------===//

// Test conv_elementwise_gemm with splitK in second gemm passes applicability
// GEMMGEMM-SPLITK: rock.threadwise_gemm_accel
func.func @conv_gemm_splitk_valid(%arg0: memref<1x128x256x1x1xf16>, %arg1: memref<2x1x256x32x32xf16>, %arg2: memref<1x128x64xf16>, %arg3: memref<1x2048x64xf16>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  rock.conv_elementwise_gemm{
    ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
    %arg3 = ab * %arg2 : memref<1x128x64xf16> -> memref<1x2048x64xf16>
  } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v3:32,32,32,32,32,32,16,8,2,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], storeMethod = #rock<StoreMethod set>, strides = [1 : index, 1 : index]}
  return
}

// Test conv_elementwise_gemm without splitK (splitK=1)
// GEMMGEMM-SPLITK: rock.threadwise_gemm_accel
func.func @conv_gemm_nosplit_valid(%arg0: memref<1x128x256x3x3xf32>, %arg1: memref<2x1x256x128x128xf32>, %arg2: memref<1x128x128xf32>, %arg3: memref<1x32768x128xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  rock.conv_elementwise_gemm{
    ab = conv(%arg0, %arg1) : memref<1x128x256x3x3xf32>, memref<2x1x256x128x128xf32>
    %arg3 = ab * %arg2 : memref<1x128x128xf32> -> memref<1x32768x128xf32>
  } {dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, perf_config = "attn:v3:32,32,32,32,32,32,16,8,1,1,2,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [1 : index, 1 : index, 1 : index, 1 : index], storeMethod = #rock<StoreMethod set>, strides = [1 : index, 1 : index]}
  return
}
