// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.attention{
    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "attn:v1:128,128,16,8,32,64,8,1", firstGemmIdx = 0 : i32}
  return
}

func.func @rock_gemm_gemm_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.gemm_elementwise_gemm{
    ab = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = ab * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "attn:v1:128,128,16,8,32,64,8,1", firstGemmIdx = 0 : i32}
  return
}

func.func @rock_conv_gemm_invalid_perf_config(%arg0: memref<1x128x256x1x1xf16>, %arg1: memref<2x1x256x32x32xf16>, %arg2: memref<1x128x128xf16>, %arg3: memref<1x2048x128xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.conv_elementwise_gemm{
    ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
    %arg3 = ab * %arg2 : memref<1x128x128xf16> -> memref<1x2048x128xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", dilations = [1 : index, 1 : index], features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_fmax_f32>, perf_config = "attn:v1:128,128,16,8,32,64,8,1", filter_layout = ["g", "k", "c", "0", "1"], firstGemmIdx = 0 : i32, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
  return
}

func.func @rock_conv_schedulev2(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) attributes {schedule_version =  #rock.schedule_version<2>} {
  // expected-error @+1 {{kernel has both perf_config and schedule_version attribute set. Please modify schedule version directly inside perf_config and remove schedule_version}}
  rock.conv(%filter, %input, %output) features = mfma|dot|atomic_add|atomic_add_f16 {
    arch = "amdgcn-amd-amdhsa:gfx942",
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
