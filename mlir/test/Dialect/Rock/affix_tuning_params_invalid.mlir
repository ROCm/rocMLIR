// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.attention{
    qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
    %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "128,16,8,32,64,8,1,1"}
  return
}
