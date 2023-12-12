// This tests the error handling in the rock-affix-params pass

// RUN: rocmlir-opt -rock-affix-params %s -verify-diagnostics

func.func @rock_attention_invalid_perf_config(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{The provided perf config is not valid}}
  rock.attention(%arg0, %arg1, %arg2, %arg3) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", kTransposed, perf_config = "128,16,8,32,64,8,1,1"} : memref<1x384x64xf16>, memref<1x384x64xf16>, memref<1x384x64xf16>, memref<1x384x64xf16>
  return
}
