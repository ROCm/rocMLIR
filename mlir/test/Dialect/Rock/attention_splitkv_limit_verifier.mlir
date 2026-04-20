// RUN: env ROCMLIR_ATTENTION_SPLITKV_MAX_EXTRA_BYTES=1024 rocmlir-opt -verify-diagnostics %s

func.func @attention_splitkv_limit_reject(
    %arg0: memref<1x64x1024xf16>,
    %arg1: memref<1x64x1024xf16>,
    %arg2: memref<1x1024x64xf16>,
    %arg3: memref<2x1024xf32>,
    %arg4: memref<2x1024x64xf16>)
    attributes {rock.kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // expected-error @+1 {{splitKV requires 135168 bytes of extra output/LSE storage, which exceeds the limit (1024 bytes). Lower splitKV or reduce sequence sizes. Override this guard with ROCMLIR_ATTENTION_SPLITKV_MAX_EXTRA_BYTES.}}
  rock.attention {
    qk = tr %arg0 * %arg1 : memref<1x64x1024xf16>, memref<1x64x1024xf16>
    lse = %arg3 : memref<2x1024xf32>
    %arg4 = softmax(qk) * %arg2 : memref<1x1024x64xf16> -> memref<2x1024x64xf16>
  } {features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, firstGemmIndices = array<i64: 0>, splitKV = 2 : i32, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, storeMethod = #rock<StoreMethod set>}
  return
}
