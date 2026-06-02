// A pure element-wise extf on the attention output is safe to fuse with
// splitKV > 1 (lossless widening commutes with the LSE combine).
// Regression test for https://github.com/ROCm/rocMLIR/issues/2376

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -emit-module-fusibility-for=attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1 - | FileCheck %s
// CHECK: fusible:1

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @attn_splitkv_output_extf_fusible(
      %queries: memref<1x64x1024xf16>,
      %keys: memref<1x64x1024xf16>,
      %values: memref<1x1024x64xf16>,
      %lse: memref<4x1024xf32>,
      %out: memref<4x1024x64xf32>
  ) attributes {rock.kernel, mhal.arch = "##TOKEN_ARCH##"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x1024x64xf16>
    rock.attention {
      qk = tr %queries * %keys : memref<1x64x1024xf16>, memref<1x64x1024xf16>
      lse = %lse : memref<4x1024xf32>
      %alloc = softmax(qk) * %values : memref<1x1024x64xf16> -> memref<4x1024x64xf16>
    } {
      firstGemmIndices = array<i64: 0>,
      splitKV = 4 : i32,
      storeMethod = #rock<StoreMethod set>,
      numHeadsKV = 1 : i32,
      numHeadsQ = 1 : i32
    }

    linalg.generic {
      indexing_maps = [#map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%alloc : memref<4x1024x64xf16>)
      outs(%out : memref<4x1024x64xf32>) {
    ^bb0(%in: f16, %out_init: f32):
      %cvt = arith.extf %in : f16 to f32
      linalg.yield %cvt : f32
    }
    return
  }
}
