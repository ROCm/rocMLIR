// Output fusions are not allowed for attention ops with splitKV > 1.
// With flash decoding, partial results need LSE-based corrections in a
// subsequent stage, so output fusions should not be applied to the attention
// kernel. Input fusions and fusions between the two GEMMs are still allowed.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -emit-module-fusibility-for=attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1 - | FileCheck %s
// CHECK: fusible:0

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @attn_splitkv_not_fusible(
      %queries: memref<1x64x1024xf32>,
      %keys: memref<1x64x1024xf32>,
      %values: memref<1x1024x64xf32>,
      %lse: memref<4x1024xf32>,
      %bias: memref<4x1024x64xf32>,
      %out: memref<4x1024x64xf32>
  ) attributes {rock.kernel, mhal.arch = "##TOKEN_ARCH##"} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x1024x64xf32>
    rock.attention {
      qk = tr %queries * %keys : memref<1x64x1024xf32>, memref<1x64x1024xf32>
      lse = %lse : memref<4x1024xf32>
      %alloc = softmax(qk) * %values : memref<1x1024x64xf32> -> memref<4x1024x64xf32>
    } {
      firstGemmIndices = array<i64: 0>,
      splitKV = 4 : i32,
      storeMethod = #rock<StoreMethod set>,
      numHeadsKV = 1 : i32,
      numHeadsQ = 1 : i32
    }

    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%alloc, %bias : memref<4x1024x64xf32>, memref<4x1024x64xf32>)
      outs(%out : memref<4x1024x64xf32>) {
    ^bb0(%in: f32, %in_bias: f32, %result: f32):
      %add = arith.addf %in, %in_bias : f32
      linalg.yield %add : f32
    }
    return
  }
}
