// Input fusions and fusions between the first and second gemm are allowed
// for attention ops with splitKV > 1. Only output fusions are disallowed.

// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-gen -emit-module-fusibility-for=attn:v3:32,32,32,32,32,32,16,1,1,1,2,0,1 - | FileCheck %s
// CHECK: fusible:1

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module {
  func.func @attn_splitkv_input_and_intergemm_fusible(
      %queries: memref<1x64x1024xf32>,
      %keys: memref<1x64x1024xf32>,
      %values: memref<1x1024x64xf32>,
      %lse: memref<4x1024xf32>,
      %scale: memref<1x64x1024xf32>,
      %out: memref<4x1024x64xf32>
  ) attributes {rock.kernel, mhal.arch = "##TOKEN_ARCH##"} {
    // Input fusion: scale the queries before feeding into attention
    %scaled_queries = memref.alloc() {alignment = 64 : i64} : memref<1x64x1024xf32>
    linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%queries, %scale : memref<1x64x1024xf32>, memref<1x64x1024xf32>)
      outs(%scaled_queries : memref<1x64x1024xf32>) {
    ^bb0(%in: f32, %in_scale: f32, %result: f32):
      %mul = arith.mulf %in, %in_scale : f32
      linalg.yield %mul : f32
    }

    rock.attention {
      qk = tr %scaled_queries * %keys : memref<1x64x1024xf32>, memref<1x64x1024xf32>
      lse = %lse : memref<4x1024xf32>
      // Inter-gemm fusion: elementwise identity between first and second gemm
      qk = elementwise {
      ^bb0(%in: memref<1x1024x1024xf32>, %out_ew: memref<1x1024x1024xf32>):
        memref.copy %in, %out_ew : memref<1x1024x1024xf32> to memref<1x1024x1024xf32>
        rock.yield
      }
      %out = softmax(qk) * %values : memref<1x1024x64xf32> -> memref<4x1024x64xf32>
    } {
      firstGemmIndices = array<i64: 0>,
      splitKV = 4 : i32,
      storeMethod = #rock<StoreMethod set>,
      numHeadsKV = 1 : i32,
      numHeadsQ = 1 : i32
    }
    return
  }
}
