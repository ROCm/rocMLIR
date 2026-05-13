// RUN: rocmlir-opt --migraphx-to-tosa %s | FileCheck %s

// Verify that MIGraphXToTosa does not modify rock.attention or its nested
// regions (which may contain linalg/arith ops from MIGraphXAttentionToRock).

// CHECK-LABEL: func.func @preserves_rock_attention
// CHECK: rock.attention
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: rock.yield
// CHECK-NOT: migraphx.mul
func.func @preserves_rock_attention(
    %q: tensor<1x7x3xf16>,
    %k: tensor<1x3x7xf16>,
    %v: tensor<1x7x3xf16>,
    %scale: tensor<1x7x7xf16>
) -> tensor<1x7x3xf16> attributes {rock.kernel, arch = ""} {
  %alloc = bufferization.alloc_tensor() : tensor<1x7x3xf16>
  %result = rock.attention {
    qk = %q * %k : tensor<1x7x3xf16>, tensor<1x3x7xf16>
    qk = elementwise otherIns(%scale : tensor<1x7x7xf16>) {
    ^bb0(%arg0: memref<1x7x7xf16>, %arg1: memref<1x7x7xf16>, %arg2: memref<1x7x7xf16>):
      %out = memref.alloc() : memref<1x7x7xf16>
      linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                         affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
      } ins(%arg0, %arg1 : memref<1x7x7xf16>, memref<1x7x7xf16>)
        outs(%out : memref<1x7x7xf16>) {
      ^bb0(%in0: f16, %in1: f16, %o: f16):
        %prod = arith.mulf %in0, %in1 : f16
        linalg.yield %prod : f16
      }
      memref.copy %out, %arg2 : memref<1x7x7xf16> to memref<1x7x7xf16>
      rock.yield
    }
    %alloc = softmax(qk) * %v : tensor<1x7x3xf16> -> tensor<1x7x3xf16>
  } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<1x7x3xf16>
  return %result : tensor<1x7x3xf16>
}
