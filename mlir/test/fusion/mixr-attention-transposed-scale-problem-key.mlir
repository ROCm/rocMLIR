// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// A transposed pre-softmax multiply operand is an attention scale fusion, not a
// transposed bias. This guards that transBias detection stays gated on the add.
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f32 -transQ false -transK false -transV false -transO false -causal false -return_lse false -split_kv 1 -num_heads_q 1 -num_heads_kv 1 -g 1 -seq_len_q 5 -seq_len_k 7 -head_dim_qk 3 -head_dim_v 3 -with-attn-scale true -with-attn-bias false -transBias false
module
{
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x5x3xf32, 15x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg3: !migraphx.shaped<1x7x5xf32, 35x5x1>)
                                    -> (!migraphx.shaped<1x5x3xf32, 15x3x1>)  attributes {rock.kernel, rock.arch = "gfx942", rock.num_cu = 304 : i64} {
    %0 = migraphx.dot %arg0, %arg1: <1x5x3xf32, 15x3x1>, <1x3x7xf32, 21x7x1> -> <1x5x7xf32, 35x7x1>
    %transposed_scale = migraphx.transpose %arg3 {permutation = [0, 2, 1]} : <1x7x5xf32, 35x5x1> -> <1x5x7xf32, 35x1x5>
    %scaled = migraphx.mul %0, %transposed_scale : <1x5x7xf32, 35x7x1>, <1x5x7xf32, 35x1x5> -> <1x5x7xf32, 35x7x1>
    %1 = migraphx.softmax %scaled{axis = 2 : i64} : <1x5x7xf32, 35x7x1> -> <1x5x7xf32, 35x7x1>
    %2 = migraphx.dot %1, %arg2: <1x5x7xf32, 35x7x1>, <1x7x3xf32, 21x3x1> -> <1x5x3xf32, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }
}
