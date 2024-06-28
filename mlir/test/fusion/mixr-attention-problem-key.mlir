
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f32 -transQ false -transK false -transV false -transO false -g 1 -seq_len_q 7 -seq_len_k 7 -head_dim_qk 3 -head_dim_v 3
module 
{
  func.func private @mlir_attention(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1> {func.read_access},
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1> {func.read_access},
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1> {func.read_access},
                                    %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1> {func.read_access}) 
                                    -> (!migraphx.shaped<1x7x3xf32, 21x3x1> {func.write_access})  attributes {kernel, arch = "gfx942", num_cu = 304 : i64} {
    %0 = migraphx.dot %arg0, %arg1: <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %1 = migraphx.softmax %biased{axis = 2 : i64} : <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %2 = migraphx.dot %1, %arg2: <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %2 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
