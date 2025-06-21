
// RUN: rocmlir-driver -kernel-pipeline=migraphx,highlevel %s | rocmlir-gen --emit-tuning-key - | FileCheck %s
// CHECK: gfx942
// CHECK-SAME: 304
// CHECK-SAME: -t f32 -transA false -transB false -transC false -transO false -g 1 -m 7 -n 7 -k 3 -gemmO 3
module 
{
  func.func private @mlir_gemm_gemm(%arg0: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg1: !migraphx.shaped<1x3x7xf32, 21x7x1>,
                                    %arg2: !migraphx.shaped<1x7x3xf32, 21x3x1>,
                                    %arg3: !migraphx.shaped<1x7x7xf32, 49x7x1>) 
                                    -> (!migraphx.shaped<1x7x3xf32, 21x3x1>)  attributes {kernel, arch = "gfx942", num_cu = 304 : i64} {
    %0 = migraphx.dot %arg0, %arg1: <1x7x3xf32, 21x3x1>, <1x3x7xf32, 21x7x1> -> <1x7x7xf32, 49x7x1>
    %biased = migraphx.add %0, %arg3 : <1x7x7xf32, 49x7x1>, <1x7x7xf32, 49x7x1> -> <1x7x7xf32, 49x7x1>
    %2 = migraphx.dot %biased, %arg2: <1x7x7xf32, 49x7x1>, <1x7x3xf32, 21x3x1> -> <1x7x3xf32, 21x3x1>
    return %2 : !migraphx.shaped<1x7x3xf32, 21x3x1>
  }
}
