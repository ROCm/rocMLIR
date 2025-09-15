// RUN: rocmlir-gen -fut mlir_unpack_dequantizelinear_dot --arch %arch --clone-harness %s | rocmlir-driver -host-pipeline=migraphx | FileCheck %s
// XFAIL: *
// COM: Check that if the pass ordering is incorrect (clone-harness first and then host-pipeline=migraphx), we get a clear error about what is wrong.
func.func private @mlir_unpack_dequantizelinear_dot(%arg0: !migraphx.shaped<1x4x8xi8, 32x8x1>, %arg1: !migraphx.shaped<1x16x4xf16, 64x4x1>) -> !migraphx.shaped<1x4x4xf16, 16x4x1>  {
  %0 = migraphx.literal (dense<[0.25]> : tensor<1xf16>) : <1xf16, 0>
  %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [1, 5, 16]} : <1xf16, 0> -> <1x4x16xf16, 0x0x0>
  %2 = migraphx.unpack %arg0 {axis = 2 : i64} : <1x4x8xi8, 32x8x1> -> <1x4x16xi8, 64x16x1>
  %3 = migraphx.dequantizelinear %2, %1 : <1x4x16xi8, 64x16x1>, <1x4x16xf16, 0x0x0> -> <1x4x16xf16, 64x16x1>
  %4 = migraphx.dot %3, %arg1 : <1x4x16xf16, 64x16x1>, <1x16x4xf16, 64x4x1> -> <1x4x4xf16, 16x4x1>
  return %4 : !migraphx.shaped<1x4x4xf16, 16x4x1>
}
