// RUN: rocmlir-opt -migraphx-realize-int4 --split-input-file %s | FileCheck %s


// CHECK-LABEL: @basic
// CHECK-SAME: (%[[x:.+]]: !migraphx.shaped<8x2x2xi4, 4x2x1>) -> !migraphx.shaped<8x4xi8, 4x1>
func.func @basic(%x: !migraphx.shaped<8x2xi8, 2x1>) -> !migraphx.shaped<8x4xi8, 4x1> {
  // CHECK: %[[extended:.+]] = migraphx.convert zero_extend %[[x]] : <8x2x2xi4, 4x2x1> to <8x2x2xi8, 4x2x1>
  // CHECK: %[[reshaped:.+]] = migraphx.reshape %[[extended]] {dims = [8, 4]}
  // CHECK: return %[[reshaped]]
  %y = migraphx.unpack %x {axis = 1 : i64} : <8x2xi8, 2x1> -> <8x4xi8, 4x1>
  func.return %y : !migraphx.shaped<8x4xi8, 4x1>
}
