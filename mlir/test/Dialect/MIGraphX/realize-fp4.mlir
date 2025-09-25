// RUN: rocmlir-opt -migraphx-realize-fp4 --split-input-file %s | FileCheck %s

//===----------------------------------------------------------------------===//
// 1. Fold argument unpack: signature + return changed to fp4; unpack removed.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @basic_fp4(
// CHECK-SAME: %[[ARG:.*]]: !migraphx.shaped<8x4xf4E2M1FN, 4x1>
// CHECK-SAME: ) -> !migraphx.shaped<8x4xf4E2M1FN, 4x1>
// CHECK-NOT: migraphx.unpack
// CHECK: return %[[ARG]] : !migraphx.shaped<8x4xf4E2M1FN, 4x1>
func.func @basic_fp4(%x: !migraphx.shaped<8x2xf8E4M3FN, 2x1>)
    -> !migraphx.shaped<8x4xf8E4M3FN, 4x1> {
  %y = migraphx.unpack %x {axis = 1 : i64}
       : <8x2xf8E4M3FN, 2x1> -> <8x4xf8E4M3FN, 4x1>
  return %y : !migraphx.shaped<8x4xf8E4M3FN, 4x1>
}

// -----

//===----------------------------------------------------------------------===//
// 2. Internal unpack only (unpack must NOT fold into the argument).
// We keep a live second use of the argument (%t) so the argument was
// originally multi-use. Expect:
//   - Input argument stays fp8
//   - Unpack rewritten to produce fp4
//   - First function result type updated to fp4
//   - Second result (from transpose) stays fp8
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @internal_only
// CHECK-SAME: (%[[ARG:.*]]: !migraphx.shaped<8x2xf8E4M3FN, 2x1>)
// CHECK-SAME: -> (!migraphx.shaped<8x4xf4E2M1FN, 4x1>, !migraphx.shaped<2x8xf8E4M3FN, 1x2>)
// CHECK: %[[T:.*]] = migraphx.transpose %[[ARG]] {permutation = [1, 0]}
// CHECK-SAME: : <8x2xf8E4M3FN, 2x1> -> <2x8xf8E4M3FN, 1x2>
// CHECK: %[[U:.*]] = migraphx.unpack %[[ARG]] {axis = 1
// CHECK-SAME: : <8x2xf8E4M3FN, 2x1> -> <8x4xf4E2M1FN, 4x1>
// CHECK: return %[[U]], %[[T]]
func.func @internal_only(%x: !migraphx.shaped<8x2xf8E4M3FN, 2x1>)
    -> (!migraphx.shaped<8x4xf8E4M3FN, 4x1>, !migraphx.shaped<2x8xf8E4M3FN, 1x2>) {
  %t = migraphx.transpose %x {permutation = [1, 0]}
       : <8x2xf8E4M3FN, 2x1> -> <2x8xf8E4M3FN, 1x2>
  %u = migraphx.unpack %x {axis = 1 : i64}
       : <8x2xf8E4M3FN, 2x1> -> <8x4xf8E4M3FN, 4x1>
  return %u, %t : !migraphx.shaped<8x4xf8E4M3FN, 4x1>, !migraphx.shaped<2x8xf8E4M3FN, 1x2>
}

// -----

//===----------------------------------------------------------------------===//
// 3. Transpose + unpack: unpack moved before transpose; result fp4.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @transpose_fp4(
// CHECK-SAME: %[[X:.*]]: !migraphx.shaped<9x4x8xf4E2M1FN, 32x1x4>
// CHECK-SAME: ) -> !migraphx.shaped<9x8x4xf4E2M1FN, 32x4x1>
// CHECK: %[[T:.*]] = migraphx.transpose %[[X]] {permutation = [0, 2, 1]}
// CHECK-SAME: : <9x4x8xf4E2M1FN, 32x1x4> -> <9x8x4xf4E2M1FN, 32x4x1>
// CHECK: return %[[T]] : !migraphx.shaped<9x8x4xf4E2M1FN, 32x4x1>
func.func @transpose_fp4(%x: !migraphx.shaped<9x2x8xf8E4M3FN, 16x1x2>)
    -> !migraphx.shaped<9x8x4xf8E4M3FN, 32x4x1> {
  %t = migraphx.transpose %x {permutation = [0, 2, 1]}
       : <9x2x8xf8E4M3FN, 16x1x2> -> <9x8x2xf8E4M3FN, 16x2x1>
  %u = migraphx.unpack %t {axis = 2 : i64}
       : <9x8x2xf8E4M3FN, 16x2x1> -> <9x8x4xf8E4M3FN, 32x4x1>
  return %u : !migraphx.shaped<9x8x4xf8E4M3FN, 32x4x1>
}

// -----

//===----------------------------------------------------------------------===//
// 4. Reshape + unpack (expand) -> unpack folded into arg, then reshape.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @reshape_expand_fp4(
// CHECK-SAME: %[[ARG:.*]]: !migraphx.shaped<9x16xf4E2M1FN, 16x1>
// CHECK-SAME: ) -> !migraphx.shaped<9x2x8xf4E2M1FN, 16x8x1>
// CHECK-NOT: migraphx.unpack
// CHECK: %[[R:.*]] = migraphx.reshape %[[ARG]] {dims = [9, 2, 8]}
// CHECK: return %[[R]] : !migraphx.shaped<9x2x8xf4E2M1FN, 16x8x1>
func.func @reshape_expand_fp4(%x: !migraphx.shaped<9x8xf8E4M3FN, 8x1>)
    -> !migraphx.shaped<9x2x8xf8E4M3FN, 16x8x1> {
  %r = migraphx.reshape %x {dims = [9, 2, 4]}
       : <9x8xf8E4M3FN, 8x1> -> <9x2x4xf8E4M3FN, 8x4x1>
  %u = migraphx.unpack %r {axis = 2 : i64}
       : <9x2x4xf8E4M3FN, 8x4x1> -> <9x2x8xf8E4M3FN, 16x8x1>
  return %u : !migraphx.shaped<9x2x8xf8E4M3FN, 16x8x1>
}

// -----

//===----------------------------------------------------------------------===//
// 5. Reshape + unpack (collapse) -> unpack folded into arg, then reshape.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @reshape_collapse_fp4(
// CHECK-SAME: %[[ARG:.*]]: !migraphx.shaped<9x2x8xf4E2M1FN, 16x8x1>
// CHECK-SAME: ) -> !migraphx.shaped<9x16xf4E2M1FN, 16x1>
// CHECK-NOT: migraphx.unpack
// CHECK: %[[R:.*]] = migraphx.reshape %[[ARG]] {dims = [9, 16]}
// CHECK: return %[[R]] : !migraphx.shaped<9x16xf4E2M1FN, 16x1>
func.func @reshape_collapse_fp4(%x: !migraphx.shaped<9x2x4xf8E4M3FN, 8x4x1>)
    -> !migraphx.shaped<9x16xf8E4M3FN, 16x1> {
  %r = migraphx.reshape %x {dims = [9, 8]}
       : <9x2x4xf8E4M3FN, 8x4x1> -> <9x8xf8E4M3FN, 8x1>
  %u = migraphx.unpack %r {axis = 1 : i64}
       : <9x8xf8E4M3FN, 8x1> -> <9x16xf8E4M3FN, 16x1>
  return %u : !migraphx.shaped<9x16xf8E4M3FN, 16x1>
}

// -----

//===----------------------------------------------------------------------===//
// 6. Multibroadcast + unpack -> unpack folded into arg, then broadcast.
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @multibroadcast_fp4(
// CHECK-SAME: %[[ARG:.*]]: !migraphx.shaped<1x8x1xf4E2M1FN, 2x1x2>
// CHECK-SAME: ) -> !migraphx.shaped<4x8x3xf4E2M1FN, 0x1x0>
// CHECK-NOT: migraphx.unpack
// CHECK: %[[B:.*]] = migraphx.multibroadcast %[[ARG]] {out_lens = [4, 8, 3]}
// CHECK: return %[[B]] : !migraphx.shaped<4x8x3xf4E2M1FN, 0x1x0>
func.func @multibroadcast_fp4(%x: !migraphx.shaped<1x4x1xf8E4M3FN, 1x1x1>)
    -> !migraphx.shaped<4x8x3xf8E4M3FN, 0x1x0> {
  %b = migraphx.multibroadcast %x {out_lens = [4, 4, 3]}
       : <1x4x1xf8E4M3FN, 1x1x1> -> <4x4x3xf8E4M3FN, 0x1x0>
  %u = migraphx.unpack %b {axis = 1 : i64}
       : <4x4x3xf8E4M3FN, 0x1x0> -> <4x8x3xf8E4M3FN, 0x1x0>
  return %u : !migraphx.shaped<4x8x3xf8E4M3FN, 0x1x0>
}