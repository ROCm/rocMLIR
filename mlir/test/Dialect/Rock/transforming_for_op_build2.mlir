// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: 1D, no transforms, no iterArgs, default flags
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @one_dim_no_transform
// CHECK: rock.transforming_for
func.func @one_dim_no_transform(%init: index) {
  %for = "rock.transforming_for"(%init) ({
    // body
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [1]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: 2D, with transforms, no iterArgs, forceUnroll and useIndexDiffs true
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @two_dim_with_transform_flags
// CHECK: rock.transforming_for
func.func @two_dim_with_transform_flags(%init0: index, %init1: index) {
  %for = "rock.transforming_for"(%init0, %init1) ({
    // body
    "rock.yield"() : () -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>],
      [#rock.transform_map<PassThrough ["a"] at [0] -> ["b"] at [0], affine_map<(d0) -> (d0)>, dense<[20]> : tensor<1xi64>, dense<[20]> : tensor<1xi64>>]
    ],
    bounds = [10, 20],
    strides = [2, 4],
    forceUnroll,
    useIndexDiffs
  } : (index, index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: 2D, no transforms, with iterArgs
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @two_dim_no_transform_iterargs
// CHECK: rock.transforming_for
func.func @two_dim_no_transform_iterargs(%init0: index, %init1: index, %iter: i32) -> i32 {
  %for = "rock.transforming_for"(%init0, %init1) iter_args(%i = %iter : i32) -> (i32) ({
    // body
    "rock.yield"(%i) : (i32) -> ()
  }) {
    transforms = [ [], [] ],
    bounds = [5, 7],
    strides = [1, 1]
  } : (index, index, i32) -> (i32)
  return %for : i32
}

//===----------------------------------------------------------------------===//
// Test: 3D, mixed transforms, no flags, no iterArgs
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @three_dim_mixed_transform
// CHECK: rock.transforming_for
func.func @three_dim_mixed_transform(%i0: index, %i1: index, %i2: index) {
  %for = "rock.transforming_for"(%i0, %i1, %i2) ({
    // body
    "rock.yield"() : () -> ()
  }) {
    transforms = [
      [],
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[8]> : tensor<1xi64>, dense<[8]> : tensor<1xi64>>],
      []
    ],
    bounds = [8, 8, 8],
    strides = [1, 2, 4]
  } : (index, index, index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: 1D, with transform, with iterArgs, forceUnroll only
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @one_dim_transform_iterargs_forceunroll
// CHECK: rock.transforming_for
func.func @one_dim_transform_iterargs_forceunroll(%i: index, %iter: i64) -> i64 {
  %for = "rock.transforming_for"(%i) iter_args(%j = %iter : i64) -> (i64) ({
    // body
    "rock.yield"(%j) : (i64) -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[4]> : tensor<1xi64>, dense<[4]> : tensor<1xi64>>]
    ],
    bounds = [4],
    strides = [1],
    forceUnroll
  } : (index, i64) -> (i64)
  return %for : i64
}

//===----------------------------------------------------------------------===//
// Test: 2D, with transforms, with iterArgs, useIndexDiffs only
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @two_dim_transform_iterargs_indexdiffs
// CHECK: rock.transforming_for
func.func @two_dim_transform_iterargs_indexdiffs(%i0: index, %i1: index, %iter: f32) -> f32 {
  %for = "rock.transforming_for"(%i0, %i1) iter_args(%x = %iter : f32) -> (f32) ({
    // body
    "rock.yield"(%x) : (f32) -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[3]> : tensor<1xi64>, dense<[3]> : tensor<1xi64>>],
      [#rock.transform_map<PassThrough ["a"] at [0] -> ["b"] at [0], affine_map<(d0) -> (d0)>, dense<[6]> : tensor<1xi64>, dense<[6]> : tensor<1xi64>>]
    ],
    bounds = [3, 6],
    strides = [1, 2],
    useIndexDiffs
  } : (index, index, f32) -> (f32)
  return %for : f32
}