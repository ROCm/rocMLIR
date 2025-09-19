// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: 1D, no transforms, stride divides bound, no iter_args
//===----------------------------------------------------------------------===//
func.func @valid_1d() {
  // CHECK: rock.transforming_for
  %init = arith.constant 0 : index
  "rock.transforming_for"(%init) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Valid: 2D, with transforms, stride divides bound, with iter_args
//===----------------------------------------------------------------------===//
func.func @valid_2d_transform_iterargs(%i: index, %j: index, %iter: i32) -> i32 {
  "rock.transforming_for"(%i, %j) iter_args(%k = %iter : i32) -> (i32) ({
    "rock.yield"(%k) : (i32) -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>],
      []
    ],
    bounds = [10, 20],
    strides = [2, 4]
  } : (index, index, i32) -> (i32)
  return %k : i32
}

//===----------------------------------------------------------------------===//
// Invalid: bounds and strides length mismatch
//===----------------------------------------------------------------------===//
func.func @bounds_strides_len_mismatch() {
  %i = arith.constant 0 : index
  // expected-error @+1 {{Bounds list and strides list must have same length}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10, 20],
    strides = [2]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: stride is zero
//===----------------------------------------------------------------------===//
func.func @stride_zero() {
  %i = arith.constant 0 : index
  // expected-error @+1 {{Negative and zero strides are not permitted}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [0]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: stride is negative
//===----------------------------------------------------------------------===//
func.func @stride_negative() {
  %i = arith.constant 0 : index
  // expected-error @+1 {{Negative and zero strides are not permitted}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [-2]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: bound not divisible by stride
//===----------------------------------------------------------------------===//
func.func @bound_not_divisible_by_stride() {
  %i = arith.constant 0 : index
  // expected-error @+1 {{Bound for dimension 0 (10) does not evenly divide the stride in that dimension (3}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [3]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: no iteration dimension (empty bounds)
//===----------------------------------------------------------------------===//
func.func @empty_bounds() {
  // expected-error @+1 {{Must have at least one iteration dimension}}
  "rock.transforming_for"() ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [],
    bounds = [],
    strides = []
  } : () -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: mismatch between number of yielded values and op results
//===----------------------------------------------------------------------===//
func.func @yielded_vs_results_mismatch(%i: index, %iter: i32) -> i32 {
  // expected-error @+1 {{Mismatch between number of yielded values and number of op results}}
  "rock.transforming_for"(%i) iter_args(%k = %iter : i32) -> (i32) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2]
  } : (index, i32) -> (i32)
  return %k : i32
}

//===----------------------------------------------------------------------===//
// Invalid: lowerStarts attribute doesn't have one entry per domain plus 2
//===----------------------------------------------------------------------===//
func.func @lower_starts_wrong_size(%i: index) {
  // expected-error @+1 {{Lower starts attribute doesn't have one entry per domain plus 2}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2],
    lower_starts = [0, 1] // should be 3 entries for 1 domain
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: validity domain doesn't contain one value per domain
//===----------------------------------------------------------------------===//
func.func @validity_domain_wrong_size(%i: index) {
  // expected-error @+1 {{Validity domain doesn't contain one value per domain}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2],
    lower_starts = [0, 1, 3] // last difference should be 1 for 1 domain
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: region args don't start with lower coords
//===----------------------------------------------------------------------===//
func.func @region_args_not_start_lower_coords(%i: index) {
  // expected-error @+1 {{Region args don't start with lower coords}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2],
    lower_starts = [1, 2, 3]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: mismatch between lower and upper coordinates without a transform
//===----------------------------------------------------------------------===//
func.func @mismatch_lower_upper_no_transform(%i: index, %j: index) {
  // expected-error @+1 {{Mismatch between number of lower and upper coordinates without a transform in domain #0}}
  "rock.transforming_for"(%i, %j) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [], [] ],
    bounds = [10, 20],
    strides = [2, 2]
  } : (index, index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: mismatch between upper initial values and inputs to transform sequence
//===----------------------------------------------------------------------===//
func.func @mismatch_upper_init_transform(%i: index) {
  // expected-error @+1 {{Mismatch between number of upper initial values and number of inputs to transform sequence in domain #0}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y"] at [0], affine_map<(d0) -> (d0)>, dense<[10]> : tensor<1xi64>, dense<[10]> : tensor<1xi64>>]
    ],
    bounds = [10],
    strides = [2]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: mismatch between lower arguments and outputs of transform sequence
//===----------------------------------------------------------------------===//
func.func @mismatch_lower_args_transform(%i: index) {
  // expected-error @+1 {{Mismatch between number of lower arguments and number of outputs of transform sequence in domain #0}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [
      [#rock.transform_map<PassThrough ["x"] at [0] -> ["y","z"] at [0,1], affine_map<(d0) -> (d0, d0)>, dense<[10]> : tensor<1xi64>, dense<[10, 10]> : tensor<2xi64>>]
    ],
    bounds = [10],
    strides = [2]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Invalid: lower starts attribute not accurate after domain
//===----------------------------------------------------------------------===//
func.func @lower_starts_not_accurate(%i: index) {
  // expected-error @+1 {{Lower starts attribute not accurate after domain #0}}
  "rock.transforming_for"(%i) ({
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [2],
    lower_starts = [0, 2, 2]
  } : (index) -> ()
  return
}