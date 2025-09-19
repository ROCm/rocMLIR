// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Valid: affine map matches upper/lower bounds, all bounds non-negative
//===----------------------------------------------------------------------===//
func.func @valid() {
  // CHECK: #rock.transform_map<PassThrough [] at [] -> [] at []>
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough [] at [] -> [] at [],
      affine_map<() -> ()>,
      dense<> : tensor<0xi64>,
      dense<> : tensor<0xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: 2D, affine map inputs/outputs match bounds, all bounds non-negative
//===----------------------------------------------------------------------===//
func.func @valid_2d() {
  // CHECK: #rock.transform_map<PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1]>,
  // CHECK-SAME: affine_map<(d0, d1) -> (d0, d1)>,
  // CHECK-SAME: dense<[4, 5]> : tensor<2xi64>,
  // CHECK-SAME: dense<[4, 5]> : tensor<2xi64>
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1],
      affine_map<(d0, d1) -> (d0, d1)>,
      dense<[4, 5]> : tensor<2xi64>,
      dense<[4, 5]> : tensor<2xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: affine map inputs != upper bounds size
//===----------------------------------------------------------------------===//
func.func @inputs_mismatch() {
  // expected-error @+1 {{Affine map has 1 inputs but there are 2 input dimensions}}
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1],
      affine_map<(d0) -> (d0, d0)>,
      dense<[4, 5]> : tensor<2xi64>,
      dense<[4, 5]> : tensor<2xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: affine map outputs != lower bounds size
//===----------------------------------------------------------------------===//
func.func @outputs_mismatch() {
  // expected-error @+1 {{Affine map has 1 outputs but there are 2 outut dimensions}}
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x","y"] at [0,1] -> ["a","b"] at [0,1],
      affine_map<(d0, d1) -> (d0)>,
      dense<[4, 5]> : tensor<2xi64>,
      dense<[4, 5]> : tensor<2xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: negative upper bound
//===----------------------------------------------------------------------===//
func.func @negative_upper_bound() {
  // expected-error @+1 {{Upper bound/shape component less than 0}}
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x"] at [0] -> ["a"] at [0],
      affine_map<(d0) -> (d0)>,
      dense<[-1]> : tensor<1xi64>,
      dense<[4]> : tensor<1xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Invalid: negative lower bound
//===----------------------------------------------------------------------===//
func.func @negative_lower_bound() {
  // expected-error @+1 {{Lower bound/shape component less than 0}}
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x"] at [0] -> ["a"] at [0],
      affine_map<(d0) -> (d0)>,
      dense<[4]> : tensor<1xi64>,
      dense<[-2]> : tensor<1xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: 0D (scalar) map
//===----------------------------------------------------------------------===//
func.func @scalar() {
  // CHECK: #rock.transform_map<PassThrough [] at [] -> [] at []>
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough [] at [] -> [] at [],
      affine_map<() -> ()>,
      dense<> : tensor<0xi64>,
      dense<> : tensor<0xi64>
    >
  } : () -> ()
}

//===----------------------------------------------------------------------===//
// Valid: 1D, upper/lower bounds zero
//===----------------------------------------------------------------------===//
func.func @zero_bounds() {
  // CHECK: #rock.transform_map<PassThrough ["x"] at [0] -> ["a"] at [0]>
  %0 = "test.use_attr"() {
    attr = #rock.transform_map<
      PassThrough ["x"] at [0] -> ["a"] at [0],
      affine_map<(d0) -> (d0)>,
      dense<[0]> : tensor<1xi64>,
      dense<[0]> : tensor<1xi64>
    >
  } : () -> ()
}