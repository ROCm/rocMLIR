// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// COM: Negative coverage for rock::TransformingForOp::parse and
// COM: rock::TransformingForOp::verify in
// COM: mlir/lib/Dialect/Rock/IR/RockDialect.cpp. Each section exercises a single
// COM: parser or verifier error branch of the rock.transforming_for op.

#unmerge = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

// COM: parse: no transforms but lower/upper arg counts differ
func.func @tfor_no_transform_arg_count() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected same number of lower and upper arguments when transforms absent}}
  rock.transforming_for (%a, %b) = [](%c0) (%v) = validity bounds [2] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: the transform list element is not a transform_map attribute
func.func @tfor_not_transform_map() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected transform map attributes}}
  rock.transforming_for (%a) = [0 : i32](%c0) (%v) = validity bounds [4] strides [1] {
    rock.yield
  }
  return
}

// -----

#unmerge0 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

// COM: parse: number of upper inits doesn't match the transform sequence inputs
func.func @tfor_wrong_num_inputs() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Transformation sequence expected 2 inputs}}
  rock.transforming_for (%l) = [#unmerge0](%c0) (%v) = validity bounds [16, 4] strides [1, 1] {
    rock.yield
  }
  return
}

// -----

#unmerge1 = #rock.transform_map<affine_map<(d0, d1) -> (d1 + 4 * d0)>
    by [<Unmerge{16, 4} ["1", "0"] at [0, 1] -> ["r"] at [0]>]
    bounds = [16, 4] -> [64]>

// COM: parse: number of lower coords doesn't match the transform sequence outputs
func.func @tfor_wrong_num_outputs() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Transformation sequence expected 1 outputs}}
  rock.transforming_for (%l0, %l1) = [#unmerge1](%c0, %c0) (%v) = validity bounds [16, 4] strides [1, 1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: number of validity arguments must equal the number of domains
func.func @tfor_wrong_num_validities() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Expected 1 validity arguments, one per domain, but found 2}}
  rock.transforming_for (%a) = [](%c0) (%v0, %v1) = validity bounds [4] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: parse: iter_args count must match the number of result types
func.func @tfor_iter_args_type_mismatch() {
  %c0 = arith.constant 0 : index
  %init = arith.constant 0.0 : f32
  // expected-error @+1 {{Mismatch between number of iter_args and types}}
  rock.transforming_for (%a) = [](%c0) (%v) = validity iter_args (%x = %init) -> (f32, f32) bounds [4] strides [1] {
    rock.yield %x : f32
  }
  return
}

// -----

// COM: verify: bounds and strides lists must have the same length
func.func @tfor_bounds_strides_length() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Bounds list and strides list must have same length}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [2, 3] strides [1] {
    rock.yield
  }
  return
}

// -----

// COM: verify: zero/negative strides are rejected
func.func @tfor_zero_stride() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{Negative and zero strides are not permitted}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [2, 3] strides [0, 1] {
    rock.yield
  }
  return
}

// -----

// COM: verify: each bound must evenly divide its stride
func.func @tfor_bound_not_divisible() {
  %c0 = arith.constant 0 : index
  // expected-error @+1 {{does not evenly divide the stride}}
  rock.transforming_for (%a, %b) = [](%c0, %c0) (%v) = validity bounds [3, 3] strides [2, 1] {
    rock.yield
  }
  return
}
