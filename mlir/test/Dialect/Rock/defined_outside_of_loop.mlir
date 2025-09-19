// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: Value defined outside the loop (should be true)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @outside_loop
func.func @outside_loop(%init: index, %external: i32) {
  // %external is defined outside the loop
  "rock.transforming_for"(%init) ({
    // CHECK: "test.is_defined_outside_of_loop"(%external)
    "test.is_defined_outside_of_loop"(%external) : (i32) -> i1
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [10],
    strides = [1]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: Value defined inside the loop (should be false)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @inside_loop
func.func @inside_loop(%init: index) {
  "rock.transforming_for"(%init) ({
    %inside = arith.constant 42 : i32
    // CHECK: "test.is_defined_outside_of_loop"(%inside)
    "test.is_defined_outside_of_loop"(%inside) : (i32) -> i1
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [5],
    strides = [1]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: Loop-carried iter_arg (should be true, as it's defined outside)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @iter_arg
func.func @iter_arg(%init: index, %iter: i64) -> i64 {
  %for = "rock.transforming_for"(%init) iter_args(%i = %iter : i64) -> (i64) ({
    // CHECK: "test.is_defined_outside_of_loop"(%i)
    "test.is_defined_outside_of_loop"(%i) : (i64) -> i1
    "rock.yield"(%i) : (i64) -> ()
  }) {
    transforms = [ [] ],
    bounds = [7],
    strides = [1]
  } : (index, i64) -> (i64)
  return %for : i64
}

//===----------------------------------------------------------------------===//
// Test: Nested loop, value defined in outer loop (should be false in inner)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nested_loop_outer
func.func @nested_loop_outer(%init: index) {
  "rock.transforming_for"(%init) ({
    %outer = arith.constant 1 : i32
    "rock.transforming_for"(%init) ({
      // CHECK: "test.is_defined_outside_of_loop"(%outer)
      "test.is_defined_outside_of_loop"(%outer) : (i32) -> i1
      "rock.yield"() : () -> ()
    }) {
      transforms = [ [] ],
      bounds = [3],
      strides = [1]
    } : (index) -> ()
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [2],
    strides = [1]
  } : (index) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Test: Nested loop, value defined outside both loops (should be true in inner)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @nested_loop_external
func.func @nested_loop_external(%init: index, %external: i32) {
  "rock.transforming_for"(%init) ({
    "rock.transforming_for"(%init) ({
      // CHECK: "test.is_defined_outside_of_loop"(%external)
      "test.is_defined_outside_of_loop"(%external) : (i32) -> i1
      "rock.yield"() : () -> ()
    }) {
      transforms = [ [] ],
      bounds = [4],
      strides = [1]
    } : (index) -> ()
    "rock.yield"() : () -> ()
  }) {
    transforms = [ [] ],
    bounds = [2],
    strides = [1]
  } : (index) -> ()
  return
}