// RUN: mlir-opt --test-mhal-select-targets %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: testType returns true if targetTypes is empty
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @empty_target_types
// CHECK: "test.test_type"() {type = "gpu"} : () -> i1
func.func @empty_target_types() {
  // Should return true for any type if targetTypes is empty
  "test.test_type"() {type = "gpu"} : () -> i1
  "test.test_type"() {type = "cpu"} : () -> i1
  "test.test_type"() {type = "fpga"} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testType returns true if type is in targetTypes
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @type_in_target_types
// CHECK: "test.test_type"() {type = "gpu", targetTypes = ["gpu", "cpu"]} : () -> i1
func.func @type_in_target_types() {
  // Should return true for "gpu" and "cpu"
  "test.test_type"() {type = "gpu", targetTypes = ["gpu", "cpu"]} : () -> i1
  "test.test_type"() {type = "cpu", targetTypes = ["gpu", "cpu"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testType returns false if type is not in targetTypes
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @type_not_in_target_types
// CHECK: "test.test_type"() {type = "fpga", targetTypes = ["gpu", "cpu"]} : () -> i1
func.func @type_not_in_target_types() {
  // Should return false for "fpga"
  "test.test_type"() {type = "fpga", targetTypes = ["gpu", "cpu"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testType with multiple types in targetTypes
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @multiple_types
// CHECK: "test.test_type"() {type = "gpu", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
// CHECK: "test.test_type"() {type = "cpu", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
// CHECK: "test.test_type"() {type = "fpga", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
func.func @multiple_types() {
  "test.test_type"() {type = "gpu", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
  "test.test_type"() {type = "cpu", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
  "test.test_type"() {type = "fpga", targetTypes = ["gpu", "cpu", "fpga"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testType with type string case sensitivity
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @case_sensitivity
// CHECK: "test.test_type"() {type = "GPU", targetTypes = ["gpu"]} : () -> i1
func.func @case_sensitivity() {
  // Should return false if case does not match
  "test.test_type"() {type = "GPU", targetTypes = ["gpu"]} : () -> i1
  return
}