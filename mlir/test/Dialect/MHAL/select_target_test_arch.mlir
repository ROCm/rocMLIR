// RUN: mlir-opt --test-mhal-select-targets %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: testArch returns true if type matches and arch is compatible
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @arch_match
// CHECK: "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
func.func @arch_match() {
  // Should return true: type matches and arch is compatible
  "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns false if type does not match
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @type_not_match
// CHECK: "test.test_arch"() {type = "cpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
func.func @type_not_match() {
  // Should return false: type does not match
  "test.test_arch"() {type = "cpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns false if arch does not parse
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @arch_parse_fail
// CHECK: "test.test_arch"() {type = "gpu", arch = "not_a_real_arch", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
func.func @arch_parse_fail() {
  // Should return false: arch does not parse
  "test.test_arch"() {type = "gpu", arch = "not_a_real_arch", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns false if no compatible arch in targetArchs
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @arch_not_compatible
// CHECK: "test.test_arch"() {type = "gpu", arch = "gfx908", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
func.func @arch_not_compatible() {
  // Should return false: arch not compatible
  "test.test_arch"() {type = "gpu", arch = "gfx908", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns true if any compatible arch in targetArchs
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @arch_any_compatible
// CHECK: "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx908", "gfx90a"]} : () -> i1
func.func @arch_any_compatible() {
  // Should return true: arch matches one of the targetArchs
  "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx908", "gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns false if targetArchs is empty
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @archs_empty
// CHECK: "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = []} : () -> i1
func.func @archs_empty() {
  // Should return false: no targetArchs to match
  "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = []} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns false if testType fails (type not in targetTypes)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @test_type_fails
// CHECK: "test.test_arch"() {type = "fpga", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
func.func @test_type_fails() {
  // Should return false: type not in targetTypes
  "test.test_arch"() {type = "fpga", arch = "gfx90a", targetTypes = ["gpu"], targetArchs = ["gfx90a"]} : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: testArch returns true if targetTypes is empty and arch matches
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @target_types_empty
// CHECK: "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = [], targetArchs = ["gfx90a"]} : () -> i1
func.func @target_types_empty() {
  // Should return true: targetTypes empty, arch matches
  "test.test_arch"() {type = "gpu", arch = "gfx90a", targetTypes = [], targetArchs = ["gfx90a"]} : () -> i1
  return
}