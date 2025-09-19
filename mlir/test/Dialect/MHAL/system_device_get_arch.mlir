// RUN: mlir-opt --test-system-device %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getArch returns chip only if only chip is set
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @chip_only
// CHECK: "test.get_arch"() {type = 1, chip = "gfx90a"} : () -> (str) // gfx90a
func.func @chip_only() {
  // Should return "gfx90a"
  "test.get_arch"() {type = 1, chip = "gfx90a"} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns triple:chip if both are set
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @triple_and_chip
// CHECK: "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a"} : () -> (str) // amdgcn-amd-amdhsa:gfx90a
func.func @triple_and_chip() {
  // Should return "amdgcn-amd-amdhsa:gfx90a"
  "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a"} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns chip:feature+ for single enabled feature
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @chip_feature_plus
// CHECK: "test.get_arch"() {type = 1, chip = "gfx90a", features = ["xnack+"]} : () -> (str) // gfx90a:xnack+
func.func @chip_feature_plus() {
  // Should return "gfx90a:xnack+"
  "test.get_arch"() {type = 1, chip = "gfx90a", features = ["xnack+"]} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns chip:feature- for single disabled feature
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @chip_feature_minus
// CHECK: "test.get_arch"() {type = 1, chip = "gfx90a", features = ["sramecc-"]} : () -> (str) // gfx90a:sramecc-
func.func @chip_feature_minus() {
  // Should return "gfx90a:sramecc-"
  "test.get_arch"() {type = 1, chip = "gfx90a", features = ["sramecc-"]} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns chip:feature+:feature- for multiple features
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @chip_multi_features
// CHECK: "test.get_arch"() {type = 1, chip = "gfx90a", features = ["xnack+", "sramecc-"]} : () -> (str) // gfx90a:xnack+:sramecc-
func.func @chip_multi_features() {
  // Should return "gfx90a:xnack+:sramecc-" (order may vary)
  "test.get_arch"() {type = 1, chip = "gfx90a", features = ["xnack+", "sramecc-"]} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns triple:chip:features for all fields
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @triple_chip_features
// CHECK: "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a", features = ["xnack+", "sramecc-"]} : () -> (str) // amdgcn-amd-amdhsa:gfx90a:xnack+:sramecc-
func.func @triple_chip_features() {
  // Should return "amdgcn-amd-amdhsa:gfx90a:xnack+:sramecc-" (order may vary)
  "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a", features = ["xnack+", "sramecc-"]} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns triple only if only triple is set (chip empty)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @triple_only
// CHECK: "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa"} : () -> (str) // amdgcn-amd-amdhsa:
func.func @triple_only() {
  // Should return "amdgcn-amd-amdhsa:"
  "test.get_arch"() {type = 1, triple = "amdgcn-amd-amdhsa"} : () -> (str)
  return
}

//===----------------------------------------------------------------------===//
// Test: getArch returns empty string if all fields are empty
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_empty
// CHECK: "test.get_arch"() {type = 1} : () -> (str) // ""
func.func @all_empty() {
  // Should return ""
  "test.get_arch"() {type = 1} : () -> (str)
  return
}