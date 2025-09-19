// RUN: mlir-opt --test-system-device %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true for exact match (type, triple, chip, features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @exact_match
func.func @exact_match() {
  // Device: type=GPU, triple="amdgcn-amd-amdhsa", chip="gfx90a", features={xnack+:true, sramecc-:false}
  // Spec:   type=GPU, triple="amdgcn-amd-amdhsa", chip="gfx90a", features={xnack+:true, sramecc-:false}
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = ["xnack+", "sramecc-"],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = ["xnack+", "sramecc-"]
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if spec omits triple (wildcard triple)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @spec_no_triple
func.func @spec_no_triple() {
  // Device: triple="amdgcn-amd-amdhsa"
  // Spec:   triple=""
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = [],
    spec_type = 1, spec_triple = "", spec_chip = "gfx90a",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if device omits triple (wildcard triple)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @device_no_triple
func.func @device_no_triple() {
  // Device: triple=""
  // Spec:   triple="amdgcn-amd-amdhsa"
  "test.is_compatible"() {
    device_type = 1, triple = "", chip = "gfx90a",
    features = [],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns false if triples differ and both are set
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @triple_mismatch
func.func @triple_mismatch() {
  // Device: triple="amdgcn-amd-amdhsa"
  // Spec:   triple="amdgcn-amd-amdhsa2"
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = [],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa2", spec_chip = "gfx90a",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if spec omits chip (wildcard chip)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @spec_no_chip
func.func @spec_no_chip() {
  // Device: chip="gfx90a"
  // Spec:   chip=""
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = [],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns false if chips differ and spec sets chip
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @chip_mismatch
func.func @chip_mismatch() {
  // Device: chip="gfx90a"
  // Spec:   chip="gfx908"
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = [],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx908",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if spec omits features (wildcard features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @spec_no_features
func.func @spec_no_features() {
  // Device: features={xnack+:true}
  // Spec:   features={}
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = ["xnack+"],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if all spec features match device features
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @features_match
func.func @features_match() {
  // Device: features={xnack+:true, sramecc-:false}
  // Spec:   features={xnack+:true}
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = ["xnack+", "sramecc-"],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = ["xnack+"]
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns false if any spec feature does not match device
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @feature_mismatch
func.func @feature_mismatch() {
  // Device: features={xnack+:true, sramecc-:false}
  // Spec:   features={xnack-:false}
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = ["xnack+", "sramecc-"],
    spec_type = 1, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = ["xnack-"]
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns false if types differ
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @type_mismatch
func.func @type_mismatch() {
  // Device: type=GPU, Spec: type=CPU
  "test.is_compatible"() {
    device_type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a",
    features = [],
    spec_type = 0, spec_triple = "amdgcn-amd-amdhsa", spec_chip = "gfx90a",
    spec_features = []
  } : () -> i1
  return
}

//===----------------------------------------------------------------------===//
// Test: isCompatible returns true if both device and spec have empty triple/chip/features
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @all_empty
func.func @all_empty() {
  "test.is_compatible"() {
    device_type = 1, triple = "", chip = "",
    features = [],
    spec_type = 1, spec_triple = "", spec_chip = "",
    spec_features = []
  } : () -> i1
  return
}