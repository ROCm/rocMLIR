// RUN: mlir-opt --mhal-select-targets="target-types=gpu target-archs=gfx90a" %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: No mhal.targets attribute, function unchanged
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_targets
func.func @no_targets() {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with one compatible target, attribute reduced to one
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @single_compatible
// CHECK: "mhal.targets" = [#mhal.kernel_package<type = "gpu", target = "gfx90a">]
func.func @single_compatible() attributes {mhal.targets = [#mhal.kernel_package<type = "gpu", target = "gfx90a">]} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with multiple targets, only compatible one kept
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @multi_targets_one_compatible
// CHECK: "mhal.targets" = [#mhal.kernel_package<type = "gpu", target = "gfx90a">]
func.func @multi_targets_one_compatible() attributes {
  mhal.targets = [
    #mhal.kernel_package<type = "gpu", target = "gfx908">,
    #mhal.kernel_package<type = "gpu", target = "gfx90a">,
    #mhal.kernel_package<type = "cpu", target = "x86_64">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with no compatible targets, attribute removed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_compatible
// CHECK-NOT: "mhal.targets"
// CHECK: func @no_compatible
func.func @no_compatible() attributes {
  mhal.targets = [
    #mhal.kernel_package<type = "gpu", target = "gfx908">,
    #mhal.kernel_package<type = "cpu", target = "x86_64">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with no compatible targets, error if targetArchs set and CPU not allowed
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_compatible_error
// expected-error@+1 {{target object not found}}
func.func @no_compatible_error() attributes {
  mhal.targets = [
    #mhal.kernel_package<type = "gpu", target = "gfx908">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with compatible CPU target if GPU not allowed
//===----------------------------------------------------------------------===//
// RUN: mlir-opt --mhal-select-targets="target-types=cpu target-archs=x86_64" %s | FileCheck %s --check-prefix=CPU
// CPU-LABEL: func @cpu_compatible
// CPU: "mhal.targets" = [#mhal.kernel_package<type = "cpu", target = "x86_64">]
func.func @cpu_compatible() attributes {
  mhal.targets = [
    #mhal.kernel_package<type = "gpu", target = "gfx90a">,
    #mhal.kernel_package<type = "cpu", target = "x86_64">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with multiple compatible targets, last one kept
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @multi_compatible_last_kept
// CHECK: "mhal.targets" = [#mhal.kernel_package<type = "gpu", target = "gfx90a">]
func.func @multi_compatible_last_kept() attributes {
  mhal.targets = [
    #mhal.kernel_package<type = "gpu", target = "gfx90a">,
    #mhal.kernel_package<type = "gpu", target = "gfx90a">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// Test: mhal.targets with empty attribute, nothing happens
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @empty_targets
// CHECK-NOT: "mhal.targets"
func.func @empty_targets() attributes {mhal.targets = []} {
  return
}