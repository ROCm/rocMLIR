// RUN: mlir-opt --test-system-device %s 2>&1 | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: dump prints type, count, triple, chip, features, and properties
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @dump_cpu
// CHECK: Device(CPU) x 1
// CHECK: triple =
// CHECK: chip = x86_64
func.func @dump_cpu() {
  // type = ECPU (0), chip = "x86_64", count = 1
  "test.dump_system_device"() {type = 0, chip = "x86_64", count = 1} : () -> ()
  return
}

// CHECK-LABEL: func @dump_gpu_triple_chip
// CHECK: Device(GPU) x 2
// CHECK: triple = amdgcn-amd-amdhsa
// CHECK: chip = gfx90a
func.func @dump_gpu_triple_chip() {
  // type = EGPU (1), triple = "amdgcn-amd-amdhsa", chip = "gfx90a", count = 2
  "test.dump_system_device"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a", count = 2} : () -> ()
  return
}

// CHECK-LABEL: func @dump_npu_features
// CHECK: Device(NPU) x 3
// CHECK: triple = npu-triple
// CHECK: chip = npu-chip
// CHECK: features = foo+:bar-:baz+
func.func @dump_npu_features() {
  // type = ENPU (2), triple = "npu-triple", chip = "npu-chip", features = ["foo+", "bar-", "baz+"], count = 3
  "test.dump_system_device"() {type = 2, triple = "npu-triple", chip = "npu-chip", features = ["foo+", "bar-", "baz+"], count = 3} : () -> ()
  return
}

// CHECK-LABEL: func @dump_alt_type
// CHECK: Device(ALT) x 1
// CHECK: triple =
// CHECK: chip = custom
func.func @dump_alt_type() {
  // type = 99 (unknown), chip = "custom", count = 1
  "test.dump_system_device"() {type = 99, chip = "custom", count = 1} : () -> ()
  return
}

// CHECK-LABEL: func @dump_with_properties
// CHECK: Device(GPU) x 1
// CHECK: triple = amdgcn-amd-amdhsa
// CHECK: chip = gfx90a
// CHECK: features = xnack+
// CHECK: {
// CHECK:     foo = 42
// CHECK:     bar = hello
// CHECK: }
func.func @dump_with_properties() {
  // type = EGPU (1), triple = "amdgcn-amd-amdhsa", chip = "gfx90a", features = ["xnack+"], count = 1, properties = {foo=42, bar="hello"}
  "test.dump_system_device"() {type = 1, triple = "amdgcn-amd-amdhsa", chip = "gfx90a", features = ["xnack+"], count = 1, properties = {foo = 42, bar = "hello"}} : () -> ()
  return
}

// CHECK-LABEL: func @dump_empty
// CHECK: Device(CPU) x 0
// CHECK: triple =
// CHECK: chip =
func.func @dump_empty() {
  // type = ECPU (0), count = 0, all other fields empty
  "test.dump_system_device"() {type = 0, count = 0} : () -> ()
  return
}