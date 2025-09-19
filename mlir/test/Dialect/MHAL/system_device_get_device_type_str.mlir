// RUN: mlir-opt --test-system-device %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getDeviceTypeStr returns correct string for each SystemDevice::Type
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @test_device_type_str
func.func @test_device_type_str() {
  // Should return "CPU"
  "test.get_device_type_str"() {type = 0} : () -> (str)
  // Should return "GPU"
  "test.get_device_type_str"() {type = 1} : () -> (str)
  // Should return "NPU"
  "test.get_device_type_str"() {type = 2} : () -> (str)
  // Should return "ALT" for unknown/other
  "test.get_device_type_str"() {type = 99} : () -> (str)
  return
}

// CHECK: "test.get_device_type_str"() {type = 0} : () -> (str) // CPU
// CHECK: "test.get_device_type_str"() {type = 1} : () -> (str) // GPU
// CHECK: "test.get_device_type_str"() {type = 2} : () -> (str) // NPU
// CHECK: "test.get_device_type_str"() {type = 99} : () -> (str) // ALT