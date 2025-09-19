// RUN: mlir-opt --test-system-device %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: parse simple chip name (no triple, no features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_chip_only
func.func @parse_chip_only() {
  // Should parse chip as "gfx90a", no triple, no features
  "test.parse_system_device"() {arch = "gfx90a"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse triple:chip (no features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_triple_chip
func.func @parse_triple_chip() {
  // Should parse triple as "amdgcn-amd-amdhsa", chip as "gfx90a"
  "test.parse_system_device"() {arch = "amdgcn-amd-amdhsa:gfx90a"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse chip:feature+ (single feature enabled)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_chip_feature_plus
func.func @parse_chip_feature_plus() {
  // Should parse chip as "gfx90a", feature "xnack+" enabled
  "test.parse_system_device"() {arch = "gfx90a:xnack+"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse chip:feature- (single feature disabled)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_chip_feature_minus
func.func @parse_chip_feature_minus() {
  // Should parse chip as "gfx90a", feature "sramecc-" disabled
  "test.parse_system_device"() {arch = "gfx90a:sramecc-"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse chip:feature+:feature- (multiple features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_chip_multi_features
func.func @parse_chip_multi_features() {
  // Should parse chip as "gfx90a", features "xnack+" enabled, "sramecc-" disabled
  "test.parse_system_device"() {arch = "gfx90a:xnack+:sramecc-"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse triple:chip:feature+:feature- (triple, chip, multiple features)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_triple_chip_multi_features
func.func @parse_triple_chip_multi_features() {
  // Should parse triple as "amdgcn-amd-amdhsa", chip as "gfx90a", features "xnack+" enabled, "sramecc-" disabled
  "test.parse_system_device"() {arch = "amdgcn-amd-amdhsa:gfx90a:xnack+:sramecc-"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse chip with whitespace in features (should trim)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_chip_features_whitespace
func.func @parse_chip_features_whitespace() {
  // Should parse chip as "gfx90a", features "xnack+" enabled, "sramecc-" disabled (with whitespace trimmed)
  "test.parse_system_device"() {arch = "gfx90a: xnack+ : sramecc- "} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse only triple (should set chip empty)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_triple_only
func.func @parse_triple_only() {
  // Should parse triple as "amdgcn-amd-amdhsa", chip as ""
  "test.parse_system_device"() {arch = "amdgcn-amd-amdhsa:"} : () -> (!test.device)
  return
}

//===----------------------------------------------------------------------===//
// Test: parse empty string (should set all fields empty)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @parse_empty
func.func @parse_empty() {
  // Should parse all fields as empty
  "test.parse_system_device"() {arch = ""} : () -> (!test.device)
  return
}