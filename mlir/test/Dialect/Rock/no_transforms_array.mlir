// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: noTransformsArray with n = 0 (should produce an empty array)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_transforms_n0
// CHECK:      %[[ARR:.*]] = "test.no_transforms_array"() {n = 0} : () -> array<array<none>>
// CHECK:      return %[[ARR]]
func.func @no_transforms_n0() -> array<array<none>> {
  %arr = "test.no_transforms_array"() {n = 0} : () -> array<array<none>>
  return %arr : array<array<none>>
}

//===----------------------------------------------------------------------===//
// Test: noTransformsArray with n = 1 (should produce array with one empty array)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_transforms_n1
// CHECK:      %[[ARR:.*]] = "test.no_transforms_array"() {n = 1} : () -> array<array<none>>
// CHECK:      return %[[ARR]]
func.func @no_transforms_n1() -> array<array<none>> {
  %arr = "test.no_transforms_array"() {n = 1} : () -> array<array<none>>
  return %arr : array<array<none>>
}

//===----------------------------------------------------------------------===//
// Test: noTransformsArray with n = 3 (should produce array of three empty arrays)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_transforms_n3
// CHECK:      %[[ARR:.*]] = "test.no_transforms_array"() {n = 3} : () -> array<array<none>>
// CHECK:      return %[[ARR]]
func.func @no_transforms_n3() -> array<array<none>> {
  %arr = "test.no_transforms_array"() {n = 3} : () -> array<array<none>>
  return %arr : array<array<none>>
}