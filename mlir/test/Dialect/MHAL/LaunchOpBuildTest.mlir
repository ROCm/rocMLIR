// Written by copilot for LaunchOp::build using the prompt "please generate a completely comprehensive test suite for LaunchOp::build in lit"

// RUN: mlir-opt %s -split-input-file | FileCheck %s

// CHECK-LABEL: func @launch_no_dependencies_no_operands
func.func @launch_no_dependencies_no_operands(%funcarg0: i32) -> (i32) {
  // Create a dummy callee function
  %0 = call @callee() : () -> (i32)
  // CHECK: "mhal.launch"
  "mhal.launch"() {callee = @callee, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> (!mhal.token, i32)
  return %0 : i32
}

// -----

// CHECK-LABEL: func @launch_with_dependencies_and_operands
func.func @launch_with_dependencies_and_operands(%dep: !mhal.token, %op1: i32, %op2: f32) -> (i32, f32) {
  // Create a dummy callee function
  %0:2 = call @callee2() : () -> (i32, f32)
  // CHECK: "mhal.launch"
  "mhal.launch"(%dep, %op1, %op2) {callee = @callee2, operand_segment_sizes = dense<[1, 2]> : vector<2xi32>} : (!mhal.token, i32, f32) -> (!mhal.token, i32, f32)
  return %0#0, %0#1 : i32, f32
}

// -----

// CHECK-LABEL: func @launch_only_dependencies
func.func @launch_only_dependencies(%dep1: !mhal.token, %dep2: !mhal.token) -> () {
  // Create a dummy callee function
  call @callee3() : () -> ()
  // CHECK: "mhal.launch"
  "mhal.launch"(%dep1, %dep2) {callee = @callee3, operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!mhal.token, !mhal.token) -> (!mhal.token)
  return
}

// -----

// CHECK-LABEL: func @launch_only_operands
func.func @launch_only_operands(%op1: i32, %op2: f32) -> (i32, f32) {
  // Create a dummy callee function
  %0:2 = call @callee4() : () -> (i32, f32)
  // CHECK: "mhal.launch"
  "mhal.launch"(%op1, %op2) {callee = @callee4, operand_segment_sizes = dense<[0, 2]> : vector<2xi32>} : (i32, f32) -> (!mhal.token, i32, f32)
  return %0#0, %0#1 : i32, f32
}

// -----

// CHECK-LABEL: func @launch_mixed_types
func.func @launch_mixed_types(%dep: !mhal.token, %op1: i32, %op2: f32, %op3: i64) -> (i32, f32, i64) {
  // Create a dummy callee function
  %0:3 = call @callee5() : () -> (i32, f32, i64)
  // CHECK: "mhal.launch"
  "mhal.launch"(%dep, %op1, %op2, %op3) {callee = @callee5, operand_segment_sizes = dense<[1, 3]> : vector<2xi32>} : (!mhal.token, i32, f32, i64) -> (!mhal.token, i32, f32, i64)
  return %0#0, %0#1, %0#2 : i32, f32, i64
}

// -----

// Dummy callee functions for symbol references
func.func private @callee() -> (i32)
func.func private @callee2() -> (i32, f32)
func.func private @callee3() -> ()
func.func private @callee4() -> (i32, f32)
func.func private @callee5() -> (i32, f32, i64)