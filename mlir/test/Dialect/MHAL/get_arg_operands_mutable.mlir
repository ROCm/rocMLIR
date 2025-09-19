// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getArgOperandsMutable returns only the launch operands (not dependencies)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @basic
module {
  func.func @kernel_func(%a: i32, %b: f32) -> i32 {
    %0 = arith.addi %a, %a : i32
    return %0 : i32
  }

  func.func @basic(%dep: !mhal.token, %a: i32, %b: f32) {
    // Create a LaunchOp with one dependency and two operands
    %launch = "mhal.launch"(%dep, %a, %b) {callee = @kernel_func} : (!mhal.token, i32, f32) -> (!mhal.token, i32)
    // CHECK: "test.get_arg_operands_mutable"(%launch)
    "test.get_arg_operands_mutable"(%launch) : (operation) -> (i32, f32)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: getArgOperandsMutable with multiple dependencies and operands
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @multi_deps
module {
  func.func @kernel_func(%a: i32, %b: f32, %c: i64) -> i32 {
    %0 = arith.addi %a, %a : i32
    return %0 : i32
  }

  func.func @multi_deps(%dep1: !mhal.token, %dep2: !mhal.token, %a: i32, %b: f32, %c: i64) {
    %launch = "mhal.launch"(%dep1, %dep2, %a, %b, %c) {callee = @kernel_func} : (!mhal.token, !mhal.token, i32, f32, i64) -> (!mhal.token, i32)
    // CHECK: "test.get_arg_operands_mutable"(%launch)
    "test.get_arg_operands_mutable"(%launch) : (operation) -> (i32, f32, i64)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: getArgOperandsMutable with no dependencies (all operands)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @no_deps
module {
  func.func @kernel_func(%a: i32, %b: f32) -> i32 {
    %0 = arith.addi %a, %a : i32
    return %0 : i32
  }

  func.func @no_deps(%a: i32, %b: f32) {
    %launch = "mhal.launch"(%a, %b) {callee = @kernel_func} : (i32, f32) -> (!mhal.token, i32)
    // CHECK: "test.get_arg_operands_mutable"(%launch)
    "test.get_arg_operands_mutable"(%launch) : (operation) -> (i32, f32)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: getArgOperandsMutable with only dependencies (no operands)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @only_deps
module {
  func.func @kernel_func() -> i32 {
    %0 = arith.constant 0 : i32
    return %0 : i32
  }

  func.func @only_deps(%dep: !mhal.token) {
    %launch = "mhal.launch"(%dep) {callee = @kernel_func} : (!mhal.token) -> (!mhal.token, i32)
    // CHECK: "test.get_arg_operands_mutable"(%launch)
    "test.get_arg_operands_mutable"(%launch) : (operation) -> ()
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: getArgOperandsMutable with mixed types and order
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @mixed_types
module {
  func.func @kernel_func(%a: f64, %b: i1, %c: i32) -> i32 {
    %0 = arith.addi %c, %c : i32
    return %0 : i32
  }

  func.func @mixed_types(%dep: !mhal.token, %a: f64, %b: i1, %c: i32) {
    %launch = "mhal.launch"(%dep, %a, %b, %c) {callee = @kernel_func} : (!mhal.token, f64, i1, i32) -> (!mhal.token, i32)
    // CHECK: "test.get_arg_operands_mutable"(%launch)
    "test.get_arg_operands_mutable"(%launch) : (operation) -> (f64, i1, i32)
    return
  }
}