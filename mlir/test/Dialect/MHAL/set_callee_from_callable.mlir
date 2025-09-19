// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: setCalleeFromCallable sets the callee attribute to the given symbol
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @set_callee_basic
module {
  func.func @kernel_func(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }

  func.func @set_callee_basic(%dep: !mhal.token, %arg0: i32) {
    // Create a LaunchOp with an initial callee
    %launch = "mhal.launch"(%dep, %arg0) {callee = @kernel_func} : (!mhal.token, i32) -> (!mhal.token, i32)
    // CHECK: "test.set_callee_from_callable"(%launch, @kernel_func)
    "test.set_callee_from_callable"(%launch, @kernel_func) : (operation, symbol_ref) -> operation
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: setCalleeFromCallable changes the callee attribute to a new symbol
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @set_callee_change
module {
  func.func @kernel_func_a(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }
  func.func @kernel_func_b(%arg0: i32) -> i32 {
    %0 = arith.muli %arg0, %arg0 : i32
    return %0 : i32
  }

  func.func @set_callee_change(%dep: !mhal.token, %arg0: i32) {
    // Create a LaunchOp with an initial callee
    %launch = "mhal.launch"(%dep, %arg0) {callee = @kernel_func_a} : (!mhal.token, i32) -> (!mhal.token, i32)
    // Change the callee to kernel_func_b
    // CHECK: "test.set_callee_from_callable"(%launch, @kernel_func_b)
    "test.set_callee_from_callable"(%launch, @kernel_func_b) : (operation, symbol_ref) -> operation
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: setCalleeFromCallable with a missing symbol (should set attribute, but may fail verification)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @set_callee_missing_symbol
module {
  func.func @kernel_func(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }

  func.func @set_callee_missing_symbol(%dep: !mhal.token, %arg0: i32) {
    %launch = "mhal.launch"(%dep, %arg0) {callee = @kernel_func} : (!mhal.token, i32) -> (!mhal.token, i32)
    // Set to a symbol that does not exist in the module
    // CHECK: "test.set_callee_from_callable"(%launch, @missing_func)
    "test.set_callee_from_callable"(%launch, @missing_func) : (operation, symbol_ref) -> operation
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: setCalleeFromCallable with a symbol in a nested module
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @set_callee_nested_symbol
module {
  module @nested {
    func.func @kernel_func_nested(%arg0: i32) -> i32 {
      %0 = arith.addi %arg0, %arg0 : i32
      return %0 : i32
    }
  }

  func.func @set_callee_nested_symbol(%dep: !mhal.token, %arg0: i32) {
    // Use a nested symbol reference
    %launch = "mhal.launch"(%dep, %arg0) {callee = @nested::@kernel_func_nested} : (!mhal.token, i32) -> (!mhal.token, i32)
    // CHECK: "test.set_callee_from_callable"(%launch, @nested::@kernel_func_nested)
    "test.set_callee_from_callable"(%launch, @nested::@kernel_func_nested) : (operation, symbol_ref) -> operation
    return
  }
}