// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: getCallResults returns all results except the leading mhal.token
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @single_result
module {
  func.func @kernel_func(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    return %0 : i32
  }

  func.func @single_result(%dep: !mhal.token, %arg0: i32) {
    %launch = "mhal.launch"(%dep, %arg0) {callee = @kernel_func} : (!mhal.token, i32) -> (!mhal.token, i32)
    // CHECK: "test.get_call_results"(%launch)
    "test.get_call_results"(%launch) : (operation) -> i32
    return
  }
}

// CHECK-LABEL: func @multi_result
module {
  func.func @kernel_func_multi(%a: i32, %b: f32) -> (i32, f32) {
    return %a, %b : i32, f32
  }

  func.func @multi_result(%dep: !mhal.token, %a: i32, %b: f32) {
    %launch = "mhal.launch"(%dep, %a, %b) {callee = @kernel_func_multi} : (!mhal.token, i32, f32) -> (!mhal.token, i32, f32)
    // CHECK: "test.get_call_results"(%launch)
    "test.get_call_results"(%launch) : (operation) -> (i32, f32)
    return
  }
}

// CHECK-LABEL: func @no_result
module {
  func.func @kernel_func_void(%a: i32) -> () {
    return
  }

  func.func @no_result(%dep: !mhal.token, %a: i32) {
    %launch = "mhal.launch"(%dep, %a) {callee = @kernel_func_void} : (!mhal.token, i32) -> (!mhal.token)
    // CHECK: "test.get_call_results"(%launch)
    "test.get_call_results"(%launch) : (operation) -> ()
    return
  }
}

// CHECK-LABEL: func @nested_launch
module {
  func.func @kernel_func(%a: i32) -> i32 {
    %0 = arith.addi %a, %a : i32
    return %0 : i32
  }

  func.func @nested_launch(%dep: !mhal.token, %a: i32) {
    %launch1 = "mhal.launch"(%dep, %a) {callee = @kernel_func} : (!mhal.token, i32) -> (!mhal.token, i32)
    %launch2 = "mhal.launch"(%launch1#0, %launch1#1) {callee = @kernel_func} : (!mhal.token, i32) -> (!mhal.token, i32)
    // CHECK: "test.get_call_results"(%launch2)
    "test.get_call_results"(%launch2) : (operation) -> i32
    return
  }
}