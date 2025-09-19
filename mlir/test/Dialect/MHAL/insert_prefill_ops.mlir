// RUN: mlir-opt --mhal-prefill-pass %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: No prefill attributes, no memset inserted
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @no_prefill
func.func @no_prefill(%arg0: memref<4xf32>) {
  %cst = arith.constant 0 : index
  gpu.launch_func @kernel_module::@kernel_func blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xf32>)
  return
}

//===----------------------------------------------------------------------===//
// Test: Single prefill attribute, memset inserted before launch_func
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @single_prefill
// CHECK: arith.constant 1.000000e+00
// CHECK: gpu.memset
// CHECK: gpu.launch_func
module {
  gpu.binary @kernel_module { objects = [#gpu.object<{ properties = { kernel_func = [#mhal.prefill<arg_index = 0, init_value = 1.0 : f32>] } }>] }
  func.func @single_prefill(%arg0: memref<4xf32>) {
    %cst = arith.constant 0 : index
    gpu.launch_func @kernel_module::@kernel_func blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xf32>)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: Multiple prefill attributes, multiple memset inserted before launch_func
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @multi_prefill
// CHECK: arith.constant 2.000000e+00
// CHECK: gpu.memset
// CHECK: arith.constant 3.000000e+00
// CHECK: gpu.memset
// CHECK: gpu.launch_func
module {
  gpu.binary @kernel_module { objects = [#gpu.object<{ properties = { kernel_func = [
    #mhal.prefill<arg_index = 0, init_value = 2.0 : f32>,
    #mhal.prefill<arg_index = 1, init_value = 3.0 : f32>
  ] } }>] }
  func.func @multi_prefill(%arg0: memref<4xf32>, %arg1: memref<4xf32>) {
    %cst = arith.constant 0 : index
    gpu.launch_func @kernel_module::@kernel_func blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xf32>, %arg1 : memref<4xf32>)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: Prefill attribute with different element type (i32)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @prefill_i32
// CHECK: arith.constant 42
// CHECK: gpu.memset
// CHECK: gpu.launch_func
module {
  gpu.binary @kernel_module { objects = [#gpu.object<{ properties = { kernel_func = [#mhal.prefill<arg_index = 0, init_value = 42 : i32>] } }>] }
  func.func @prefill_i32(%arg0: memref<4xi32>) {
    %cst = arith.constant 0 : index
    gpu.launch_func @kernel_module::@kernel_func blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xi32>)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: Prefill attribute with multiple kernel functions, only correct one used
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @multi_kernel_func
// CHECK: arith.constant 7
// CHECK: gpu.memset
// CHECK: gpu.launch_func
module {
  gpu.binary @kernel_module { objects = [#gpu.object<{ properties = {
    kernel_func_a = [#mhal.prefill<arg_index = 0, init_value = 7 : i32>],
    kernel_func_b = [#mhal.prefill<arg_index = 0, init_value = 9 : i32>]
  } }>] }
  func.func @multi_kernel_func(%arg0: memref<4xi32>) {
    %cst = arith.constant 0 : index
    gpu.launch_func @kernel_module::@kernel_func_a blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xi32>)
    return
  }
}

//===----------------------------------------------------------------------===//
// Test: Prefill attribute with out-of-bounds arg_index (should assert/fail)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @out_of_bounds
module {
  gpu.binary @kernel_module { objects = [#gpu.object<{ properties = { kernel_func = [#mhal.prefill<arg_index = 2, init_value = 1.0 : f32>] } }>] }
  func.func @out_of_bounds(%arg0: memref<4xf32>) {
    %cst = arith.constant 0 : index
    // expected-remark@+1 {{provided arg index is out of bounds}}
    gpu.launch_func @kernel_module::@kernel_func blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%arg0 : memref<4xf32>)
    return
  }
}