// RUN: rocmlir-opt -split-input-file --migraphx-to-linalg --linalg-to-rock -verify-diagnostics %s | FileCheck %s
// RUN: rocmlir-driver --kernel-pipeline=migraphx-linalg,highlevel-linalg %s | FileCheck %s --check-prefixes="DRIVER"

// Checking for driver as well
// DRIVER-LABEL: func.func @dot_one
// DRIVER:  rock.transform
// DRIVER:  rock.transform
// DRIVER:  rock.gemm

// CHECK-LABEL: func.func @dot_one(
// CHECK-SAME: %[[arg0:.*]]: tensor<6xf32>, 
// CHECK-SAME: %[[arg1:.*]]: tensor<6xf32>)
// CHECK-NEXT:      %[[expanded:.*]] = tensor.expand_shape %[[arg1]]
// CHECK-NEXT:      %[[expanded_0:.*]] = tensor.expand_shape %[[arg0]]
// CHECK-NEXT:      %[[cst:.*]] = arith.constant dense<0.000000e+00> : tensor<1x3x3xf32>
// CHECK-NEXT:      %[[zero:.*]] = bufferization.alloc_tensor() : tensor<1x3x3xf32>
// CHECK-NEXT:      %[[one:.*]] = rock.gemm %[[zero]] = %[[expanded_0]] * %[[expanded]] storeMethod = set
// CHECK-NEXT:      %[[collapsed:.*]] = tensor.collapse_shape %[[one]]
// CHECK-NEXT:      return %[[collapsed]]
func.func @dot_one(%arg0 : !migraphx.shaped<1x3x2xf32, 6x2x1>, %arg1: !migraphx.shaped<1x2x3xf32, 6x3x1>)
  -> !migraphx.shaped<1x3x3xf32, 9x3x1> attributes {kernel, arch="gfx950"}{
    %0 = migraphx.dot %arg0, %arg1 : <1x3x2xf32, 6x2x1>, <1x2x3xf32, 6x3x1> -> <1x3x3xf32, 9x3x1>
      func.return %0 : !migraphx.shaped<1x3x3xf32, 9x3x1>
}

// -----

// DRIVER-LABEL: func.func @dot_two
// DRIVER:  rock.transform
// DRIVER:  rock.transform
// DRIVER:  rock.gemm

// CHECK-LABEL: func.func @dot_two
// CHECK: rock.gemm
func.func @dot_two(%arg0 : !migraphx.shaped<3x2xf32, 2x1>, %arg1: !migraphx.shaped<2x3xf32, 3x1>)
  -> !migraphx.shaped<3x3xf32, 3x1> attributes {kernel, arch="gfx950"}{
    %0 = migraphx.dot %arg0, %arg1 : <3x2xf32, 2x1>, <2x3xf32, 3x1> -> <3x3xf32, 3x1>
      func.return %0 : !migraphx.shaped<3x3xf32, 3x1>
}
