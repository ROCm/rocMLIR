// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @expand_strides_basic
func.func @expand_strides_basic(%arg0: tensor<4x24x24xf16>) -> tensor<4x48x24xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<4x48x24xf16>
  // CHECK: %[[RESULT:.*]] = rock.expand_strides %arg0 into %[[ALLOC]] : tensor<4x24x24xf16> into tensor<4x48x24xf16> -> tensor<4x48x24xf16>
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "expand_strides"} : (tensor<4x24x24xf16>) -> tensor<4x48x24xf16>
  // CHECK: return %[[RESULT]] : tensor<4x48x24xf16>
  return %0 : tensor<4x48x24xf16>
}


// CHECK-LABEL: func.func @expand_strides_with_gemm
func.func @expand_strides_with_gemm(%arg0: tensor<4x24x16xf16>, %arg1: tensor<4x16x24xf16>) -> tensor<4x48x24xf16> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %cst = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  // CHECK: bufferization.alloc_tensor() : tensor<4x24x24xf16>
  // CHECK: rock.gemm
  %0 = tosa.matmul %arg0, %arg1, %cst, %cst {acc_type = f32} : (tensor<4x24x16xf16>, tensor<4x16x24xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<4x24x24xf16>
  %1 = tosa.sigmoid %0 : (tensor<4x24x24xf16>) -> tensor<4x24x24xf16>
  // CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<4x48x24xf16>
  // CHECK: rock.expand_strides %{{.*}} into %[[ALLOC]] : tensor<4x24x24xf16> into tensor<4x48x24xf16> -> tensor<4x48x24xf16>
  %2 = tosa.custom %1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "expand_strides"} : (tensor<4x24x24xf16>) -> tensor<4x48x24xf16>
  return %2 : tensor<4x48x24xf16>
}


// CHECK-LABEL: func.func @expand_strides_different_types
func.func @expand_strides_different_types(%arg0: tensor<2x8x8xf32>) -> tensor<2x16x16xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  // CHECK: %[[ALLOC:.*]] = bufferization.alloc_tensor() : tensor<2x16x16xf32>
  // CHECK: rock.expand_strides %arg0 into %[[ALLOC]] : tensor<2x8x8xf32> into tensor<2x16x16xf32> -> tensor<2x16x16xf32>
  %0 = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "expand_strides"} : (tensor<2x8x8xf32>) -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

