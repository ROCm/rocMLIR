// RUN: rocmlir-opt --migraphx-to-tosa %s | FileCheck %s

// CHECK-LABEL: func.func @mlir_dot_sigmoid
func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {kernel = "mixr"} {
  // CHECK: tosa.matmul
  // CHECK-SAME: -> tensor<4x24x24xf16>
  %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
  // CHECK: %[[SIGMOID:.*]] = tosa.sigmoid
  // CHECK-SAME: -> tensor<4x24x24xf16>
  %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x48x24xf16>
  // CHECK: tensor.insert_slice %[[SIGMOID]] into %[[EMPTY]]
  // CHECK-SAME: tensor<4x24x24xf16> into tensor<4x48x24xf16>
  // CHECK: tosa.reshape
  // CHECK-SAME: -> tensor<4608xf16>
  return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
}

