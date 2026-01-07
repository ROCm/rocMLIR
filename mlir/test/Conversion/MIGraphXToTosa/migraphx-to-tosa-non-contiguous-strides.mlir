// RUN: rocmlir-opt --migraphx-to-tosa -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @expand_single_stride
func.func @expand_single_stride(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {arch = "gfx950", kernel = "mixr"} {
  %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
  %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x24x1>
  %2 = migraphx.expand_strides %1 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x48x24xf16>
  // CHECK: tensor.insert_slice {{.*}} into %[[EMPTY]][0, 0, 0] [4, 24, 24] [1, 1, 1] : tensor<4x24x24xf16> into tensor<4x48x24xf16>
  return %2 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
}

// CHECK-LABEL: func.func @expand_multiple_strides
func.func @expand_multiple_strides(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 2304x48x1> attributes {arch = "gfx950", kernel = "mixr"} {
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x24x1>
  %1 = migraphx.expand_strides %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 2304x48x1>
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<4x48x48xf16>
  // CHECK: tensor.insert_slice {{.*}} into %[[EMPTY]][0, 0, 0] [4, 24, 24] [1, 1, 1] : tensor<4x24x24xf16> into tensor<4x48x48xf16>
  return %1 : !migraphx.shaped<4x24x24xf16, 2304x48x1>
}
