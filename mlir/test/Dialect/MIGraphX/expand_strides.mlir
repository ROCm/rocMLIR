// RUN: rocmlir-opt %s -migraphx-transform | FileCheck %s

func.func @mlir_dot_sigmoid(%arg0: !migraphx.shaped<4x24x16xf16, 384x16x1>, %arg1: !migraphx.shaped<4x16x24xf16, 384x24x1>) -> !migraphx.shaped<4x24x24xf16, 1152x24x1> attributes {arch = "gfx950", kernel = "mixr"} {
  %0 = migraphx.dot %arg0, %arg1 : <4x24x16xf16, 384x16x1>, <4x16x24xf16, 384x24x1> -> <4x24x24xf16, 576x24x1>
  %1 = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  // CHECK: %[[SIGMOID:.*]] = migraphx.sigmoid %0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x24x1>
  // CHECK: migraphx.expand_strides %[[SIGMOID]] : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 1152x24x1>
  return %1 : !migraphx.shaped<4x24x24xf16, 1152x24x1>
}

// CHECK-LABEL: func.func @expand_multiple_strides
func.func @expand_multiple_strides(%arg0: !migraphx.shaped<4x24x24xf16, 576x24x1>) -> !migraphx.shaped<4x24x24xf16, 2304x48x1> attributes {arch = "gfx950", kernel = "mixr"} {
  %0 = migraphx.sigmoid %arg0 : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 2304x48x1>
  // CHECK: %[[SIGMOID:.*]] = migraphx.sigmoid {{.*}} : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 576x24x1>
  // CHECK: migraphx.expand_strides %[[SIGMOID]] : <4x24x24xf16, 576x24x1> -> <4x24x24xf16, 2304x48x1>
  return %0 : !migraphx.shaped<4x24x24xf16, 2304x48x1>
}
