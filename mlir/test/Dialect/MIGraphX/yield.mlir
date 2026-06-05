// RUN: rocmlir-opt --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func.func @migraphx_yield_no_value
// CHECK: migraphx.yield
func.func @migraphx_yield_no_value() {
  "test.container"() ({
    migraphx.yield
  }) : () -> ()
  return
}

// CHECK-LABEL: func.func @migraphx_yield_with_value
// CHECK: migraphx.yield %{{.*}} : !migraphx.shaped<2x64x256xf16, 16384x256x1>
func.func @migraphx_yield_with_value(
    %arg0: !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
  "test.container"() ({
    migraphx.yield %arg0 : !migraphx.shaped<2x64x256xf16, 16384x256x1>
  }) : () -> ()
  return
}
