// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// RUN: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

// CHECK-LABEL: func.func @migraphx_dot
// CHECK-NEXT: migraphx.dot 
func.func @migraphx_dot(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
  %0 = migraphx.dot %arg0, %arg1 : <1x16x512xf4E2M1FN, 8192x512x1>, <1x512x16xf4E2M1FN, 8192x16x1> -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// CHECK-LABEL: func.func @migraphx_quant_dot
// CHECK-NEXT: migraphx.quant_dot
func.func @migraphx_quant_dot(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
 %0 = migraphx.quant_dot
      %arg0,
      %arg1 
    : <1x16x512xf4E2M1FN, 8192x512x1>, 
      <1x512x16xf4E2M1FN, 8192x16x1>
    -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// CHECK-LABEL: func.func @migraphx_quant_dot_scaled
// CHECK-NEXT: migraphx.quant_dot
func.func @migraphx_quant_dot_scaled(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>, %arg2: !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>, %arg3: !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
 %0 = migraphx.quant_dot
      %arg0 scaled by %arg2,
      %arg1 scaled by %arg3
    : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
      !migraphx.shaped<1x16x512xf8E8M0FNU, 8192x512x1>,
      !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
      !migraphx.shaped<1x512x16xf8E8M0FNU, 8192x16x1>
    -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}
