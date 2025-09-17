// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -targets %arch | rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-gridwise-gemm-to-blockwise -o /dev/null 2>&1 -debug-only=rock-gridwise-to-blockwise | FileCheck %s
// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: i4
// CHECK: elemTypeVLoad: f16
module {
  func.func private @mlir_attention_int4(%arg0: !migraphx.shaped<4096x4096xf16, 8192x1>, %arg1: !migraphx.shaped<4096x4096xf16, 8192x1>, %arg2: !migraphx.shaped<4096x4096xf16, 4096x1>, %arg3: !migraphx.shaped<4096x2048xui8, 2048x1>, %arg4: !migraphx.shaped<4096x4096xf16, 4096x1>) -> !migraphx.shaped<4096x4096xf16, 4096x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.unpack %arg3 {axis = 1 : i64} : <4096x2048xui8, 2048x1> -> <4096x4096xi8, 4096x1>
    %1 = migraphx.reshape %arg1 {dims = [64, 128]} : <4096x4096xf16, 8192x1> -> <4096x4096xf16, 16536x2>
    %2 = migraphx.reshape %arg0 {dims = [64, 128]} : <4096x4096xf16, 8192x1> -> <4096x4096xf16, 16536x2>
    %3 = migraphx.dequantizelinear %0, %1, %2 : <4096x4096xi8, 4096x1>, <4096x4096xf16, 16536x2>, !migraphx.shaped<4096x4096xf16, 16536x2> -> <4096x4096xf16, 4096x1>
    %4 = migraphx.dot %arg2, %3 : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %5 = migraphx.softmax %4 {axis = 1 : i64} : <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    %6 = migraphx.dot %5, %arg4 {perf_config = "attn:v2:64,128,128,128,16,16,1,1,1,2,1"} : <4096x4096xf16, 4096x1>, <4096x4096xf16, 4096x1> -> <4096x4096xf16, 4096x1>
    return %6 : !migraphx.shaped<4096x4096xf16, 4096x1>
  }
}
