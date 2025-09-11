// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx,highlevel -targets %arch | rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-gridwise-gemm-to-blockwise -o /dev/null 2>&1 -debug-only=rock-gridwise-to-blockwise | FileCheck %s
// CHECK: elemTypeQLoad: f16
// CHECK: elemTypeKLoad: i4
// CHECK: elemTypeVLoad: f16
module {
  func.func private @mlir_attention(%arg0: !migraphx.shaped<64x64xf16, 128x1>, %arg1: !migraphx.shaped<64x64xf16, 128x1>, %arg2: !migraphx.shaped<64x64xf16, 64x1>, %arg3: !migraphx.shaped<64x32xui8, 32x1>, %arg4: !migraphx.shaped<64x64xf16, 64x1>) -> !migraphx.shaped<64x64xf16, 64x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.unpack %arg3 {axis = 1 : i64} : <64x32xui8, 32x1> -> <64x64xui8, 64x1>
    %1 = migraphx.reshape %arg1 {dims = [64, 128]} : <64x64xf16, 128x1> -> <64x64xf16, 256x2>
    %2 = migraphx.reshape %arg0 {dims = [64, 128]} : <64x64xf16, 128x1> -> <64x64xf16, 256x2>
    %3 = migraphx.dequantizelinear %0, %1, %2 : <64x64xui8, 64x1>, <64x64xf16, 256x2>, !migraphx.shaped<64x64xf16, 256x2> -> <64x64xf16, 64x1>
    %4 = migraphx.dot %arg2, %3 : <64x64xf16, 64x1>, <64x64xf16, 64x1> -> <64x64xf16, 64x1>
    %5 = migraphx.softmax %4 {axis = 1 : i64} : <64x64xf16, 64x1> -> <64x64xf16, 64x1>
    %6 = migraphx.dot %5, %arg4 : <64x64xf16, 64x1>, <64x64xf16, 64x1> -> <64x64xf16, 64x1>
    return %6 : !migraphx.shaped<64x64xf16, 64x1>
  }
}