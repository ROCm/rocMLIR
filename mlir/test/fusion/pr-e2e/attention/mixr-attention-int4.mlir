module {
  func.func private @mlir_attention(%9 : !migraphx.shaped<64x128xf16, 256x2>, %8 : !migraphx.shaped<64x128xf16, 256x2>, %arg0: !migraphx.shaped<64x64xf16, 64x1>, %arg1: !migraphx.shaped<64x32xui8, 32x1>, %arg2: !migraphx.shaped<64x64xf16, 64x1>) -> (!migraphx.shaped<64x64xf16, 64x1>) {
    %0 = migraphx.unpack %arg1 {axis = 1 : i64} : <64x32xui8, 32x1> -> <64x64xui8, 64x1>
    %10 = migraphx.dequantizelinear %0, %8, %9 : <64x64xui8, 64x1>, <64x128xf16, 256x2>, !migraphx.shaped<64x128xf16, 256x2> -> !migraphx.shaped<64x64xf16, 64x1>
    %x0 = migraphx.dot %arg0, %10: !migraphx.shaped<64x64xf16, 64x1>, !migraphx.shaped<64x64xf16, 64x1> -> !migraphx.shaped<64x64xf16, 64x1>
    %x1 = migraphx.softmax %x0{axis = 2 : i64} : !migraphx.shaped<64x64xf16, 64x1> -> !migraphx.shaped<64x64xf16, 64x1>
    %x2 = migraphx.dot %x1, %arg2: !migraphx.shaped<64x64xf16, 64x1>, !migraphx.shaped<64x64xf16, 64x1> -> !migraphx.shaped<64x64xf16, 64x1>
    return %x2 : !migraphx.shaped<64x64xf16, 64x1>
  }
}
