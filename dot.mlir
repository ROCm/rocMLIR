module {
  func.func @mlir_dot(%arg0: !migraphx.shaped<64x128xf32, 128x1>, %arg1: !migraphx.shaped<128x32xf32, 32x1>) -> !migraphx.shaped<64x32xf32, 32x1> attributes {arch = "gfx1200", kernel = "mixr", num_cu = 16 : i64} {
    %0 = migraphx.dot %arg0, %arg1 : <64x128xf32, 128x1>, <128x32xf32, 32x1> -> <64x32xf32, 32x1>
    return %0 : !migraphx.shaped<64x32xf32, 32x1>
  }
}