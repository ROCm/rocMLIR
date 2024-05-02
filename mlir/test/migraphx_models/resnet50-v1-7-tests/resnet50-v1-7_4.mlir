// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -kernel-pipeline full  -arch gfx1100 --verify-passes | rocmlir-opt
module {
  func.func @mlir_reshape_dot_add(%arg0: !migraphx.shaped<1x1000xf32, 0x1>, %arg1: !migraphx.shaped<1x2048x1x1xf32, 2048x1x1x1>, %arg2: !migraphx.shaped<2048x1000xf32, 1000x1>) -> !migraphx.shaped<1x1000xf32, 1000x1> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 35 : i64} {
    %0 = migraphx.reshape %arg1 {dims = [1, 2048]} : <1x2048x1x1xf32, 2048x1x1x1> -> <1x2048xf32, 2048x1>
    %1 = migraphx.dot %0, %arg2 : <1x2048xf32, 2048x1>, <2048x1000xf32, 1000x1> -> <1x1000xf32, 1000x1>
    %2 = migraphx.add %1, %arg0 : <1x1000xf32, 1000x1>, <1x1000xf32, 0x1> -> <1x1000xf32, 1000x1>
    return %2 : !migraphx.shaped<1x1000xf32, 1000x1>
  }
}
