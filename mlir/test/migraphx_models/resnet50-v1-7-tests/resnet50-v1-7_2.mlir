// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -kernel-pipeline full  -arch gfx1100 --verify-passes | rocmlir-opt
module {
  func.func @mlir_convolution_add_add_relu(%arg0: !migraphx.shaped<1x512x28x28xf32, 0x1x0x0>, %arg1: !migraphx.shaped<1x512x28x28xf32, 401408x784x28x1>, %arg2: !migraphx.shaped<1x128x28x28xf32, 100352x784x28x1>, %arg3: !migraphx.shaped<512x128x1x1xf32, 128x1x1x1>) -> !migraphx.shaped<1x512x28x28xf32, 401408x784x28x1> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 35 : i64} {
    %0 = migraphx.convolution %arg2, %arg3 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x128x28x28xf32, 100352x784x28x1>, <512x128x1x1xf32, 128x1x1x1> -> <1x512x28x28xf32, 401408x784x28x1>
    %1 = migraphx.add %0, %arg0 : <1x512x28x28xf32, 401408x784x28x1>, <1x512x28x28xf32, 0x1x0x0> -> <1x512x28x28xf32, 401408x784x28x1>
    %2 = migraphx.add %1, %arg1 : <1x512x28x28xf32, 401408x784x28x1>, <1x512x28x28xf32, 401408x784x28x1> -> <1x512x28x28xf32, 401408x784x28x1>
    %3 = migraphx.relu %2 : <1x512x28x28xf32, 401408x784x28x1> -> <1x512x28x28xf32, 401408x784x28x1>
    return %3 : !migraphx.shaped<1x512x28x28xf32, 401408x784x28x1>
  }
}
