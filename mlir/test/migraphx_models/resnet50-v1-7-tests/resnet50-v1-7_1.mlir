// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -kernel-pipeline full  -arch gfx1100 --verify-passes | rocmlir-opt
module {
  func.func @mlir_convolution_add_relu(%arg0: !migraphx.shaped<1x64x56x56xf32, 0x1x0x0>, %arg1: !migraphx.shaped<1x256x56x56xf32, 802816x3136x56x1>, %arg2: !migraphx.shaped<64x256x1x1xf32, 256x1x1x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 35 : i64} {
    %0 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x256x56x56xf32, 802816x3136x56x1>, <64x256x1x1xf32, 256x1x1x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    %1 = migraphx.add %0, %arg0 : <1x64x56x56xf32, 200704x3136x56x1>, <1x64x56x56xf32, 0x1x0x0> -> <1x64x56x56xf32, 200704x3136x56x1>
    %2 = migraphx.relu %1 : <1x64x56x56xf32, 200704x3136x56x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    return %2 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  }
}
