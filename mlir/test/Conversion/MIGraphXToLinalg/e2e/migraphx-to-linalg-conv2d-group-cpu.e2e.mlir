
func.func @conv_2d_group(%in: !migraphx.shaped<2x4x123x124xf32, 61008x15252x124x1>, %fil: !migraphx.shaped<8x2x4x5xf32, >) -> !migraphx.shaped<2x8x27x19xf32, > {
  %out = migraphx.convolution %in, %fil {dilation = [2, 3], group = 2 : i64, padding = [2, 2, 2, 2], padding_mode = 0 : i64, stride = [4, 5]} : 
    <1x4x5x5xf32, 100x1x20x4>, <8x4x3x3xf32, 36x1x12x4> -> <1x8x3x3xf32, 63x1x21x7>
  func.return %out : !migraphx.shaped<1x8x3x3xf32, 63x1x21x7>
}
