module {
func.func @mlir_slice_sigmoid_mul_convolution_broadcast_add_sigmoid_mul(%arg0: !migraphx.shaped<1x256x80x80xf16, 1638400x1x20480x256>, %arg1: !migraphx.shaped<128x128x3x3xf16, 1152x9x3x1>, %arg2: !migraphx.shaped<128xf16, 1>) -> !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128> attributes {arch = "gfx942:sramecc+:xnack-", kernel = "mixr", num_cu = 304 : i64} {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [256], starts = [128]} : <1x256x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 1638400x1x20480x256>
    %1 = migraphx.sigmoid %0 : <1x128x80x80xf16, 1638400x1x20480x256> -> <1x128x80x80xf16, 819200x1x10240x128>
    %2 = migraphx.mul %0, %1 : <1x128x80x80xf16, 1638400x1x20480x256>, <1x128x80x80xf16, 819200x1x10240x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [1, 1], perf_config = "v3:64,64,8,16,16,4,1,1,2,1,1"} : <1x128x80x80xf16, 819200x1x10240x128>, <128x128x3x3xf16, 1152x9x3x1> -> <1x128x80x80xf16, 819200x1x10240x128>
    %4 = migraphx.broadcast %arg2 {axis = 1 : i64, out_lens = [1, 128, 80, 80]} : <128xf16, 1> -> <1x128x80x80xf16, 0x1x0x0>
    %5 = migraphx.add %3, %4 : <1x128x80x80xf16, 819200x1x10240x128>, <1x128x80x80xf16, 0x1x0x0> -> <1x128x80x80xf16, 819200x1x10240x128>
    %6 = migraphx.sigmoid %5 : <1x128x80x80xf16, 819200x1x10240x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    %7 = migraphx.mul %5, %6 : <1x128x80x80xf16, 819200x1x10240x128>, <1x128x80x80xf16, 819200x1x10240x128> -> <1x128x80x80xf16, 819200x1x10240x128>
    return %7 : !migraphx.shaped<1x128x80x80xf16, 819200x1x10240x128>
  }
}

