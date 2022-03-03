module  {
  //func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x1024x1024x1x1xi8>, %arg1: memref<128x1x1024x14x14xi8>, %arg2: memref<128x1x1024x14x14xi8>) attributes {kernel = 0 : i32} {
  //  miopen.conv2d(%arg0, %arg1, %arg2) {arch = "gfx908", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32], xdlopsV2 = true} : memref<1x1024x1024x1x1xi8>, memref<128x1x1024x14x14xi8>, memref<128x1x1024x14x14xi8>
  //  return
  //}
  func @miopen_conv2d_i8(%filter : memref<1x128x8x3x3xi8>, %input : memref<128x1x8x32x32xi8>, %output : memref<128x1x128x30x30xi32>) attributes {kernel = 0 : i32} {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx908",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    xdlopsV2 = true
  } : memref<1x128x8x3x3xi8>, memref<128x1x8x32x32xi8>, memref<128x1x128x30x30xi32>
  return
}
}
