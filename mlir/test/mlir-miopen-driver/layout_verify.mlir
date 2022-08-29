// Check the guards of tensor layouts in MIOpenOps
// RUN: miopen-opt %s -split-input-file -verify-diagnostics

func.func @miopen_conv2d_gkcyx_ngchw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel} {
  miopen.conv2d(%arg0, %arg1, %arg2) features = none {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 64 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @miopen_conv2d_gkycx_ngchw_ngkhw(%arg0: memref<1x128x3x8x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  miopen.conv2d(%arg0, %arg1, %arg2) features = none {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "y", "c", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 64 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x3x8x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @miopen_conv2d_gkcyx_gnhcw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<1x128x32x8x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  miopen.conv2d(%arg0, %arg1, %arg2) features = none {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["gi", "ni", "hi", "ci", "wi"], numCu = 64 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf32>, memref<1x128x32x8x32xf32>, memref<128x1x128x30x30xf32>
  return
}
