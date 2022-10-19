// Check the guards of tensor layouts in RockOps
// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

func.func @rock_conv2d_gkcyx_ngchw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "gfx906"} {
  rock.conv2d(%arg0, %arg1, %arg2) features = dot, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {arch = "gfx906", filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"]} : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @rock_conv2d_gkycx_ngchw_ngkhw(%arg0: memref<1x128x3x8x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "gfx906"} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  rock.conv2d(%arg0, %arg1, %arg2) features = dot, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {arch = "gfx906", filter_layout = ["g", "k", "y", "c", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"]} : memref<1x128x3x8x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @rock_conv2d_gkcyx_gnhcw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<1x128x32x8x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "gfx906"} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  rock.conv2d(%arg0, %arg1, %arg2) features = dot, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {arch = "gfx906", filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["gi", "ni", "hi", "ci", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"]} : memref<1x128x8x3x3xf32>, memref<1x128x32x8x32xf32>, memref<128x1x128x30x30xf32>
  return
}
