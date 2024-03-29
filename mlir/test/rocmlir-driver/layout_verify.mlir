// Check the guards of tensor layouts in RockOps
// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

func.func @rock_conv_gkcyx_ngchw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
  rock.conv(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @rock_conv_gkycx_ngchw_ngkhw(%arg0: memref<1x128x3x8x3xf32>, %arg1: memref<128x1x8x32x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  rock.conv(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "c", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x128x3x8x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// -----

func.func @rock_conv_gkcyx_gnhcw_ngkhw(%arg0: memref<1x128x8x3x3xf32>, %arg1: memref<1x128x32x8x32xf32>, %arg2: memref<128x1x128x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
  // expected-error@+1 {{Disjointed yx or hw!}}
  rock.conv(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["gi", "ni", "0i", "ci", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x128x8x3x3xf32>, memref<1x128x32x8x32xf32>, memref<128x1x128x30x30xf32>
  return
}
