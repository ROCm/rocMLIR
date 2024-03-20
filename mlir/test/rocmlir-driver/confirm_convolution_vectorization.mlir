// RUN: rocmlir-driver -kernel-pipeline=gpu %s | FileCheck %s

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//  func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x32x48x3x3xf32>, %arg1: memref<64x1x48x14x14xf32>, %arg2: memref<64x1x32x12x12xf32>) attributes {kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
//    rock.conv2d(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCU = 104 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x32x48x3x3xf32>, memref<64x1x48x14x14xf32>, memref<64x1x32x12x12xf32>
//    return
//  }

  func.func @rock_conv2d_gkyxc_nhwgc_nhwgk_0(%arg0: memref<1x32x3x3x48xf32>, %arg1: memref<64x14x14x1x48xf32>, %arg2: memref<64x12x12x1x32xf32>) attributes {kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    rock.conv2d(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "hi", "wi", "gi", "ci"], numCU = 104 : i32, output_layout = ["no", "ho", "wo", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x32x3x3x48xf32>, memref<64x14x14x1x48xf32>, memref<64x12x12x1x32xf32>
    return
  }
}
