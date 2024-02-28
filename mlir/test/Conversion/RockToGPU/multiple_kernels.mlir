// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gridwise-gemm-to-blockwise -rock-blockwise-gemm-to-threadwise -rock-threadwise-gemm-lowering -rock-sugar-to-loops -rock-clean-math -rock-buffer-load-merge -rock-loops-to-cf -convert-rock-to-gpu %s | rocmlir-opt

// The last kernel be converted would appear as the first.

// CHECK-NOT: func.func @step1
// CHECK-NOT: func.func @step2
// CHECK-NOT: func.func @step3
// CHECK-NOT: func.func @step4
// CHECK-LABEL: gpu.func @step4
// CHECK-LABEL: gpu.func @step3
// CHECK-LABEL: gpu.func @step2
// CHECK-LABEL: gpu.func @step1

module  {
  func.func @step1(%arg0: memref<1x512x16x3x3xf32>, %arg1: memref<512x1x16x32x32xf32>, %arg2: memref<512x1x512x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    rock.conv2d(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1, 1], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0, 0, 0, 0], strides = [1, 1]} : memref<1x512x16x3x3xf32>, memref<512x1x16x32x32xf32>, memref<512x1x512x30x30xf32>
    return
  }

  func.func @step2(%arg0: memref<1x64x16x3x3xf32>, %arg1: memref<64x1x16x32x32xf32>, %arg2: memref<64x1x64x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    rock.conv2d(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1, 1], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0, 0, 0, 0], strides = [1, 1]} : memref<1x64x16x3x3xf32>, memref<64x1x16x32x32xf32>, memref<64x1x64x30x30xf32>
    return
  }

  func.func @step3(%arg0: memref<1x32x16x3x3xf32>, %arg1: memref<32x1x16x32x32xf32>, %arg2: memref<32x1x32x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    rock.conv2d(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1, 1], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0, 0, 0, 0], strides = [1, 1]} : memref<1x32x16x3x3xf32>, memref<32x1x16x32x32xf32>, memref<32x1x32x30x30xf32>
    return
  }

  func.func @step4(%arg0: memref<1x32x8x3x3xf32>, %arg1: memref<32x1x8x32x32xf32>, %arg2: memref<32x1x32x30x30xf32>) attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx906"} {
    rock.conv2d(%arg0, %arg1, %arg2) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1, 1], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0, 0, 0, 0], strides = [1, 1]} : memref<1x32x8x3x3xf32>, memref<32x1x8x32x32xf32>, memref<32x1x32x30x30xf32>
    return
  }
}
