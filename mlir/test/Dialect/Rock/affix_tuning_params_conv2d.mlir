// This tests checks the following aspects of the lowering:
// * convolution tuning parameters are set as expected
// If versions of these tests appear in lowering_top_level, then changes to the tuning
// parameters made here should be reflected in that file

// RUN: rock-opt -rock-affix-params %s | FileCheck %s

// CHECK-DAG: #[[$GENERAL_PARAMS_0:.*]] = #rock.general_gemm_params<kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, mThreadsPerCuwave = 4, nThreadsPerCuwave = 4, mCuwavesPerBlock = 4, nCuwavesPerBlock = 4>
// CHECK-DAG: #[[$GENERAL_PARAMS_1:.*]] = #rock.general_gemm_params<kPerBlock = 16, mPerBlock = 64, nPerBlock = 64, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, mThreadsPerCuwave = 4, nThreadsPerCuwave = 4, mCuwavesPerBlock = 2, nCuwavesPerBlock = 2>
// CHECK-DAG: #[[$XDLOPS_PARAMS_0:.*]] = #rock.xdlops_gemm_params<kPerBlock = 8, mPerBlock = 64, nPerBlock = 64, kpack = 1, mPerWave = 32, nPerWave = 32>
// CHECK-DAG: #[[$XDLOPS_PARAMS_1:.*]] = #rock.xdlops_gemm_params<kPerBlock = 4, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 64, nPerWave = 64>

// CHECK-LABEL: @rock_conv2d
func.func @rock_conv2d(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  // CHECK: rock.conv2d
  // CHECK-SAME: blockSize = 256
  // CHECK-SAME: gridSize = 900
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_0]]
  rock.conv2d(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_f16
func.func @rock_conv2d_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  // CHECK: rock.conv2d
  // CHECK-SAME: blockSize = 256
  // CHECK-SAME: gridSize = 900
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_0]]
  rock.conv2d(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_i8
func.func @rock_conv2d_i8(%filter : memref<1x128x8x3x3xi8>, %input : memref<128x1x8x32x32xi8>, %output : memref<128x1x128x30x30xi32>) {
  // CHECK: rock.conv2d
  // CHECK-SAME: blockSize = 256
  // CHECK-SAME: gridSize = 3600
  // CHECK-SAME: params = #[[$XDLOPS_PARAMS_0]]
  rock.conv2d(%filter, %input, %output) features = xdlops {
    arch = "gfx908",
    numCu = 120 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xi8>, memref<128x1x8x32x32xi8>, memref<128x1x128x30x30xi32>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_bwd_data
func.func @rock_conv2d_bwd_data(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
  // CHECK: rock.conv2d_bwd_data
  // CHECK-SAME: blockSize = 256
  // CHECK-SAME: gridSize = 1568
  // CHECK-SAME: params = #[[$XDLOPS_PARAMS_1]]
  rock.conv2d_bwd_data(%filter, %input, %output) features = xdlops {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "k", "c", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    numCu = 120 : i32,
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [0 , 0 , 0 , 0],
    strides = [1 : i32, 1 : i32]
  } : memref<1x1024x1024x1x1xf32>, memref<128x1x1024x14x14xf32>, memref<128x1x1024x14x14xf32>
  return
}

// CHECK-LABEL: @rock_conv2d_bwd_data_f16
func.func @rock_conv2d_bwd_data_f16(%filter: memref<1x1024x1024x1x1xf16>, %input: memref<128x1x1024x14x14xf16>, %output: memref<128x1x1024x14x14xf16>) attributes {kernel = 0 : i32} {
  // CHECK: rock.conv2d_bwd_data
  // CHECK-SAME: blockSize = 256
  // CHECK-SAME: gridSize = 1568
  // CHECK-SAME: params = #[[$XDLOPS_PARAMS_1]]
  rock.conv2d_bwd_data(%filter, %input, %output) features = xdlops {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "k", "c", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    numCu = 120 : i32,
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [0 , 0 , 0 , 0],
    strides = [1 : i32, 1 : i32]
  } : memref<1x1024x1024x1x1xf16>, memref<128x1x1024x14x14xf16>, memref<128x1x1024x14x14xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_bwd_data_padMN
func.func @rock_conv2d_bwd_data_padMN(%filter : memref<1x64x3x1x1xf32>, %input : memref<11x1x3x15x15xf32>, %output : memref<11x1x64x15x15xf32>) {
  // CHECK: rock.conv2d_bwd_data
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 39
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_data(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x64x3x1x1xf32>, memref<11x1x3x15x15xf32>, memref<11x1x64x15x15xf32>
  return
}

// CHECK-LABEL: @rock_conv2d_bwd_data_padMK
func.func @rock_conv2d_bwd_data_padMK(%filter : memref<1x11x3x1x1xf32>, %input : memref<128x1x3x15x15xf32>, %output : memref<128x1x11x15x15xf32>) {
  // CHECK: rock.conv2d_bwd_data
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 450
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_data(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x11x3x1x1xf32>, memref<128x1x3x15x15xf32>, memref<128x1x11x15x15xf32>
  return
}

// CHECK-LABEL: @rock_conv2d_bwd_weight
func.func @rock_conv2d_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  // CHECK: rock.conv2d_bwd_weight
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 4
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: @rock_conv2d_bwd_weight_f16
func.func @rock_conv2d_bwd_weight_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  // CHECK: rock.conv2d_bwd_weight
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 4
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv2d_bwd_weight_padALL
func.func @rock_conv2d_bwd_weight_padALL(%filter : memref<1x20x8x3x3xf32>, %input : memref<7x1x8x32x32xf32>, %output : memref<7x1x20x30x30xf32>) {
  // CHECK: rock.conv2d_bwd_weight
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 2
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x20x8x3x3xf32>, memref<7x1x8x32x32xf32>, memref<7x1x20x30x30xf32>
  return
}

// CHECK-LABEL: @rock_conv2d_bwd_weight_padALL_f16
func.func @rock_conv2d_bwd_weight_padALL_f16(%filter : memref<1x20x8x3x3xf16>, %input : memref<7x1x8x32x32xf16>, %output : memref<7x1x20x30x30xf16>) {
  // CHECK: rock.conv2d_bwd_weight
  // CHECK-SAME: blockSize = 64
  // CHECK-SAME: gridSize = 2
  // CHECK-SAME: params = #[[$GENERAL_PARAMS_1]]
  rock.conv2d_bwd_weight(%filter, %input, %output) features = none {
    arch = "gfx906",
    numCu = 64 : i32,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0],
    gemm_id = 0
  } : memref<1x20x8x3x3xf16>, memref<7x1x8x32x32xf16>, memref<7x1x20x30x30xf16>
  return
}
