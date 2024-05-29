// This tests checks the following aspects of the lowering:
// * convolution tuning parameters are set as expected
// If versions of these tests appear in lowering_top_level, then changes to the tuning
// parameters made here should be reflected in that file

// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -verify-passes %s | FileCheck %s --check-prefix=CHECK
// RUN: rocmlir-driver -mlir-print-local-scope -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise %s | FileCheck %s --check-prefix=GRID

// CHECK-LABEL: @rock_conv
// GRID-LABEL: rock_conv
func.func @rock_conv(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 900
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: func.func @rock_conv_f16
// GRID-LABEL: func.func @rock_conv_f16
func.func @rock_conv_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  // CHECK: rock.conv
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 900
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv_i8
// GRID-LABEL: func.func @rock_conv_i8
func.func @rock_conv_i8(%filter : memref<1x128x8x3x3xi8>, %input : memref<128x1x8x32x32xi8>, %output : memref<128x1x128x30x30xi32>) {
  // CHECK: rock.conv
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 256, kpack = 4, mPerWave = 128, nPerWave = 64, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 450
  rock.conv(%filter, %input, %output) features = mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xi8>, memref<128x1x8x32x32xi8>, memref<128x1x128x30x30xi32>
  return
}

// CHECK-LABEL: func.func @rock_conv_bwd_data
// GRID-LABEL: func.func @rock_conv_bwd_data
func.func @rock_conv_bwd_data(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 2, mPerBlock = 256, nPerBlock = 256, kpack = 4, mPerWave = 128, nPerWave = 128, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 392
  rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index]
  } : memref<1x1024x1024x1x1xf32>, memref<128x1x1024x14x14xf32>, memref<128x1x1024x14x14xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_data_f16
// GRID-LABEL: @rock_conv_bwd_data_f16
func.func @rock_conv_bwd_data_f16(%filter: memref<1x1024x1024x1x1xf16>, %input: memref<128x1x1024x14x14xf16>, %output: memref<128x1x1024x14x14xf16>) attributes {kernel = 0 : i32} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 256, kpack = 4, mPerWave = 64, nPerWave = 128, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 784
  rock.conv_bwd_data(%filter, %input, %output) features = mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [1 : index, 1 : index]
  } : memref<1x1024x1024x1x1xf16>, memref<128x1x1024x14x14xf16>, memref<128x1x1024x14x14xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv_bwd_data_padMN
// GRID-LABEL: func.func @rock_conv_bwd_data_padMN
func.func @rock_conv_bwd_data_padMN(%filter : memref<1x64x3x1x1xf32>, %input : memref<11x1x3x15x15xf32>, %output : memref<11x1x64x15x15xf32>) {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 78
  rock.conv_bwd_data(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    kernelId = 0 : index
  } : memref<1x64x3x1x1xf32>, memref<11x1x3x15x15xf32>, memref<11x1x64x15x15xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_data_padMK
// GRID-LABEL: @rock_conv_bwd_data_padMK
func.func @rock_conv_bwd_data_padMK(%filter : memref<1x11x3x1x1xf32>, %input : memref<128x1x3x15x15xf32>, %output : memref<128x1x11x15x15xf32>) {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 64, kPerBlock = 4, mPerBlock = 32, nPerBlock = 64, kPerThread = 1, mPerThread = 2, nPerThread = 4, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 450
  rock.conv_bwd_data(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    kernelId = 0 : index
  } : memref<1x11x3x1x1xf32>, memref<128x1x3x15x15xf32>, memref<128x1x11x15x15xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_weight
// GRID-LABEL: @rock_conv_bwd_weight
func.func @rock_conv_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 12
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    numCU = 64 : i32,
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_weight_f16
// GRID-LABEL: @rock_conv_bwd_weight_f16
func.func @rock_conv_bwd_weight_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 128, kPerBlock = 16, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 12
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    numCU = 64 : i32,
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}

// CHECK-LABEL: func.func @rock_conv_bwd_weight_padALL
// GRID-LABEL: func.func @rock_conv_bwd_weight_padALL
func.func @rock_conv_bwd_weight_padALL(%filter : memref<1x20x8x3x3xf32>, %input : memref<7x1x8x32x32xf32>, %output : memref<7x1x20x30x30xf32>) {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 64, kPerBlock = 4, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 3
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    numCU = 64 : i32,
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x20x8x3x3xf32>, memref<7x1x8x32x32xf32>, memref<7x1x20x30x30xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_weight_padALL_f16
// GRID-LABEL: @rock_conv_bwd_weight_padALL_f16
func.func @rock_conv_bwd_weight_padALL_f16(%filter : memref<1x20x8x3x3xf16>, %input : memref<7x1x8x32x32xf16>, %output : memref<7x1x20x30x30xf16>) {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 64, kPerBlock = 4, mPerBlock = 32, nPerBlock = 32, kPerThread = 1, mPerThread = 2, nPerThread = 2, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 3
  rock.conv_bwd_weight(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    numCU = 64 : i32,
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x20x8x3x3xf16>, memref<7x1x8x32x32xf16>, memref<7x1x20x30x30xf16>
  return
}

// CHECK-LABEL: @rock_conv_7x7_tuning
// GRID-LABEL: @rock_conv_7x7_tuning
func.func @rock_conv_7x7_tuning(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>) {
  // CHECK: rock.conv
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 64, nPerBlock = 256, kpack = 1, mPerWave = 64, nPerWave = 64, mnPerXdl = 64, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 12544
  rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    // Restore this once the kPack + padding support works
    // perf_config = "v2:64,256,8,64,64,4,1,1,1",
    perf_config = "v2:64,256,8,64,64,1,1,1,2",
    strides = [2 : index, 2 : index]
  } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>
  return
}

// CHECK-LABEL: @rock_conv_7x7
// GRID-LABEL: @rock_conv_7x7
func.func @rock_conv_7x7(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>) {
  // CHECK: rock.conv
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 128, kpack = 1, mPerWave = 64, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 25088
  rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index]
  } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_weight_7x7
// GRID-LABEL: @rock_conv_bwd_weight_7x7
func.func @rock_conv_bwd_weight_7x7(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>) attributes {kernel = 0 : i32} {
  // CHECK: rock.conv_bwd_weight
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 32, kpack = 8, mPerWave = 32, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 5
  rock.conv_bwd_weight(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    numCU = 120 : i32,
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index]
  } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_data_7x7_tuning
// GRID-LABEL: @rock_conv_bwd_data_7x7_tuning
func.func @rock_conv_bwd_data_7x7_tuning(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>) attributes {kernel = 1 : i32} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 16, nPerBlock = 128, kpack = 4, mPerWave = 16, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 26450
  rock.conv_bwd_data(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 1 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    perf_config = "v2:16,128,8,16,16,4,1,1,1",
    strides = [2 : index, 2 : index]
  } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>
  return
}

// CHECK-LABEL: @rock_conv_bwd_data_7x7
// GRID-LABEL: @rock_conv_bwd_data_7x7
func.func @rock_conv_bwd_data_7x7(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>) attributes {kernel = 1 : i32} {
  // CHECK: rock.conv_bwd_data
  // CHECK-SAME: derivedBlockSize = 128
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 16, nPerBlock = 32, kpack = 4, mPerWave = 16, nPerWave = 16, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 105800
  rock.conv_bwd_data(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx908",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 1 : index,
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    strides = [2 : index, 2 : index]
  } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>
  return
}

// CHECK-LABEL: @rock_gemm_from_conv
// GRID-LABEL: @rock_gemm_from_conv
func.func @rock_gemm_from_conv(%a : memref<1x72x128xf32>, %b : memref<1x72x115200xf32>, %c : memref<1x128x115200xf32>) {
  // CHECK: rock.gemm
  // CHECK-SAME: params = #rock.general_gemm_params<blockSize = 256, kPerBlock = 8, mPerBlock = 128, nPerBlock = 128, kPerThread = 1, mPerThread = 4, nPerThread = 4, kpack = 1, splitKFactor = 1>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 900
  rock.gemm %c = tr %a * %b features = none storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx906",
    numCU = 64 : i32
  } : memref<1x128x115200xf32> = memref<1x72x128xf32> * memref<1x72x115200xf32>
  return
}

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv
// GRID-LABEL: func.func @rock_gemm_from_i8_conv
func.func @rock_gemm_from_i8_conv(%a : memref<1x72x128xi8>, %b : memref<1x72x115200xi8>, %c : memref<1x128x115200xi32>) {
  // CHECK: rock.gemm
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 128, nPerBlock = 256, kpack = 4, mPerWave = 128, nPerWave = 64, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 450
  rock.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx908",
    numCU = 120 : i32
  } : memref<1x128x115200xi32> = memref<1x72x128xi8> * memref<1x72x115200xi8>
  return
}

// The available xdlops for int8 change on gfx940, verify that different tuning
// parameters are picked.

// CHECK-LABEL: func.func @rock_gemm_from_i8_conv_gfx940
// GRID-LABEL: func.func @rock_gemm_from_i8_conv_gfx940
func.func @rock_gemm_from_i8_conv_gfx940(%a : memref<1x72x128xi8>, %b : memref<1x72x115200xi8>, %c : memref<1x128x115200xi32>) {
  // CHECK: rock.gemm
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 3600
  rock.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx940",
    numCU = 120 : i32
  } : memref<1x128x115200xi32> = memref<1x72x128xi8> * memref<1x72x115200xi8>
  return
}

// And verify that 8-bit floats have the same tuning behavior as i8.
// CHECK-LABEL: func.func @rock_gemm_xdlops_fp8_bf8
// GRID-LABEL: func.func @rock_gemm_xdlops_fp8_bf8
func.func @rock_gemm_xdlops_fp8_bf8(%a : memref<1x72x128xf8E4M3FNUZ>, %b : memref<1x72x115200xf8E5M2FNUZ>, %c : memref<1x128x115200xf32>) {
  // CHECK: rock.gemm
  // CHECK-SAME: derivedBlockSize = 256
  // CHECK-SAME: params = #rock.xdlops_gemm_derived_params<kpackPerBlock = 4, mPerBlock = 64, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, forceUnroll = true>
  // GRID: rock.gridwise_gemm
  // GRID-SAME: gridSize = 3600
  rock.gemm %c = tr %a * %b features = mfma|dot|atomic_add storeMethod = set {
    arch = "amdgcn-amd-amdhsa:gfx940",
    numCU = 120 : i32
  } : memref<1x128x115200xf32> = memref<1x72x128xf8E4M3FNUZ> * memref<1x72x115200xf8E5M2FNUZ>
  return
}

// CHECK-LABEL: func.func @rock_attention_default
// CHECK-SAME: block_size = 32
// GRID-LABEL: func.func @rock_attention_default
// GRID-SAME: grid_size = 12
func.func @rock_attention_default(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: #rock.wmma_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, splitKFactor = 1, forceUnroll = true>
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>}
  return
}

// CHECK-LABEL: func.func @rock_attention_large
// CHECK-SAME: block_size = 256
// GRID-LABEL: func.func @rock_attention_large
// GRID-SAME: grid_size = 128
func.func @rock_attention_large(%arg0: memref<1x16384x512xf32>, %arg1: memref<1x512x16384xf32>, %arg2: memref<1x16384x512xf32>, %arg3: memref<1x16384x512xf32>) {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x16384x512xf32>
  // CHECK: rock.attention
  // CHECK: params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 2, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 64, splitKFactor = 1, forceUnroll = true>
  // CHECK: params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 64, splitKFactor = 1, forceUnroll = true>
  rock.attention{
    qk = %arg0 * %arg1 : memref<1x16384x512xf32>, memref<1x512x16384xf32>
    %arg3 = softmax(qk) * %arg2 : memref<1x16384x512xf32> -> memref<1x16384x512xf32>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, perf_config = "attn:v1:128,128,128,2,64,64,8,1"}
  return
}

// CHECK-LABEL: func.func @rock_attention_mperblockg1
// CHECK-SAME: block_size = 128
// GRID-LABEL: func.func @rock_attention_mperblockg1
// GRID-SAME: grid_size = 3
func.func @rock_attention_mperblockg1_wmma(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: #rock.wmma_gemm_params<kpackPerBlock = 2, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, forceUnroll = true>
  // CHECK: #rock.wmma_gemm_params<kpackPerBlock = 16, mPerBlock = 256, nPerBlock = 128, kpack = 8, mPerWave = 128, nPerWave = 64, splitKFactor = 1, forceUnroll = true>
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "amdgcn-amd-amdhsa:gfx1100", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|wmma>, perf_config = "attn:v1:128,256,128,2,64,64,8,1"}
  return
}

// CHECK-LABEL: func.func @rock_attention_mperblockg1
// CHECK-SAME: block_size = 256
// GRID-LABEL: func.func @rock_attention_mperblockg1
// GRID-SAME: grid_size = 3
func.func @rock_attention_mperblockg1_mfma(%arg0: memref<1x384x64xf16>, %arg1: memref<1x384x64xf16>, %arg2: memref<1x384x64xf16>, %arg3: memref<1x384x64xf16>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK: rock.attention
  // CHECK: #rock.xdlops_gemm_derived_params<kpackPerBlock = 2, mPerBlock = 128, nPerBlock = 128, kpack = 8, mPerWave = 64, nPerWave = 64, mnPerXdl = 64, splitKFactor = 1, forceUnroll = true>
  // CHECK: #rock.xdlops_gemm_derived_params<kpackPerBlock = 16, mPerBlock = 256, nPerBlock = 128, kpack = 8, mPerWave = 128, nPerWave = 64, mnPerXdl = 64, splitKFactor = 1, forceUnroll = true>
  rock.attention{
   qk = %arg0 * tr %arg1 : memref<1x384x64xf16>, memref<1x384x64xf16>
   %arg3 = softmax(qk) * %arg2 : memref<1x384x64xf16> -> memref<1x384x64xf16>
  } {arch = "gfx942:sramecc+:xnack-", features = #rock<GemmFeatures mfma|dot|atomic_add>, perf_config = "attn:v1:128,256,128,2,64,64,8,1"}
  return
}

// CHECK-LABEL: func.func @rock_conv_tuning
// GRID-LABEL: func.func @rock_conv_tuning
func.func @rock_conv_tuning(%arg0: memref<1x1x1x3x3xf32>, %arg1: memref<64x1x1x14x14xf32>, %arg2: memref<64x1x1x14x14xf32>) attributes {kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  rock.conv(%arg0, %arg1, %arg2) features =  mfma|dot|atomic_add {
    arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-",
    dilations = [1 : index, 1 : index],
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    numCU = 110 : i32,
    output_layout = ["no", "go", "ko", "0o", "1o"],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    perf_config = "v2:32,128,4,32,32,4,1,1,1",
    strides = [1 : index, 1 : index]} : memref<1x1x1x3x3xf32>, memref<64x1x1x14x14xf32>, memref<64x1x1x14x14xf32>

  return
}
