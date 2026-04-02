// RUN: rocmlir-opt --rock-affix-params --rock-conv-to-winograd --mlir-print-local-scope %s | FileCheck %s

// Eligible: 3x3, stride=1, dilation=1, f32, C*K=4096 >= 2048, gfx942
// CHECK-LABEL: @eligible_3x3_f32
// CHECK: rock.winograd_conv
// CHECK-SAME: fmr = 0
// CHECK-NOT: rock.conv
func.func @eligible_3x3_f32(%filter: memref<1x128x64x3x3xf32>, %input: memref<1x1x64x32x32xf32>, %output: memref<1x1x128x30x30xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x3x3xf32>, memref<1x1x64x32x32xf32>, memref<1x1x128x30x30xf32>
  return
}

// -----

// Eligible: f16, gfx942
// CHECK-LABEL: @eligible_3x3_f16
// CHECK: rock.winograd_conv
func.func @eligible_3x3_f16(%filter: memref<1x128x64x3x3xf16>, %input: memref<1x1x64x32x32xf16>, %output: memref<1x1x128x30x30xf16>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [1 : index, 1 : index, 1 : index, 1 : index]
  } : memref<1x128x64x3x3xf16>, memref<1x1x64x32x32xf16>, memref<1x1x128x30x30xf16>
  return
}

// -----

// NOT eligible: 5x5 filter
// CHECK-LABEL: @ineligible_5x5
// CHECK: rock.conv
// CHECK-NOT: rock.winograd_conv
func.func @ineligible_5x5(%filter: memref<1x128x64x5x5xf32>, %input: memref<1x1x64x32x32xf32>, %output: memref<1x1x128x28x28xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x5x5xf32>, memref<1x1x64x32x32xf32>, memref<1x1x128x28x28xf32>
  return
}

// -----

// NOT eligible: stride=2
// CHECK-LABEL: @ineligible_stride2
// CHECK: rock.conv
// CHECK-NOT: rock.winograd_conv
func.func @ineligible_stride2(%filter: memref<1x128x64x3x3xf32>, %input: memref<1x1x64x32x32xf32>, %output: memref<1x1x128x15x15xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [2 : index, 2 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x3x3xf32>, memref<1x1x64x32x32xf32>, memref<1x1x128x15x15xf32>
  return
}

// -----

// NOT eligible: dilation=2
// CHECK-LABEL: @ineligible_dilation2
// CHECK: rock.conv
// CHECK-NOT: rock.winograd_conv
func.func @ineligible_dilation2(%filter: memref<1x128x64x3x3xf32>, %input: memref<1x1x64x32x32xf32>, %output: memref<1x1x128x28x28xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [2 : index, 2 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x3x3xf32>, memref<1x1x64x32x32xf32>, memref<1x1x128x28x28xf32>
  return
}

// -----

// NOT eligible: 1x1 filter
// CHECK-LABEL: @ineligible_1x1
// CHECK: rock.conv
// CHECK-NOT: rock.winograd_conv
func.func @ineligible_1x1(%filter: memref<1x128x64x1x1xf32>, %input: memref<1x1x64x32x32xf32>, %output: memref<1x1x128x32x32xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x1x1xf32>, memref<1x1x64x32x32xf32>, memref<1x1x128x32x32xf32>
  return
}

// -----

// NOT eligible: C*K < 2048 (C=4, K=8, C*K=32)
// CHECK-LABEL: @ineligible_small_ck
// CHECK: rock.conv
// CHECK-NOT: rock.winograd_conv
func.func @ineligible_small_ck(%filter: memref<1x8x4x3x3xf32>, %input: memref<1x1x4x8x8xf32>, %output: memref<1x1x8x6x6xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  rock.conv(%filter, %input, %output) features = mfma {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x8x4x3x3xf32>, memref<1x1x4x8x8xf32>, memref<1x1x8x6x6xf32>
  return
}
