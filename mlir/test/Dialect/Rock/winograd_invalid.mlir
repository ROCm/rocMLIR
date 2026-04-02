// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// -----

// Test: stride != 1 rejected
func.func @winograd_conv_bad_stride(
    %filter: memref<1x128x64x3x3xf16>,
    %input: memref<1x1x64x32x32xf16>,
    %output: memref<1x1x128x15x15xf16>
) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  // expected-error@+1 {{'rock.winograd_conv' op Winograd requires stride=1}}
  rock.winograd_conv(%filter, %input, %output) features = mfma {
    dilations = [1 : index, 1 : index],
    strides = [2 : index, 2 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    filterPreTransformed = false,
    fmr = 0 : i32
  } : memref<1x128x64x3x3xf16>, memref<1x1x64x32x32xf16>, memref<1x1x128x15x15xf16>
  return
}

// -----

// Test: dilation != 1 rejected
func.func @winograd_conv_bad_dilation(
    %filter: memref<1x128x64x3x3xf16>,
    %input: memref<1x1x64x32x32xf16>,
    %output: memref<1x1x128x28x28xf16>
) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  // expected-error@+1 {{'rock.winograd_conv' op Winograd requires dilation=1}}
  rock.winograd_conv(%filter, %input, %output) features = mfma {
    dilations = [2 : index, 2 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    filterPreTransformed = false,
    fmr = 0 : i32
  } : memref<1x128x64x3x3xf16>, memref<1x1x64x32x32xf16>, memref<1x1x128x28x28xf16>
  return
}

// -----

// Test: unsupported element type (bf16) rejected
func.func @winograd_conv_bad_dtype(
    %filter: memref<1x128x64x3x3xbf16>,
    %input: memref<1x1x64x32x32xbf16>,
    %output: memref<1x1x128x30x30xbf16>
) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  // expected-error@+1 {{op operand #0 must be}}
  rock.winograd_conv(%filter, %input, %output) features = mfma {
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [1 : index, 1 : index, 1 : index, 1 : index],
    filterPreTransformed = false,
    fmr = 0 : i32
  } : memref<1x128x64x3x3xbf16>, memref<1x1x64x32x32xbf16>, memref<1x1x128x30x30xbf16>
  return
}

// -----

// Test: F_4_3 with f16 rejected (condition number too high)
func.func @winograd_conv_f43_f16(
    %filter: memref<1x128x64x3x3xf16>,
    %input: memref<1x1x64x32x32xf16>,
    %output: memref<1x1x128x30x30xf16>
) attributes {arch = "amdgcn-amd-amdhsa:gfx942", kernel} {
  // expected-error@+1 {{'rock.winograd_conv' op Winograd F_4_3/F_2_5 requires f32}}
  rock.winograd_conv(%filter, %input, %output) features = mfma {
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index],
    filterPreTransformed = false,
    fmr = 1 : i32
  } : memref<1x128x64x3x3xf16>, memref<1x1x64x32x32xf16>, memref<1x1x128x30x30xf16>
  return
}
