// This tests checks the following aspects of lowering component:
// * Input tensor has non-zero padding.

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform %s | FileCheck %s --check-prefix=AFFINE

func @miopen_conv2d_cyxk_cnhw_knhw(%filter : memref<8x3x3x128xf16>, %input : memref<8x128x32x32xf16>, %output : memref<128x128x32x32xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["c", "y", "x", "k"],
    input_layout = ["ci", "ni", "hi", "wi"],
    output_layout = ["ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [1, 1]
  } : memref<8x3x3x128xf16>, memref<8x128x32x32xf16>, memref<128x128x32x32xf16>
  return
}

// LOWERING-LABEL: func @miopen_conv2d
// LOWERING:  miopen.transform(%arg1)
// LOWERING:  output_layout = ["ci", "ni", "hipad", "wipad"]
// LOWERING:  memref<8x128x34x34xf16>

// AFFINE: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3, d1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 - 1, d3 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d3 - 1, d4 + d5 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0 floordiv 9, d1 floordiv 1024, (d0 mod 9) floordiv 3 + (d1 mod 1024) floordiv 32 - 1, (d0 mod 9) mod 3 + (d1 mod 1024) mod 32 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1) -> (d0, d1 floordiv 1024, (d1 mod 1024) floordiv 32, (d1 mod 1024) mod 32)>
