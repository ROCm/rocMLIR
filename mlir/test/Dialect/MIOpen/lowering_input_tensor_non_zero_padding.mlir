// This tests checks the following aspects of lowering component:
// * Input tensor has non-zero padding.

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform %s | FileCheck %s --check-prefix=AFFINE

func @miopen_conv2d_gcyxk_gcnhw_gknhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x32x32xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [1, 1]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x32x32xf32>
  return
}

// LOWERING-LABEL: func @miopen_conv2d
// LOWERING:  miopen.transform(%arg1)
// LOWERING:  output_layout = ["gi", "ci", "ni", "hipad", "wipad"]
// LOWERING:  memref<1x8x128x34x34xf32>

// AFFINE: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d1 mod 9) mod 3, d2)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3, d4) -> (0, 0, 0, 1, 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4 - 1, d5 + d6 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d0, d1 floordiv 9, d2 floordiv 1024, (d1 mod 9) floordiv 3 + (d2 mod 1024) floordiv 32 - 1, (d1 mod 9) mod 3 + (d2 mod 1024) mod 32 - 1)>
// AFFINE-NEXT: #map{{[0-9]+}} = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 1024, (d2 mod 1024) floordiv 32, (d2 mod 1024) mod 32)>
