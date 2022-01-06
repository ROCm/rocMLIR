// This tests checks the following aspects of lowering component:
// * Input tensor has non-zero padding.
// * Memrefs get the correct affine map attached after transforms

// RUN: miopen-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s

// CHECK-DAG: #[[$AFFINE:map[0-9]+]] = #map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)>
// CHECK-DAG: #[[$MAP:transform_map[0-9]+]] = #transform_map1 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 1, d4 - 1)> by[#miopen.transform<PassThrough ["ni"] at [2] -> ["ni"] at [2]>, #miopen.transform<PassThrough ["gi"] at [0] -> ["gi"] at [0]>, #miopen.transform<PassThrough ["ci"] at [1] -> ["ci"] at [1]>, #miopen.transform<Pad{1, 1, 1, 1} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [1, 8, 128, 34, 34] -> [1, 8, 128, 32, 32]>
// CHECK-LABEL: func @miopen_conv2d_gcyxk_gcnhw_gknhw
// CHECK: miopen.transform %arg1 by [#[[MAP]]] : memref<1x8x128x32x32xf32> to memref<1x8x128x34x34xf32, #[[$AFFINE]]>
func @miopen_conv2d_gcyxk_gcnhw_gknhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x32x32xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [1, 1, 1, 1]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x32x32xf32>
  return
}
