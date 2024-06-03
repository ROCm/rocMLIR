// RUN: rocmlir-driver -kernel-pipeline migraphx,highlevel %s | FileCheck %s

// CHECK-LABEL: @convNHWC
// CHECK-SAME: memref<100xf32>
// CHECK-SAME: memref<252xf32>
// CHECK-SAME: memref<63xf32>
func.func @convNHWC(%in: !migraphx.shaped<1x4x5x5xf32, 100x1x20x4>, %fil: !migraphx.shaped<7x4x3x3xf32, 36x1x12x4>) -> !migraphx.shaped<1x7x3x3xf32, 63x1x21x7> attributes {kernel, arch = "gfx1100", num_cu = 48 : i64} {
  // CHECK: rock.conv
  // CHECK-SAME: filter_layout = ["g", "k", "y", "x", "c"]
  // CHECK-SAME: input_layout = ["ni", "hi", "wi", "gi", "ci"]
  // CHECK-SAME: output_layout = ["no", "ho", "wo", "go", "ko"]
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x5x5xf32, 100x1x20x4>, <7x4x3x3xf32, 36x1x12x4> -> <1x7x3x3xf32, 63x1x21x7>
  func.return %out : !migraphx.shaped<1x7x3x3xf32, 63x1x21x7>
}
