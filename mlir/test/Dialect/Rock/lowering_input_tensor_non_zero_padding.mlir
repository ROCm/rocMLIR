// This tests checks the following aspects of lowering component:
// * Input tensor has non-zero padding.
// * Memrefs get the correct affine map attached after transforms

// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm %s | FileCheck %s

// CHECK-DAG: #[[$MAP:transform_map[0-9]*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["ni"] at [2] -> ["ni"] at [2]>, <PassThrough ["gi"] at [0] -> ["gi"] at [0]>, <PassThrough ["ci"] at [1] -> ["ci"] at [1]>, <Pad{1, 1, 1, 1} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [1, 8, 128, 34, 34] -> [1, 8, 128, 32, 32]>
// CHECK-LABEL: func.func @rock_conv_gcyxk_gcnhw_gknhw
// CHECK: rock.transform %arg1 by #[[$MAP]] : memref<1x8x128x32x32xf32> to memref<1x8x128x34x34xf32>
func.func @rock_conv_gcyxk_gcnhw_gknhw(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x32x32xf32>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "c", "y", "x", "k"],
    input_layout = ["gi", "ci", "ni", "hi", "wi"],
    output_layout = ["go", "ko", "no", "ho", "wo"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [1 : index,  1 : index,  1 : index,  1 : index]
  } : memref<1x8x3x3x128xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x32x32xf32>
  return
}
