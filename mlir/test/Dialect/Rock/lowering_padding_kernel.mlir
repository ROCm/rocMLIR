// This tests checks the following aspects of lowering component:
// * The correct padding transformations are generated and added to the gemm

// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise %s | FileCheck %s
// CHECK-DAG: #[[$PAD_GEMMK:.*]] = #rock.transform_map{{.*}}Pad{0, 2} ["gemmKPad"] at [1] -> ["gemmK"] at [1]

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_padding_kernel
// CHECK-SAME: %[[filter:.*]]: memref<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]]
// CHECK: %[[padK:.*]] = rock.transform %[[gemmFilter]] by #[[$PAD_GEMMK]]
// CHECK: rock.gridwise_gemm %{{.*}} = %[[padK]] * %{{.*}}
func.func @rock_conv_kcyx_nchw_nkhw_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<64x32x2x11x11xf32>, %output : memref<64x32x128x9x9xf32>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<32x128x2x3x3xf32>, memref<64x32x2x11x11xf32>, memref<64x32x128x9x9xf32>
  return
}

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_no_extra_padding
// CHECK-SAME: %[[filter:.*]]:  memref<1x128x64x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]]
// CHECK: rock.gridwise_gemm %{{.*}} = %[[gemmFilter]] * %{{.*}}
func.func @rock_conv_kcyx_nchw_nkhw_no_extra_padding(%filter : memref<1x128x64x3x3xf32>, %input : memref<128x1x64x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x64x3x3xf32>, memref<128x1x64x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}

// CHECK-LABEL: func.func @rock_conv_kcyx_nchw_nkhw_partial_padding_kernel
// CHECK-SAME: %[[filter:.*]]: memref<32x128x2x3x3xf32>
// CHECK: %[[gemmFilter:.*]] = rock.transform %[[filter]]
// CHECK: %[[padK:.*]] = rock.transform %[[gemmFilter]] by #[[$PAD_GEMMK]]
// CHECK: rock.gridwise_gemm %{{.*}} = %[[padK]] * %{{.*}}

func.func @rock_conv_kcyx_nchw_nkhw_partial_padding_kernel(%filter : memref<32x128x2x3x3xf32>, %input : memref<128x32x2x11x11xf32>, %output : memref<128x32x128x9x9xf32>) {
  rock.conv(%filter, %input, %output) features = none {
    arch = "amdgcn-amd-amdhsa:gfx906",
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni","gi", "ci", "hi", "wi"],
    output_layout = ["no", "go",  "ko", "ho", "wo"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<32x128x2x3x3xf32>, memref<128x32x2x11x11xf32>, memref<128x32x128x9x9xf32>
  return
}
