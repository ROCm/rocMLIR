// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly
// * Can pass arguments in the right sequence
// * Have, in most cases, the correct transformations
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: miopen-opt -miopen-affix-params -miopen-conv-to-gemm %s | FileCheck %s

// CHECK-DAG: #[[$MAP_FILTER_FWD:transform_map[0-9]+]] = #miopen.transform_map<{{.*}} bounds = [1, 72, 128] -> [1, 128, 8, 3, 3]>
// CHECK-DAG: #[[$MAP_INPUT1_FWD:transform_map[0-9]+]] = #miopen.transform_map<{{.*}} bounds = [128, 1, 8, 32, 32] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT2_FWD:transform_map[0-9]+]] = #miopen.transform_map<{{.*}} bounds = [128, 1, 8, 3, 30, 3, 30] -> [128, 1, 8, 32, 32]>
// CHECK-DAG: #[[$MAP_INPUT3_FWD:transform_map[0-9]+]] = #miopen.transform_map<{{.*}} bounds = [1, 72, 115200] -> [128, 1, 8, 3, 30, 3, 30]>
// CHECK-DAG: #[[$MAP_OUTPUT_FWD:transform_map[0-9]+]] = #miopen.transform_map<{{.*}} bounds = [1, 128, 115200] -> [128, 1, 128, 30, 30]>

// CHECK-DAG: #[[$MAP_BWD_DATA_FIL1_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Embed{1, 1} ["ydot", "ytilda"] at [3, 4] -> ["y"] at [3]>, <Embed{1, 1} ["xdot", "xtilda"] at [5, 6] -> ["x"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL2_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["g", "k", "c"] at [0, 1, 2] -> ["g", "k", "c"] at [0, 1, 2]>, <Slice{0, 1, 0, 1} ["ydotslice", "xdotslice"] at [3, 5] -> ["ydot", "xdot"] at [3, 5]>, <Slice{0, 1, 0, 1} ["ytildaslice", "xtildaslice"] at [4, 6] -> ["ytilda", "xtilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL3_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["k", "ydotslice", "xdotslice"] at [1, 3, 5]>, <Merge{1024, 1, 1} ["gemmM"] at [2] -> ["c", "ytildaslice", "xtildaslice"] at [2, 4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_FIL4_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [2] -> ["gemmM"] at [2]>, <Unmerge{256, 4} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>]

// CHECK-DAG: #[[$MAP_BWD_DATA_IN1_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Pad{0, 0, 0, 0} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>
// CHECK-DAG: #[[$MAP_BWD_DATA_IN2_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Embed{1, 1} ["ytilda", "htilda"] at [3, 4] -> ["hipad"] at [3]>, <Embed{1, 1} ["xtilda", "wtilda"] at [5, 6] -> ["wipad"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN3_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gi", "ni", "ci"] at [1, 0, 2] -> ["gi", "ni", "ci"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["yslice", "xslice"] at [3, 5] -> ["ytilda", "xtilda"] at [3, 5]>, <Slice{0, 14, 0, 14} ["hslice", "wslice"] at [4, 6] -> ["htilda", "wtilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN4_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{1024, 1, 1} ["gemmM"] at [1] -> ["ci", "yslice", "xslice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "hslice", "wslice"] at [0, 4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT1_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Embed{-1, 1} ["ydot", "htilda"] at [3, 4] -> ["ho"] at [3]>, <Embed{-1, 1} ["xdot", "wtilda"] at [5, 6] -> ["wo"] at [4]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT2_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["go", "no", "ko"] at [1, 0, 2] -> ["go", "no", "ko"] at [1, 0, 2]>, <Slice{0, 1, 0, 1} ["yslice", "xslice"] at [3, 5] -> ["ydot", "xdot"] at [3, 5]>, <Slice{0, 14, 0, 14} ["hslice", "wslice"] at [4, 6] -> ["htilda", "wtilda"] at [4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT3_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ko", "yslice", "xslice"] at [2, 3, 5]>, <Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "hslice", "wslice"] at [0, 4, 6]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT4_NO_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>, <Unmerge{256, 4} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>]

// CHECK-DAG: #[[$MAP_BWD_DATA_FIL_PAD_MN:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 61} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_IN_PAD_MN:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 61} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 21} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT_PAD_MN:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 21} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]

// CHECK-DAG: #[[$MAP_BWD_DATA_FIL_PAD_MK:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 5} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 61} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>] bounds = [1, 16, 64] -> [1, 11, 3]>
// CHECK-DAG: #[[$MAP_BWD_DATA_IN_PAD_MK:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 61} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>] bounds = [1, 64, 28800] -> [1, 3, 28800]>
// CHECK-DAG: #[[$MAP_BWD_DATA_OUT_PAD_MK:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 5} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>]

// CHECK-DAG: #[[$MAP_BWD_WEIGHT_FIL1:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["c", "y", "x"] at [2, 3, 4]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_FIL_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 56} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_IN3:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["ni", "ho", "wo"] at [0, 4, 6]>, <Merge{8, 3, 3} ["gemmN"] at [2] -> ["ci", "y", "x"] at [2, 3, 5]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_IN_PAD:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 56} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_OUT:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{128, 30, 30} ["gemmK"] at [1] -> ["no", "ho", "wo"] at [0, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [2]>]

// CHECK-DAG: #[[$MAP_BWD_WEIGHT_FIL_PAD_ALL:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 44} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 56} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_IN_PAD_ALL:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 4} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 56} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>]
// CHECK-DAG: #[[$MAP_BWD_WEIGHT_OUT_PAD_ALL:transform_map[0-9]+]] = {{.*}}by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 4} ["gemmKPad"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 44} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>]
func.func @miopen_conv2d(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  %[[FILTER:.*]] = miopen.transform %arg0 by [#[[$MAP_FILTER_FWD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_INPUT1_FWD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_INPUT2_FWD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_INPUT3_FWD]]]
// CHECK-NEXT:  %[[OUT:.*]] = miopen.transform %arg2 by [#[[$MAP_OUTPUT_FWD]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[OUT]] = %[[FILTER]] * %[[IN3]]

func.func @miopen_conv2d_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  miopen.conv2d(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  %[[FILTER:.*]] = miopen.transform %arg0 by [#[[$MAP_FILTER_FWD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_INPUT1_FWD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_INPUT2_FWD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_INPUT3_FWD]]]
// CHECK-NEXT:  %[[OUT:.*]] = miopen.transform %arg2 by [#[[$MAP_OUTPUT_FWD]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[OUT]] = %[[FILTER]] * %[[IN3]]

func.func @miopen_conv2d_i8(%filter : memref<1x128x8x3x3xi8>, %input : memref<128x1x8x32x32xi8>, %output : memref<128x1x128x30x30xi32>) {
  miopen.conv2d(%filter, %input, %output) features = xdlops {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_i8.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  %[[FILTER:.*]] = miopen.transform %arg0 by [#[[$MAP_FILTER_FWD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_INPUT1_FWD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_INPUT2_FWD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_INPUT3_FWD]]]
// CHECK-NEXT:  %[[OUT:.*]] = miopen.transform %arg2 by [#[[$MAP_OUTPUT_FWD]]]
// CHECK-NEXT:  miopen.gridwise_gemm_v2(%[[FILTER]], %[[IN3]], %[[OUT]])


func.func @miopen_conv2d_bwd_data(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
miopen.conv2d_bwd_data(%filter, %input, %output) features = xdlops {
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

// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0 by [#[[$MAP_BWD_DATA_FIL1_NO_PAD]]]
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_DATA_FIL2_NO_PAD]]]
// CHECK-NEXT:  %[[FIL3:.*]] = miopen.transform %[[FIL2]] by [#[[$MAP_BWD_DATA_FIL3_NO_PAD]]]
// CHECK-NEXT:  %[[FIL4:.*]] = miopen.transform %[[FIL3]] by [#[[$MAP_BWD_DATA_FIL4_NO_PAD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_BWD_DATA_IN1_NO_PAD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_BWD_DATA_IN2_NO_PAD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_BWD_DATA_IN3_NO_PAD]]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_DATA_IN4_NO_PAD]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2 by [#[[$MAP_BWD_DATA_OUT1_NO_PAD]]]
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]] by [#[[$MAP_BWD_DATA_OUT2_NO_PAD]]]
// CHECK-NEXT:  %[[OUT3:.*]] = miopen.transform %[[OUT2]] by [#[[$MAP_BWD_DATA_OUT3_NO_PAD]]]
// CHECK-NEXT:  %[[OUT4:.*]] = miopen.transform %[[OUT3]] by [#[[$MAP_BWD_DATA_OUT4_NO_PAD]]]
// CHECK-NEXT:  miopen.gridwise_gemm_v2(%[[FIL4]], %[[OUT4]], %[[IN4]]){{.*}}

func.func @miopen_conv2d_bwd_data_f16(%filter: memref<1x1024x1024x1x1xf16>, %input: memref<128x1x1024x14x14xf16>, %output: memref<128x1x1024x14x14xf16>) attributes {kernel = 0 : i32} {
miopen.conv2d_bwd_data(%filter, %input, %output) features = xdlops {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_data_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0 by [#[[$MAP_BWD_DATA_FIL1_NO_PAD]]]
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_DATA_FIL2_NO_PAD]]]
// CHECK-NEXT:  %[[FIL3:.*]] = miopen.transform %[[FIL2]] by [#[[$MAP_BWD_DATA_FIL3_NO_PAD]]]
// CHECK-NEXT:  %[[FIL4:.*]] = miopen.transform %[[FIL3]] by [#[[$MAP_BWD_DATA_FIL4_NO_PAD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_BWD_DATA_IN1_NO_PAD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_BWD_DATA_IN2_NO_PAD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_BWD_DATA_IN3_NO_PAD]]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_DATA_IN4_NO_PAD]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2 by [#[[$MAP_BWD_DATA_OUT1_NO_PAD]]]
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]] by [#[[$MAP_BWD_DATA_OUT2_NO_PAD]]]
// CHECK-NEXT:  %[[OUT3:.*]] = miopen.transform %[[OUT2]] by [#[[$MAP_BWD_DATA_OUT3_NO_PAD]]]
// CHECK-NEXT:  %[[OUT4:.*]] = miopen.transform %[[OUT3]] by [#[[$MAP_BWD_DATA_OUT4_NO_PAD]]]
// CHECK-NEXT:  miopen.gridwise_gemm_v2(%[[FIL4]], %[[OUT4]], %[[IN4]]){{.*}}

func.func @miopen_conv2d_bwd_data_padMN(%filter : memref<1x64x3x1x1xf32>, %input : memref<11x1x3x15x15xf32>, %output : memref<11x1x64x15x15xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) features = none {
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

// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_data_padMN.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]]
// CHECK-NEXT:  %[[FIL3:.*]] = miopen.transform %[[FIL2]]
// CHECK-NEXT:  %[[FIL4:.*]] = miopen.transform %[[FIL3]] by [#[[$MAP_BWD_DATA_FIL_PAD_MN]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]]
// CHECK-NEXT:  %[[IN5:.*]] = miopen.transform %[[IN4]] by [#[[$MAP_BWD_DATA_IN_PAD_MN]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]]
// CHECK-NEXT:  %[[OUT3:.*]] = miopen.transform %[[OUT2]]
// CHECK-NEXT:  %[[OUT4:.*]] = miopen.transform %[[OUT3]] by [#[[$MAP_BWD_DATA_OUT_PAD_MN]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[IN5]] = %[[FIL4]] * %[[OUT4]]{{.*}}

func.func @miopen_conv2d_bwd_data_padMK(%filter : memref<1x11x3x1x1xf32>, %input : memref<128x1x3x15x15xf32>, %output : memref<128x1x11x15x15xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) features = none {
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

// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_data_padMK.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]]
// CHECK-NEXT:  %[[FIL3:.*]] = miopen.transform %[[FIL2]]
// CHECK-NEXT:  %[[FIL4:.*]] = miopen.transform %[[FIL3]] by [#[[$MAP_BWD_DATA_FIL_PAD_MK]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]]
// CHECK-NEXT:  %[[IN5:.*]] = miopen.transform %[[IN4]] by [#[[$MAP_BWD_DATA_IN_PAD_MK]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]]
// CHECK-NEXT:  %[[OUT3:.*]] = miopen.transform %[[OUT2]]
// CHECK-NEXT:  %[[OUT4:.*]] = miopen.transform %[[OUT3]] by [#[[$MAP_BWD_DATA_OUT_PAD_MK]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[IN5]] = %[[FIL4]] * %[[OUT4]]{{.*}}

func.func @miopen_conv2d_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0 by [#[[$MAP_BWD_WEIGHT_FIL1]]]
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_WEIGHT_FIL_PAD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_INPUT1_FWD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_INPUT2_FWD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_BWD_WEIGHT_IN3]]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_WEIGHT_IN_PAD]]]
// CHECK-NEXT:  %[[OUT:.*]] = miopen.transform %arg2 by [#[[$MAP_BWD_WEIGHT_OUT]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[FIL2]] = %[[OUT]] * %[[IN4]]{{.*}}

func.func @miopen_conv2d_bwd_weight_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_weight_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0 by [#[[$MAP_BWD_WEIGHT_FIL1]]]
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_WEIGHT_FIL_PAD]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1 by [#[[$MAP_INPUT1_FWD]]]
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]] by [#[[$MAP_INPUT2_FWD]]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]] by [#[[$MAP_BWD_WEIGHT_IN3]]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_WEIGHT_IN_PAD]]]
// CHECK-NEXT:  %[[OUT:.*]] = miopen.transform %arg2 by [#[[$MAP_BWD_WEIGHT_OUT]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[FIL2]] = %[[OUT]] * %[[IN4]]{{.*}}

func.func @miopen_conv2d_bwd_weight_padALL(%filter : memref<1x20x8x3x3xf32>, %input : memref<7x1x8x32x32xf32>, %output : memref<7x1x20x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_weight_padALL.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_WEIGHT_FIL_PAD_ALL]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_WEIGHT_IN_PAD_ALL]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]] by [#[[$MAP_BWD_WEIGHT_OUT_PAD_ALL]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[FIL2]] = %[[OUT2]] * %[[IN4]]{{.*}}

func.func @miopen_conv2d_bwd_weight_padALL_f16(%filter : memref<1x20x8x3x3xf16>, %input : memref<7x1x8x32x32xf16>, %output : memref<7x1x20x30x30xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) features = none {
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
// CHECK-LABEL: func.func {{@miopen_conv2d_bwd_weight_padALL_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  %[[FIL1:.*]] = miopen.transform %arg0
// CHECK-NEXT:  %[[FIL2:.*]] = miopen.transform %[[FIL1]] by [#[[$MAP_BWD_WEIGHT_FIL_PAD_ALL]]]
// CHECK-NEXT:  %[[IN1:.*]] = miopen.transform %arg1
// CHECK-NEXT:  %[[IN2:.*]] = miopen.transform %[[IN1]]
// CHECK-NEXT:  %[[IN3:.*]] = miopen.transform %[[IN2]]
// CHECK-NEXT:  %[[IN4:.*]] = miopen.transform %[[IN3]] by [#[[$MAP_BWD_WEIGHT_IN_PAD_ALL]]]
// CHECK-NEXT:  %[[OUT1:.*]] = miopen.transform %arg2
// CHECK-NEXT:  %[[OUT2:.*]] = miopen.transform %[[OUT1]] by [#[[$MAP_BWD_WEIGHT_OUT_PAD_ALL]]]
// CHECK-NEXT:  miopen.gridwise_gemm %[[FIL2]] = %[[OUT2]] * %[[IN4]]{{.*}}
