// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly 
// * Can pass arguments in the right sequence
// * Have the right number of transforms
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: miopen-opt -miopen-affix-params -miopen-lowering %s | FileCheck %s

func @miopen_conv2d(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg0) {{{.*}}lower_layer_bounds = [1 : i32, 128 : i32, 8 : i32, 3 : i32, 3 : i32]{{.*}}upper_layer_bounds = [1 : i32, 72 : i32, 128 : i32]{{.*}}} : memref<1x128x8x3x3xf32> to memref<1x72x128xf32>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg1) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}upper_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}} : memref<128x1x8x32x32xf32> to memref<128x1x8x32x32xf32>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}upper_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 3 : i32, 30 : i32, 3 : i32, 30 : i32]{{.*}}} : memref<128x1x8x32x32xf32> to memref<128x1x8x3x30x3x30xf32>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 3 : i32, 30 : i32, 3 : i32, 30 : i32]{{.*}}upper_layer_bounds = [1 : i32, 72 : i32, 115200 : i32]{{.*}}} : memref<128x1x8x3x30x3x30xf32> to memref<1x72x115200xf32>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg2) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 128 : i32, 30 : i32, 30 : i32]{{.*}}upper_layer_bounds = [1 : i32, 128 : i32, 115200 : i32]{{.*}}} : memref<128x1x128x30x30xf32> to memref<1x128x115200xf32>
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  miopen.conv2d(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg0) {{{.*}}lower_layer_bounds = [1 : i32, 128 : i32, 8 : i32, 3 : i32, 3 : i32]{{.*}}upper_layer_bounds = [1 : i32, 72 : i32, 128 : i32]{{.*}}} : memref<1x128x8x3x3xf16> to memref<1x72x128xf16>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg1) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}upper_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}} : memref<128x1x8x32x32xf16> to memref<128x1x8x32x32xf16>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 32 : i32, 32 : i32]{{.*}}upper_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 3 : i32, 30 : i32, 3 : i32, 30 : i32]{{.*}}} : memref<128x1x8x32x32xf16> to memref<128x1x8x3x30x3x30xf16>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%{{.*}}) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 8 : i32, 3 : i32, 30 : i32, 3 : i32, 30 : i32]{{.*}}upper_layer_bounds = [1 : i32, 72 : i32, 115200 : i32]{{.*}}} : memref<128x1x8x3x30x3x30xf16> to memref<1x72x115200xf16>
// CHECK-NEXT:  %{{.*}} = miopen.transform(%arg2) {{{.*}}lower_layer_bounds = [128 : i32, 1 : i32, 128 : i32, 30 : i32, 30 : i32]{{.*}}upper_layer_bounds = [1 : i32, 128 : i32, 115200 : i32]{{.*}}} : memref<128x1x128x30x30xf16> to memref<1x128x115200xf16>
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data(%filter: memref<1x1024x1024x1x1xf32>, %input: memref<128x1x1024x14x14xf32>, %output: memref<128x1x1024x14x14xf32>) attributes {kernel = 0 : i32} {
miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "k", "c", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    num_cu = 120 : i32,
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [0 , 0 , 0 , 0],
    strides = [1 : i32, 1 : i32],
    xdlopsV2 = true
  } : memref<1x1024x1024x1x1xf32>, memref<128x1x1024x14x14xf32>, memref<128x1x1024x14x14xf32>
  return
}

// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}      
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* upper_layer_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}     
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}     
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}      
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}       
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_f16(%filter: memref<1x1024x1024x1x1xf16>, %input: memref<128x1x1024x14x14xf16>, %output: memref<128x1x1024x14x14xf16>) attributes {kernel = 0 : i32} {
miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx908",
    dilations = [1 : i32, 1 : i32],
    filter_layout = ["g", "k", "c", "y", "x"],
    gemm_id = 0 : i32,
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    num_cu = 120 : i32,
    output_layout = ["no", "go", "ko", "ho", "wo"],
    padding = [0 , 0 , 0 , 0],
    strides = [1 : i32, 1 : i32],
    xdlopsV2 = true
  } : memref<1x1024x1024x1x1xf16>, memref<128x1x1024x14x14xf16>, memref<128x1x1024x14x14xf16>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}      
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* upper_layer_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}     
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}     
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}      
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}       
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_padMN(%filter : memref<1x64x3x1x1xf32>, %input : memref<11x1x3x15x15xf32>, %output : memref<11x1x64x15x15xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
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

// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmMPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* upper_layer_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmMPad", "gemmNPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmNPad"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_padMK(%filter : memref<1x11x3x1x1xf32>, %input : memref<128x1x3x15x15xf32>, %output : memref<128x1x11x15x15xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
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

// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmMPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* upper_layer_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmMPad", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmN"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm




func @miopen_conv2d_bwd_weight(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmNPad"\].*}}
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmNPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmM", "gemmNPad"\].*}}
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmNPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight_padALL(%filter : memref<1x20x8x3x3xf32>, %input : memref<7x1x8x32x32xf32>, %output : memref<7x1x20x30x30xf32>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x20x8x3x3xf32>, memref<7x1x8x32x32xf32>, memref<7x1x20x30x30xf32>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight_padALL.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmMPad", "gemmNPad"\].*}}
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmNPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmMPad"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_weight_padALL_f16(%filter : memref<1x20x8x3x3xf16>, %input : memref<7x1x8x32x32xf16>, %output : memref<7x1x20x30x30xf16>) {
  miopen.conv2d_bwd_weight(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    filter_layout = ["g", "k", "c", "y", "x"],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    strides = [1, 1],
    padding = [0, 0, 0 ,0]
  } : memref<1x20x8x3x3xf16>, memref<7x1x8x32x32xf16>, memref<7x1x20x30x30xf16>
  return
}
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight_padALL_f16.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_weight
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* upper_layer_layout = \["gemmG", "gemmM", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmMPad", "gemmNPad"\].*}}
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmK", "gemmN"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmNPad"\].*}}
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* upper_layer_layout = \["gemmG", "gemmK", "gemmM"\].*}}
// CHECK-NEXT:  {{miopen.transform.* upper_layer_layout = \["gemmG", "gemmKPad", "gemmMPad"\].*}}
// CHECK-NEXT:  miopen.gridwise_gemm
