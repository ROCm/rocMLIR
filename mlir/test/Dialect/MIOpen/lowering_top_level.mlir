// This tests checks the following aspects of lowering component:
// * Can pass arguments correctly 
// * Can pass arguments in the right sequence
// * Have the right number of transforms
// * Have one gridwise_gemm
// * Can support F32 and F16

// RUN: mlir-opt -miopen-lowering %s | FileCheck %s

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
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
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
// CHECK-LABEL: func {{@miopen_conv2d.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
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
// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* output_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmK", "gemmM"\].*}}      
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* output_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}     
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmM", "gemmN"\].*}}     
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* output_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}      
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmK", "gemmN"\].*}}       
// CHECK-NEXT:  miopen.gridwise_gemm

func @miopen_conv2d_bwd_data_f16(%filter : memref<1x128x8x3x3xf16>, %input : memref<128x1x8x32x32xf16>, %output : memref<128x1x128x30x30xf16>) {
  miopen.conv2d_bwd_data(%filter, %input, %output) {
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
// CHECK-LABEL: func {{@miopen_conv2d_bwd_data.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  {{miopen.transform\(%arg0\).* output_layout = \["g", "k", "c", "ydot", "ytilda", "xdot", "xtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["g", "k", "c", "ydotslice", "ytildaslice", "xdotslice", "xtildaslice"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmK", "gemmM"\].*}}      
// CHECK-NEXT:  {{miopen.transform\(%arg1\).* output_layout = \["gi", "ni", "ci", "hipad", "wipad"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gi", "ni", "ci", "ytilda", "htilda", "xtilda", "wtilda"\].*}}    
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gi", "ni", "ci", "ytildaslice", "htildaslice", "xtildaslice", "wtildaslice"\].*}}     
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmM", "gemmN"\].*}}     
// CHECK-NEXT:  {{miopen.transform\(%arg2\).* output_layout = \["go", "no", "ko", "ydot", "htilda", "xdot", "wtilda"\].*}}
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["go", "no", "ko", "ydotslice", "htildaslice", "xdotslice", "wtildaslice"\].*}}      
// CHECK-NEXT:  {{miopen.transform.* output_layout = \["gemmG", "gemmK", "gemmN"\].*}}       
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
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
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
// CHECK-LABEL: func {{@miopen_conv2d_bwd_weight.*%arg0.*%arg1.*%arg2}}
// CHECK-NOT:   miopen.conv2d_bwd_data
// CHECK-NEXT:  miopen.transform(%arg0)
// CHECK-NEXT:  miopen.transform(%arg1)
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform
// CHECK-NEXT:  miopen.transform(%arg2)
// CHECK-NEXT:  miopen.gridwise_gemm
