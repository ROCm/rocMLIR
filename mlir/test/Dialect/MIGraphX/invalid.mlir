// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

func.func @mlir_reshape_inconsistent_dims(%arg0: !migraphx.shaped<4096x4096xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op dimValue: 64 inconsistent with result dimension 4096}}
  %0 = migraphx.reshape %arg0 {dims = [64, 128]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
  return
}

func.func @mlir_reshape_dynamic_shape(%arg0: !migraphx.shaped<4096x?xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op Dynamic shapes are not supported}}
  %0 = migraphx.reshape %arg0 {dims = [4096, 4096]} : <4096x?xf16, 0x1> -> <4096x?xf16, 16536x2>
  return
}

func.func @mlir_reshape_rank(%arg0: !migraphx.shaped<4096x4096xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op number of dims (3) does not match result rank (2)}}
  %0 = migraphx.reshape %arg0 {dims = [1, 4096, 4096]} : <4096x4096xf16, 0x1> -> <4096x4096xf16, 16536x2>
  return
}

func.func @mlir_num_input_elements(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op input and output element counts do not match}}
  %0 = migraphx.reshape %arg0 {dims = [3, 5]} : <2x4xf16, 0x1> -> <3x5xf16, 0x1>
  return
}

func.func @mlir_element_type(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op failed to verify that all of {input, output} have same element type}}
  %0 = migraphx.reshape %arg0 {dims = [4, 2]} : <2x4xf16, 0x1> -> <4x2xf32, 0x1>
  return
}

func.func @mlir_multiple_neg_one(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op expected at most one target dimension to be -1}}
  %0 = migraphx.reshape %arg0 {dims = [-1, -1]} : <2x4xf16, 0x1> -> <4x2xf16, 0x1>
  return
}

func.func @mlir_neg_one_with_zero(%arg0: !migraphx.shaped<2x4xf16, 0x1>) {
  // expected-error@+1 {{'migraphx.reshape' op Cannot mix missing dimensions with zero dimension}}
  %0 = migraphx.reshape %arg0 {dims = [0, -1]} : <2x4xf16, 0x1> -> <4x2xf16, 0x1>
  return
}

func.func @func_equal(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
  %cst = migraphx.literal (dense<1> : tensor<1x36x384x64xi32>) : <1x36x384x64xi32, 884736x24576x64x1>
  %0 = migraphx.add %arg0, %cst : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
  // expected-error@+1 {{'migraphx.equal' op failed to verify that all of {inA, inB, output} have same element type}}
  %1 = migraphx.equal %arg0, %0 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi16, 884736x24576x64x1>
  return %1 : !migraphx.shaped<1x36x384x64xi16, 884736x24576x64x1>
}

// Test: Only scaleA provided (should fail - both scales required)
func.func @quant_dot_only_scale_a(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf32, 8192x512x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf32, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: Only scaleB provided (should fail - both scales required)
func.func @quant_dot_only_scale_b(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // expected-error @+1 {{both scaleA and scaleB must be provided or neither}}
  %0 = migraphx.quant_dot
       %arg0,
       %arg1 scaled by %arg2
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: Scaled quant_dot with non-f4E2M1FN inputs (should fail)
func.func @quant_dot_wrong_input_type(
  %arg0: !migraphx.shaped<2x32x64xi8, 2048x64x1>, 
  %arg1: !migraphx.shaped<2x64x32xi8, 2048x32x1>,
  %arg2: !migraphx.shaped<2x32x64xf32, 2048x64x1>,
  %arg3: !migraphx.shaped<2x64x32xf32, 2048x32x1>
) -> !migraphx.shaped<2x32x32xf32, 1024x32x1> {
  // expected-error @+1 {{'migraphx.quant_dot' op Scaled quant dot ops only support f4E2M1FN element type}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<2x32x64xi8, 2048x64x1> scaled by
       !migraphx.shaped<2x32x64xf32, 2048x64x1>,
       !migraphx.shaped<2x64x32xi8, 2048x32x1> scaled by
       !migraphx.shaped<2x64x32xf32, 2048x32x1>
     -> !migraphx.shaped<2x32x32xf32, 1024x32x1>
  return %0 : !migraphx.shaped<2x32x32xf32, 1024x32x1>
}

// -----

// Test: Scaled quant_dot with FP8 inputs (should fail - requires f4E2M1FN)
func.func @quant_dot_fp8_inputs(
  %arg0: !migraphx.shaped<4x8x16xf8E4M3FNUZ, 128x16x1>, 
  %arg1: !migraphx.shaped<4x16x8xf8E5M2FNUZ, 128x8x1>,
  %arg2: !migraphx.shaped<4x8x16xf8E8M0FNU, 128x16x1>,
  %arg3: !migraphx.shaped<4x16x8xf8E8M0FNU, 128x8x1>
) -> !migraphx.shaped<4x8x8xf32, 64x8x1> {
  // expected-error @+1 {{input types must have the same element type}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<4x8x16xf8E4M3FNUZ, 128x16x1> scaled by
       !migraphx.shaped<4x8x16xf8E8M0FNU, 128x16x1>,
       !migraphx.shaped<4x16x8xf8E5M2FNUZ, 128x8x1> scaled by
       !migraphx.shaped<4x16x8xf8E8M0FNU, 128x8x1>
     -> !migraphx.shaped<4x8x8xf32, 64x8x1>
  return %0 : !migraphx.shaped<4x8x8xf32, 64x8x1>
}

// -----

// Test: Mismatched input types for scaled quant_dot
func.func @quant_dot_mismatched_inputs(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf8E4M3FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf32, 8192x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // expected-error @+1 {{input types must have the same element type}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf32, 8192x512x1>,
       !migraphx.shaped<1x512x16xf8E4M3FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: Scale shape doesn't match input shape (rank mismatch)
func.func @quant_dot_mismatched_scale_shape(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16xf32, 16x1>,
  %arg3: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // expected-error @+1 {{scaleA shape must have the same number of dimensions as the input types}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16xf32, 16x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: Scale dimensions don't match input dimensions
func.func @quant_dot_wrong_scale_dims(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x32x512xf32, 16384x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xf32, 256x16x1> {
  // expected-error @+1 {{scaleA shape must have the same dimensions as the input types}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x32x512xf32, 16384x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: Result type must be float for scaled quant_dot
func.func @quant_dot_result_not_float(
  %arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, 
  %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>,
  %arg2: !migraphx.shaped<1x16x512xf32, 8192x512x1>,
  %arg3: !migraphx.shaped<1x512x16xf32, 8192x16x1>
) -> !migraphx.shaped<1x16x16xi32, 256x16x1> {
  // expected-error @+1 {{result type must be a float32 type for scaled quant dot ops}}
  %0 = migraphx.quant_dot
       %arg0 scaled by %arg2,
       %arg1 scaled by %arg3
     : !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1> scaled by
       !migraphx.shaped<1x16x512xf32, 8192x512x1>,
       !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1> scaled by
       !migraphx.shaped<1x512x16xf32, 8192x16x1>
     -> !migraphx.shaped<1x16x16xi32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xi32, 256x16x1>
}

// -----

// CHECK-LABEL: func.func @migraphx_quant_dot_f4_no_scales
func.func @migraphx_quant_dot_f4_n_scales(%arg0: !migraphx.shaped<1x16x512xf4E2M1FN, 8192x512x1>, %arg1: !migraphx.shaped<1x512x16xf4E2M1FN, 8192x16x1>) -> !migraphx.shaped<1x16x16xf32, 256x16x1>  {
 // expected-error @+1 {{Quant Dot ops requires scales to be provided to use f4E2M1FN element type}}
 %0 = migraphx.quant_dot
      %arg0,
      %arg1 
    : <1x16x512xf4E2M1FN, 8192x512x1>, 
      <1x512x16xf4E2M1FN, 8192x16x1>
    -> <1x16x16xf32, 256x16x1>
  return %0 : !migraphx.shaped<1x16x16xf32, 256x16x1>
}

// -----

// Test: squeeze with axis out of bounds (positive)
func.func @squeeze_axis_out_of_bounds(%arg0: !migraphx.shaped<1x3x1x4xf16, 12x4x4x1>) -> !migraphx.shaped<3x1x4xf16, 4x4x1> {
  // expected-error @+1 {{'migraphx.squeeze' op axis 5 is out of bounds for input with rank 4 (valid range is [-4, 3])}}
  %0 = migraphx.squeeze %arg0 {axes = [5]} : <1x3x1x4xf16, 12x4x4x1> -> <3x1x4xf16, 4x4x1>
  return %0 : !migraphx.shaped<3x1x4xf16, 4x4x1>
}

// -----

// Test: squeeze with axis out of bounds (negative)
func.func @squeeze_axis_out_of_bounds_negative(%arg0: !migraphx.shaped<1x3x1x4xf16, 12x4x4x1>) -> !migraphx.shaped<3x1x4xf16, 4x4x1> {
  // expected-error @+1 {{'migraphx.squeeze' op axis -5 is out of bounds for input with rank 4 (valid range is [-4, 3])}}
  %0 = migraphx.squeeze %arg0 {axes = [-5]} : <1x3x1x4xf16, 12x4x4x1> -> <3x1x4xf16, 4x4x1>
  return %0 : !migraphx.shaped<3x1x4xf16, 4x4x1>
}

// -----

// Test: squeeze on axis that doesn't have size 1
func.func @squeeze_non_unit_dim(%arg0: !migraphx.shaped<1x3x1x4xf16, 12x4x4x1>) -> !migraphx.shaped<1x1x4xf16, 4x4x1> {
  // expected-error @+1 {{'migraphx.squeeze' op cannot squeeze axis 1 (normalized to 1) which has size 3 (expected size 1)}}
  %0 = migraphx.squeeze %arg0 {axes = [1]} : <1x3x1x4xf16, 12x4x4x1> -> <1x1x4xf16, 4x4x1>
  return %0 : !migraphx.shaped<1x1x4xf16, 4x4x1>
}

// -----

// Test: gather with axis out of bounds (positive)
func.func @gather_axis_out_of_bounds(%data: !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>, %indices: !migraphx.shaped<5xi32, 1>) -> !migraphx.shaped<5x2x2x16xf16, 64x32x16x1> {
  // expected-error @+1 {{'migraphx.gather' op axis 4 is out of bounds for data with rank 4 (valid range is [-4, 3])}}
  %0 = migraphx.gather %data, %indices {axis = 4} : <5x2x2x16xf16, 64x32x16x1>, <5xi32, 1> -> <5x2x2x16xf16, 64x32x16x1>
  return %0 : !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>
}

// -----

// Test: gather with axis out of bounds (negative)
func.func @gather_axis_out_of_bounds_negative(%data: !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>, %indices: !migraphx.shaped<5xi32, 1>) -> !migraphx.shaped<5x2x2x16xf16, 64x32x16x1> {
  // expected-error @+1 {{'migraphx.gather' op axis -5 is out of bounds for data with rank 4 (valid range is [-4, 3])}}
  %0 = migraphx.gather %data, %indices {axis = -5} : <5x2x2x16xf16, 64x32x16x1>, <5xi32, 1> -> <5x2x2x16xf16, 64x32x16x1>
  return %0 : !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>
}

// -----

// Test: scatter_none with axis out of bounds (positive)
func.func @scatter_none_axis_out_of_bounds(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x2x16xi32, 32x16x1>,
    %updates: !migraphx.shaped<8x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // expected-error @+1 {{'migraphx.scatter_none' op axis 3 is out of bounds for data with rank 3 (valid range is [-3, 2])}}
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = 3}
      : <10x2x16xf16, 32x16x1>, <8x2x16xi32, 32x16x1>, <8x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

// -----

// Test: scatter_none with axis out of bounds (negative)
func.func @scatter_none_axis_out_of_bounds_negative(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x2x16xi32, 32x16x1>,
    %updates: !migraphx.shaped<8x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // expected-error @+1 {{'migraphx.scatter_none' op axis -4 is out of bounds for data with rank 3 (valid range is [-3, 2])}}
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = -4}
      : <10x2x16xf16, 32x16x1>, <8x2x16xi32, 32x16x1>, <8x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

// -----

// Test: scatter_none with mismatched ranks
func.func @scatter_none_rank_mismatch(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x16xi32, 16x1>,
    %updates: !migraphx.shaped<8x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // expected-error @+1 {{'migraphx.scatter_none' op data, indices, and updates must have the same rank, got 3, 2, and 3}}
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = 0}
      : <10x2x16xf16, 32x16x1>, <8x16xi32, 16x1>, <8x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

// -----

// Test: scatter_none with mismatched indices/updates shapes
func.func @scatter_none_shape_mismatch(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x2x16xi32, 32x16x1>,
    %updates: !migraphx.shaped<4x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // expected-error @+1 {{'migraphx.scatter_none' op indices and updates must have the same shape}}
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = 0}
      : <10x2x16xf16, 32x16x1>, <8x2x16xi32, 32x16x1>, <4x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

