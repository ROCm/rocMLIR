// RUN: rocmlir-opt %s -split-input-file -verify-diagnostics

// ---- migraphx.shaped type ----

// expected-error @+1 {{migraphx.shaped type has 1 elements in its shape but 2 strides defined}}
func.func @invalid_more_strides_than_shapes(%arg: !migraphx.shaped<1xf32, 1x1>)  {
  func.return
}

// -----

// expected-error @+1 {{migraphx.shaped type has 2 elements in its shape but 1 strides defined}}
func.func @invalid_more_shapes_than_strides(%arg: !migraphx.shaped<1x1xf32, 1>)  {
  func.return
}

// -----

// ---- migraphx.reshape ----

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

// -----

// ---- migraphx.equal ----

func.func @func_equal(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{rock.kernel, rock.arch = ""} {
  %cst = migraphx.literal (dense<1> : tensor<1x36x384x64xi32>) : <1x36x384x64xi32, 884736x24576x64x1>
  %0 = migraphx.add %arg0, %cst : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
  // expected-error@+1 {{'migraphx.equal' op failed to verify that all of {inA, inB, output} have same element type}}
  %1 = migraphx.equal %arg0, %0 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi16, 884736x24576x64x1>
  return %1 : !migraphx.shaped<1x36x384x64xi16, 884736x24576x64x1>
}

// -----

// ---- migraphx.clip ----

func.func @clip_mismatched_element_types(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x8xf16, 8x1>, %arg2: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  // expected-error @+1 {{op failed to verify that all of {x, minVals, maxVals, output} have same element type}}
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <4x8xf32, 8x1>, <4x8xf16, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// -----

func.func @clip_mismatched_shapes(%arg0: !migraphx.shaped<4x8xf32, 8x1>, %arg1: !migraphx.shaped<4x4xf32, 4x1>, %arg2: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  // expected-error @+1 {{op failed to verify that all of {x, minVals, maxVals, output} have same shape}}
  %0 = migraphx.clip %arg0, %arg1, %arg2 : <4x8xf32, 8x1>, <4x4xf32, 4x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// -----

// ---- migraphx.where ----

func.func @where_cond_not_bool(%arg0: !migraphx.shaped<4x4xf32, 4x1>, %arg1: !migraphx.shaped<4x4xf32, 4x1>, %arg2: !migraphx.shaped<4x4xf32, 4x1>) -> !migraphx.shaped<4x4xf32, 4x1> {
  // expected-error @+1 {{'migraphx.where' op operand #0 must be !migraphx.shaped of 8-bit signless integer or 8-bit signed integer or 8-bit unsigned integer values, but got '!migraphx.shaped<4x4xf32, 4x1>'}}
  %0 = migraphx.where %arg0, %arg1, %arg2 : <4x4xf32, 4x1>, <4x4xf32, 4x1>, <4x4xf32, 4x1> -> <4x4xf32, 4x1>
  return %0 : !migraphx.shaped<4x4xf32, 4x1>
}

// -----

func.func @where_mismatched_types(%arg0: !migraphx.shaped<4x4xi8, 4x1>, %arg1: !migraphx.shaped<4x4xf32, 4x1>, %arg2: !migraphx.shaped<4x4xf16, 4x1>) -> !migraphx.shaped<4x4xf32, 4x1> {
  // expected-error @+1 {{op failed to verify that all of {inA, inB, output} have same element type}}
  %0 = migraphx.where %arg0, %arg1, %arg2 : <4x4xi8, 4x1>, <4x4xf32, 4x1>, <4x4xf16, 4x1> -> <4x4xf32, 4x1>
  return %0 : !migraphx.shaped<4x4xf32, 4x1>
}

// -----

func.func @where_mismatched_shapes(%arg0: !migraphx.shaped<4x4xi8, 4x1>, %arg1: !migraphx.shaped<4x8xf32, 8x1>, %arg2: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x8xf32, 8x1> {
  // expected-error @+1 {{op failed to verify that all of {inA, inB, output, cond} have same shape}}
  %0 = migraphx.where %arg0, %arg1, %arg2 : <4x4xi8, 4x1>, <4x8xf32, 8x1>, <4x8xf32, 8x1> -> <4x8xf32, 8x1>
  return %0 : !migraphx.shaped<4x8xf32, 8x1>
}

// -----

// ---- migraphx.convert ----

func.func @convert_shape_mismatch(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x4xf16, 4x1> {
  // expected-error @+1 {{op failed to verify that all of {inA, output} have same shape}}
  %0 = migraphx.convert %arg0 : <4x8xf32, 8x1> to <4x4xf16, 4x1>
  return %0 : !migraphx.shaped<4x4xf16, 4x1>
}

// -----

// ---- migraphx.abs ----

func.func @abs_shape_mismatch(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<4x4xf32, 4x1> {
  // expected-error @+1 {{op failed to verify that all of {inA, output} have same shape}}
  %0 = migraphx.abs %arg0 : <4x8xf32, 8x1> -> <4x4xf32, 4x1>
  return %0 : !migraphx.shaped<4x4xf32, 4x1>
}

// -----

// ---- migraphx.exp ----

func.func @exp_rank_mismatch(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<32xf32, 1> {
  // expected-error @+1 {{op failed to verify that all of {inA, output} have same shape}}
  %0 = migraphx.exp %arg0 : <4x8xf32, 8x1> -> <32xf32, 1>
  return %0 : !migraphx.shaped<32xf32, 1>
}

// -----

// ---- migraphx.relu ----

func.func @relu_shape_mismatch(%arg0: !migraphx.shaped<4x8xf32, 8x1>) -> !migraphx.shaped<2x8xf32, 8x1> {
  // expected-error @+1 {{op failed to verify that all of {inA, output} have same shape}}
  %0 = migraphx.relu %arg0 : <4x8xf32, 8x1> -> <2x8xf32, 8x1>
  return %0 : !migraphx.shaped<2x8xf32, 8x1>
}

// -----

// ---- migraphx.sigmoid ----

func.func @func_sigmoid_2d_i32(%arg0: !migraphx.shaped<4x8xi32, 8x1>) -> !migraphx.shaped<4x8xi32, 8x1> {
  // expected-error @+1 {{only support floating point}}
  %0 = migraphx.sigmoid %arg0 : <4x8xi32, 8x1> -> <4x8xi32, 8x1>
  return %0 : !migraphx.shaped<4x8xi32, 8x1>
}

// -----

// ---- migraphx.quant_dot ----

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

// ---- migraphx.dot ----

// CHECK-LABEL: func.func @dot_rank_less_than_2
func.func @dot_rank_less_than_2(%arg0: !migraphx.shaped<320xf16, 1>, %arg1: !migraphx.shaped<320x64xf16, 64x1>) -> !migraphx.shaped<64xf16, 1> {
  // expected-error @+1 {{expect operand to have rank greater or equal to 2}}
  %0 = migraphx.dot %arg0, %arg1 : <320xf16, 1>, <320x64xf16, 64x1> -> <64xf16, 1>
  return %0 : !migraphx.shaped<64xf16, 1>
}

// -----

// CHECK-LABEL: func.func @dot_incompatible_inner_dim
func.func @dot_incompatible_inner_dim(%arg0: !migraphx.shaped<2x64x320xf16, 20480x320x1>, %arg1: !migraphx.shaped<2x256x64xf16, 16384x64x1>) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{contraction dimension mismatch: the first operand}}
  %0 = migraphx.dot %arg0, %arg1 : <2x64x320xf16, 20480x320x1>, <2x256x64xf16, 16384x64x1> -> <2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// CHECK-LABEL: func.func @dot_invalid_batch
func.func @dot_invalid_batch(%arg0: !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<6x2x2xf32, 4x2x1>) -> !migraphx.shaped<3x2x2x2xf32, 8x4x2x1> attributes {rock.kernel, rock.arch="gfx950"} {
  // expected-error@+1 {{batch dimension mismatch: the first operand ('!migraphx.shaped<3x2x2x2xf32, 8x4x2x1>') and the second operand ('!migraphx.shaped<6x2x2xf32, 4x2x1>') have incompatible batch dimensions}}
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf32, 8x4x2x1>, <6x2x2xf32, 4x2x1> -> <3x2x2x2xf32, 8x4x2x1>
  func.return %0 : !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>
}

// -----

// CHECK-LABEL: func.func @dot_invalid_broadcast
func.func @dot_invalid_broadcast(%arg0: !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>, %arg1: !migraphx.shaped<2x3x2x2xf32, 12x4x2x1>) -> !migraphx.shaped<3x2x2x2xf32, 8x4x2x1> attributes {rock.kernel, rock.arch="gfx950"} {
  // expected-error@+1 {{batch dimension mismatch: the first operand}}
  %0 = migraphx.dot %arg0, %arg1 : <3x2x2x2xf32, 8x4x2x1>, <2x3x2x2xf32, 12x4x2x1> -> <3x2x2x2xf32, 8x4x2x1>
  func.return %0 : !migraphx.shaped<3x2x2x2xf32, 8x4x2x1>
}

// -----

// CHECK-LABEL: func.func @dot_result_shape_mismatch
func.func @dot_result_shape_mismatch(%arg0: !migraphx.shaped<2x3x4xf16, 12x4x1>, %arg1: !migraphx.shaped<2x4x5xf16, 20x5x1>) -> !migraphx.shaped<2x3x4xf16, 12x4x1> {
  // expected-error @+1 {{result type is inconsistent with input shapes}}
  %0 = migraphx.dot %arg0, %arg1 : <2x3x4xf16, 12x4x1>, <2x4x5xf16, 20x5x1> -> <2x3x4xf16, 12x4x1>
  return %0 : !migraphx.shaped<2x3x4xf16, 12x4x1>
}

// -----

// ---- migraphx.slice ----

func.func @invalid_attr_size_mismatch(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{op axes, starts, and ends must have the same size}}
  %result = migraphx.slice %input {axes = [0, 1], starts = [0], ends = [2, 2]} : <10x10xf32, 10x1> -> <2x2xf32, 2x1>
  func.return
}

// -----

func.func @invalid_rank_mismatch(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{input and output shapes must have the same rank}}
  %result = migraphx.slice %input {axes = [0], starts = [0], ends = [5]} : <10x10xf32, 10x1> -> <5xf32, 1>
  func.return
}

// -----

func.func @invalid_negative_axis(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{all attribute must non non-negative}}
  %result = migraphx.slice %input {axes = [-1], starts = [0], ends = [5]} : <10x10xf32, 10x1> -> <10x5xf32, 5x1>
  func.return
}

// -----

func.func @invalid_axis_out_of_range(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{axes is greater than input rank}}
  %result = migraphx.slice %input {axes = [0, 10], starts = [0, 0], ends = [2, 2]} : <10x10xf32, 10x1> -> <2x2xf32, 2x1>
  func.return
}

// -----

func.func @invalid_axis_equals_rank(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{axes is greater than input rank}}
  %result = migraphx.slice %input {axes = [2], starts = [0], ends = [5]} : <10x10xf32, 10x1> -> <10x10xf32, 10x1>
  func.return
}

// -----

func.func @invalid_start_greater_than_end(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{op start is greater or equal to end}}
  %result = migraphx.slice %input {axes = [0], starts = [5], ends = [2]} : <10x10xf32, 10x1> -> <10x10xf32, 10x1>
  func.return
}

// -----

func.func @invalid_end_exceeds_input(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{end is greater than input shape}}
  %result = migraphx.slice %input {axes = [0, 1], starts = [0, 0], ends = [11, 10]} : <10x10xf32, 10x1> -> <11x10xf32, 11x1>
  func.return
}

// -----

func.func @invalid_shape_mismatch(%input: !migraphx.shaped<10x10xf32, 10x1>) {
  // expected-error @+1 {{input shape and attribute does not infer output shape}}
  %result = migraphx.slice %input {axes = [0], starts = [0], ends = [5]} : <10x10xf32, 10x1> -> <3x10xf32, 10x1>
  func.return
}

// -----

// ---- migraphx.quantizelinear ----

func.func @quantize_scale_bias_ui32(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>,
    %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>,
    %bias: !migraphx.shaped<1x1x1x64xi32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1> attributes {rock.kernel = "mixr"} {
  // expected-error @+1 {{failed to verify that output and bias must have the same element type}}
  %1 = migraphx.quantizelinear %arg, %scale, %bias :
    <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xi32, 64x64x64x1> -> <1x112x112x64xf16, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>
}

// -----

// ---- migraphx.attention ----

// Operand rank: all of Q, K, V must have rank >= 2.
func.func @attention_rank_too_low(
    %q: !migraphx.shaped<128xf16, 1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<64xf16, 1> {
  // expected-error @+1 {{'migraphx.attention' op operands must have rank >= 2}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <128xf16, 1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<64xf16, 1>
  return %0 : !migraphx.shaped<64xf16, 1>
}

// -----

// First GEMM: Q's last dim (head_qk) must equal K's second-to-last dim.
func.func @attention_contraction_mismatch(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op head dimension mismatched for first gemm}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x64x256xf16, 16384x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// Second GEMM: K's last dim (seq_k) must equal V's second-to-last dim.
func.func @attention_second_gemm_mismatch(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x128x64xf16, 8192x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op sequence length dimension mismatch for second gemm}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x128x64xf16, 8192x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// K and V must have identical leading (batch/heads) dims.
func.func @attention_kv_leading_dim_mismatch(
    %q: !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>,
    %k: !migraphx.shaped<2x2x64x32xf16, 4096x2048x32x1>,
    %v: !migraphx.shaped<2x3x32x64xf16, 6144x2048x64x1>
) -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1> {
  // expected-error @+1 {{'migraphx.attention' op leading dimension mismatch at dimension 1: keys=2 != values=3}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x4x32x64xf16, 8192x2048x64x1>, <2x2x64x32xf16, 4096x2048x32x1>, <2x3x32x64xf16, 6144x2048x64x1>
    -> !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
  return %0 : !migraphx.shaped<2x4x32x64xf16, 8192x2048x64x1>
}

// -----

// GQA: Q's leading dims must be a multiple of K/V's (here 3 vs 2).
func.func @attention_q_not_divisible_by_k(
    %q: !migraphx.shaped<2x3x32x64xf16, 6144x2048x64x1>,
    %k: !migraphx.shaped<2x2x64x32xf16, 4096x2048x32x1>,
    %v: !migraphx.shaped<2x2x32x64xf16, 4096x2048x64x1>
) -> !migraphx.shaped<2x3x32x64xf16, 6144x2048x64x1> {
  // expected-error @+1 {{'migraphx.attention' op leading dimension mismatch at dimension 1: queries=3 is not equal to or divisible by keys=2}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x3x32x64xf16, 6144x2048x64x1>, <2x2x64x32xf16, 4096x2048x32x1>, <2x2x32x64xf16, 4096x2048x64x1>
    -> !migraphx.shaped<2x3x32x64xf16, 6144x2048x64x1>
  return %0 : !migraphx.shaped<2x3x32x64xf16, 6144x2048x64x1>
}

// -----

// Result shape (non-splitkv path): expected [B, seq_q, head_v].
func.func @attention_output_shape_mismatch(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x128xf16, 8192x128x1> {
  // expected-error @+1 {{'migraphx.attention' op result shape is inconsistent}}
  %0 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x128xf16, 8192x128x1>
  return %0 : !migraphx.shaped<2x64x128xf16, 8192x128x1>
}

// -----

// softmaxType attribute must be a float type.
func.func @attention_invalid_softmax_type(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op softmaxType must be a float type}}
  %0 = migraphx.attention %q, %k, %v {
  } softmax_type = i32
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_lse_shape_mismatch(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<4x64xf32, 64x1>) {
  // expected-error @+1 {{'migraphx.attention' op lse shape is inconsistent}}
  %0, %1 = migraphx.attention %q, %k, %v {
  }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<4x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<4x64xf32, 64x1>
}

// -----

func.func @attention_inputs_but_empty_body(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op preSoftmaxElemWiseInputs are provided but preSoftmaxBody contains no operations}}
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%bias : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_pre_softmax_non_elementwise_migraphx_op(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %a: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
    %b: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%a, %b
      : !migraphx.shaped<2x64x256xf16, 16384x256x1>,
        !migraphx.shaped<2x256x64xf16, 16384x64x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %aa: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %bb: !migraphx.shaped<2x256x64xf16, 16384x64x1>):
      // expected-error @+1 {{'migraphx.dot' op preSoftmaxBody must only contain elementwise migraphx ops}}
      %dot = migraphx.dot %aa, %bb
        : <2x64x256xf16, 16384x256x1>, <2x256x64xf16, 16384x64x1>
        -> <2x64x64xf16, 4096x64x1>
      migraphx.yield %dot : !migraphx.shaped<2x64x64xf16, 4096x64x1>
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_pre_softmax_reduce_in_body(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%bias : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %b: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      // expected-error @+1 {{'migraphx.reduce_sum' op preSoftmaxBody must only contain elementwise migraphx ops}}
      %reduced = migraphx.reduce_sum %qk {axes = [2]}
        : <2x64x256xf16, 16384x256x1> -> <2x64x1xf16, 64x1x1>
      migraphx.yield %reduced : !migraphx.shaped<2x64x1xf16, 64x1x1>
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_kvcache_without_current_seq_len(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op feature 'kvcache' requires 'currentSeqLen' operand}}
  %0 = migraphx.attention %q, %k, %v {
  } features = kvcache
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_sliding_window_without_kvcache(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op feature 'sliding_window' requires 'kvcache' to be set}}
  %0 = migraphx.attention %q, %k, %v {
  } features = sliding_window slidingWindowSize = 64
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// slidingWindowSize must be positive
func.func @attention_sliding_window_negative_size(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op slidingWindowSize must be positive}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|sliding_window" slidingWindowSize = -1
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// slidingWindowSize must not exceed key sequence length (256)
func.func @attention_sliding_window_exceeds_seqlen(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op slidingWindowSize must not exceed max sequence length}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|sliding_window" slidingWindowSize = 999
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// slidingWindowSize attribute without sliding_window feature
func.func @attention_orphan_sliding_window_size(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op 'slidingWindowSize' attribute requires feature 'sliding_window'}}
  %0 = migraphx.attention %q, %k, %v {
  } slidingWindowSize = 64
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_prefix_offset_without_causal(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op feature 'prefix_offset' requires 'causal' to be set}}
  %0 = migraphx.attention %q, %k, %v {
  } features = prefix_offset
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_splitkv_without_lse(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op feature 'splitkv' requires LSE result}}
  %0 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>
  return %0 : !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>
}

// -----

func.func @attention_orphan_current_seq_len(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op 'currentSeqLen' operand requires feature 'kvcache'}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_orphan_prefix_offset(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %po: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op 'prefixOffset' operand requires feature 'prefix_offset'}}
  %0 = migraphx.attention %q, %k, %v
    prefix_offset(%po : !migraphx.shaped<2xi32, 1>) {
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

func.func @attention_current_seq_len_rank_too_high(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<1x2x1xi32, 2x1x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op 'currentSeqLen' shape must match Q leading dims}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<1x2x1xi32, 2x1x1>) {
    } features = kvcache
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// 4D Q with rank-1 [batch] currentSeqLen is rejected when numHeads > 1.
// Producer must broadcast across heads explicitly via migraphx.multibroadcast.
func.func @attention_current_seq_len_missing_head_broadcast(
    %q: !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<2x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<2x2x16x8xf16, 256x128x8x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x2x4x8xf16, 64x32x8x1> {
  // expected-error @+1 {{'migraphx.attention' op 'currentSeqLen' shape must match Q leading dims}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = kvcache
    : <2x2x4x8xf16, 64x32x8x1>, <2x2x8x16xf16, 256x128x16x1>, <2x2x16x8xf16, 256x128x8x1>
    -> !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>
}

// -----

// splitKV must evenly divide key sequence length
func.func @attention_splitkv_not_divisible(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x3x64x64xf16, 12288x4096x64x1>, !migraphx.shaped<2x3x64xf32, 192x64x1>) {
  // expected-error @+1 {{'migraphx.attention' op key sequence length (256) must be divisible by splitKV (3)}}
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 3
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x3x64x64xf16, 12288x4096x64x1>, !migraphx.shaped<2x3x64xf32, 192x64x1>
  return %0, %1 : !migraphx.shaped<2x3x64x64xf16, 12288x4096x64x1>, !migraphx.shaped<2x3x64xf32, 192x64x1>
}

// -----

// prefix_offset feature requires prefixOffset operand
func.func @attention_prefix_offset_no_operand(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op feature 'prefix_offset' requires 'prefixOffset' operand}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// splitkv feature requires splitKV > 1
func.func @attention_splitkv_no_attr(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  // expected-error @+1 {{'migraphx.attention' op feature 'splitkv' requires splitKV > 1}}
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// -----

// orphan splitKV attribute without splitkv feature
func.func @attention_orphan_splitkv(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  // expected-error @+1 {{'migraphx.attention' op 'splitKV' attribute requires feature 'splitkv'}}
  %0, %1 = migraphx.attention %q, %k, %v {
  } splitKV = 2
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// -----

// prefixOffset shape must match Q leading dims exactly
func.func @attention_prefix_offset_rank3(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %sl: !migraphx.shaped<2xi32, 1>,
    %po: !migraphx.shaped<1x2x1xi32, 2x1x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op 'prefixOffset' shape must match Q leading dims}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2xi32, 1>)
    prefix_offset(%po : !migraphx.shaped<1x2x1xi32, 2x1x1>) {
    } features = "kvcache|causal|prefix_offset"
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}

// -----

// 4D Q with rank-1 [batch] prefixOffset is rejected when numHeads > 1.
func.func @attention_prefix_offset_missing_head_broadcast(
    %q: !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<2x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<2x2x16x8xf16, 256x128x8x1>,
    %sl: !migraphx.shaped<2x2xi32, 2x1>,
    %po: !migraphx.shaped<2xi32, 1>
) -> !migraphx.shaped<2x2x4x8xf16, 64x32x8x1> {
  // expected-error @+1 {{'migraphx.attention' op 'prefixOffset' shape must match Q leading dims}}
  %0 = migraphx.attention %q, %k, %v
    current_seq_len(%sl : !migraphx.shaped<2x2xi32, 2x1>)
    prefix_offset(%po : !migraphx.shaped<2xi32, 1>) {
    } features = "kvcache|causal|prefix_offset"
    : <2x2x4x8xf16, 64x32x8x1>, <2x2x8x16xf16, 256x128x16x1>, <2x2x16x8xf16, 256x128x8x1>
    -> !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>
  return %0 : !migraphx.shaped<2x2x4x8xf16, 64x32x8x1>
}


// -----

// splitKV: result shape missing split dimension (got [2,64,64] but expected [2,2,64,64])
func.func @attention_splitkv_wrong_result_shape(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>) {
  // expected-error @+1 {{'migraphx.attention' op result shape is inconsistent with attention dimensions: expected [2, 2, 64, 64] but got [2, 64, 64]}}
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>
  return %0, %1 : !migraphx.shaped<2x64x64xf16, 4096x64x1>, !migraphx.shaped<2x2x64xf32, 128x64x1>
}

// -----

// splitKV: LSE shape missing split dimension (got [2,64] but expected [2,2,64])
func.func @attention_splitkv_wrong_lse_shape(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>
) -> (!migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>) {
  // expected-error @+1 {{'migraphx.attention' op lse shape is inconsistent with attention dimensions: expected [2, 2, 64] but got [2, 64]}}
  %0, %1 = migraphx.attention %q, %k, %v {
  } features = splitkv splitKV = 2
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> <2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
  return %0, %1 : !migraphx.shaped<2x2x64x64xf16, 8192x4096x64x1>, !migraphx.shaped<2x64xf32, 64x1>
}

// -----

// splitKV + preSoftmaxBody: inputs must have split-space rank (5D not 4D)
func.func @attention_splitkv_presoftmax_wrong_rank(
    %q: !migraphx.shaped<1x2x4x8xf16, 64x32x8x1>,
    %k: !migraphx.shaped<1x2x8x16xf16, 256x128x16x1>,
    %v: !migraphx.shaped<1x2x16x8xf16, 256x128x8x1>,
    %s: !migraphx.shaped<1x2x4x16xf16, 128x64x16x1>
) -> (!migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>) {
  // expected-error @+1 {{'migraphx.attention' op preSoftmaxElemWiseInput shape rank (4) must match split QK shape rank (5) when splitkv is enabled}}
  %0, %1 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%s : !migraphx.shaped<1x2x4x16xf16, 128x64x16x1>) {
    ^bb0(%qk: !migraphx.shaped<1x2x4x16xf16, 128x64x16x1>,
         %ss: !migraphx.shaped<1x2x4x16xf16, 128x64x16x1>):
      %scaled = migraphx.mul %qk, %ss
        : <1x2x4x16xf16, 128x64x16x1>, <1x2x4x16xf16, 128x64x16x1> -> <1x2x4x16xf16, 128x64x16x1>
      migraphx.yield %scaled : !migraphx.shaped<1x2x4x16xf16, 128x64x16x1>
    } features = splitkv splitKV = 2
    : <1x2x4x8xf16, 64x32x8x1>, <1x2x8x16xf16, 256x128x16x1>, <1x2x16x8xf16, 256x128x8x1>
    -> <1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
  return %0, %1 : !migraphx.shaped<1x2x2x4x8xf16, 128x64x32x8x1>, !migraphx.shaped<1x2x2x4xf32, 16x8x4x1>
}

// -----

// preSoftmaxBody block arg count mismatch: 1 input but 3 block args (expected 2)
func.func @attention_block_arg_count_mismatch(
    %q: !migraphx.shaped<2x64x128xf16, 8192x128x1>,
    %k: !migraphx.shaped<2x128x256xf16, 32768x256x1>,
    %v: !migraphx.shaped<2x256x64xf16, 16384x64x1>,
    %bias: !migraphx.shaped<2x64x256xf16, 16384x256x1>
) -> !migraphx.shaped<2x64x64xf16, 4096x64x1> {
  // expected-error @+1 {{'migraphx.attention' op preSoftmaxBody block must have exactly 2 arguments (1 for QK result + 1 preSoftmaxElemWiseInputs), got 3}}
  %0 = migraphx.attention %q, %k, %v
    pre_softmax_inputs(%bias : !migraphx.shaped<2x64x256xf16, 16384x256x1>) {
    ^bb0(%qk: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %b: !migraphx.shaped<2x64x256xf16, 16384x256x1>,
         %extra: !migraphx.shaped<2x64x256xf16, 16384x256x1>):
      %sum = migraphx.add %qk, %b
        : <2x64x256xf16, 16384x256x1>, <2x64x256xf16, 16384x256x1>
        -> <2x64x256xf16, 16384x256x1>
      migraphx.yield %sum : !migraphx.shaped<2x64x256xf16, 16384x256x1>
    }
    : <2x64x128xf16, 8192x128x1>, <2x128x256xf16, 32768x256x1>, <2x256x64xf16, 16384x64x1>
    -> !migraphx.shaped<2x64x64xf16, 4096x64x1>
  return %0 : !migraphx.shaped<2x64x64xf16, 4096x64x1>
}
