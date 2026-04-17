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

func.func @quantize_scale_bias_ui32(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>,
    %scale: !migraphx.shaped<1x1x1x64xf32, 64x64x64x1>,
    %bias: !migraphx.shaped<1x1x1x64xi32, 64x64x64x1>) -> !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1> attributes {rock.kernel = "mixr"} {
  // expected-error @+1 {{output and bias must have the same element type}}
  %1 = migraphx.quantizelinear %arg, %scale, %bias :
    <1x112x112x64xf32, 802816x7168x64x1>, <1x1x1x64xf32, 64x64x64x1>, !migraphx.shaped<1x1x1x64xi32, 64x64x64x1> -> <1x112x112x64xf16, 802816x7168x64x1>
  return %1 : !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>
}
