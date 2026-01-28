// RUN: rocmlir-opt %s | FileCheck %s
// Test DXGML operations

dxgml.module {
  // CHECK-LABEL: @test_constants
  dxgml.function @test_constants() -> !dxgml.tensor<3x!dxgml.int64> {
    // CHECK: dxgml_op.constant
    %c = dxgml_op.constant(#dxgml.dense_integer_elements<[1, 2, 3]> : !dxgml.tensor<3x!dxgml.int64>)
    dxgml.return %c : !dxgml.tensor<3x!dxgml.int64>
  }

  // CHECK-LABEL: @test_elementwise_unary
  dxgml.function @test_elementwise_unary(%arg0: !dxgml.tensor<1x32x224x224x!dxgml.float32>) 
      -> !dxgml.tensor<1x32x224x224x!dxgml.float32> {
    // CHECK: dxgml_op.relu
    %0 = dxgml_op.relu(%arg0) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    // CHECK: dxgml_op.sigmoid
    %1 = dxgml_op.sigmoid(%0) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    // CHECK: dxgml_op.tanh
    %2 = dxgml_op.tanh(%1) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    // CHECK: dxgml_op.abs
    %3 = dxgml_op.abs(%2) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    dxgml.return %3 : !dxgml.tensor<1x32x224x224x!dxgml.float32>
  }

  // CHECK-LABEL: @test_elementwise_binary
  dxgml.function @test_elementwise_binary(
      %arg0: !dxgml.tensor<1x32x224x224x!dxgml.float32>,
      %arg1: !dxgml.tensor<1x32x224x224x!dxgml.float32>)
      -> !dxgml.tensor<1x32x224x224x!dxgml.float32> {
    // CHECK: dxgml_op.add
    %0 = dxgml_op.add(%arg0, %arg1) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>, !dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    // CHECK: dxgml_op.multiply
    %1 = dxgml_op.multiply(%0, %arg1) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>, !dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    // CHECK: dxgml_op.max
    %2 = dxgml_op.max(%1, %arg0) : (!dxgml.tensor<1x32x224x224x!dxgml.float32>, !dxgml.tensor<1x32x224x224x!dxgml.float32>) -> !dxgml.tensor<1x32x224x224x!dxgml.float32>
    dxgml.return %2 : !dxgml.tensor<1x32x224x224x!dxgml.float32>
  }

  // CHECK-LABEL: @test_convolution
  dxgml.function @test_convolution(
      %input: !dxgml.tensor<1x3x224x224x!dxgml.float16>,
      %filter: !dxgml.tensor<64x3x7x7x!dxgml.float16>,
      %bias: !dxgml.tensor<64x!dxgml.float16>)
      -> !dxgml.tensor<1x64x112x112x!dxgml.float16> {
    // CHECK: dxgml_op.convolution
    %0 = dxgml_op.convolution(%input, %filter, %bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>,
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[3, 3]> : !dxgml.tensor<2x!dxgml.int64>,
      end_padding = #dxgml.dense_integer_elements<[3, 3]> : !dxgml.tensor<2x!dxgml.int64>,
      strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x3x224x224x!dxgml.float16>, !dxgml.tensor<64x3x7x7x!dxgml.float16>, !dxgml.tensor<64x!dxgml.float16>) -> !dxgml.tensor<1x64x112x112x!dxgml.float16>
    dxgml.return %0 : !dxgml.tensor<1x64x112x112x!dxgml.float16>
  }

  // CHECK-LABEL: @test_depth_to_space
  dxgml.function @test_depth_to_space(%arg0: !dxgml.tensor<1x16x540x960x!dxgml.float16>)
      -> !dxgml.tensor<1x4x1080x1920x!dxgml.float16> {
    // CHECK: dxgml_op.depth_to_space
    %0 = dxgml_op.depth_to_space(%arg0) {
      block_size = #dxgml.integer<2 : !dxgml.int64>,
      depth_space_order = #dxgml_op.depth_space_order_enum_attr<depth_space_order_column_row_depth>
    } : (!dxgml.tensor<1x16x540x960x!dxgml.float16>) -> !dxgml.tensor<1x4x1080x1920x!dxgml.float16>
    dxgml.return %0 : !dxgml.tensor<1x4x1080x1920x!dxgml.float16>
  }
}
