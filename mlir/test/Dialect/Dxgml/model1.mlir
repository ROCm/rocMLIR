// RUN: rocmlir-opt %s | FileCheck %s
// Test model1 - Conv + ReLU residual blocks for image upscaling

// CHECK-LABEL: dxgml.module
dxgml.module {

  // CHECK-LABEL: @torch_jit
  dxgml.entry_point @torch_jit(%arg0: !dxgml.tensor<1x4x2160x3840x!dxgml.float16>) -> !dxgml.tensor<1x4x2160x3840x!dxgml.float16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.opset_versions = {aimet_torch = 1 : si64}, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "1.13.1"} {
    %_conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_conv1.weight : !dxgml.tensor<32x4x3x3x!dxgml.float16>>)
    %_conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    %_RDB1.conv1.weight = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv1.weight : !dxgml.tensor<32x32x3x3x!dxgml.float16>>)
    %_RDB1.conv1.bias = dxgml_op.constant(#dxgml.constant_resource<_RDB1.conv1.bias : !dxgml.tensor<32x!dxgml.float16>>)
    
    // CHECK: dxgml_op.convolution
    %0 = dxgml_op.convolution(%arg0, %_conv1.weight, %_conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[2, 2]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x4x2160x3840x!dxgml.float16>, !dxgml.tensor<32x4x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    
    // CHECK: dxgml_op.convolution
    %1 = dxgml_op.convolution(%0, %_RDB1.conv1.weight, %_RDB1.conv1.bias) {
      group_count = #dxgml.integer<1 : !dxgml.int64>, 
      dilations = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>,
      start_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      end_padding = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>, 
      strides = #dxgml.dense_integer_elements<[1, 1]> : !dxgml.tensor<2x!dxgml.int64>
    } : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<32x32x3x3x!dxgml.float16>, !dxgml.tensor<32x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    
    // CHECK: dxgml_op.relu
    %2 = dxgml_op.relu(%1) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    
    // CHECK: dxgml_op.add
    %3 = dxgml_op.add(%2, %0) : (!dxgml.tensor<1x32x1080x1920x!dxgml.float16>, !dxgml.tensor<1x32x1080x1920x!dxgml.float16>) -> !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
    
    // CHECK: dxgml.return
    dxgml.return %3 : !dxgml.tensor<1x32x1080x1920x!dxgml.float16>
  }
}
