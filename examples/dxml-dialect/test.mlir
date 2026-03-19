// RUN: rocmlir-opt %s | FileCheck %s
// Test model1 - Conv + ReLU residual blocks for image upscaling

DxGML.Module @torch_jit(
    %arg0: !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
  ) -> !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16> 
  attributes {
    version = "v0.0.1",
    torch.onnx_meta.ir_version = 6 : si64, 
    torch.onnx_meta.opset_version = 12 : si64, 
    torch.onnx_meta.opset_versions = {aimet_torch = 1 : si64}, 
    torch.onnx_meta.producer_name = "pytorch", 
    torch.onnx_meta.producer_version = "1.13.1"
  } 
{
    %_conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_conv1.weight>) : !DxGML.Tensor<32x4x3x3x!DxGML.Float16>
    %_conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
    %_RDB1.conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv1.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
    %_RDB1.conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
    
    // CHECK: dxgml_op.convolution
    %0 = DxGML.Convolution(%arg0, %_conv1.weight, %_conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>, 
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>, 
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>, 
      strides = #DxGML.ConstantValue<[2, 2]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>, !DxGML.Tensor<32x4x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    
    // CHECK: dxgml_op.convolution
    %1 = DxGML.Convolution(%0, %_RDB1.conv1.weight, %_RDB1.conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>, 
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>, 
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>, 
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    

    // CHECK: dxgml_op.relu
    %2 = DxGML.Relu(%1) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    
    // CHECK: dxgml_op.add
    %3 = DxGML.Add(%2, %0) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    
    // CHECK: dxgml.return
    DxGML.Return %3 : !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
}
