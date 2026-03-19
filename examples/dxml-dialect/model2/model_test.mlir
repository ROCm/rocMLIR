DxGML.Module @model2(
    %arg0: !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
  ) -> !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
    attributes {
      version = "v0.0.1",
      producer_name = "pytorch",
      producer_version = "1.13.1"
    }
{
  %_conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_conv1.weight>) : !DxGML.Tensor<32x4x3x3x!DxGML.Float16>
  %_conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB1.conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv1.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB1.conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB1.conv2.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv2.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB1.conv2.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv2.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB1.conv3.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv3.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB1.conv3.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB1.conv3.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB2.conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv1.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB2.conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB2.conv2.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv2.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB2.conv2.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv2.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB2.conv3.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv3.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB2.conv3.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB2.conv3.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB3.conv1.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv1.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB3.conv1.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv1.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB3.conv2.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv2.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB3.conv2.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv2.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_RDB3.conv3.weight = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv3.weight>) : !DxGML.Tensor<32x32x3x3x!DxGML.Float16>
  %_RDB3.conv3.bias = DxGML.Constant(#DxGML.ConstantResource<_RDB3.conv3.bias>) : !DxGML.Tensor<32x!DxGML.Float16>
  %_conv_post.weight = DxGML.Constant(#DxGML.ConstantResource<_conv_post.weight>) : !DxGML.Tensor<96x32x3x3x!DxGML.Float16>
  %_conv_post.bias = DxGML.Constant(#DxGML.ConstantResource<_conv_post.bias>) : !DxGML.Tensor<96x!DxGML.Float16>
  %_conv_final.weight = DxGML.Constant(#DxGML.ConstantResource<_conv_final.weight>) : !DxGML.Tensor<16x96x1x1x!DxGML.Float16>
  %_conv_final.bias = DxGML.Constant(#DxGML.ConstantResource<_conv_final.bias>) : !DxGML.Tensor<16x!DxGML.Float16>
    %0 = DxGML.Convolution(%arg0, %_conv1.weight, %_conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[2, 2]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>, !DxGML.Tensor<32x4x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %1 = DxGML.Convolution(%0, %_RDB1.conv1.weight, %_RDB1.conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %2 = DxGML.Relu(%1) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %3 = DxGML.Add(%2, %0) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %4 = DxGML.Convolution(%3, %_RDB1.conv2.weight, %_RDB1.conv2.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %5 = DxGML.Relu(%4) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %6 = DxGML.Add(%3, %5) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %7 = DxGML.Convolution(%6, %_RDB1.conv3.weight, %_RDB1.conv3.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %8 = DxGML.Add(%7, %0) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %9 = DxGML.Convolution(%8, %_RDB2.conv1.weight, %_RDB2.conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %10 = DxGML.Relu(%9) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %11 = DxGML.Add(%10, %8) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %12 = DxGML.Convolution(%11, %_RDB2.conv2.weight, %_RDB2.conv2.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %13 = DxGML.Relu(%12) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %14 = DxGML.Add(%11, %13) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %15 = DxGML.Convolution(%14, %_RDB2.conv3.weight, %_RDB2.conv3.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %16 = DxGML.Add(%15, %8) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %17 = DxGML.Convolution(%16, %_RDB3.conv1.weight, %_RDB3.conv1.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %18 = DxGML.Relu(%17) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %19 = DxGML.Add(%18, %16) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %20 = DxGML.Convolution(%19, %_RDB3.conv2.weight, %_RDB3.conv2.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %21 = DxGML.Relu(%20) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %22 = DxGML.Add(%19, %21) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %23 = DxGML.Convolution(%22, %_RDB3.conv3.weight, %_RDB3.conv3.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<32x32x3x3x!DxGML.Float16>, !DxGML.Tensor<32x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %24 = DxGML.Add(%23, %16) : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>
    %25 = DxGML.Convolution(%24, %_conv_post.weight, %_conv_post.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x32x1080x1920x!DxGML.Float16>, !DxGML.Tensor<96x32x3x3x!DxGML.Float16>, !DxGML.Tensor<96x!DxGML.Float16>) -> !DxGML.Tensor<1x96x1080x1920x!DxGML.Float16>
    %26 = DxGML.Relu(%25) : (!DxGML.Tensor<1x96x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x96x1080x1920x!DxGML.Float16>
    %27 = DxGML.Convolution(%26, %_conv_final.weight, %_conv_final.bias) {
      group_count = #DxGML.ConstantValue<[1]> : !DxGML.Tensor<1x!DxGML.Int64>,
      dilations = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>,
      start_padding = #DxGML.ConstantValue<[0, 0]> : !DxGML.Tensor<2x!DxGML.Int64>,
      end_padding = #DxGML.ConstantValue<[0, 0]> : !DxGML.Tensor<2x!DxGML.Int64>,
      strides = #DxGML.ConstantValue<[1, 1]> : !DxGML.Tensor<2x!DxGML.Int64>
    } : (!DxGML.Tensor<1x96x1080x1920x!DxGML.Float16>, !DxGML.Tensor<16x96x1x1x!DxGML.Float16>, !DxGML.Tensor<16x!DxGML.Float16>) -> !DxGML.Tensor<1x16x1080x1920x!DxGML.Float16>
    %28 = DxGML.DepthToSpace(%27) {
      block_size = #DxGML.ConstantValue<[2]> : !DxGML.Tensor<1x!DxGML.Int64>,
      depth_space_order = #DxGML.DepthSpaceOrderEnumAttr<DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH>
     } : (!DxGML.Tensor<1x16x1080x1920x!DxGML.Float16>) -> !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
     DxGML.Return %28 : !DxGML.Tensor<1x4x2160x3840x!DxGML.Float16>
}


{-#
  
#-}
