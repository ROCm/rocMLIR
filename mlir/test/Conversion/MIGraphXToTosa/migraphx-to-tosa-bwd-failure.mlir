// This test checks that we emit an error when trying to convert 3-D
// backwards convolution ops.

// RUN: not rocmlir-opt -split-input-file --migraphx-to-tosa %s 2>&1 | FileCheck %s

module {
    func.func @bwd_data_conv_3d(
        %arg0: !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1>,
        %arg1: !migraphx.shaped<16x16x1x1x1xf32, 16x1x1x1x1>,
        %arg2: !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1>
    ) -> !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1> {
        // CHECK: Only 1-D and 2-D backwards convolution ops are supported
        %0 = migraphx.backwards_data_convolution %arg1, %arg0 {
            dilation = [1, 1, 1],
            group = 1 : i64,
            padding = [0, 0, 0, 0, 0, 0],
            padding_mode = 0 : i64,
            stride = [1, 1, 1],
            kernelId = 0 : i64
        } : <16x16x1x1x1xf32, 16x1x1x1x1>, <1x16x4x4x4xf32, 1024x64x16x4x1> -> <1x16x4x4x4xf32, 1024x64x16x4x1>
        return %0 : !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1>
    }
}

