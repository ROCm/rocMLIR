module {
  dxgml.module @simple_gemm {
    dxgml.entry_point @simple_gemm_relu(
        %A    : !dxgml.tensor<2x4x!dxgml.float32>,
        %B    : !dxgml.tensor<4x3x!dxgml.float32>,
        %bias : !dxgml.tensor<2x3x!dxgml.float32>
    ) -> !dxgml.tensor<2x3x!dxgml.float32>
    attributes {
      torch.onnx_meta.ir_version = 6 : si64,
      torch.onnx_meta.opset_version = 12 : si64,
      torch.onnx_meta.producer_name = "pytorch",
      torch.onnx_meta.producer_version = "2.0.0"
    } {
      // Step 1: Matrix multiplication  A(2x4) @ B(4x3) -> gemm(2x3)
      %gemm = dxgml_op.gemm (%A, %B)
        : (!dxgml.tensor<2x4x!dxgml.float32>, !dxgml.tensor<4x3x!dxgml.float32>)
        -> !dxgml.tensor<2x3x!dxgml.float32>

      // Step 2: Add bias
      %biased = dxgml_op.add (%gemm, %bias)
        : (!dxgml.tensor<2x3x!dxgml.float32>, !dxgml.tensor<2x3x!dxgml.float32>)
        -> !dxgml.tensor<2x3x!dxgml.float32>

      // Step 3: ReLU activation
      %result = dxgml_op.relu (%biased)
        : (!dxgml.tensor<2x3x!dxgml.float32>) -> !dxgml.tensor<2x3x!dxgml.float32>

      dxgml.return %result : !dxgml.tensor<2x3x!dxgml.float32>
    }
  }
}