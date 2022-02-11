// mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-cf -convert-std-to-llvm conv2d.mlir
// filter: YXCK
// input : NHWC
// output: NHWK
func @conv2d(%filter: memref<3x3x8x128xf32>,
             %input : memref<128x32x32x8xf32>,
             %output: memref<128x30x30x128xf32>) {
  linalg.conv(%filter, %input, %output) {strides=[1,1], dilations=[1,1], padding=dense<0> : tensor<2x2xi64>} : memref<3x3x8x128xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>
  return
}
