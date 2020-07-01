// mlir-opt -convert-linalg-to-loops -lower-affine -convert-scf-to-std -convert-std-to-llvm conv2d.mlir
func @conv2d(%arg0: memref<?x?x?x?xf32>,
             %arg1: memref<?x?x?x?xf32>,
             %arg2: memref<?x?x?x?xf32>) {
  linalg.conv(%arg0, %arg1, %arg2) {filter=[1,1], dilations=[1,1], padding=dense<0> : tensor<2x2xi64>} : memref<?x?x?x?xf32>,
                                     memref<?x?x?x?xf32>,
                                     memref<?x?x?x?xf32>
  return
}
