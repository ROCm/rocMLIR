// RUN: mlir-miopen-driver -host-pipeline full %s | FileCheck %s

module {
// CHECK: func @resnet50(%[[ARG0:.*]]: memref<1x32x32x64xf32>, %[[ARG1:.*]]: memref<64x3x3x64xf32>, %[[ARG2:.*]]: memref<64x3x3x64xf32>, %[[ARG3:.*]]: memref<1x32x32x64xf32>) 
// CHECK: %[[MEM0:.*]] = memref.alloc
// CHECK: %[[TOKEN0:.*]] = async.launch @resnet50_outlined_part_0 (%{{.*}}, %{{.*}}, %[[MEM0]])
// CHECK: %[[TOKEN1:.*]] = async.launch @resnet50_outlined_part_1 [%[[TOKEN0]]] (%[[MEM0]], %{{.*}}, %{{.*}}, %{{.*}})
// CHECK: async.await %[[TOKEN1]] : !async.token

  func @resnet50(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x3x3x64xf32>, %arg2: tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32> {

    %cst = arith.constant dense<0.0> : tensor<64xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %1 = "tosa.clamp"(%0) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    //%cst1 = arith.constant dense<0.0> : tensor<64xf32>
    %2 = "tosa.conv2d"(%1, %arg2, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %3 = "tosa.clamp"(%2) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %4 = "tosa.add"(%arg0, %3)
     : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    return %4 : tensor<1x32x32x64xf32>
  }
}

