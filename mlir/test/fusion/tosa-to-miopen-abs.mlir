// RUN: mlir-opt --tosa-to-miopen --tosa-to-linalg-on-tensors --linalg-fuse-elementwise-ops --linalg-bufferize --func-bufferize --buffer-results-to-out-params --finalizing-bufferize -miopen-copy-opt -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 -miopen-linalg-align  %s -o -| FileCheck %s

// CHECK-LABEL: test_fusion
// CHECK: linalg.generic {indexing_maps = [#map{{.*}}, #map{{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins(%{{.*}} : memref<1x8x8xf32, 5>) outs(%{{.*}} : memref<1x8x8xf32, 5>)

func @test_fusion(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x3x3x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x32x32x8xf32>, tensor<128x3x3x8xf32>, tensor<8xf32>) -> tensor<128x30x30x128xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x30x30x128xf32>) -> tensor<128x30x30x128xf32>

  return %1 : tensor<128x30x30x128xf32>
}

// -----

