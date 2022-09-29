// RUN: mlir-opt --split-input-file --tosa-partition %s -verify-each=0 -o - | FileCheck %s

// CHECK-LABEL: func private @test_fusion__part_0
// CHECK: tosa.conv2d
// CHECK: tosa.abs
// CHECK: return
// CHECK: func @test_fusion
// CHECK: call @test_fusion__part_0
func.func @test_fusion(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %1 : tensor<128x128x30x30xf32>
}


// CHECK-LABEL: func private @test_fusion2__part_0
// CHECK: tosa.conv2d
// CHECK: tosa.negate
// CHECK: return
// CHECK: func @test_fusion2
// CHECK: call @test_fusion2__part_0
func.func @test_fusion2(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.negate"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %1 : tensor<128x128x30x30xf32>
}


// CHECK-LABEL: func private @test_fusion3__part_0
// CHECK: tosa.conv2d
// CHECK: tosa.abs
// CHECK: tosa.negate
// CHECK: return
// CHECK: func @test_fusion3
// CHECK: call @test_fusion3__part_0
func.func @test_fusion3(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.negate"(%1) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %2 : tensor<128x128x30x30xf32>
}


// CHECK-LABEL: func private @test_fusion4__part_0
// CHECK: tosa.conv2d
// CHECK: tosa.abs
// +++pf:  This test used to absorb the tosa.add, too, but doesn't now.
// CHECK: return
// CHECK: func @test_fusion4
// CHECK: call @test_fusion4__part_0
func.func @test_fusion4(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.add"(%0, %1) {} : (tensor<128x128x30x30xf32>, tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %2 : tensor<128x128x30x30xf32>
}


// CHECK-LABEL: func private @test_fusion5__part_0
// CHECK-NEXT: tosa.conv2d
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: tosa.add
// CHECK-NEXT: return
// CHECK: func private @test_fusion5__part_1
// CHECK-NEXT: tosa.conv2d
// CHECK-NEXT: return
// CHECK: func @test_fusion5
// CHECK-NEXT: call @test_fusion5__part_1
// CHECK-NEXT: call @test_fusion5__part_0
func.func @test_fusion5(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>, %arg3: tensor<128x8x32x32xf32>, %arg4: tensor<128x8x3x3xf32>, %arg5: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %1 = "tosa.conv2d"(%arg3, %arg4, %arg5) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.abs"(%0) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %3 = "tosa.add"(%2, %1) {} : (tensor<128x128x30x30xf32>, tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %3 : tensor<128x128x30x30xf32>
}


// CHECK-LABEL: func private @test_fusion6__part_0
// CHECK-NEXT: tosa.conv2d
// CHECK-NEXT: return
// CHECK: func @test_fusion6
// CHECK-NEXT: call @test_fusion6__part_0
// CHECK-NEXT: return
func.func @test_fusion6(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  return %0 : tensor<128x128x30x30xf32>
}

// CHECK-LABEL: func private @test_fusion7__part_0
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: tosa.conv2d
// CHECK-NEXT: return
// CHECK: func @test_fusion7
// CHECK-NEXT: call @test_fusion7__part_0
// CHECK-NEXT: return
func.func @test_fusion7(%arg0: tensor<128x8x32x32xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %0 = "tosa.abs"(%arg0) {} : (tensor<128x8x32x32xf32>) -> tensor<128x8x32x32xf32>
  %1 = "tosa.conv2d"(%0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  return %1 : tensor<128x128x30x30xf32>
}
