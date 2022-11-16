// RUN: mlir-opt --tosa-partition %s | FileCheck %s
// RUN: mlir-opt --tosa-partition=partition-tag=one %s | FileCheck %s --check-prefix=ONE
// RUN: mlir-opt --tosa-partition='anchor-ops=tosa.depthwise_conv2d partition-tag=two' %s | FileCheck %s --check-prefix=TWO
// RUN: mlir-opt --tosa-partition='anchor-ops=tosa.depthwise_conv2d trailing-only partition-tag=three' %s | FileCheck %s --check-prefix=THREE
// RUN: mlir-opt --tosa-partition='anchor-ops=tosa.conv2d partition-tag=four' %s | FileCheck %s --check-prefix=FOUR

// RUN: mlir-opt --test-tosa-partition-options=default %s | FileCheck %s --check-prefix=CHECK
// RUN: mlir-opt --test-tosa-partition-options=depthwise-only %s | FileCheck %s --check-prefix=TWO
// RUN: mlir-opt --test-tosa-partition-options=conv-only %s | FileCheck %s --check-prefix=FOUR
// RUN: mlir-opt --test-tosa-partition-options=attr-one %s | FileCheck %s --check-prefix=ONE
// RUN: mlir-opt --test-tosa-partition-options=nofront-arg %s | FileCheck %s --check-prefix=THREE

// CHECK-LABEL: func private @test_fusion8__part_0
// CHECK-SAME: attributes {{{.*}}kernel}
// CHECK-NEXT: arith.constant
// CHECK-NEXT: tosa.transpose
// CHECK-NEXT: tosa.depthwise_conv2d
// CHECK-NEXT: tosa.abs
// CHECK-NEXT: tosa.add
// CHECK-NEXT: return
// CHECK: func private @test_fusion8__part_1
// CHECK-NEXT: tosa.conv2d
// CHECK-NEXT: return
// CHECK: func @test_fusion8
// CHECK: call @test_fusion8__part_1
// CHECK: call @test_fusion8__part_0

// ONE-LABEL: func private @test_fusion8__part_0
// ONE-SAME: attributes {{{.*}}one}
// ONE-NEXT: arith.constant
// ONE-NEXT: tosa.transpose
// ONE-NEXT: tosa.depthwise_conv2d
// ONE-NEXT: tosa.abs
// ONE-NEXT: tosa.add
// ONE-NEXT: return
// ONE: func @test_fusion8
// ONE: call @test_fusion8__part_0

// TWO-LABEL: func private @test_fusion8__part_0
// TWO-NEXT: arith.constant
// TWO-NEXT: tosa.transpose
// TWO-NEXT: tosa.depthwise_conv2d
// TWO-NEXT: tosa.abs
// TWO-NEXT: tosa.add
// TWO-NEXT: return
// TWO: func @test_fusion8
// TWO: tosa.conv2d
// TWO: call @test_fusion8__part_0

// THREE-LABEL: func private @test_fusion8__part_0
// THREE-NEXT: arith.constant
// THREE-NEXT: tosa.transpose
// THREE-NEXT: tosa.depthwise_conv2d
// THREE-NEXT: tosa.abs
// THREE-NEXT: tosa.add
// THREE-NEXT: return
// THREE: func @test_fusion8
// THREE: tosa.conv2d
// THREE: call @test_fusion8__part_0

// FOUR-LABEL: func private @test_fusion8__part_0
// FOUR-SAME: attributes {{{.*}}four}
// FOUR-NEXT: tosa.conv2d
// FOUR-NEXT: tosa.add
// FOUR-NEXT: return
// FOUR: func @test_fusion8
// FOUR: call @test_fusion8__part_0

func.func @test_fusion8(%arg0: tensor<128x32x32x8xf32>, %arg1: tensor<128x8x3x3xf32>, %arg2: tensor<8xf32>, %arg3: tensor<128x8x32x32xf32>, %arg4: tensor<128x8x3x3xf32>, %arg5: tensor<8xf32>) -> tensor<128x128x30x30xf32> {
  %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
  %0 = "tosa.transpose"(%arg0, %cst) {changing_layout_root = false} : (tensor<128x32x32x8xf32>, tensor<4xi64>) -> tensor<128x8x32x32xf32>
  %1 = "tosa.depthwise_conv2d"(%0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %2 = "tosa.conv2d"(%arg3, %arg4, %arg5) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<128x8x32x32xf32>, tensor<128x8x3x3xf32>, tensor<8xf32>) -> tensor<128x128x30x30xf32>
  %3 = "tosa.abs"(%1) {} : (tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  %4 = "tosa.add"(%3, %2) {} : (tensor<128x128x30x30xf32>, tensor<128x128x30x30xf32>) -> tensor<128x128x30x30xf32>
  return %4 : tensor<128x128x30x30xf32>
}
