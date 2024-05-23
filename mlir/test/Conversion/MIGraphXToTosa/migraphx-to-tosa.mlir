// RUN: rocmlir-opt -split-input-file --migraphx-to-tosa %s | FileCheck %s

module  {
  // CHECK-LABEL: func.func @ConvBias
  func.func @ConvBias(%arg0: !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> {
    %0 = migraphx.literal (dense<1.000000e+00> : tensor<64x64x1x1xf32>) : <64x64x1x1xf32, 64x1x1x1>
    %1 = migraphx.convolution %arg0, %0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x64x56x56xf32, 200704x3136x56x1>, <64x64x1x1xf32, 64x1x1x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    %2 = migraphx.literal (dense<2.000000e+00> : tensor<1x64x56x56xf32>) : <1x64x56x56xf32, 200704x3136x56x1>
    %3 = migraphx.add %1, %2 : <1x64x56x56xf32, 200704x3136x56x1>, <1x64x56x56xf32, 200704x3136x56x1> -> <1x64x56x56xf32, 200704x3136x56x1>
     return %3 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  }
  // CHECK-LABEL: func.func @ConvNoBias
  // CHECK-SAME: ([[arg0:%.+]]: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  func.func @ConvNoBias(%arg0: !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> {
    %0 = migraphx.literal (dense<3.000000e+00> : tensor<64x64x1x1xf32>) : <1x64x56x56xf32, 200704x3136x56x1>
    // CHECK: [[trIn:%.+]] = tosa.transpose {{.*}}[[arg0]]{{.*}} : (tensor<1x64x56x56xf32>, tensor<4xi64>) -> tensor<1x56x56x64xf32>
    // CHECK: [[conv:%.+]] = tosa.conv2d {{.*}}[[trIn]]
    %1 = migraphx.convolution %arg0, %0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x64x56x56xf32, 200704x3136x56x1>, <1x64x56x56xf32, 200704x3136x56x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    // CHECK: [[trOut:%.+]] = tosa.transpose {{.*}}[[conv]]
     return %1 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  }

}

// -----

// Note: if we start running constant folding for transposes in migraphx-to-tosa
// all the transposes should go away
// CHECK-LABEL: @convNHWC
// CHECK-SAME: (%{{.*}}: tensor<1x5x5x4xf32>, %{{.*}}: tensor<7x3x3x4xf32>) -> tensor<1x3x3x7xf32>
func.func @convNHWC(%in: !migraphx.shaped<1x4x5x5xf32, 100x1x20x4>, %fil: !migraphx.shaped<7x4x3x3xf32, 36x1x12x4>) -> !migraphx.shaped<1x7x3x3xf32, 63x1x21x7> {
  // CHECK-4 tosa.transpose
  // CHECK: tosa.conv2d
  // CHECK-SAME: (tensor<1x5x5x4xf32>, tensor<7x3x3x4xf32>, tensor<7xf32>) -> tensor<1x3x3x7xf32>
  // CHECK-2: tosa.transpose
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x5x5xf32, 100x1x20x4>, <7x4x3x3xf32, 36x1x12x4> -> <1x7x3x3xf32, 63x1x21x7>
  func.return %out : !migraphx.shaped<1x7x3x3xf32, 63x1x21x7>
}

// -----

// Tests for non-standard shapes.

// CHECK-LABEL: @transposed
// CHECK-SAME: ([[arg0:%.+]]: tensor<3x4xf32>) -> tensor<3x4xf32>
// CHECK: [[perm:%.+]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi64>}>
// CHECK: [[logical:%.+]] = tosa.transpose [[arg0]], [[perm]]
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]], [[perm]]
// CHECK: return [[outMem]]
func.func @transposed(%arg0: !migraphx.shaped<4x3xf32, 1x4>) -> !migraphx.shaped<4x3xf32, 1x4> {
  %op = migraphx.floor %arg0 : <4x3xf32, 1x4> -> <4x3xf32, 1x4>
  func.return %op : !migraphx.shaped<4x3xf32, 1x4>
}

// CHECK-LABEL: @broadcast
// CHECK-SAME: ([[arg0:%.+]]: tensor<4x1xf32>, [[arg1:%.+]]: tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK: [[zero:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4x3xf32>}>
// CHECK: [[broadcast:%.+]] = tosa.add [[zero]], [[arg0]]
// CHECK: [[op:%.+]] = tosa.sub [[broadcast]], [[arg1]]
// CHECK: return [[op]]
func.func @broadcast(%arg0: !migraphx.shaped<4x3xf32, 1x0>, %arg1: !migraphx.shaped<4x3xf32, 3x1>) -> !migraphx.shaped<4x3xf32, 3x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3xf32, 1x0>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
  func.return %op : !migraphx.shaped<4x3xf32, 3x1>
}

// CHECK-LABEL: @sliced
// CHECK-SAME: ([[arg0:%.+]]: tensor<4x5xf32>, [[arg1:%.+]]: tensor<4x3xf32>) -> tensor<4x3xf32>
// CHECK: [[sliced:%.+]] = tosa.slice [[arg0]] {size = array<i64: 4, 3>, start = array<i64: 0, 0>}
// CHECK: [[op:%.+]] = tosa.sub [[sliced]], [[arg1]]
// CHECK: return [[op]]
func.func @sliced(%arg0: !migraphx.shaped<4x3xf32, 5x1>, %arg1: !migraphx.shaped<4x3xf32, 3x1>) -> !migraphx.shaped<4x3xf32, 3x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3xf32, 5x1>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
  func.return %op : !migraphx.shaped<4x3xf32, 3x1>
}

// CHECK-LABEL: @everything
// CHECK-SAME: ([[arg0:%.+]]: tensor<5x6x1xf32>, [[arg1:%.+]]: tensor<4x3x5xf32>) -> tensor<4x3x5xf32>
// CHECK: [[perm:%.+]] = "tosa.const"() <{value = dense<[1, 2, 0]> : tensor<3xi64>}>
// CHECK: [[transposed:%.+]] = tosa.transpose [[arg0]], [[perm]]
// CHECK: [[sliced:%.+]] = tosa.slice [[transposed]] {size = array<i64: 4, 1, 5>, start = array<i64: 0, 0, 0>}
// CHECK: [[zero:%.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<4x3x5xf32>}>
// CHECK: [[broadcast:%.+]] = tosa.add [[zero]], [[sliced]]
// CHECK: [[op:%.+]] = tosa.sub [[broadcast]], [[arg1]]
// CHECK: return [[op]]
func.func @everything(%arg0: !migraphx.shaped<4x3x5xf32, 1x0x6>, %arg1: !migraphx.shaped<4x3x5xf32, 15x5x1>) -> !migraphx.shaped<4x3x5xf32, 15x5x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3x5xf32, 1x0x6>, <4x3x5xf32, 15x5x1> -> <4x3x5xf32, 15x5x1>
  func.return %op : !migraphx.shaped<4x3x5xf32, 15x5x1>
}

// CHECK-LABEL: @matchingLogicalTypes
// CHECK-SAME: ([[arg0:%.+]]: tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: [[perm:%.+]] = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi64>}>
// CHECK: [[logical:%.+]] = tosa.transpose [[arg0]], [[perm]]
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]], [[perm]]
// CHECK: return [[outMem]]
func.func @matchingLogicalTypes(%arg0: !migraphx.shaped<3x3xf32, 1x3>) -> !migraphx.shaped<3x3xf32, 1x3> {
  %op = migraphx.floor %arg0 : <3x3xf32, 1x3> -> <3x3xf32, 1x3>
  func.return %op : !migraphx.shaped<3x3xf32, 1x3>
}

// CHECK-LABEL: @transposeWithUnitDims
// CHECK-SAME: ([[arg0:%.+]]: tensor<3x1x3xf32>) -> tensor<3x1x3xf32>
// CHECK: [[perm:%.+]] = "tosa.const"() <{value = dense<[2, 1, 0]> : tensor<3xi64>}>
// CHECK: [[logical:%.+]] = tosa.transpose [[arg0]], [[perm]]
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]], [[perm]]
// CHECK: return [[outMem]]
func.func @transposeWithUnitDims(%arg0: !migraphx.shaped<3x1x3xf32, 1x3x3>) -> !migraphx.shaped<3x1x3xf32, 1x3x3> {
  %op = migraphx.floor %arg0 : <3x1x3xf32, 1x3x3> -> <3x1x3xf32, 1x3x3>
  func.return %op : !migraphx.shaped<3x1x3xf32, 1x3x3>
}

// CHECK-LABEL: @needStableSort
// CHECK-SAME: ([[arg0:%.+]]: tensor<3x1x1xf32>) -> tensor<3x1x1xf32>
// CHECK: [[op:%.+]] = tosa.floor [[arg0]]
// CHECK: return [[op]]
func.func @needStableSort(%arg0: !migraphx.shaped<3x1x1xf32, 1x1x1>) -> !migraphx.shaped<3x1x1xf32, 1x1x1> {
  %op = migraphx.floor %arg0 : <3x1x1xf32, 1x1x1> -> <3x1x1xf32, 1x1x1>
  func.return %op : !migraphx.shaped<3x1x1xf32, 1x1x1>
}

// CHECK-LABEL: @scalar
// CHECK-SAME: ([[arg0:%.+]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: [[op:%.+]] = tosa.floor [[arg0]]
// CHECK: return [[op]]
func.func @scalar(%arg0: !migraphx.shaped<1xf32, 0>) -> !migraphx.shaped<1xf32, 0> {
  %op = migraphx.floor %arg0 : <1xf32, 0> -> <1xf32, 0>
  func.return %op : !migraphx.shaped<1xf32, 0>
}

// CHECK-LABEL: @scalar0d
// CHECK-SAME: ([[arg0:%.+]]: tensor<f32>) -> tensor<f32>
// CHECK: [[op:%.+]] = tosa.floor [[arg0]]
// CHECK: return [[op]]
func.func @scalar0d(%arg0: !migraphx.shaped<f32>) -> !migraphx.shaped<f32> {
  %op = migraphx.floor %arg0 : <f32> -> <f32>
  func.return %op : !migraphx.shaped<f32>
}


// -----

// CHECK-LABEL: @conv3d_add
// CHECK-SAME: (%{{.*}}: tensor<4x1x1x1x1xf32>, %{{.*}}: tensor<2x3x5x5x5xf32>, %{{.*}}: tensor<4x3x2x2x2xf32>) -> tensor<2x4x2x2x2xf32>
func.func @conv3d_add(%arg0: !migraphx.shaped<2x4x2x2x2xf32, 0x1x0x0x0>, %arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>  {
  // CHECK-COUNT-3: tosa.transpose
  // CHECK: tosa.conv3d
  // CHECK-SAME: (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>) -> tensor<2x2x2x2x4xf32>
  // CHECK-2: tosa.transpose
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  %1 = migraphx.add %0, %arg0 : <2x4x2x2x2xf32, 32x8x4x2x1>, <2x4x2x2x2xf32, 0x1x0x0x0> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  return %1 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}
