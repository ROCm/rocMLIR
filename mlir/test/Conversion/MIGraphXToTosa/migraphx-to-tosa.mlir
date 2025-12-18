// RUN: rocmlir-opt -split-input-file --migraphx-to-tosa %s | FileCheck %s

module  {
  // CHECK-LABEL: func.func @ConvNoBias
  // CHECK-SAME: ([[arg0:%.+]]: tensor<200704xf32>) -> tensor<200704xf32>
  func.func @ConvNoBias(%arg0: !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>) -> !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1> {
    // CHECK: [[inExp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<200704xf32>, !tosa.shape<4>) -> tensor<1x64x56x56xf32>
    %0 = migraphx.literal (dense<3.000000e+00> : tensor<64x64x1x1xf32>) : <1x64x56x56xf32, 200704x3136x56x1>
    // CHECK: [[trIn:%.+]] = tosa.transpose {{.*}}[[inExp]]{{.*}} : (tensor<1x64x56x56xf32>) -> tensor<1x56x56x64xf32>
    // CHECK: [[conv:%.+]] = tosa.conv2d {{.*}}[[trIn]]
    %1 = migraphx.convolution %arg0, %0 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x64x56x56xf32, 200704x3136x56x1>, <1x64x56x56xf32, 200704x3136x56x1> -> <1x64x56x56xf32, 200704x3136x56x1>
    // CHECK: [[trOut:%.+]] = tosa.transpose {{.*}}[[conv]]
    // CHECK: [[outFlat:%.+]] = tosa.reshape [[trOut]], %{{.*}} : (tensor<1x64x56x56xf32>, !tosa.shape<1>) -> tensor<200704xf32>
     return %1 : !migraphx.shaped<1x64x56x56xf32, 200704x3136x56x1>
  }
}

// -----

// Note: if we start running constant folding for transposes in migraphx-to-tosa
// all the transposes should go away
// CHECK-LABEL: @convNHWC
// CHECK-SAME: ([[arg0:%.+]]: tensor<100xf32>, [[arg1:%.+]]: tensor<252xf32>) -> tensor<63xf32>
func.func @convNHWC(%in: !migraphx.shaped<1x4x5x5xf32, 100x1x20x4>, %fil: !migraphx.shaped<7x4x3x3xf32, 36x1x12x4>) -> !migraphx.shaped<1x7x3x3xf32, 63x1x21x7> {
  // CHECK: [[arg1Exp:%.+]] = tosa.reshape [[arg1]]
  // CHECK-SAME: (tensor<252xf32>, !tosa.shape<4>) -> tensor<7x3x3x4xf32>
  // CHECK: [[arg1Tr1:%.+]] = tosa.transpose [[arg1Exp]]
  // CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]]
  // CHECK-SAME: (tensor<100xf32>, !tosa.shape<4>) -> tensor<1x5x5x4xf32>
  // CHECK: [[arg0Tr1:%.+]] = tosa.transpose [[arg0Exp]]
  // CHECK: [[arg0Tr2:%.+]] = tosa.transpose [[arg0Tr1]]
  // CHECK: [[arg1Tr2:%.+]] = tosa.transpose [[arg1Tr1]]
  // CHECK: [[conv:%.+]] = tosa.conv2d
  // CHECK-SAME: (tensor<1x5x5x4xf32>, tensor<7x3x3x4xf32>, tensor<7xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x7xf32>
  // CHECK: [[convTr1:%.+]] = tosa.transpose [[conv]]
  // CHECK: [[convTr2:%.+]] = tosa.transpose [[convTr1]]
  // CHECK: [[convFlat:%.+]] = tosa.reshape [[convTr2]]
  // CHECK-SAME: (tensor<1x3x3x7xf32>, !tosa.shape<1>) -> tensor<63xf32>
  // CHECK: return [[convFlat]]
  %out = migraphx.convolution %in, %fil {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x4x5x5xf32, 100x1x20x4>, <7x4x3x3xf32, 36x1x12x4> -> <1x7x3x3xf32, 63x1x21x7>
  func.return %out : !migraphx.shaped<1x7x3x3xf32, 63x1x21x7>
}

// -----

// Tests for non-standard shapes.

// CHECK-LABEL: @transposed
// CHECK-SAME: ([[arg0:%.+]]: tensor<12xf32>) -> tensor<12xf32>
// CHECK: [[exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<12xf32>, !tosa.shape<2>) -> tensor<3x4xf32>
// CHECK: [[logical:%.+]] = tosa.transpose [[exp]] {perms = array<i32: 1, 0>}
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]] {perms = array<i32: 1, 0>}
// CHECK: [[outFlat:%.+]] = tosa.reshape [[outMem]], %{{.*}} : (tensor<3x4xf32>, !tosa.shape<1>) -> tensor<12xf32>
// CHECK: return [[outFlat]]
func.func @transposed(%arg0: !migraphx.shaped<4x3xf32, 1x4>) -> !migraphx.shaped<4x3xf32, 1x4> {
  %op = migraphx.floor %arg0 : <4x3xf32, 1x4> -> <4x3xf32, 1x4>
  func.return %op : !migraphx.shaped<4x3xf32, 1x4>
}

// CHECK-LABEL: @broadcast
// CHECK-SAME: ([[arg0:%.+]]: tensor<4xf32>, [[arg1:%.+]]: tensor<12xf32>) -> tensor<12xf32>
// CHECK: [[arg1Exp:%.+]] = tosa.reshape [[arg1]], %{{.*}} : (tensor<12xf32>, !tosa.shape<2>) -> tensor<4x3xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<4xf32>, !tosa.shape<2>) -> tensor<4x1xf32>
// CHECK: [[one:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4x3xf32>}>
// CHECK: [[broadcast:%.+]] = tosa.mul [[one]], [[arg0Exp]], %{{.*}}
// CHECK: [[op:%.+]] = tosa.sub [[broadcast]], [[arg1Exp]]
// CHECK: [[opFlat:%.+]] = tosa.reshape [[op]], %{{.*}} : (tensor<4x3xf32>, !tosa.shape<1>) -> tensor<12xf32>
// CHECK: return [[opFlat]]
func.func @broadcast(%arg0: !migraphx.shaped<4x3xf32, 1x0>, %arg1: !migraphx.shaped<4x3xf32, 3x1>) -> !migraphx.shaped<4x3xf32, 3x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3xf32, 1x0>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
  func.return %op : !migraphx.shaped<4x3xf32, 3x1>
}

// CHECK-LABEL: @sliced
// CHECK-SAME: ([[arg0:%.+]]: tensor<20xf32>, [[arg1:%.+]]: tensor<12xf32>) -> tensor<12xf32>
// CHECK: [[arg1Exp:%.+]] = tosa.reshape [[arg1]], %{{.*}} : (tensor<12xf32>, !tosa.shape<2>) -> tensor<4x3xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<20xf32>, !tosa.shape<2>) -> tensor<4x5xf32>
// CHECK: [[sliced:%.+]] = tosa.slice [[arg0Exp]], %{{.*}}, %{{.*}} : (tensor<4x5xf32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<4x3xf32>
// CHECK: [[op:%.+]] = tosa.sub [[sliced]], [[arg1Exp]]
// CHECK: [[opFlat:%.+]] = tosa.reshape [[op]], %{{.*}} : (tensor<4x3xf32>, !tosa.shape<1>) -> tensor<12xf32>
// CHECK: return [[opFlat]]
func.func @sliced(%arg0: !migraphx.shaped<4x3xf32, 5x1>, %arg1: !migraphx.shaped<4x3xf32, 3x1>) -> !migraphx.shaped<4x3xf32, 3x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3xf32, 5x1>, <4x3xf32, 3x1> -> <4x3xf32, 3x1>
  func.return %op : !migraphx.shaped<4x3xf32, 3x1>
}

// CHECK-LABEL: @everything
// CHECK-SAME: ([[arg0:%.+]]: tensor<30xf32>, [[arg1:%.+]]: tensor<60xf32>) -> tensor<60xf32>
// CHECK: [[arg1Exp:%.+]] = tosa.reshape [[arg1]], %{{.*}} : (tensor<60xf32>, !tosa.shape<3>) -> tensor<4x3x5xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<30xf32>, !tosa.shape<3>) -> tensor<5x6x1xf32>
// CHECK: [[transposed:%.+]] = tosa.transpose [[arg0Exp]] {perms = array<i32: 1, 2, 0>}
// CHECK: [[sliced:%.+]] = tosa.slice [[transposed]], %{{.*}}, %{{.*}} : (tensor<6x1x5xf32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<4x1x5xf32>
// CHECK: [[one:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<4x3x5xf32>}>
// CHECK: [[broadcast:%.+]] = tosa.mul [[one]], [[sliced]], %{{.*}}
// CHECK: [[op:%.+]] = tosa.sub [[broadcast]], [[arg1Exp]]
// CHECK: [[opFlat:%.+]] = tosa.reshape [[op]], %{{.*}} : (tensor<4x3x5xf32>, !tosa.shape<1>) -> tensor<60xf32>
// CHECK: return [[opFlat]]
func.func @everything(%arg0: !migraphx.shaped<4x3x5xf32, 1x0x6>, %arg1: !migraphx.shaped<4x3x5xf32, 15x5x1>) -> !migraphx.shaped<4x3x5xf32, 15x5x1> {
  %op = migraphx.sub %arg0, %arg1 : <4x3x5xf32, 1x0x6>, <4x3x5xf32, 15x5x1> -> <4x3x5xf32, 15x5x1>
  func.return %op : !migraphx.shaped<4x3x5xf32, 15x5x1>
}

// CHECK-LABEL: @matchingLogicalTypes
// CHECK-SAME: ([[arg0:%.+]]: tensor<9xf32>) -> tensor<9xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<9xf32>, !tosa.shape<2>) -> tensor<3x3xf32>
// CHECK: [[logical:%.+]] = tosa.transpose [[arg0Exp]] {perms = array<i32: 1, 0>}
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]] {perms = array<i32: 1, 0>}
// CHECK: [[outFlat:%.+]] = tosa.reshape [[outMem]], %{{.*}} : (tensor<3x3xf32>, !tosa.shape<1>) -> tensor<9xf32>
// CHECK: return [[outFlat]]
func.func @matchingLogicalTypes(%arg0: !migraphx.shaped<3x3xf32, 1x3>) -> !migraphx.shaped<3x3xf32, 1x3> {
  %op = migraphx.floor %arg0 : <3x3xf32, 1x3> -> <3x3xf32, 1x3>
  func.return %op : !migraphx.shaped<3x3xf32, 1x3>
}

// CHECK-LABEL: @transposeWithUnitDims
// CHECK-SAME: ([[arg0:%.+]]: tensor<9xf32>) -> tensor<9xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<9xf32>, !tosa.shape<3>) -> tensor<3x1x3xf32>
// CHECK: [[logical:%.+]] = tosa.transpose [[arg0Exp]] {perms = array<i32: 2, 1, 0>}
// CHECK: [[op:%.+]] = tosa.floor [[logical]]
// CHECK: [[outMem:%.+]] = tosa.transpose [[op]] {perms = array<i32: 2, 1, 0>}
// CHECK: [[outFlat:%.+]] = tosa.reshape [[outMem]], %{{.*}} : (tensor<3x1x3xf32>, !tosa.shape<1>) -> tensor<9xf32>
// CHECK: return [[outFlat]]
func.func @transposeWithUnitDims(%arg0: !migraphx.shaped<3x1x3xf32, 1x3x3>) -> !migraphx.shaped<3x1x3xf32, 1x3x3> {
  %op = migraphx.floor %arg0 : <3x1x3xf32, 1x3x3> -> <3x1x3xf32, 1x3x3>
  func.return %op : !migraphx.shaped<3x1x3xf32, 1x3x3>
}

// CHECK-LABEL: @needStableSort
// CHECK-SAME: ([[arg0:%.+]]: tensor<3xf32>) -> tensor<3xf32>
// CHECK: [[arg0Exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<3xf32>, !tosa.shape<3>) -> tensor<3x1x1xf32>
// CHECK: [[op:%.+]] = tosa.floor [[arg0Exp]]
// CHECK: [[opFlat:%.+]] = tosa.reshape [[op]], %{{.*}} : (tensor<3x1x1xf32>, !tosa.shape<1>) -> tensor<3xf32>
// CHECK: return [[opFlat]]
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
// CHECK-SAME: ([[arg0:%.+]]: tensor<1xf32>) -> tensor<1xf32>
// CHECK: [[arg0Col:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<1xf32>, !tosa.shape<0>) -> tensor<f32>
// CHECK: [[op:%.+]] = tosa.floor [[arg0Col]]
// CHECK: [[opExp:%.+]] = tosa.reshape [[op]], {{.*}} : (tensor<f32>, !tosa.shape<1>) -> tensor<1xf32>
// CHECK: return [[opExp]]
func.func @scalar0d(%arg0: !migraphx.shaped<f32>) -> !migraphx.shaped<f32> {
  %op = migraphx.floor %arg0 : <f32> -> <f32>
  func.return %op : !migraphx.shaped<f32>
}


// -----

// CHECK-LABEL: @conv3d_add
// CHECK-SAME: (%{{.*}}: tensor<4xf32>, %{{.*}}: tensor<750xf32>, %{{.*}}: tensor<96xf32>) -> tensor<64xf32>
func.func @conv3d_add(%arg0: !migraphx.shaped<2x4x2x2x2xf32, 0x1x0x0x0>, %arg1: !migraphx.shaped<2x3x5x5x5xf32, 375x125x25x5x1>, %arg2: !migraphx.shaped<4x3x2x2x2xf32, 24x8x4x2x1>) -> !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1> {
  // CHECK-COUNT-3: tosa.transpose
  // CHECK: tosa.conv3d
  // CHECK-SAME: (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x2x2x2x4xf32>
  // CHECK-2: tosa.transpose
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [2, 2, 2], group = 1 : i64, padding = [0, 0, 0, 0, 0, 0], padding_mode = 0 : i64, stride = [2, 2, 2]} : <2x3x5x5x5xf32, 375x125x25x5x1>, <4x3x2x2x2xf32, 24x8x4x2x1> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  %1 = migraphx.add %0, %arg0 : <2x4x2x2x2xf32, 32x8x4x2x1>, <2x4x2x2x2xf32, 0x1x0x0x0> -> <2x4x2x2x2xf32, 32x8x4x2x1>
  return %1 : !migraphx.shaped<2x4x2x2x2xf32, 32x8x4x2x1>
}

// CHECK-LABEL: @conv1d_add
// CHECK-SAME: (%{{.*}}: tensor<64xf32>, %{{.*}}: tensor<672xf32>, %{{.*}}: tensor<1344xf32>) -> tensor<14336xf32>
func.func @conv1d_add(%arg0: !migraphx.shaped<1x64x224xf32, 0x1x0>, %arg1: !migraphx.shaped<1x3x224xf32, 672x224x1>, %arg2: !migraphx.shaped<64x3x7xf32, 21x7x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  // CHECK-COUNT-3: tosa.transpose
  // CHECK: tosa.conv2d
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 3, 3, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x224x1x3xf32>, tensor<64x7x1x3xf32>, tensor<64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x224x1x64xf32>
  // CHECK-2: tosa.transpose
  // CHECK: tosa.add
  %0 = migraphx.convolution %arg1, %arg2 {dilation = [1], group = 1 : i64, padding = [3, 3], padding_mode = 0 : i64, stride = [1]} : <1x3x224xf32, 672x224x1>, <64x3x7xf32, 21x7x1> -> <1x64x224xf32, 14336x224x1>
  %1 = migraphx.add %0, %arg0 : <1x64x224xf32, 14336x224x1>, <1x64x224xf32, 0x1x0> -> <1x64x224xf32, 14336x224x1>
  return %1 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}

// CHECK-LABEL: @conv2d_float8
// CHECK-SAME: (%{{.*}}: tensor<256xf8E5M2>, %{{.*}}: tensor<256xf8E5M2>) -> tensor<256xf8E5M2>
func.func @conv2d_float8(%arg0: !migraphx.shaped<1x16x4x4xf8E5M2, 256x16x4x1>, %arg1: !migraphx.shaped<16x16x1x1xf8E5M2, 16x1x1x1>) -> !migraphx.shaped<1x16x4x4xf8E5M2, 256x16x4x1> {
  // CHECK-COUNT-2: tosa.transpose
  // CHECK: tosa.conv2d
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x16xf8E5M2>, tensor<16x1x1x16xf8E5M2>, tensor<16xf8E5M2>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x16xf8E5M2>
  %0 = migraphx.convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x16x4x4xf8E5M2, 256x16x4x1>, <16x16x1x1xf8E5M2, 16x1x1x1> -> <1x16x4x4xf8E5M2, 256x16x4x1>
  return %0 : !migraphx.shaped<1x16x4x4xf8E5M2, 256x16x4x1>
}

// CHECK-LABEL: @quant_conv2d_float8
// CHECK-SAME: (%{{.*}}: tensor<256xf8E5M2>, %{{.*}}: tensor<256xf8E5M2>) -> tensor<256xf32>
func.func @quant_conv2d_float8(%arg0: !migraphx.shaped<1x16x4x4xf8E5M2, 256x16x4x1>, %arg1: !migraphx.shaped<16x16x1x1xf8E5M2, 16x1x1x1>) -> !migraphx.shaped<1x16x4x4xf32, 256x16x4x1> {
  // CHECK-COUNT-2: tosa.transpose
  // CHECK: tosa.conv2d
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x16xf8E5M2>, tensor<16x1x1x16xf8E5M2>, tensor<16xf32>, tensor<1xf8E5M2>, tensor<1xf8E5M2>) -> tensor<1x4x4x16xf32>
  %0 = migraphx.quant_convolution %arg0, %arg1 {dilation = [1, 1], group = 1 : i64, padding = [0, 0, 0, 0], padding_mode = 0 : i64, stride = [1, 1]} : <1x16x4x4xf8E5M2, 256x16x4x1>, <16x16x1x1xf8E5M2, 16x1x1x1> -> <1x16x4x4xf32, 256x16x4x1>
  return %0 : !migraphx.shaped<1x16x4x4xf32, 256x16x4x1>
}

// CHECK-LABEL: @bwd_data_conv3d
func.func @bwd_data_conv3d(%arg0: !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1>, %arg1: !migraphx.shaped<16x16x1x1x1xf32, 16x1x1x1x1>) -> !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1> {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>}
  %0 = migraphx.backwards_data_convolution %arg1, %arg0 {
    dilation = [1, 1, 1],
    group = 1 : i64,
    padding = [0, 0, 0, 0, 0, 0],
    padding_mode = 0 : i64,
    stride = [1, 1, 1]} : <16x16x1x1x1xf32, 16x1x1x1x1>, <1x16x4x4x4xf32, 1024x64x16x4x1> -> <1x16x4x4x4xf32, 1024x64x16x4x1>
  return %0 : !migraphx.shaped<1x16x4x4x4xf32, 1024x64x16x4x1>
}

// CHECK-LABEL: @bwd_data_conv2d
func.func @bwd_data_conv2d(%arg0: !migraphx.shaped<1x512x16x16xf32,131072x256x16x1>,
                                        %arg1: !migraphx.shaped<512x512x4x4xf32, 8192x16x4x1>) -> !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1> attributes {arch = "gfx942", kernel = "mixr", num_cu = 0 : i64} {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
    dilation = [1, 1],
    group = 1 : i64,
    padding = [1, 1, 1, 1],
    padding_mode = 0 : i64,
    stride = [1, 1]} : <1x512x16x16xf32, 131072x256x16x1>, <512x512x4x4xf32, 8192x16x4x1> -> <1x512x32x32xf32, 524288x1024x32x1>
  return %0 : !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1>
}

// CHECK-LABEL: @bwd_data_conv2d_stride
func.func @bwd_data_conv2d_stride(%arg0: !migraphx.shaped<1x512x16x16xf32,131072x256x16x1>,
                                        %arg1: !migraphx.shaped<512x512x4x4xf32, 8192x16x4x1>) -> !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1> attributes {arch = "gfx942", kernel = "mixr", num_cu = 0 : i64} {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
    dilation = [1, 1],
    group = 1 : i64,
    padding = [1, 1, 1, 1],
    padding_mode = 0 : i64,
    stride = [2, 2]} : <1x512x16x16xf32, 131072x256x16x1>, <512x512x4x4xf32, 8192x16x4x1> -> <1x512x32x32xf32, 524288x1024x32x1>
  return %0 : !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1>
}

// CHECK-LABEL: @bwd_data_conv2d_group
func.func @bwd_data_conv2d_group(%arg0: !migraphx.shaped<1x512x16x16xf32,131072x256x16x1>,
                                        %arg1: !migraphx.shaped<512x512x4x4xf32, 8192x16x4x1>) -> !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1> attributes {arch = "gfx942", kernel = "mixr", num_cu = 0 : i64} {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 2 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
    dilation = [1, 1],
    group = 2 : i64,
    padding = [1, 1, 1, 1],
    padding_mode = 0 : i64,
    stride = [1, 1]} : <1x512x16x16xf32, 131072x256x16x1>, <512x512x4x4xf32, 8192x16x4x1> -> <1x512x32x32xf32, 524288x1024x32x1>
  return %0 : !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1>
}

// CHECK-LABEL: @bwd_data_conv2d_dilation
func.func @bwd_data_conv2d_dilation(%arg0: !migraphx.shaped<1x512x16x16xf32,131072x256x16x1>,
                                        %arg1: !migraphx.shaped<512x512x4x4xf32, 8192x16x4x1>) -> !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1> attributes {arch = "gfx942", kernel = "mixr", num_cu = 0 : i64} {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 2, 2>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>}
  %0 = migraphx.backwards_data_convolution %arg0, %arg1 {
    dilation = [2, 2],
    group = 1 : i64,
    padding = [1, 1, 1, 1],
    padding_mode = 0 : i64,
    stride = [1, 1]} : <1x512x16x16xf32, 131072x256x16x1>, <512x512x4x4xf32, 8192x16x4x1> -> <1x512x32x32xf32, 524288x1024x32x1>
  return %0 : !migraphx.shaped<1x512x32x32xf32, 524288x1024x32x1>
}

// CHECK-LABEL: @bwd_data_conv1d
func.func @bwd_data_conv1d(%arg0: !migraphx.shaped<1x64x224xf32, 0x1x0>, %arg1: !migraphx.shaped<1x3x224xf32, 672x224x1>, %arg2: !migraphx.shaped<64x3x1xf32, 3x1x1>) -> !migraphx.shaped<1x64x224xf32, 14336x224x1> {
  // CHECK: tosa.custom
  // CHECK-SAME: {acc_type = f32, dilation = array<i64: 1, 1>, domain_name = "rocmlir", group = 1 : i64, implementation_attrs = "", operator_name = "conv_bwd_data", out_pad = array<i64: 0, 0, 0, 0>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}
  %0 = migraphx.backwards_data_convolution %arg1, %arg2 {
    dilation = [1],
    group = 1 : i64,
    padding = [0, 0],
    padding_mode = 0 : i64,
    stride = [1]} : <1x3x224xf32, 672x224x1>, <64x3x1xf32, 3x1x1> -> <1x64x224xf32, 14336x224x1>
  %1 = migraphx.add %0, %arg0 : <1x64x224xf32, 14336x224x1>, <1x64x224xf32, 0x1x0> -> <1x64x224xf32, 14336x224x1>
  return %1 : !migraphx.shaped<1x64x224xf32, 14336x224x1>
}

// -----

// CHECK-LABEL: @dot_f16
func.func @dot_f16(%arg0: !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>, %arg1: !migraphx.shaped<8x64x320x320xf16, 6553600x102400x320x1>) -> !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>  attributes {kernel = "mixr"} {
 // CHECK: tosa.matmul
 // CHECK-SAME: {acc_type = f32, perf_config = "v2:16,16,8,16,16,4,1,1,1"} : (tensor<512x64x320xf16>, tensor<512x320x320xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<512x64x320xf16>
  %4 = migraphx.dot %arg0, %arg1 {perf_config = "v2:16,16,8,16,16,4,1,1,1"} : <8x64x64x320xf16, 1310720x20480x320x1>, <8x64x320x320xf16, 6553600x102400x320x1> -> <8x64x64x320xf16, 1310720x20480x320x1>
  return %4 : !migraphx.shaped<8x64x64x320xf16, 1310720x20480x320x1>
}

// -----

// Tests for squeeze

// CHECK-LABEL: @squeeze_single_axis
// CHECK-SAME: ([[arg0:%.+]]: tensor<320xf16>) -> tensor<320xf16>
func.func @squeeze_single_axis(%arg0: !migraphx.shaped<1x5x2x2x16xf16, 320x64x32x16x1>) -> !migraphx.shaped<5x2x2x16xf16, 64x32x16x1> {
  // CHECK: [[exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<320xf16>, !tosa.shape<5>) -> tensor<1x5x2x2x16xf16>
  // CHECK: [[squeezed:%.+]] = tosa.reshape [[exp]], %{{.*}} : (tensor<1x5x2x2x16xf16>, !tosa.shape<4>) -> tensor<5x2x2x16xf16>
  // CHECK: [[flat:%.+]] = tosa.reshape [[squeezed]], %{{.*}} : (tensor<5x2x2x16xf16>, !tosa.shape<1>) -> tensor<320xf16>
  // CHECK: return [[flat]]
  %0 = migraphx.squeeze %arg0 {axes = [0]} : <1x5x2x2x16xf16, 320x64x32x16x1> -> <5x2x2x16xf16, 64x32x16x1>
  func.return %0 : !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>
}

// CHECK-LABEL: @squeeze_no_axes
// CHECK-SAME: ([[arg0:%.+]]: tensor<12xf16>) -> tensor<12xf16>
func.func @squeeze_no_axes(%arg0: !migraphx.shaped<1x3x1x4xf16, 12x4x4x1>) -> !migraphx.shaped<3x4xf16, 4x1> {
  // CHECK: [[exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<12xf16>, !tosa.shape<4>) -> tensor<1x3x1x4xf16>
  // CHECK: [[squeezed:%.+]] = tosa.reshape [[exp]], %{{.*}} : (tensor<1x3x1x4xf16>, !tosa.shape<2>) -> tensor<3x4xf16>
  // CHECK: [[flat:%.+]] = tosa.reshape [[squeezed]], %{{.*}} : (tensor<3x4xf16>, !tosa.shape<1>) -> tensor<12xf16>
  // CHECK: return [[flat]]
  %0 = migraphx.squeeze %arg0 : <1x3x1x4xf16, 12x4x4x1> -> <3x4xf16, 4x1>
  func.return %0 : !migraphx.shaped<3x4xf16, 4x1>
}

// CHECK-LABEL: @squeeze_negative_axis
// CHECK-SAME: ([[arg0:%.+]]: tensor<12xf16>) -> tensor<12xf16>
func.func @squeeze_negative_axis(%arg0: !migraphx.shaped<1x3x1x4xf16, 12x4x4x1>) -> !migraphx.shaped<1x3x4xf16, 12x4x1> {
  // CHECK: [[exp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<12xf16>, !tosa.shape<4>) -> tensor<1x3x1x4xf16>
  // CHECK: [[squeezed:%.+]] = tosa.reshape [[exp]], %{{.*}} : (tensor<1x3x1x4xf16>, !tosa.shape<3>) -> tensor<1x3x4xf16>
  // CHECK: [[flat:%.+]] = tosa.reshape [[squeezed]], %{{.*}} : (tensor<1x3x4xf16>, !tosa.shape<1>) -> tensor<12xf16>
  // CHECK: return [[flat]]
  %0 = migraphx.squeeze %arg0 {axes = [-2]} : <1x3x1x4xf16, 12x4x4x1> -> <1x3x4xf16, 12x4x1>
  func.return %0 : !migraphx.shaped<1x3x4xf16, 12x4x1>
}

// -----

// Tests for gather

// CHECK-LABEL: @gather_axis0
// CHECK-SAME: ([[arg0:%.+]]: tensor<320xf16>, [[arg1:%.+]]: tensor<5xi32>) -> tensor<320xf16>
func.func @gather_axis0(%data: !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>, %indices: !migraphx.shaped<5xi32, 1>) -> !migraphx.shaped<5x2x2x16xf16, 64x32x16x1> {
  // CHECK: [[dataExp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<320xf16>, !tosa.shape<4>) -> tensor<5x2x2x16xf16>
  // CHECK: [[dataReshaped:%.+]] = tosa.reshape [[dataExp]], %{{.*}} : (tensor<5x2x2x16xf16>, !tosa.shape<3>) -> tensor<1x5x64xf16>
  // Negative index normalization
  // CHECK: [[zeroConst:%.+]] = "tosa.const"() <{values = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK: [[dimSizeConst:%.+]] = "tosa.const"() <{values = dense<5> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK: [[isNonNeg:%.+]] = tosa.greater_equal [[arg1]], [[zeroConst]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi1>
  // CHECK: [[indicesPlusK:%.+]] = tosa.add [[arg1]], [[dimSizeConst]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK: [[normalizedIndices:%.+]] = tosa.select [[isNonNeg]], [[arg1]], [[indicesPlusK]] : (tensor<5xi1>, tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK: [[indicesReshaped:%.+]] = tosa.reshape [[normalizedIndices]], %{{.*}} : (tensor<5xi32>, !tosa.shape<2>) -> tensor<1x5xi32>
  // CHECK: [[gathered:%.+]] = tosa.gather [[dataReshaped]], [[indicesReshaped]]
  // CHECK: [[reshapedOut:%.+]] = tosa.reshape [[gathered]], %{{.*}} : (tensor<1x5x64xf16>, !tosa.shape<4>) -> tensor<5x2x2x16xf16>
  // CHECK: [[flat:%.+]] = tosa.reshape [[reshapedOut]], %{{.*}} : (tensor<5x2x2x16xf16>, !tosa.shape<1>) -> tensor<320xf16>
  // CHECK: return [[flat]]
  %0 = migraphx.gather %data, %indices {axis = 0} : <5x2x2x16xf16, 64x32x16x1>, <5xi32, 1> -> <5x2x2x16xf16, 64x32x16x1>
  func.return %0 : !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>
}

// CHECK-LABEL: @gather_axis1
// CHECK-SAME: ([[arg0:%.+]]: tensor<24xf32>, [[arg1:%.+]]: tensor<2xi32>) -> tensor<16xf32>
func.func @gather_axis1(%data: !migraphx.shaped<2x3x4xf32, 12x4x1>, %indices: !migraphx.shaped<2xi32, 1>) -> !migraphx.shaped<2x2x4xf32, 8x4x1> {
  // CHECK: [[dataExp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<24xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  // CHECK: [[dataReshaped:%.+]] = tosa.reshape [[dataExp]], %{{.*}} : (tensor<2x3x4xf32>, !tosa.shape<3>) -> tensor<2x3x4xf32>
  // Negative index normalization
  // CHECK: [[zeroConst:%.+]] = "tosa.const"() <{values = dense<0> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK: [[dimSizeConst:%.+]] = "tosa.const"() <{values = dense<3> : tensor<2xi32>}> : () -> tensor<2xi32>
  // CHECK: [[isNonNeg:%.+]] = tosa.greater_equal [[arg1]], [[zeroConst]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK: [[indicesPlusK:%.+]] = tosa.add [[arg1]], [[dimSizeConst]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: [[normalizedIndices:%.+]] = tosa.select [[isNonNeg]], [[arg1]], [[indicesPlusK]] : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  // CHECK: [[indicesBatched:%.+]] = tosa.reshape [[normalizedIndices]], %{{.*}} : (tensor<2xi32>, !tosa.shape<2>) -> tensor<1x2xi32>
  // CHECK: [[indicesTiled:%.+]] = tosa.tile [[indicesBatched]], %{{.*}} : (tensor<1x2xi32>, !tosa.shape<2>) -> tensor<2x2xi32>
  // CHECK: [[gathered:%.+]] = tosa.gather [[dataReshaped]], [[indicesTiled]]
  // CHECK: [[reshapedOut:%.+]] = tosa.reshape [[gathered]], %{{.*}} : (tensor<2x2x4xf32>, !tosa.shape<3>) -> tensor<2x2x4xf32>
  // CHECK: [[flat:%.+]] = tosa.reshape [[reshapedOut]], %{{.*}} : (tensor<2x2x4xf32>, !tosa.shape<1>) -> tensor<16xf32>
  // CHECK: return [[flat]]
  %0 = migraphx.gather %data, %indices {axis = 1} : <2x3x4xf32, 12x4x1>, <2xi32, 1> -> <2x2x4xf32, 8x4x1>
  func.return %0 : !migraphx.shaped<2x2x4xf32, 8x4x1>
}

// CHECK-LABEL: @gather_negative_axis
// CHECK-SAME: ([[arg0:%.+]]: tensor<320xf16>, [[arg1:%.+]]: tensor<5xi32>) -> tensor<320xf16>
func.func @gather_negative_axis(%data: !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>, %indices: !migraphx.shaped<5xi32, 1>) -> !migraphx.shaped<5x2x2x16xf16, 64x32x16x1> {
  // CHECK: [[dataExp:%.+]] = tosa.reshape [[arg0]], %{{.*}} : (tensor<320xf16>, !tosa.shape<4>) -> tensor<5x2x2x16xf16>
  // CHECK: [[dataReshaped:%.+]] = tosa.reshape [[dataExp]], %{{.*}} : (tensor<5x2x2x16xf16>, !tosa.shape<3>) -> tensor<1x5x64xf16>
  // Negative index normalization
  // CHECK: [[zeroConst:%.+]] = "tosa.const"() <{values = dense<0> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK: [[dimSizeConst:%.+]] = "tosa.const"() <{values = dense<5> : tensor<5xi32>}> : () -> tensor<5xi32>
  // CHECK: [[isNonNeg:%.+]] = tosa.greater_equal [[arg1]], [[zeroConst]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi1>
  // CHECK: [[indicesPlusK:%.+]] = tosa.add [[arg1]], [[dimSizeConst]] : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK: [[normalizedIndices:%.+]] = tosa.select [[isNonNeg]], [[arg1]], [[indicesPlusK]] : (tensor<5xi1>, tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK: [[indicesReshaped:%.+]] = tosa.reshape [[normalizedIndices]], %{{.*}} : (tensor<5xi32>, !tosa.shape<2>) -> tensor<1x5xi32>
  // CHECK: [[gathered:%.+]] = tosa.gather [[dataReshaped]], [[indicesReshaped]]
  // CHECK: [[reshapedOut:%.+]] = tosa.reshape [[gathered]], %{{.*}} : (tensor<1x5x64xf16>, !tosa.shape<4>) -> tensor<5x2x2x16xf16>
  // CHECK: [[flat:%.+]] = tosa.reshape [[reshapedOut]], %{{.*}} : (tensor<5x2x2x16xf16>, !tosa.shape<1>) -> tensor<320xf16>
  // CHECK: return [[flat]]
  %0 = migraphx.gather %data, %indices {axis = -4} : <5x2x2x16xf16, 64x32x16x1>, <5xi32, 1> -> <5x2x2x16xf16, 64x32x16x1>
  func.return %0 : !migraphx.shaped<5x2x2x16xf16, 64x32x16x1>
}

// -----

// Tests for scatter

// CHECK-LABEL: @scatter_none_axis0
func.func @scatter_none_axis0(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x2x16xi32, 1x0x0>,
    %updates: !migraphx.shaped<8x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<10x2x16xf16>, !tosa.shape<3>) -> tensor<1x10x32xf16>
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<8x2x16xf16>, !tosa.shape<3>) -> tensor<1x8x32xf16>
  // CHECK: tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x8x32xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x8x1xi32>
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1x8x1xi32>, !tosa.shape<2>) -> tensor<1x8xi32>
  // Negative index normalization
  // CHECK: "tosa.const"() <{values = dense<0> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  // CHECK: "tosa.const"() <{values = dense<10> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  // CHECK: tosa.greater_equal
  // CHECK: tosa.add
  // CHECK: tosa.select
  // CHECK: tosa.scatter
  // CHECK-SAME: tensor<1x10x32xf16>
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = 0}
      : <10x2x16xf16, 32x16x1>, <8x2x16xi32, 1x0x0>, <8x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  func.return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

// CHECK-LABEL: @scatter_none_negative_axis
func.func @scatter_none_negative_axis(
    %data: !migraphx.shaped<10x2x16xf16, 32x16x1>,
    %indices: !migraphx.shaped<8x2x16xi32, 1x0x0>,
    %updates: !migraphx.shaped<8x2x16xf16, 32x16x1>
) -> !migraphx.shaped<10x2x16xf16, 32x16x1> {
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<10x2x16xf16>, !tosa.shape<3>) -> tensor<1x10x32xf16>
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<8x2x16xf16>, !tosa.shape<3>) -> tensor<1x8x32xf16>
  // CHECK: tosa.slice %{{.*}}, %{{.*}}, %{{.*}} : (tensor<1x8x32xi32>, !tosa.shape<3>, !tosa.shape<3>) -> tensor<1x8x1xi32>
  // CHECK: tosa.reshape %{{.*}}, %{{.*}} : (tensor<1x8x1xi32>, !tosa.shape<2>) -> tensor<1x8xi32>
  // Negative index normalization
  // CHECK: "tosa.const"() <{values = dense<0> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  // CHECK: "tosa.const"() <{values = dense<10> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
  // CHECK: tosa.greater_equal
  // CHECK: tosa.add
  // CHECK: tosa.select
  // CHECK: tosa.scatter
  // CHECK-SAME: tensor<1x10x32xf16>
  %0 = migraphx.scatter_none %data, %indices, %updates {axis = -3}
      : <10x2x16xf16, 32x16x1>, <8x2x16xi32, 1x0x0>, <8x2x16xf16, 32x16x1>
      -> <10x2x16xf16, 32x16x1>
  func.return %0 : !migraphx.shaped<10x2x16xf16, 32x16x1>
}

