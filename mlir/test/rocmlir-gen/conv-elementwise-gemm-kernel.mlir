// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation conv_gemm -groupsize=1 -batchsize=2 -in_channels=256 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h_l=0 -padding_h_r=0 -padding_w_l=0 -padding_w_r=0 -gemmO=128 --transC=true --transO=false -fil_layout=gkcyx -in_layout=ngchw -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_conv_gemm
// CHECK-SAME: (%[[filterRaw:.*0]]: memref<32768xf32>,
// CHECK-SAME: %[[inputRaw:.*1]]: memref<524288xf32>,
// CHECK-SAME: %[[cRaw:.*2]]: memref<16384xf32>,
// CHECK-SAME: %[[outputRaw:.*3]]: memref<262144xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[filter:.*]] = rock.transform %[[filterRaw]] {{.*}} : memref<32768xf32> to memref<1x128x256x1x1xf32>
// CHECK-NEXT: %[[input:.*]] = rock.transform %[[inputRaw]] {{.*}} : memref<524288xf32> to memref<2x1x256x32x32xf32>
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] {{.*}} : memref<16384xf32> to memref<1x128x128xf32>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<262144xf32> to memref<1x2048x128xf32>

// CHECK-NEXT: rock.conv_elementwise_gemm
// CHECK-NEXT: ab = conv(%[[filter]], %[[input]])
// CHECK: %[[output]] = ab * tr %[[c]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_conv_gemm
// CHECK: %[[convTensor:.*]] = tosa.conv2d %[[inputTensor:.*]], %[[filterTensor:.*]], %{{.*}}, %{{.*}}, %{{.*}} {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<2x32x32x256xf32>, tensor<128x1x1x256xf32>, tensor<128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x32x32x128xf32>
// CHECK-DAG: %[[abTensor:.*]] = tosa.reshape %[[convTensor]], %{{.*}} : (tensor<2x32x32x128xf32>, !tosa.shape<3>) -> tensor<1x2048x128xf32>
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[abTensor]], %[[cTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : (tensor<1x2048x128xf32>, tensor<1x128x128xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2048x128xf32>
// CHECK: return
