// RUN: rocmlir-driver -host-pipeline highlevel %s | FileCheck %s

module {
  // CHECK-LABEL: @dot_tr_collapse_reshape1
  func.func @dot_tr_collapse_reshape1(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x12x384x64xf32>, %arg3: tensor<1x12x384x64xf32>) -> tensor<1x12x384x384xf32> attributes {arch = "", kernel} {
    // CHECK-DAG: %[[TRANSFORM0:.*]] = rock.transform %arg3 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %0 = "tosa.transpose"(%arg3) {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x384x64xf32>) -> tensor<1x12x64x384xf32>
    // CHECK-DAG: %[[TRANSFORM1:.*]] = rock.transform %arg2 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[12, 384, 64]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %1 = "tosa.reshape"(%arg2, %const_shape) : (tensor<1x12x384x64xf32>, !tosa.shape<3>) -> tensor<12x384x64xf32>
    %const_shape2 = "tosa.const_shape"() { values = dense<[12, 64, 384]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %2 = "tosa.reshape"(%0, %const_shape2) : (tensor<1x12x64x384xf32>, !tosa.shape<3>) -> tensor<12x64x384xf32>
    // CHECK: rock.gemm {{.*}} = %[[TRANSFORM1]] * tr %[[TRANSFORM0]]
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "tosa.matmul"(%1, %2, %a_zp, %b_zp) : (tensor<12x384x64xf32>, tensor<12x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x384x384xf32>
    %const_shape3 = "tosa.const_shape"() { values = dense<[1, 12, 384, 384]> : tensor<4xindex> } : () -> !tosa.shape<4>
    %4 = "tosa.reshape"(%3, %const_shape3) : (tensor<12x384x384xf32>, !tosa.shape<4>) -> tensor<1x12x384x384xf32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.mul"(%4, %arg0, %shift) : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x12x384x384xf32>
    %6 = "tosa.add"(%5, %arg1) : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    return %6 : tensor<1x12x384x384xf32>
  }

  // CHECK-LABEL: @dot_tr_collapse_reshape2
  func.func private @dot_tr_collapse_reshape2(%arg0: tensor<2x320x64x64xf32>, %arg1: tensor<1x320x320xf32>) -> tensor<2x4096x320xf32> attributes {kernel} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x320x320xf32>
    // CHECK-DAG: %[[TRANSFORM_ARG1_0:.*]] = rock.transform %arg1 {{.*}} : memref<1x320x320xf32> to memref<2x320x320xf32>
    %0 = "tosa.add"(%cst, %arg1) : (tensor<2x320x320xf32>, tensor<1x320x320xf32>) -> tensor<2x320x320xf32>
    // CHECK-DAG: %[[TRANSFORM_ARG0_0:.*]] = rock.transform %arg0 {{.*}} : memref<2x320x64x64xf32> to memref<2x64x64x320xf32>
    // CHECK-DAG: %[[TRANSFORM_ARG0_1:.*]] = rock.transform %[[TRANSFORM_ARG0_0]] by {{.*}} : memref<2x64x64x320xf32> to memref<2x4096x320xf32>
    // CHECK-DAG: %[[TRANSFORM_ARG0_2:.*]] = rock.transform %[[TRANSFORM_ARG0_1]] by {{.*}} : memref<2x4096x320xf32> to memref<8192x320xf32>
    // CHECK-DAG: %[[TRANSFORM_ARG1_1:.*]] = rock.transform %[[TRANSFORM_ARG1_0]] by {{.*}} : memref<2x320x320xf32> to memref<320x320xf32>
    %1 = "tosa.transpose"(%arg0) {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x320x64x64xf32>) -> tensor<2x64x64x320xf32>
    %const_shape = "tosa.const_shape"() { values = dense<[2, 4096, 320]> : tensor<3xindex> } : () -> !tosa.shape<3>
    %2 = "tosa.reshape"(%1, %const_shape) : (tensor<2x64x64x320xf32>, !tosa.shape<3>) -> tensor<2x4096x320xf32>
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    // CHECK:  rock.gemm {{.*}} = %[[TRANSFORM_ARG0_2]] * %[[TRANSFORM_ARG1_1]]
    %3 = "tosa.matmul"(%2, %0, %a_zp, %b_zp) : (tensor<2x4096x320xf32>, tensor<2x320x320xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x4096x320xf32>
    return %3 : tensor<2x4096x320xf32>
  }
}
