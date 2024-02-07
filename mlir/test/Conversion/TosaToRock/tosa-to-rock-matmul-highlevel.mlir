// RUN: rocmlir-driver -host-pipeline highlevel %s | FileCheck %s

module {
  // CHECK-LABEL: @dot_tr_collapse_reshape1
  func.func @dot_tr_collapse_reshape1(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x12x384x64xf32>, %arg3: tensor<1x12x384x64xf32>) -> tensor<1x12x384x384xf32> attributes {arch = "", kernel} {
    %cst = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi64>
    // CHECK-DAG: %[[TRANSFORM0:.*]] = rock.transform %arg3 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %0 = "tosa.transpose"(%arg3, %cst) : (tensor<1x12x384x64xf32>, tensor<4xi64>) -> tensor<1x12x64x384xf32>
    // CHECK-DAG: %[[TRANSFORM1:.*]] = rock.transform %arg2 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %1 = "tosa.reshape"(%arg2) {new_shape = array<i64: 12, 384, 64>} : (tensor<1x12x384x64xf32>) -> tensor<12x384x64xf32>
    %2 = "tosa.reshape"(%0) {new_shape = array<i64: 12, 64, 384>} : (tensor<1x12x64x384xf32>) -> tensor<12x64x384xf32>
    // CHECK: rock.gemm {{.*}} = %[[TRANSFORM1]] * tr %[[TRANSFORM0]]
    %3 = "tosa.matmul"(%1, %2) : (tensor<12x384x64xf32>, tensor<12x64x384xf32>) -> tensor<12x384x384xf32>
    %4 = "tosa.reshape"(%3) {new_shape = array<i64: 1, 12, 384, 384>} : (tensor<12x384x384xf32>) -> tensor<1x12x384x384xf32>
    %5 = "tosa.mul"(%4, %arg0) {shift = 0 : i8} : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    %6 = "tosa.add"(%5, %arg1) : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    return %6 : tensor<1x12x384x384xf32>
  }

  // CHECK-LABEL: @dot_tr_collapse_reshape
  func.func private @dot_tr_collapse_reshape2(%arg0: tensor<2x320x64x64xf32>, %arg1: tensor<1x320x320xf32>) -> tensor<2x4096x320xf32> attributes {kernel} {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x320x320xf32>
    // CHECK-DAG: %[[TRANSFORM0:.*]] = rock.transform %arg1 {{.*}} : memref<1x320x320xf32> to memref<2x320x320xf32>
    %0 = "tosa.add"(%cst, %arg1) : (tensor<2x320x320xf32>, tensor<1x320x320xf32>) -> tensor<2x320x320xf32>
    %cst_0 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    // CHECK-DAG: %[[TRANSFORM1:.*]] = rock.transform %arg0 {{.*}} : memref<2x320x64x64xf32> to memref<2x320x4096xf32>
    // CHECK-DAG: %[[TRANSFORM2:.*]] = rock.transform %[[TRANSFORM1]] by {{.*}} : memref<2x320x4096xf32> to memref<320x8192xf32>
    // CHECK-DAG: %[[TRANSFORM3:.*]] = rock.transform %[[TRANSFORM0]] by {{.*}} : memref<2x320x320xf32> to memref<320x320xf32>
    %1 = "tosa.transpose"(%arg0, %cst_0) : (tensor<2x320x64x64xf32>, tensor<4xi64>) -> tensor<2x64x64x320xf32>
    %2 = "tosa.reshape"(%1) <{new_shape = array<i64: 2, 4096, 320>}> : (tensor<2x64x64x320xf32>) -> tensor<2x4096x320xf32>
    // CHECK:  rock.gemm {{.*}} = tr %[[TRANSFORM2]] * %[[TRANSFORM3]]
    %3 = "tosa.matmul"(%2, %0) : (tensor<2x4096x320xf32>, tensor<2x320x320xf32>) -> tensor<2x4096x320xf32>
    return %3 : tensor<2x4096x320xf32>
  }
}
