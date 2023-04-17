// RUN: rocmlir-driver -host-pipeline highlevel %s | FileCheck %s

module {
  // CHECK-LABEL: @dot_tr_collapse_reshape
  func.func @dot_tr_collapse_reshape(%arg0: tensor<1x1x1x1xf32>, %arg1: tensor<1x1x1x1xf32>, %arg2: tensor<1x12x384x64xf32>, %arg3: tensor<1x12x384x64xf32>) -> tensor<1x12x384x384xf32> attributes {arch = "", kernel} {
    %cst = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi64>
    // CHECK-DAG: %[[TRANSFORM0:.*]] = rock.transform %arg3 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %0 = "tosa.transpose"(%arg3, %cst) : (tensor<1x12x384x64xf32>, tensor<4xi64>) -> tensor<1x12x64x384xf32>
    // CHECK-DAG: %[[TRANSFORM1:.*]] = rock.transform %arg2 {{.*}} : memref<1x12x384x64xf32> to memref<12x384x64xf32>
    %1 = "tosa.reshape"(%arg2) {new_shape = array<i64: 12, 384, 64>} : (tensor<1x12x384x64xf32>) -> tensor<12x384x64xf32>
    %2 = "tosa.reshape"(%0) {new_shape = array<i64: 12, 64, 384>} : (tensor<1x12x64x384xf32>) -> tensor<12x64x384xf32>
    // CHECK: rock.gemm {{.*}} = %[[TRANSFORM1]] * tr %[[TRANSFORM0]]
    %3 = "tosa.matmul"(%1, %2) : (tensor<12x384x64xf32>, tensor<12x64x384xf32>) -> tensor<12x384x384xf32>
    %4 = "tosa.reshape"(%3) {new_shape = array<i64: 1, 12, 384, 384>} : (tensor<12x384x384xf32>) -> tensor<1x12x384x384xf32>
    %5 = "tosa.mul"(%4, %arg0) {shift = 0 : i32} : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    %6 = "tosa.add"(%5, %arg1) : (tensor<1x12x384x384xf32>, tensor<1x1x1x1xf32>) -> tensor<1x12x384x384xf32>
    return %6 : tensor<1x12x384x384xf32>
  }
}
