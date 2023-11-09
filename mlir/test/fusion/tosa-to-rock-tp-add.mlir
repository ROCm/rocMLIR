// RUN: rocmlir-driver --host-pipeline highlevel %s | rocmlir-opt --rock-affix-params --rock-conv-to-gemm --rock-gemm-to-gridwise -rock-regularize --rock-gridwise-gemm-to-blockwise --rock-linalg-align | FileCheck %s

// CHECK-DAG: #[[MAP2:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["{{.*}}", "{{.*}}", "{{.*}}", "{{.*}}"] at [0, 1, 2, 3] -> ["{{.*}}", "{{.*}}", "{{.*}}", "{{.*}}"] at [0, 2, 3, 1]>] bounds = [256, 28, 28, 64] -> [256, 64, 28, 28]>
// CHECK-COUNT-4: rock.threadwise_read_into {{.*}}
// CHECK: rock.threadwise_read_into {{.*}} -> [[lain:%.*]] :
// CHECK: linalg.generic{{.*}} ins({{.*}}, [[lain]] :{{.*}}) outs(%[[outBuf:.*]] : memref<64xf32, #gpu.address_space<private>>)
// CHECK: rock.threadwise_write_all {{.*}} %[[outBuf]] ->
// to test transpose is converted as transform and fused.

func.func @test_fusion(%arg0: tensor<256x28x28x128xf32>, %arg1: tensor<64x3x3x128xf32>, %arg2: tensor<256x64x28x28xf32>) -> tensor<256x28x28x64xf32> attributes {kernel, mhal.arch = ""} {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst_0) {dilation = array<i64: 1, 1>, expected_filter_layout = "kyxc", expected_input_layout = "nhwc", expected_output_layout = "nhwk", pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<256x28x28x128xf32>, tensor<64x3x3x128xf32>, tensor<1xf32>) -> tensor<256x28x28x64xf32>
    %1 = "tosa.transpose"(%arg2, %cst) : (tensor<256x64x28x28xf32>, tensor<4xi64>) -> tensor<256x28x28x64xf32>
    %2 = "tosa.add"(%0, %1) : (tensor<256x28x28x64xf32>, tensor<256x28x28x64xf32>) -> tensor<256x28x28x64xf32>
    return %2 : tensor<256x28x28x64xf32>
}

