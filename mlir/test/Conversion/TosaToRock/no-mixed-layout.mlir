// RUN: rocmlir-driver -host-pipeline partition,highlevel -targets %arch %s | FileCheck %s

// CHECK: rock.conv2d({{.*}}) {{.*}} {{{.*}}, filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"]{{.*}}}

module {
  func.func @test(%arg0: tensor<1x512x1x1xf32>, %arg1: tensor<1x384x28x28xf32>, %arg2: tensor<512x384x1x1xf32>) -> tensor<1x512x28x28xf32> {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %0 = "tosa.transpose"(%arg1, %cst) : (tensor<1x384x28x28xf32>, tensor<4xi64>) -> tensor<1x28x28x384xf32>
    %1 = "tosa.transpose"(%arg2, %cst) : (tensor<512x384x1x1xf32>, tensor<4xi64>) -> tensor<512x1x1x384xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<512xf32>
    %2 = "tosa.conv2d"(%0, %1, %cst_0) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> : (tensor<1x28x28x384xf32>, tensor<512x1x1x384xf32>, tensor<512xf32>) -> tensor<1x28x28x512xf32>
    %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi64>
    %3 = "tosa.transpose"(%2, %cst_1) : (tensor<1x28x28x512xf32>, tensor<4xi64>) -> tensor<1x512x28x28xf32>
    %4 = "tosa.add"(%3, %arg0) : (tensor<1x512x28x28xf32>, tensor<1x512x1x1xf32>) -> tensor<1x512x28x28xf32>
    %5 = "tosa.clamp"(%4) <{max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64}> : (tensor<1x512x28x28xf32>) -> tensor<1x512x28x28xf32>
    return %5 : tensor<1x512x28x28xf32>
  }
}
