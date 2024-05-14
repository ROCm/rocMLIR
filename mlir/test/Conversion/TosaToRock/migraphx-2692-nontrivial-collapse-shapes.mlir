// RUN: rocmlir-opt -tosa-to-rock %s -o - | FileCheck %s
// COM: From the MIGraphX-generated module
// COM: func.func @mlir_reshape_transpose_reshape_convolution(%arg0: !migraphx.shaped<1x116x28x28xf32, 90944x784x28x1>, %arg1: !migraphx.shaped<116x1x3x3xf32, 9x9x3x1>) -> !migraphx.shaped<1x116x14x14xf32, 22736x196x14x1> attributes {arch = "gfx1100", kernel = "mixr"} {    %0 = migraphx.reshape %arg0 {dims = [1, 2, 58, 28, 28]} : <1x116x28x28xf32, 90944x784x28x1> -> <1x2x58x28x28xf32, 90944x45472x784x28x1>
// COM:   %1 = migraphx.transpose %0 {permutation = [0, 2, 1, 3, 4]} : <1x2x58x28x28xf32, 90944x45472x784x28x1> -> <1x58x2x28x28xf32, 90944x784x45472x28x1>
// COM:   %2 = migraphx.reshape %1 {dims = [1, -1, 28, 28]} : <1x58x2x28x28xf32, 90944x784x45472x28x1> -> <1x116x28x28xf32, 90944x784x28x1>
// COM:   %3 = migraphx.convolution %2, %arg1 {dilation = [1, 1], group = 116 : i64, padding = [1, 1, 1, 1], padding_mode = 0 : i64, stride = [2, 2]} : <1x116x28x28xf32, 90944x784x28x1>, <116x1x3x3xf32, 9x9x3x1> -> <1x116x14x14xf32, 22736x196x14x1>
// COM:   return %3 : !migraphx.shaped<1x116x14x14xf32, 22736x196x14x1>
// COM: }
// COM: which contains non-trivial slicing-and-dicing of a dimension,
// COM: showing a previous unsoundness in tosa-to-rock's handling of
// COM: transpose/collapse_shape pairs.

// CHECK-LABEL: @mlir_reshape_transpose_reshape_convolution
func.func @mlir_reshape_transpose_reshape_convolution(%arg0: tensor<1x116x28x28xf32>, %arg1: tensor<116x1x3x3xf32>) -> tensor<1x116x14x14xf32> attributes {arch = "gfx1100", kernel = "mixr"} {
  // COM: These'll get turned to transforms by -rock-view-to-transform in real compilations
  // CHECK: [[EXPANDED:%.+]] = tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2], [3], [4]]
  // CHECK: [[GC_TR:%.+]] = tosa.transpose [[EXPANDED]], %{{.*}} : (tensor<1x2x58x28x28xf32>, tensor<5xi64>) -> tensor<1x58x2x28x28xf32>
  // CHECK: [[COLLAPSED:%.+]] = tensor.collapse_shape [[GC_TR]] {{\[}}[0], [1, 2], [3], [4]]
  // CHECK: [[GROUP_SPLIT:%.+]] = rock.transform [[COLLAPSED]] {{.*}} : tensor<1x116x28x28xf32> to tensor<1x116x1x28x28xf32>
  // CHECK: rock.conv(%{{.*}}, [[GROUP_SPLIT]], %{{.*}})
  // CHECK-SAME: input_layout = ["ni", "gi", "ci", "hi", "wi"]
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2], [3], [4]] : tensor<1x116x28x28xf32> into tensor<1x2x58x28x28xf32>
  %0 = "tosa.const"() <{value = dense<[0, 2, 1, 3, 4]> : tensor<5xi64>}> : () -> tensor<5xi64>
  %1 = tosa.transpose %expanded, %0 : (tensor<1x2x58x28x28xf32>, tensor<5xi64>) -> tensor<1x58x2x28x28xf32>
  %collapsed = tensor.collapse_shape %1 [[0], [1, 2], [3], [4]] : tensor<1x58x2x28x28xf32> into tensor<1x116x28x28xf32>
  %2 = "tosa.const"() <{value = dense<[0, 2, 3, 1]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %3 = tosa.transpose %collapsed, %2 : (tensor<1x116x28x28xf32>, tensor<4xi64>) -> tensor<1x28x28x116xf32>
  %4 = tosa.transpose %arg1, %2 : (tensor<116x1x3x3xf32>, tensor<4xi64>) -> tensor<116x3x3x1xf32>
  %5 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<116xf32>}> : () -> tensor<116xf32>
  %6 = "tosa.conv2d"(%3, %4, %5) <{dilation = array<i64: 1, 1>, group = 116 : i64, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 2, 2>}> : (tensor<1x28x28x116xf32>, tensor<116x3x3x1xf32>, tensor<116xf32>) -> tensor<1x14x14x116xf32>
  %7 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi64>}> : () -> tensor<4xi64>
  %8 = "tosa.transpose"(%6, %7) : (tensor<1x14x14x116xf32>, tensor<4xi64>) -> tensor<1x116x14x14xf32>
  return %8 : tensor<1x116x14x14xf32>
}
