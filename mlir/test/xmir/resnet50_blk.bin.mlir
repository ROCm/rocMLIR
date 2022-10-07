// RUN: rocmlir-driver -host-pipeline full -kernel-pipeline full -triple amdgcn-amd-amdhsa -target gfx908 %s | FileCheck %s

module {
// CHECK: func.func private @resnet50__part_0(%arg0: memref<1x32x32x64xf32> {func.read_access}, %arg1: memref<64x3x3x64xf32> {func.read_access}, %arg2: memref<1x32x32x64xf32> {func.write_access}) attributes {async.targets = [{arch = "gfx908", binary = {{.*}}, block_size = 64 : i32, grid_size = 32 : i32, type = "gpu"}]}
// CHECK: func.func private @resnet50__part_1(%arg0: memref<1x32x32x64xf32> {func.read_access}, %arg1: memref<64x3x3x64xf32> {func.read_access}, %arg2: memref<1x32x32x64xf32> {func.read_access}, %arg3: memref<1x32x32x64xf32> {func.write_access}) attributes {async.targets = [{arch = "gfx908", binary = {{.*}}, block_size = 64 : i32, grid_size = 32 : i32, type = "gpu"}]}

  func.func @resnet50(%arg0: tensor<1x32x32x64xf32>, %arg1: tensor<64x3x3x64xf32>, %arg2: tensor<64x3x3x64xf32>) -> tensor<1x32x32x64xf32> {

    %cst = arith.constant dense<0.0> : tensor<64xf32>
    %0 = "tosa.conv2d"(%arg0, %arg1, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %1 = "tosa.clamp"(%0) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    //%cst1 = arith.constant dense<0.0> : tensor<64xf32>
    %2 = "tosa.conv2d"(%1, %arg2, %cst) {
      dilation = [1, 1],
      pad = [1, 1, 1, 1],
      stride = [1, 1]
    }
     : (tensor<1x32x32x64xf32>, tensor<64x3x3x64xf32>, tensor<64xf32>) -> tensor<1x32x32x64xf32>

    %3 = "tosa.clamp"(%2) {
      min_fp = 0.0 : f32,
      max_fp = 6.0 : f32,
      min_int = 0 : i64,
      max_int = 6 : i64
    }
     : (tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    %4 = "tosa.add"(%arg0, %3)
     : (tensor<1x32x32x64xf32>, tensor<1x32x32x64xf32>) -> tensor<1x32x32x64xf32>

    return %4 : tensor<1x32x32x64xf32>
  }
}

