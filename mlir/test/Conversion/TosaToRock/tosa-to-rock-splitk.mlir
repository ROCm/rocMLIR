// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
// CHECK-LABEL: @test_basic
// CHECK-SAME: -> (tensor<2x128x256xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_basic(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x128x256xf32> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf32>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  // CHECK: rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf32> = tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  return %c : tensor<2x128x256xf32>
}

// CHECK-LABEL: @test_basic_f16
// CHECK-SAME: -> (tensor<2x128x256xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16})
func.func @test_basic_f16(%a: tensor<2x128x64xf16>, %b: tensor<2x64x256xf16>) -> tensor<2x128x256xf16> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf16>
  // CHECK: rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf16> = tensor<2x128x64xf16> * tensor<2x64x256xf16> -> tensor<2x128x256xf16>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf16>, tensor<2x64x256xf16>) -> tensor<2x128x256xf16>
  return %c : tensor<2x128x256xf16>
}

// CHECK-LABEL: @test_basic_bf16
// CHECK-SAME: -> (tensor<2x128x256xbf16> {mhal.read_access, rock.prefill = 0.000000e+00 : bf16})
func.func @test_basic_bf16(%a: tensor<2x128x64xbf16>, %b: tensor<2x64x256xbf16>) -> tensor<2x128x256xbf16> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xbf16>
  // CHECK: rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xbf16> = tensor<2x128x64xbf16> * tensor<2x64x256xbf16> -> tensor<2x128x256xbf16>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xbf16>, tensor<2x64x256xbf16>) -> tensor<2x128x256xbf16>
  return %c : tensor<2x128x256xbf16>
}

// CHECK-LABEL: @test_reduce
// CHECK-SAME: -> (tensor<2x128x1xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x128x1xf32> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf32>
  // CHECK: %[[gemmOut:.*]] = rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf32> = tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: %[[outBuf2:.*]] = bufferization.alloc_tensor() : tensor<2x128x1xf32>
  // CHECK: rock.reduce  sum %[[gemmOut]] into %[[outBuf2]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 256 : i32} : tensor<2x128x256xf32> into tensor<2x128x1xf32> -> tensor<2x128x1xf32>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  return %1 : tensor<2x128x1xf32>
}

// CHECK-LABEL: @test_reduce_two_outputs
// CHECK-SAME: -> (tensor<2x128x1xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x1x256xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce_two_outputs(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> (tensor<2x128x1xf32>, tensor<2x1x256xf32>) attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf32>
  // CHECK: %[[outGemm:.*]] = rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf32> = tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: %[[outBuf2:.*]] = bufferization.alloc_tensor() : tensor<2x128x1xf32>
  // CHECK: rock.reduce  sum %[[outGemm]] into %[[outBuf2]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 256 : i32} : tensor<2x128x256xf32> into tensor<2x128x1xf32> -> tensor<2x128x1xf32>
  // CHECK: %[[outBuf3:.*]] = bufferization.alloc_tensor() : tensor<2x1x256xf32>
  // CHECK: rock.reduce  sum %[[outGemm]] into %[[outBuf3]] {{.*}} {axis = 1 : index, blockSize = 256 : i32, gridSize = 256 : i32} : tensor<2x128x256xf32> into tensor<2x1x256xf32> -> tensor<2x1x256xf32>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  %2 = "tosa.reduce_sum"(%c) {axis = 1 : i32} : (tensor<2x128x256xf32>) -> tensor<2x1x256xf32>
  return %1, %2 : tensor<2x128x1xf32>, tensor<2x1x256xf32>
}

// CHECK-LABEL: @test_reduce_two_outputs2
// CHECK-SAME: -> (tensor<2x128x1xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x128x256xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce_two_outputs2(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> (tensor<2x128x1xf32>, tensor<2x128x256xf32>) attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf32>
  // CHECK: %[[outGemm:.*]] = rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf32> = tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: %[[outBuf2:.*]] = bufferization.alloc_tensor() : tensor<2x128x1xf32>
  // CHECK: rock.reduce  sum %[[outGemm]] into %[[outBuf2]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 256 : i32} : tensor<2x128x256xf32> into tensor<2x128x1xf32> -> tensor<2x128x1xf32>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.reduce_sum"(%c) {axis = 2 : i32} : (tensor<2x128x256xf32>) -> tensor<2x128x1xf32>
  return %1, %c : tensor<2x128x1xf32>, tensor<2x128x256xf32>
}

// CHECK-LABEL: @test_add_two_outputs
// CHECK-SAME: -> (tensor<2x128x256xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: tensor<2x128x256xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_add_two_outputs(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>, %arg3: tensor<2x128x256xf32>) -> (tensor<2x128x256xf32>, tensor<2x128x256xf32>) attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x128x256xf32>
  // CHECK: %[[outGemm:.*]] = rock.gemm %[[outBuf]] = %arg0 * %arg1 {{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : tensor<2x128x256xf32> = tensor<2x128x64xf32> * tensor<2x64x256xf32> -> tensor<2x128x256xf32>
  // CHECK: tosa.add %[[outGemm]], %arg2 : (tensor<2x128x256xf32>, tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  %c = "tosa.matmul"(%a, %b) {perf_config="v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  %1 = "tosa.add"(%c, %arg3) {} : (tensor<2x128x256xf32>, tensor<2x128x256xf32>) -> tensor<2x128x256xf32>
  return %1, %c : tensor<2x128x256xf32>, tensor<2x128x256xf32>
}

// CHECK-LABEL: @mlir_convolution_multi_reduce
// CHECK-SAME: -> (tensor<64xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16}
// CHECK-SAME: tensor<2621440xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16})
func.func @mlir_convolution_multi_reduce(%arg0: tensor<320xf16>, %arg1: tensor<32768xf16>, %arg2: tensor<11520xf16>) -> (tensor<64xf16>, tensor<2621440xf16>) attributes {kernel} {
  %0 = tosa.const_shape  {value = dense<[32, 10, 1, 1, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %expanded = tensor.expand_shape %arg0 [[0, 1, 2, 3, 4]] output_shape [32, 10, 1, 1, 1] : tensor<320xf16> into tensor<32x10x1x1x1xf16>
  %1 = tosa.transpose %expanded {perms = array<i32: 4, 0, 1, 2, 3>} : (tensor<32x10x1x1x1xf16>) -> tensor<1x32x10x1x1xf16>
  %2 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x32x10x64x64xf16>}> : () -> tensor<2x32x10x64x64xf16>
  %3 = tosa.add %1, %2 : (tensor<1x32x10x1x1xf16>, tensor<2x32x10x64x64xf16>) -> tensor<2x32x10x64x64xf16>
  %4 = tosa.const_shape  {value = dense<[320, 4, 3, 3]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [320, 4, 3, 3] : tensor<11520xf16> into tensor<320x4x3x3xf16>
  %5 = tosa.const_shape  {value = dense<[2, 4, 64, 64]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [2, 4, 64, 64] : tensor<32768xf16> into tensor<2x4x64x64xf16>
  %6 = tosa.transpose %expanded_1 {perms = array<i32: 0, 2, 3, 1>} : (tensor<2x4x64x64xf16>) -> tensor<2x64x64x4xf16>
  %7 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<320x4x3x3xf16>) -> tensor<320x3x3x4xf16>
  %8 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %9 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<320xf16>}> : () -> tensor<320xf16>
  // CHECK: rock.conv{{.*}} {arch = "amdgcn-amd-amdhsa:gfx950", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", strides = [1 : index, 1 : index]} : tensor<1x320x4x3x3xf16>, tensor<2x1x4x64x64xf16>, tensor<2x1x320x64x64xf16> -> tensor<2x1x320x64x64xf16>
  %10 = tosa.conv2d %6, %7, %9, %8, %8 {acc_type = f32, dilation = array<i64: 1, 1>, group = 1 : i64, pad = array<i64: 1, 1, 1, 1>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1", stride = array<i64: 1, 1>} : (tensor<2x64x64x4xf16>, tensor<320x3x3x4xf16>, tensor<320xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<2x64x64x320xf16>
  %11 = tosa.transpose %10 {perms = array<i32: 0, 3, 1, 2>} : (tensor<2x64x64x320xf16>) -> tensor<2x320x64x64xf16>
  %12 = tosa.const_shape  {value = dense<[2, 32, 10, 64, 64]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %expanded_2 = tensor.expand_shape %11 [[0], [1, 2], [3], [4]] output_shape [2, 32, 10, 64, 64] : tensor<2x320x64x64xf16> into tensor<2x32x10x64x64xf16>
  %13 = tosa.add %expanded_2, %3 : (tensor<2x32x10x64x64xf16>, tensor<2x32x10x64x64xf16>) -> tensor<2x32x10x64x64xf16>
  %14 = tosa.const_shape  {value = dense<2621440> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed = tensor.collapse_shape %13 [[0, 1, 2, 3, 4]] : tensor<2x32x10x64x64xf16> into tensor<2621440xf16>
  %15 = "tosa.const"() <{value = dense<2.443790e-05> : tensor<2x32x10x64x64xf16>}> : () -> tensor<2x32x10x64x64xf16>
  %16 = "tosa.const"() <{value = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %17 = tosa.mul %13, %15, %16 : (tensor<2x32x10x64x64xf16>, tensor<2x32x10x64x64xf16>, tensor<1xi8>) -> tensor<2x32x10x64x64xf16>
  %18 = tosa.const_shape  {value = dense<[2, 32, 40960, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %collapsed_3 = tensor.collapse_shape %17 [[0], [1], [2, 3, 4]] : tensor<2x32x10x64x64xf16> into tensor<2x32x40960xf16>
  %expanded_4 = tensor.expand_shape %collapsed_3 [[0], [1], [2, 3]] output_shape [2, 32, 40960, 1] : tensor<2x32x40960xf16> into tensor<2x32x40960x1xf16>
  // CHECK: rock.reduce  sum {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 10240 : i32} : tensor<2x32x40960x1xf16> into tensor<2x32x1x1xf16> -> tensor<2x32x1x1xf16>
  %19 = tosa.reduce_sum %expanded_4 {axis = 2 : i32} : (tensor<2x32x40960x1xf16>) -> tensor<2x32x1x1xf16>
  %20 = tosa.const_shape  {value = dense<[2, 32, 1, 1, 1]> : tensor<5xindex>} : () -> !tosa.shape<5>
  %expanded_5 = tensor.expand_shape %19 [[0], [1], [2], [3, 4]] output_shape [2, 32, 1, 1, 1] : tensor<2x32x1x1xf16> into tensor<2x32x1x1x1xf16>
  %21 = tosa.const_shape  {value = dense<64> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_6 = tensor.collapse_shape %19 [[0, 1, 2, 3]] : tensor<2x32x1x1xf16> into tensor<64xf16>
  return %collapsed_6, %collapsed : tensor<64xf16>, tensor<2621440xf16>
}

}
