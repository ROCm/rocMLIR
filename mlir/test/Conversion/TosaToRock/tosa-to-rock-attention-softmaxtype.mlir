// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: func @mlir_softmaxf32_attention
// CHECK: rock.attention
// CHECK: softmaxType = f32
func.func @mlir_softmaxf32_attention(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xi8>, %arg3: tensor<786432xf16>, %arg4: tensor<786432xf16>) -> (tensor<786432xf16>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[12, 256, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xi8> into tensor<12x256x256xi8>
  %1 = tosa.const_shape  {values = dense<[1, 12, 256, 256]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_3 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %2 = "tosa.const"() <{values = dense<1.000000e+01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %3 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %4 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %5 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %collapsed = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %collapsed_5 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %expanded_4, %collapsed_5, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %expanded_6 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = tosa.mul %7, %3, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %10 = tosa.cast %expanded_0 : (tensor<12x256x256xi8>) -> tensor<12x256x256xi1>
  %11 = tosa.select %10, %9, %2 : (tensor<12x256x256xi1>, tensor<12x256x256xf16>, tensor<12x256x256xf16>) -> tensor<12x256x256xf16>
  %12 = tosa.cast %11 : (tensor<12x256x256xf16>) -> tensor<12x256x256xf32>
  %13 = tosa.reduce_max %12 {axis = 2 : i32} : (tensor<12x256x256xf32>) -> tensor<12x256x1xf32>
  %14 = tosa.const_shape  {values = dense<[12, 256, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %15 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<12x256x256xf32>}> : () -> tensor<12x256x256xf32>
  %16 = tosa.add %13, %15 : (tensor<12x256x1xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %17 = tosa.sub %12, %16 : (tensor<12x256x256xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %18 = tosa.exp %17 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %19 = tosa.reduce_sum %18 {axis = 2 : i32} : (tensor<12x256x256xf32>) -> tensor<12x256x1xf32>
  %20 = tosa.add %19, %15 : (tensor<12x256x1xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %21 = tosa.reciprocal %20 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %22 = tosa.mul %18, %21, %8 : (tensor<12x256x256xf32>, tensor<12x256x256xf32>, tensor<1xi8>) -> tensor<12x256x256xf32>
  %23 = tosa.cast %22 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf16>
  %24 = tosa.matmul %23, %collapsed, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %25 = tosa.mul %24, %expanded, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %26 = tosa.const_shape  {values = dense<786432> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %25 [[0, 1, 2]] : tensor<12x256x256xf16> into tensor<786432xf16>
  return %collapsed_7 : tensor<786432xf16>
}

// CHECK-LABEL: func @mlir_softmaxf64_attention
// CHECK: rock.attention
// CHECK: softmaxType = f64
func.func @mlir_softmaxf64_attention(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xi8>, %arg3: tensor<786432xf16>, %arg4: tensor<786432xf16>) -> (tensor<786432xf16>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[12, 256, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xi8> into tensor<12x256x256xi8>
  %1 = tosa.const_shape  {values = dense<[1, 12, 256, 256]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_3 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %2 = "tosa.const"() <{values = dense<1.000000e+01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %3 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %4 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %5 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %collapsed = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %collapsed_5 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %expanded_4, %collapsed_5, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %expanded_6 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = tosa.mul %7, %3, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %10 = tosa.cast %expanded_0 : (tensor<12x256x256xi8>) -> tensor<12x256x256xi1>
  %11 = tosa.select %10, %9, %2 : (tensor<12x256x256xi1>, tensor<12x256x256xf16>, tensor<12x256x256xf16>) -> tensor<12x256x256xf16>
  %12 = tosa.cast %11 : (tensor<12x256x256xf16>) -> tensor<12x256x256xf64>
  %13 = tosa.reduce_max %12 {axis = 2 : i32} : (tensor<12x256x256xf64>) -> tensor<12x256x1xf64>
  %14 = tosa.const_shape  {values = dense<[12, 256, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %15 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<12x256x256xf64>}> : () -> tensor<12x256x256xf64>
  %16 = tosa.add %13, %15 : (tensor<12x256x1xf64>, tensor<12x256x256xf64>) -> tensor<12x256x256xf64>
  %17 = tosa.sub %12, %16 : (tensor<12x256x256xf64>, tensor<12x256x256xf64>) -> tensor<12x256x256xf64>
  %18 = tosa.exp %17 : (tensor<12x256x256xf64>) -> tensor<12x256x256xf64>
  %19 = tosa.reduce_sum %18 {axis = 2 : i32} : (tensor<12x256x256xf64>) -> tensor<12x256x1xf64>
  %20 = tosa.add %19, %15 : (tensor<12x256x1xf64>, tensor<12x256x256xf64>) -> tensor<12x256x256xf64>
  %21 = tosa.reciprocal %20 : (tensor<12x256x256xf64>) -> tensor<12x256x256xf64>
  %22 = tosa.mul %18, %21, %8 : (tensor<12x256x256xf64>, tensor<12x256x256xf64>, tensor<1xi8>) -> tensor<12x256x256xf64>
  %23 = tosa.cast %22 : (tensor<12x256x256xf64>) -> tensor<12x256x256xf16>
  %24 = tosa.matmul %23, %collapsed, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %25 = tosa.mul %24, %expanded, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %26 = tosa.const_shape  {values = dense<786432> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %25 [[0, 1, 2]] : tensor<12x256x256xf16> into tensor<786432xf16>
  return %collapsed_7 : tensor<786432xf16>
}

// CHECK-LABEL: func @mlir_softmaxf32_lse_attention
// CHECK: %[[lseBuffer:.+]] = bufferization.alloc_tensor() : tensor<12x256xf16>
// CHECK: %{{.*}}, %[[lseOut:.*]] = rock.attention
// CHECK: lse = %[[lseBuffer]] : tensor<12x256xf16>
// CHECK: softmaxType = f32
// CHECK: %[[lseExpanded:.*]] = tensor.expand_shape %[[lseOut]]
// CHECK: %[[lseCollapsed:.*]] = tensor.collapse_shape %[[lseExpanded]]
// CHECK: return %{{.*}}, %[[lseCollapsed]] : tensor<786432xf16>, tensor<3072xf16>
func.func @mlir_softmaxf32_lse_attention(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xi8>, %arg3: tensor<786432xf16>, %arg4: tensor<786432xf16>) -> (tensor<786432xf16>, tensor<3072xf16>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = tosa.const_shape  {values = dense<[12, 256, 256]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %expanded_0 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xi8> into tensor<12x256x256xi8>
  %1 = tosa.const_shape  {values = dense<[1, 12, 256, 256]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_2 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_3 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %2 = "tosa.const"() <{values = dense<1.000000e+01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %3 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<12x256x256xf16>}> : () -> tensor<12x256x256xf16>
  %4 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %5 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %collapsed = tensor.collapse_shape %5 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %collapsed_5 = tensor.collapse_shape %4 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %expanded_4, %collapsed_5, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %expanded_6 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = tosa.mul %7, %3, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %10 = tosa.cast %expanded_0 : (tensor<12x256x256xi8>) -> tensor<12x256x256xi1>
  %11 = tosa.select %10, %9, %2 : (tensor<12x256x256xi1>, tensor<12x256x256xf16>, tensor<12x256x256xf16>) -> tensor<12x256x256xf16>
  %12 = tosa.cast %11 : (tensor<12x256x256xf16>) -> tensor<12x256x256xf32>
  %13 = tosa.reduce_max %12 {axis = 2 : i32} : (tensor<12x256x256xf32>) -> tensor<12x256x1xf32>
  %14 = tosa.const_shape  {values = dense<[12, 256, 1]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %15 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<12x256x256xf32>}> : () -> tensor<12x256x256xf32>
  %16 = tosa.add %13, %15 : (tensor<12x256x1xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %17 = tosa.sub %12, %16 : (tensor<12x256x256xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %18 = tosa.exp %17 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %19 = tosa.reduce_sum %18 {axis = 2 : i32} : (tensor<12x256x256xf32>) -> tensor<12x256x1xf32>
  %20 = tosa.add %19, %15 : (tensor<12x256x1xf32>, tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %21 = tosa.reciprocal %20 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf32>
  %22 = tosa.mul %18, %21, %8 : (tensor<12x256x256xf32>, tensor<12x256x256xf32>, tensor<1xi8>) -> tensor<12x256x256xf32>
  %23 = tosa.cast %22 : (tensor<12x256x256xf32>) -> tensor<12x256x256xf16>
  %24 = tosa.cast %19 : (tensor<12x256x1xf32>) -> tensor<12x256x1xf16>
  %25 = tosa.cast %13 : (tensor<12x256x1xf32>) -> tensor<12x256x1xf16>
  %26 = tosa.log %24 : (tensor<12x256x1xf16>) -> tensor<12x256x1xf16>
  %27 = tosa.add %26, %25 : (tensor<12x256x1xf16>, tensor<12x256x1xf16>) -> tensor<12x256x1xf16>
  %28 = tosa.const_shape  {values = dense<3072> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %27 [[0, 1, 2]] : tensor<12x256x1xf16> into tensor<3072xf16>
  %29 = tosa.matmul %23, %collapsed, %6, %6 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %30 = tosa.mul %29, %expanded, %8 : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xi8>) -> tensor<12x256x256xf16>
  %31 = tosa.const_shape  {values = dense<786432> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_8 = tensor.collapse_shape %30 [[0, 1, 2]] : tensor<12x256x256xf16> into tensor<786432xf16>
  return %collapsed_8, %collapsed_7 : tensor<786432xf16>, tensor<3072xf16>
}

// CHECK-LABEL: func @mlir_softmaxf32_attention_with_scaling
// CHECK: rock.attention
// CHECK: softmaxType = f32
func.func @mlir_softmaxf32_attention_with_scaling(%arg0: tensor<48225280xf16>, %arg1: tensor<48225280xf16>, %arg2: tensor<48225280xf16>) -> tensor<48225280xf16> attributes {arch = "gfx942", kernel = "mixr", num_cu = 304 : i64} {
  %0 = tosa.const_shape  {values = dense<[1, 75352, 5, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %1 = tosa.transpose %expanded_1 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %2 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x128x75352xf16>
  %3 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %4 = tosa.const_shape  {values = dense<[5, 75352, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %5 = tosa.const_shape  {values = dense<[5, 128, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_2 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<1x5x128x75352xf16> into tensor<5x128x75352xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %collapsed, %collapsed_2, %6, %6 {acc_type = f32} : (tensor<5x75352x128xf16>, tensor<5x128x75352xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x75352xf16>
  %8 = tosa.const_shape  {values = dense<[1, 5, 75352, 75352]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_3 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 75352] : tensor<5x75352x75352xf16> into tensor<1x5x75352x75352xf16>
  %9 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x5x75352x75352xf16>}> : () -> tensor<1x5x75352x75352xf16>
  %10 = tosa.cast %expanded_3 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %11 = tosa.cast %9 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %13 = tosa.mul %10, %11, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %scaled_cast1 = tosa.cast %13 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf16>
  %scaled_cast2 = tosa.cast %scaled_cast1 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %14 = tosa.reduce_max %13 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %15 = tosa.const_shape  {values = dense<[1, 5, 75352, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x5x75352x75352xf32>}> : () -> tensor<1x5x75352x75352xf32>
  %17 = tosa.add %14, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %18 = tosa.sub %13, %17 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %19 = tosa.exp %18 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %20 = tosa.reduce_sum %19 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %21 = tosa.add %20, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %22 = tosa.reciprocal %21 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %23 = tosa.mul %19, %22, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %24 = tosa.cast %23 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf16>
  %25 = tosa.const_shape  {values = dense<[5, 75352, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_4 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x5x75352x75352xf16> into tensor<5x75352x75352xf16>
  %collapsed_5 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %26 = tosa.matmul %collapsed_4, %collapsed_5, %6, %6 {acc_type = f32} : (tensor<5x75352x75352xf16>, tensor<5x75352x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x128xf16>
  %27 = tosa.const_shape  {values = dense<[1, 5, 75352, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_6 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 128] : tensor<5x75352x128xf16> into tensor<1x5x75352x128xf16>
  %28 = tosa.const_shape  {values = dense<48225280> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %26 [[0, 1, 2]] : tensor<5x75352x128xf16> into tensor<48225280xf16>
  return %collapsed_7 : tensor<48225280xf16>
}

// CHECK-LABEL: func @mlir_softmaxf32_attention_with_scaling_multiple_converts
// CHECK: rock.attention
// CHECK: softmaxType = f32
func.func @mlir_softmaxf32_attention_with_scaling_multiple_converts(%arg0: tensor<48225280xf16>, %arg1: tensor<48225280xf16>, %arg2: tensor<48225280xf16>) -> tensor<48225280xf16> attributes {arch = "gfx942", kernel = "mixr", num_cu = 304 : i64} {
  %0 = tosa.const_shape  {values = dense<[1, 75352, 5, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %1 = tosa.transpose %expanded_1 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %2 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x128x75352xf16>
  %3 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %4 = tosa.const_shape  {values = dense<[5, 75352, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %5 = tosa.const_shape  {values = dense<[5, 128, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_2 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<1x5x128x75352xf16> into tensor<5x128x75352xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %collapsed, %collapsed_2, %6, %6 {acc_type = f32} : (tensor<5x75352x128xf16>, tensor<5x128x75352xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x75352xf16>
  %matmaul_cast1 = tosa.cast %7 : (tensor<5x75352x75352xf16>) -> tensor<5x75352x75352xf32>
  %bias = "tosa.const"() <{values = dense<8.837890e-02> : tensor<5x75352x75352xf32>}> : () -> tensor<5x75352x75352xf32> 
  %bias_add_f32 = tosa.add %matmaul_cast1, %bias : (tensor<5x75352x75352xf32>, tensor<5x75352x75352xf32>) -> tensor<5x75352x75352xf32>
  %matmul_cast2 = tosa.cast %bias_add_f32 : (tensor<5x75352x75352xf32>) -> tensor<5x75352x75352xf16>
  %8 = tosa.const_shape  {values = dense<[1, 5, 75352, 75352]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_3 = tensor.expand_shape %matmul_cast2 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 75352] : tensor<5x75352x75352xf16> into tensor<1x5x75352x75352xf16>
  %9 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x5x75352x75352xf16>}> : () -> tensor<1x5x75352x75352xf16>
  %10 = tosa.cast %expanded_3 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %11 = tosa.cast %9 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %13 = tosa.mul %10, %11, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %14 = tosa.reduce_max %13 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %15 = tosa.const_shape  {values = dense<[1, 5, 75352, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x5x75352x75352xf32>}> : () -> tensor<1x5x75352x75352xf32>
  %17 = tosa.add %14, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %18 = tosa.sub %13, %17 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %19 = tosa.exp %18 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %20 = tosa.reduce_sum %19 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %21 = tosa.add %20, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %22 = tosa.reciprocal %21 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %23 = tosa.mul %19, %22, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %24 = tosa.cast %23 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf16>
  %25 = tosa.const_shape  {values = dense<[5, 75352, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_4 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x5x75352x75352xf16> into tensor<5x75352x75352xf16>
  %collapsed_5 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %26 = tosa.matmul %collapsed_4, %collapsed_5, %6, %6 {acc_type = f32} : (tensor<5x75352x75352xf16>, tensor<5x75352x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x128xf16>
  %27 = tosa.const_shape  {values = dense<[1, 5, 75352, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_6 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 128] : tensor<5x75352x128xf16> into tensor<1x5x75352x128xf16>
  %28 = tosa.const_shape  {values = dense<48225280> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %26 [[0, 1, 2]] : tensor<5x75352x128xf16> into tensor<48225280xf16>
  return %collapsed_7 : tensor<48225280xf16>
}

// CHECK-LABEL: func @mlir_softmaxf32_attention_with_scaling_with_one_convert
// CHECK: rock.attention
// CHECK: softmaxType = f32
// COM: here scale is already converted to f32 and order of operands to mul is changed
func.func @mlir_softmaxf32_attention_with_scaling_with_one_convert(%arg0: tensor<48225280xf16>, %arg1: tensor<48225280xf16>, %arg2: tensor<48225280xf16>) -> tensor<48225280xf16> attributes {arch = "gfx942", kernel = "mixr", num_cu = 304 : i64} {
  %0 = tosa.const_shape  {values = dense<[1, 75352, 5, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_0 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 75352, 5, 128] : tensor<48225280xf16> into tensor<1x75352x5x128xf16>
  %1 = tosa.transpose %expanded_1 {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %2 = tosa.transpose %expanded_0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x128x75352xf16>
  %3 = tosa.transpose %expanded {perms = array<i32: 0, 2, 1, 3>} : (tensor<1x75352x5x128xf16>) -> tensor<1x5x75352x128xf16>
  %4 = tosa.const_shape  {values = dense<[5, 75352, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %5 = tosa.const_shape  {values = dense<[5, 128, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_2 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<1x5x128x75352xf16> into tensor<5x128x75352xf16>
  %6 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %7 = tosa.matmul %collapsed, %collapsed_2, %6, %6 {acc_type = f32} : (tensor<5x75352x128xf16>, tensor<5x128x75352xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x75352xf16>
  %8 = tosa.const_shape  {values = dense<[1, 5, 75352, 75352]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_3 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 75352] : tensor<5x75352x75352xf16> into tensor<1x5x75352x75352xf16>
  %9 = "tosa.const"() <{values = dense<8.837890e-02> : tensor<1x5x75352x75352xf32>}> : () -> tensor<1x5x75352x75352xf32>
  %10 = tosa.cast %expanded_3 : (tensor<1x5x75352x75352xf16>) -> tensor<1x5x75352x75352xf32>
  %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %13 = tosa.mul %9, %10, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %14 = tosa.reduce_max %13 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %15 = tosa.const_shape  {values = dense<[1, 5, 75352, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %16 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1x5x75352x75352xf32>}> : () -> tensor<1x5x75352x75352xf32>
  %17 = tosa.add %14, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %18 = tosa.sub %13, %17 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %19 = tosa.exp %18 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %20 = tosa.reduce_sum %19 {axis = 3 : i32} : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x1xf32>
  %21 = tosa.add %20, %16 : (tensor<1x5x75352x1xf32>, tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %22 = tosa.reciprocal %21 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf32>
  %23 = tosa.mul %19, %22, %12 : (tensor<1x5x75352x75352xf32>, tensor<1x5x75352x75352xf32>, tensor<1xi8>) -> tensor<1x5x75352x75352xf32>
  %24 = tosa.cast %23 : (tensor<1x5x75352x75352xf32>) -> tensor<1x5x75352x75352xf16>
  %25 = tosa.const_shape  {values = dense<[5, 75352, 75352]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %collapsed_4 = tensor.collapse_shape %24 [[0, 1], [2], [3]] : tensor<1x5x75352x75352xf16> into tensor<5x75352x75352xf16>
  %collapsed_5 = tensor.collapse_shape %3 [[0, 1], [2], [3]] : tensor<1x5x75352x128xf16> into tensor<5x75352x128xf16>
  %26 = tosa.matmul %collapsed_4, %collapsed_5, %6, %6 {acc_type = f32} : (tensor<5x75352x75352xf16>, tensor<5x75352x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<5x75352x128xf16>
  %27 = tosa.const_shape  {values = dense<[1, 5, 75352, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded_6 = tensor.expand_shape %26 [[0, 1], [2], [3]] output_shape [1, 5, 75352, 128] : tensor<5x75352x128xf16> into tensor<1x5x75352x128xf16>
  %28 = tosa.const_shape  {values = dense<48225280> : tensor<1xindex>} : () -> !tosa.shape<1>
  %collapsed_7 = tensor.collapse_shape %26 [[0, 1, 2]] : tensor<5x75352x128xf16> into tensor<48225280xf16>
  return %collapsed_7 : tensor<48225280xf16>
}

// CHECK-LABEL: func @mlir_attention_convert_scale_bias_kvcache_softmax_convert
// CHECK: rock.attention
// CHECK: currentSeqLen = (%arg3 : tensor<32xi32>)
// CHECK: softmaxType = f32
func.func @mlir_attention_convert_scale_bias_kvcache_softmax_convert(%arg0: tensor<12288xf16>, %arg1: tensor<4194304xf16>, %arg2: tensor<4194304xf16>, %arg3: tensor<32xi32>, %arg4: tensor<32768xf32>, %arg5: tensor<32768xf32>) -> tensor<4096xf16> attributes {arch = "gfx942", kernel} {
  %0 = "tosa.const"() <{values = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000800100008101000082010000830100008401000085010000860100008701000088010000890100008A0100008B0100008C0100008D0100008E0100008F010000900100009101000092010000930100009401000095010000960100009701000098010000990100009A0100009B0100009C0100009D0100009E0100009F010000A0010000A1010000A2010000A3010000A4010000A5010000A6010000A7010000A8010000A9010000AA010000AB010000AC010000AD010000AE010000AF010000B0010000B1010000B2010000B3010000B4010000B5010000B6010000B7010000B8010000B9010000BA010000BB010000BC010000BD010000BE010000BF010000C0010000C1010000C2010000C3010000C4010000C5010000C6010000C7010000C8010000C9010000CA010000CB010000CC010000CD010000CE010000CF010000D0010000D1010000D2010000D3010000D4010000D5010000D6010000D7010000D8010000D9010000DA010000DB010000DC010000DD010000DE010000DF010000E0010000E1010000E2010000E3010000E4010000E5010000E6010000E7010000E8010000E9010000EA010000EB010000EC010000ED010000EE010000EF010000F0010000F1010000F2010000F3010000F4010000F5010000F6010000F7010000F8010000F9010000FA010000FB010000FC010000FD010000FE010000FF010000000200000102000002020000030200000402000005020000060200000702000008020000090200000A0200000B0200000C0200000D0200000E0200000F020000100200001102000012020000130200001402000015020000160200001702000018020000190200001A0200001B0200001C0200001D0200001E0200001F020000200200002102000022020000230200002402000025020000260200002702000028020000290200002A0200002B0200002C0200002D0200002E0200002F020000300200003102000032020000330200003402000035020000360200003702000038020000390200003A0200003B0200003C0200003D0200003E0200003F020000400200004102000042020000430200004402000045020000460200004702000048020000490200004A0200004B0200004C0200004D0200004E0200004F020000500200005102000052020000530200005402000055020000560200005702000058020000590200005A0200005B0200005C0200005D0200005E0200005F020000600200006102000062020000630200006402000065020000660200006702000068020000690200006A0200006B0200006C0200006D0200006E0200006F020000700200007102000072020000730200007402000075020000760200007702000078020000790200007A0200007B0200007C0200007D0200007E0200007F020000800200008102000082020000830200008402000085020000860200008702000088020000890200008A0200008B0200008C0200008D0200008E0200008F020000900200009102000092020000930200009402000095020000960200009702000098020000990200009A0200009B0200009C0200009D0200009E0200009F020000A0020000A1020000A2020000A3020000A4020000A5020000A6020000A7020000A8020000A9020000AA020000AB020000AC020000AD020000AE020000AF020000B0020000B1020000B2020000B3020000B4020000B5020000B6020000B7020000B8020000B9020000BA020000BB020000BC020000BD020000BE020000BF020000C0020000C1020000C2020000C3020000C4020000C5020000C6020000C7020000C8020000C9020000CA020000CB020000CC020000CD020000CE020000CF020000D0020000D1020000D2020000D3020000D4020000D5020000D6020000D7020000D8020000D9020000DA020000DB020000DC020000DD020000DE020000DF020000E0020000E1020000E2020000E3020000E4020000E5020000E6020000E7020000E8020000E9020000EA020000EB020000EC020000ED020000EE020000EF020000F0020000F1020000F2020000F3020000F4020000F5020000F6020000F7020000F8020000F9020000FA020000FB020000FC020000FD020000FE020000FF020000000300000103000002030000030300000403000005030000060300000703000008030000090300000A0300000B0300000C0300000D0300000E0300000F030000100300001103000012030000130300001403000015030000160300001703000018030000190300001A0300001B0300001C0300001D0300001E0300001F030000200300002103000022030000230300002403000025030000260300002703000028030000290300002A0300002B0300002C0300002D0300002E0300002F030000300300003103000032030000330300003403000035030000360300003703000038030000390300003A0300003B0300003C0300003D0300003E0300003F030000400300004103000042030000430300004403000045030000460300004703000048030000490300004A0300004B0300004C0300004D0300004E0300004F030000500300005103000052030000530300005403000055030000560300005703000058030000590300005A0300005B0300005C0300005D0300005E0300005F030000600300006103000062030000630300006403000065030000660300006703000068030000690300006A0300006B0300006C0300006D0300006E0300006F030000700300007103000072030000730300007403000075030000760300007703000078030000790300007A0300007B0300007C0300007D0300007E0300007F030000800300008103000082030000830300008403000085030000860300008703000088030000890300008A0300008B0300008C0300008D0300008E0300008F030000900300009103000092030000930300009403000095030000960300009703000098030000990300009A0300009B0300009C0300009D0300009E0300009F030000A0030000A1030000A2030000A3030000A4030000A5030000A6030000A7030000A8030000A9030000AA030000AB030000AC030000AD030000AE030000AF030000B0030000B1030000B2030000B3030000B4030000B5030000B6030000B7030000B8030000B9030000BA030000BB030000BC030000BD030000BE030000BF030000C0030000C1030000C2030000C3030000C4030000C5030000C6030000C7030000C8030000C9030000CA030000CB030000CC030000CD030000CE030000CF030000D0030000D1030000D2030000D3030000D4030000D5030000D6030000D7030000D8030000D9030000DA030000DB030000DC030000DD030000DE030000DF030000E0030000E1030000E2030000E3030000E4030000E5030000E6030000E7030000E8030000E9030000EA030000EB030000EC030000ED030000EE030000EF030000F0030000F1030000F2030000F3030000F4030000F5030000F6030000F7030000F8030000F9030000FA030000FB030000FC030000FD030000FE030000FF030000"> : tensor<1x1x1x1024xi32>}> : () -> tensor<1x1x1x1024xi32>
  %1 = tosa.const_shape  {values = dense<4096> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[32, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[32, 1, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[1, 32, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %5 = "tosa.const"() <{values = dense<0> : tensor<1x32x1x1024xi32>}> : () -> tensor<1x32x1x1024xi32>
  %6 = "tosa.const"() <{values = dense<0.0883788987> : tensor<1x32x1x1024xf32>}> : () -> tensor<1x32x1x1024xf32>
  %7 = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x32x1x1024xf32>}> : () -> tensor<1x32x1x1024xf32>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.const_shape  {values = dense<[32, 128, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %11 = tosa.const_shape  {values = dense<[32, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %12 = tosa.const_shape  {values = dense<[1, 96, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %13 = tosa.const_shape  {values = dense<[1, 32, 1, 1024]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %14 = tosa.const_shape  {values = dense<[1, 32, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg5 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf32> into tensor<1x32x1x1024xf32>
  %expanded_0 = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf32> into tensor<1x32x1x1024xf32>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 96, 1, 128] : tensor<12288xf16> into tensor<1x96x1x128xf16>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 1024, 128] : tensor<4194304xf16> into tensor<1x32x1024x128xf16>
  %15 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x1024x128xf16>) -> tensor<1x32x128x1024xf16>
  %extracted_slice = tensor.extract_slice %expanded_1[0, 0, 0, 0] [1, 32, 1, 128] [1, 1, 1, 1] : tensor<1x96x1x128xf16> to tensor<1x32x1x128xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
  %collapsed_3 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x32x128x1024xf16> into tensor<32x128x1024xf16>
  %16 = tosa.matmul %collapsed, %collapsed_3, %9, %9 {acc_type = f32} : (tensor<32x1x128xf16>, tensor<32x128x1024xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x1024xf16>
  %expanded_4 = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [1, 32, 1, 1024] : tensor<32x1x1024xf16> into tensor<1x32x1x1024xf16>
  %17 = tosa.cast %expanded_4 : (tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf32>
  %18 = tosa.mul %17, %expanded_0, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %19 = tosa.add %18, %expanded : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %20 = tosa.add %0, %5 : (tensor<1x1x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xi32> into tensor<1x32x1x1xi32>
  %21 = tosa.add %expanded_5, %5 : (tensor<1x32x1x1xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %22 = tosa.greater %20, %21 : (tensor<1x32x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi1>
  %23 = tosa.mul %19, %6, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %24 = tosa.select %22, %7, %23 : (tensor<1x32x1x1024xi1>, tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %25 = tosa.reduce_max %24 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %26 = tosa.sub %24, %25 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1024xf32>
  %27 = tosa.exp %26 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %28 = tosa.reduce_sum %27 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %29 = tosa.reciprocal %28 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
  %30 = tosa.mul %27, %29, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %31 = tosa.cast %30 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf16>
  %collapsed_6 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x32x1x1024xf16> into tensor<32x1x1024xf16>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 1024, 128] : tensor<4194304xf16> into tensor<32x1024x128xf16>
  %32 = tosa.matmul %collapsed_6, %expanded_7, %9, %9 {acc_type = f32} : (tensor<32x1x1024xf16>, tensor<32x1024x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x128xf16>
  %collapsed_8 = tensor.collapse_shape %32 [[0, 1, 2]] : tensor<32x1x128xf16> into tensor<4096xf16>
  return %collapsed_8 : tensor<4096xf16>
}

// CHECK-LABEL: func @mlir_attention_scale_bias_convert_kvcache_softmax_convert
// CHECK: rock.attention
// CHECK: currentSeqLen = (%arg3 : tensor<32xi32>)
// CHECK: softmaxType = f32
func.func @mlir_attention_scale_bias_convert_kvcache_softmax_convert(%arg0: tensor<12288xf16>, %arg1: tensor<4194304xf16>, %arg2: tensor<4194304xf16>, %arg3: tensor<32xi32>, %arg4: tensor<32768xf16>, %arg5: tensor<32768xf16>) -> tensor<4096xf16> attributes {arch = "gfx942", kernel} {
  %0 = "tosa.const"() <{values = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000800100008101000082010000830100008401000085010000860100008701000088010000890100008A0100008B0100008C0100008D0100008E0100008F010000900100009101000092010000930100009401000095010000960100009701000098010000990100009A0100009B0100009C0100009D0100009E0100009F010000A0010000A1010000A2010000A3010000A4010000A5010000A6010000A7010000A8010000A9010000AA010000AB010000AC010000AD010000AE010000AF010000B0010000B1010000B2010000B3010000B4010000B5010000B6010000B7010000B8010000B9010000BA010000BB010000BC010000BD010000BE010000BF010000C0010000C1010000C2010000C3010000C4010000C5010000C6010000C7010000C8010000C9010000CA010000CB010000CC010000CD010000CE010000CF010000D0010000D1010000D2010000D3010000D4010000D5010000D6010000D7010000D8010000D9010000DA010000DB010000DC010000DD010000DE010000DF010000E0010000E1010000E2010000E3010000E4010000E5010000E6010000E7010000E8010000E9010000EA010000EB010000EC010000ED010000EE010000EF010000F0010000F1010000F2010000F3010000F4010000F5010000F6010000F7010000F8010000F9010000FA010000FB010000FC010000FD010000FE010000FF010000000200000102000002020000030200000402000005020000060200000702000008020000090200000A0200000B0200000C0200000D0200000E0200000F020000100200001102000012020000130200001402000015020000160200001702000018020000190200001A0200001B0200001C0200001D0200001E0200001F020000200200002102000022020000230200002402000025020000260200002702000028020000290200002A0200002B0200002C0200002D0200002E0200002F020000300200003102000032020000330200003402000035020000360200003702000038020000390200003A0200003B0200003C0200003D0200003E0200003F020000400200004102000042020000430200004402000045020000460200004702000048020000490200004A0200004B0200004C0200004D0200004E0200004F020000500200005102000052020000530200005402000055020000560200005702000058020000590200005A0200005B0200005C0200005D0200005E0200005F020000600200006102000062020000630200006402000065020000660200006702000068020000690200006A0200006B0200006C0200006D0200006E0200006F020000700200007102000072020000730200007402000075020000760200007702000078020000790200007A0200007B0200007C0200007D0200007E0200007F020000800200008102000082020000830200008402000085020000860200008702000088020000890200008A0200008B0200008C0200008D0200008E0200008F020000900200009102000092020000930200009402000095020000960200009702000098020000990200009A0200009B0200009C0200009D0200009E0200009F020000A0020000A1020000A2020000A3020000A4020000A5020000A6020000A7020000A8020000A9020000AA020000AB020000AC020000AD020000AE020000AF020000B0020000B1020000B2020000B3020000B4020000B5020000B6020000B7020000B8020000B9020000BA020000BB020000BC020000BD020000BE020000BF020000C0020000C1020000C2020000C3020000C4020000C5020000C6020000C7020000C8020000C9020000CA020000CB020000CC020000CD020000CE020000CF020000D0020000D1020000D2020000D3020000D4020000D5020000D6020000D7020000D8020000D9020000DA020000DB020000DC020000DD020000DE020000DF020000E0020000E1020000E2020000E3020000E4020000E5020000E6020000E7020000E8020000E9020000EA020000EB020000EC020000ED020000EE020000EF020000F0020000F1020000F2020000F3020000F4020000F5020000F6020000F7020000F8020000F9020000FA020000FB020000FC020000FD020000FE020000FF020000000300000103000002030000030300000403000005030000060300000703000008030000090300000A0300000B0300000C0300000D0300000E0300000F030000100300001103000012030000130300001403000015030000160300001703000018030000190300001A0300001B0300001C0300001D0300001E0300001F030000200300002103000022030000230300002403000025030000260300002703000028030000290300002A0300002B0300002C0300002D0300002E0300002F030000300300003103000032030000330300003403000035030000360300003703000038030000390300003A0300003B0300003C0300003D0300003E0300003F030000400300004103000042030000430300004403000045030000460300004703000048030000490300004A0300004B0300004C0300004D0300004E0300004F030000500300005103000052030000530300005403000055030000560300005703000058030000590300005A0300005B0300005C0300005D0300005E0300005F030000600300006103000062030000630300006403000065030000660300006703000068030000690300006A0300006B0300006C0300006D0300006E0300006F030000700300007103000072030000730300007403000075030000760300007703000078030000790300007A0300007B0300007C0300007D0300007E0300007F030000800300008103000082030000830300008403000085030000860300008703000088030000890300008A0300008B0300008C0300008D0300008E0300008F030000900300009103000092030000930300009403000095030000960300009703000098030000990300009A0300009B0300009C0300009D0300009E0300009F030000A0030000A1030000A2030000A3030000A4030000A5030000A6030000A7030000A8030000A9030000AA030000AB030000AC030000AD030000AE030000AF030000B0030000B1030000B2030000B3030000B4030000B5030000B6030000B7030000B8030000B9030000BA030000BB030000BC030000BD030000BE030000BF030000C0030000C1030000C2030000C3030000C4030000C5030000C6030000C7030000C8030000C9030000CA030000CB030000CC030000CD030000CE030000CF030000D0030000D1030000D2030000D3030000D4030000D5030000D6030000D7030000D8030000D9030000DA030000DB030000DC030000DD030000DE030000DF030000E0030000E1030000E2030000E3030000E4030000E5030000E6030000E7030000E8030000E9030000EA030000EB030000EC030000ED030000EE030000EF030000F0030000F1030000F2030000F3030000F4030000F5030000F6030000F7030000F8030000F9030000FA030000FB030000FC030000FD030000FE030000FF030000"> : tensor<1x1x1x1024xi32>}> : () -> tensor<1x1x1x1024xi32>
  %1 = tosa.const_shape  {values = dense<4096> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[32, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[32, 1, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[1, 32, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %5 = "tosa.const"() <{values = dense<0> : tensor<1x32x1x1024xi32>}> : () -> tensor<1x32x1x1024xi32>
  %6 = "tosa.const"() <{values = dense<0.0883788987> : tensor<1x32x1x1024xf32>}> : () -> tensor<1x32x1x1024xf32>
  %7 = "tosa.const"() <{values = dense<0xFF800000> : tensor<1x32x1x1024xf32>}> : () -> tensor<1x32x1x1024xf32>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.const_shape  {values = dense<[32, 128, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %11 = tosa.const_shape  {values = dense<[32, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %12 = tosa.const_shape  {values = dense<[1, 96, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %13 = tosa.const_shape  {values = dense<[1, 32, 1, 1024]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %14 = tosa.const_shape  {values = dense<[1, 32, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg5 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf16> into tensor<1x32x1x1024xf16>
  %expanded_0 = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf16> into tensor<1x32x1x1024xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 96, 1, 128] : tensor<12288xf16> into tensor<1x96x1x128xf16>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 1024, 128] : tensor<4194304xf16> into tensor<1x32x1024x128xf16>
  %15 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x1024x128xf16>) -> tensor<1x32x128x1024xf16>
  %extracted_slice = tensor.extract_slice %expanded_1[0, 0, 0, 0] [1, 32, 1, 128] [1, 1, 1, 1] : tensor<1x96x1x128xf16> to tensor<1x32x1x128xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
  %collapsed_3 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x32x128x1024xf16> into tensor<32x128x1024xf16>
  %16 = tosa.matmul %collapsed, %collapsed_3, %9, %9 {acc_type = f32} : (tensor<32x1x128xf16>, tensor<32x128x1024xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x1024xf16>
  %expanded_4 = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [1, 32, 1, 1024] : tensor<32x1x1024xf16> into tensor<1x32x1x1024xf16>
  %17 = tosa.mul %expanded_4, %expanded_0, %8 : (tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>, tensor<1xi8>) -> tensor<1x32x1x1024xf16>
  %18 = tosa.add %17, %expanded : (tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf16>
  %19 = tosa.cast %18 : (tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf32>
  %20 = tosa.add %0, %5 : (tensor<1x1x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xi32> into tensor<1x32x1x1xi32>
  %21 = tosa.add %expanded_5, %5 : (tensor<1x32x1x1xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %22 = tosa.greater %20, %21 : (tensor<1x32x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi1>
  %23 = tosa.mul %19, %6, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %24 = tosa.select %22, %7, %23 : (tensor<1x32x1x1024xi1>, tensor<1x32x1x1024xf32>, tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %25 = tosa.reduce_max %24 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %26 = tosa.sub %24, %25 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1024xf32>
  %27 = tosa.exp %26 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %28 = tosa.reduce_sum %27 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %29 = tosa.reciprocal %28 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
  %30 = tosa.mul %27, %29, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %31 = tosa.cast %30 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf16>
  %collapsed_6 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x32x1x1024xf16> into tensor<32x1x1024xf16>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 1024, 128] : tensor<4194304xf16> into tensor<32x1024x128xf16>
  %32 = tosa.matmul %collapsed_6, %expanded_7, %9, %9 {acc_type = f32} : (tensor<32x1x1024xf16>, tensor<32x1024x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x128xf16>
  %collapsed_8 = tensor.collapse_shape %32 [[0, 1, 2]] : tensor<32x1x128xf16> into tensor<4096xf16>
  return %collapsed_8 : tensor<4096xf16>
}

// CHECK-LABEL: func @mlir_attention_scale_bias_kvcache_convert_softmax_convert
// CHECK: rock.attention
// CHECK: currentSeqLen = (%arg3 : tensor<32xi32>)
// CHECK: softmaxType = f32
func.func @mlir_attention_scale_bias_kvcache_convert_softmax_convert(%arg0: tensor<12288xf16>, %arg1: tensor<4194304xf16>, %arg2: tensor<4194304xf16>, %arg3: tensor<32xi32>, %arg4: tensor<32768xf16>, %arg5: tensor<32768xf16>) -> tensor<4096xf16> attributes {arch = "gfx942", kernel} {
  %0 = "tosa.const"() <{values = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000800100008101000082010000830100008401000085010000860100008701000088010000890100008A0100008B0100008C0100008D0100008E0100008F010000900100009101000092010000930100009401000095010000960100009701000098010000990100009A0100009B0100009C0100009D0100009E0100009F010000A0010000A1010000A2010000A3010000A4010000A5010000A6010000A7010000A8010000A9010000AA010000AB010000AC010000AD010000AE010000AF010000B0010000B1010000B2010000B3010000B4010000B5010000B6010000B7010000B8010000B9010000BA010000BB010000BC010000BD010000BE010000BF010000C0010000C1010000C2010000C3010000C4010000C5010000C6010000C7010000C8010000C9010000CA010000CB010000CC010000CD010000CE010000CF010000D0010000D1010000D2010000D3010000D4010000D5010000D6010000D7010000D8010000D9010000DA010000DB010000DC010000DD010000DE010000DF010000E0010000E1010000E2010000E3010000E4010000E5010000E6010000E7010000E8010000E9010000EA010000EB010000EC010000ED010000EE010000EF010000F0010000F1010000F2010000F3010000F4010000F5010000F6010000F7010000F8010000F9010000FA010000FB010000FC010000FD010000FE010000FF010000000200000102000002020000030200000402000005020000060200000702000008020000090200000A0200000B0200000C0200000D0200000E0200000F020000100200001102000012020000130200001402000015020000160200001702000018020000190200001A0200001B0200001C0200001D0200001E0200001F020000200200002102000022020000230200002402000025020000260200002702000028020000290200002A0200002B0200002C0200002D0200002E0200002F020000300200003102000032020000330200003402000035020000360200003702000038020000390200003A0200003B0200003C0200003D0200003E0200003F020000400200004102000042020000430200004402000045020000460200004702000048020000490200004A0200004B0200004C0200004D0200004E0200004F020000500200005102000052020000530200005402000055020000560200005702000058020000590200005A0200005B0200005C0200005D0200005E0200005F020000600200006102000062020000630200006402000065020000660200006702000068020000690200006A0200006B0200006C0200006D0200006E0200006F020000700200007102000072020000730200007402000075020000760200007702000078020000790200007A0200007B0200007C0200007D0200007E0200007F020000800200008102000082020000830200008402000085020000860200008702000088020000890200008A0200008B0200008C0200008D0200008E0200008F020000900200009102000092020000930200009402000095020000960200009702000098020000990200009A0200009B0200009C0200009D0200009E0200009F020000A0020000A1020000A2020000A3020000A4020000A5020000A6020000A7020000A8020000A9020000AA020000AB020000AC020000AD020000AE020000AF020000B0020000B1020000B2020000B3020000B4020000B5020000B6020000B7020000B8020000B9020000BA020000BB020000BC020000BD020000BE020000BF020000C0020000C1020000C2020000C3020000C4020000C5020000C6020000C7020000C8020000C9020000CA020000CB020000CC020000CD020000CE020000CF020000D0020000D1020000D2020000D3020000D4020000D5020000D6020000D7020000D8020000D9020000DA020000DB020000DC020000DD020000DE020000DF020000E0020000E1020000E2020000E3020000E4020000E5020000E6020000E7020000E8020000E9020000EA020000EB020000EC020000ED020000EE020000EF020000F0020000F1020000F2020000F3020000F4020000F5020000F6020000F7020000F8020000F9020000FA020000FB020000FC020000FD020000FE020000FF020000000300000103000002030000030300000403000005030000060300000703000008030000090300000A0300000B0300000C0300000D0300000E0300000F030000100300001103000012030000130300001403000015030000160300001703000018030000190300001A0300001B0300001C0300001D0300001E0300001F030000200300002103000022030000230300002403000025030000260300002703000028030000290300002A0300002B0300002C0300002D0300002E0300002F030000300300003103000032030000330300003403000035030000360300003703000038030000390300003A0300003B0300003C0300003D0300003E0300003F030000400300004103000042030000430300004403000045030000460300004703000048030000490300004A0300004B0300004C0300004D0300004E0300004F030000500300005103000052030000530300005403000055030000560300005703000058030000590300005A0300005B0300005C0300005D0300005E0300005F030000600300006103000062030000630300006403000065030000660300006703000068030000690300006A0300006B0300006C0300006D0300006E0300006F030000700300007103000072030000730300007403000075030000760300007703000078030000790300007A0300007B0300007C0300007D0300007E0300007F030000800300008103000082030000830300008403000085030000860300008703000088030000890300008A0300008B0300008C0300008D0300008E0300008F030000900300009103000092030000930300009403000095030000960300009703000098030000990300009A0300009B0300009C0300009D0300009E0300009F030000A0030000A1030000A2030000A3030000A4030000A5030000A6030000A7030000A8030000A9030000AA030000AB030000AC030000AD030000AE030000AF030000B0030000B1030000B2030000B3030000B4030000B5030000B6030000B7030000B8030000B9030000BA030000BB030000BC030000BD030000BE030000BF030000C0030000C1030000C2030000C3030000C4030000C5030000C6030000C7030000C8030000C9030000CA030000CB030000CC030000CD030000CE030000CF030000D0030000D1030000D2030000D3030000D4030000D5030000D6030000D7030000D8030000D9030000DA030000DB030000DC030000DD030000DE030000DF030000E0030000E1030000E2030000E3030000E4030000E5030000E6030000E7030000E8030000E9030000EA030000EB030000EC030000ED030000EE030000EF030000F0030000F1030000F2030000F3030000F4030000F5030000F6030000F7030000F8030000F9030000FA030000FB030000FC030000FD030000FE030000FF030000"> : tensor<1x1x1x1024xi32>}> : () -> tensor<1x1x1x1024xi32>
  %1 = tosa.const_shape  {values = dense<4096> : tensor<1xindex>} : () -> !tosa.shape<1>
  %2 = tosa.const_shape  {values = dense<[32, 1024, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %3 = tosa.const_shape  {values = dense<[32, 1, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<[1, 32, 1, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %5 = "tosa.const"() <{values = dense<0> : tensor<1x32x1x1024xi32>}> : () -> tensor<1x32x1x1024xi32>
  %6 = "tosa.const"() <{values = dense<0.0883788987> : tensor<1x32x1x1024xf16>}> : () -> tensor<1x32x1x1024xf16>
  %7 = "tosa.const"() <{values = dense<0xFC00> : tensor<1x32x1x1024xf16>}> : () -> tensor<1x32x1x1024xf16>
  %8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %9 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %10 = tosa.const_shape  {values = dense<[32, 128, 1024]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %11 = tosa.const_shape  {values = dense<[32, 1, 128]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %12 = tosa.const_shape  {values = dense<[1, 96, 1, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %13 = tosa.const_shape  {values = dense<[1, 32, 1, 1024]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %14 = tosa.const_shape  {values = dense<[1, 32, 1024, 128]> : tensor<4xindex>} : () -> !tosa.shape<4>
  %expanded = tensor.expand_shape %arg5 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf16> into tensor<1x32x1x1024xf16>
  %expanded_0 = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1024] : tensor<32768xf16> into tensor<1x32x1x1024xf16>
  %expanded_1 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 96, 1, 128] : tensor<12288xf16> into tensor<1x96x1x128xf16>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 32, 1024, 128] : tensor<4194304xf16> into tensor<1x32x1024x128xf16>
  %15 = tosa.transpose %expanded_2 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x32x1024x128xf16>) -> tensor<1x32x128x1024xf16>
  %extracted_slice = tensor.extract_slice %expanded_1[0, 0, 0, 0] [1, 32, 1, 128] [1, 1, 1, 1] : tensor<1x96x1x128xf16> to tensor<1x32x1x128xf16>
  %collapsed = tensor.collapse_shape %extracted_slice [[0, 1], [2], [3]] : tensor<1x32x1x128xf16> into tensor<32x1x128xf16>
  %collapsed_3 = tensor.collapse_shape %15 [[0, 1], [2], [3]] : tensor<1x32x128x1024xf16> into tensor<32x128x1024xf16>
  %16 = tosa.matmul %collapsed, %collapsed_3, %9, %9 {acc_type = f32} : (tensor<32x1x128xf16>, tensor<32x128x1024xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x1024xf16>
  %expanded_4 = tensor.expand_shape %16 [[0, 1], [2], [3]] output_shape [1, 32, 1, 1024] : tensor<32x1x1024xf16> into tensor<1x32x1x1024xf16>
  %17 = tosa.mul %expanded_4, %expanded_0, %8 : (tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>, tensor<1xi8>) -> tensor<1x32x1x1024xf16>
  %18 = tosa.add %17, %expanded : (tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf16>
  %19 = tosa.add %0, %5 : (tensor<1x1x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %expanded_5 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 32, 1, 1] : tensor<32xi32> into tensor<1x32x1x1xi32>
  %20 = tosa.add %expanded_5, %5 : (tensor<1x32x1x1xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi32>
  %21 = tosa.greater %19, %20 : (tensor<1x32x1x1024xi32>, tensor<1x32x1x1024xi32>) -> tensor<1x32x1x1024xi1>
  %22 = tosa.mul %18, %6, %8 : (tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>, tensor<1xi8>) -> tensor<1x32x1x1024xf16>
  %23 = tosa.select %21, %7, %22 : (tensor<1x32x1x1024xi1>, tensor<1x32x1x1024xf16>, tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf16>
  %24 = tosa.cast %23 : (tensor<1x32x1x1024xf16>) -> tensor<1x32x1x1024xf32>
  %25 = tosa.reduce_max %24 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %26 = tosa.sub %24, %25 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>) -> tensor<1x32x1x1024xf32>
  %27 = tosa.exp %26 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf32>
  %28 = tosa.reduce_sum %27 {axis = 3 : i32} : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1xf32>
  %29 = tosa.reciprocal %28 : (tensor<1x32x1x1xf32>) -> tensor<1x32x1x1xf32>
  %30 = tosa.mul %27, %29, %8 : (tensor<1x32x1x1024xf32>, tensor<1x32x1x1xf32>, tensor<1xi8>) -> tensor<1x32x1x1024xf32>
  %31 = tosa.cast %30 : (tensor<1x32x1x1024xf32>) -> tensor<1x32x1x1024xf16>
  %collapsed_6 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<1x32x1x1024xf16> into tensor<32x1x1024xf16>
  %expanded_7 = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [32, 1024, 128] : tensor<4194304xf16> into tensor<32x1024x128xf16>
  %32 = tosa.matmul %collapsed_6, %expanded_7, %9, %9 {acc_type = f32} : (tensor<32x1x1024xf16>, tensor<32x1024x128xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<32x1x128xf16>
  %collapsed_8 = tensor.collapse_shape %32 [[0, 1, 2]] : tensor<32x1x128xf16> into tensor<4096xf16>
  return %collapsed_8 : tensor<4096xf16>
}

// CHECK-LABEL: func @mlir_attention_i8_convert_softmax_convert_f16
// CHECK: rock.attention
// CHECK: qk = {{.*}} * {{.*}} : tensor<1x64x32xi8>, tensor<1x32x64xi8>
// CHECK: %1 = softmax(qk) * {{.*}} : tensor<1x64x32xf16>  -> tensor<1x64x32xf16>
// CHECK: softmaxType = f32
func.func @mlir_attention_i8_convert_softmax_convert_f16(%arg0: tensor<2048xi8>, %arg1: tensor<2048xi8>, %arg2: tensor<2048xf16>, %arg3: tensor<4096xf16>, %arg4: tensor<1xf16>) -> tensor<2048xf16> attributes {arch = "gfx942", kernel} {
  %0 = tosa.const_shape  {values = dense<2048> : tensor<1xindex>} : () -> !tosa.shape<1>
  %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  %2 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
  %3 = tosa.const_shape  {values = dense<[1, 32, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %4 = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
  %5 = tosa.const_shape  {values = dense<[1, 64, 64]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %6 = tosa.const_shape  {values = dense<[1, 64, 32]> : tensor<3xindex>} : () -> !tosa.shape<3>
  %expanded = tensor.expand_shape %arg2 [[0, 1, 2]] output_shape [1, 64, 32] : tensor<2048xf16> into tensor<1x64x32xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2]] output_shape [1, 64, 64] : tensor<4096xf16> into tensor<1x64x64xf16>
  %expanded_1 = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [1, 1, 1] : tensor<1xf16> into tensor<1x1x1xf16>
  %expanded_2 = tensor.expand_shape %arg1 [[0, 1, 2]] output_shape [1, 32, 64] : tensor<2048xi8> into tensor<1x32x64xi8>
  %expanded_3 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [1, 64, 32] : tensor<2048xi8> into tensor<1x64x32xi8>
  %7 = tosa.matmul %expanded_3, %expanded_2, %2, %2 {acc_type = i32} : (tensor<1x64x32xi8>, tensor<1x32x64xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x64x64xi32>
  %8 = tosa.cast %7 : (tensor<1x64x64xi32>) -> tensor<1x64x64xf16>
  %9 = tosa.mul %8, %expanded_1, %2 : (tensor<1x64x64xf16>, tensor<1x1x1xf16>, tensor<1xi8>) -> tensor<1x64x64xf16>
  %10 = tosa.add %9, %expanded_0 : (tensor<1x64x64xf16>, tensor<1x64x64xf16>) -> tensor<1x64x64xf16>
  %11 = tosa.cast %10 : (tensor<1x64x64xf16>) -> tensor<1x64x64xf32>
  %12 = tosa.reduce_max %11 {axis = 2 : i32} : (tensor<1x64x64xf32>) -> tensor<1x64x1xf32>
  %13 = tosa.sub %11, %12 : (tensor<1x64x64xf32>, tensor<1x64x1xf32>) -> tensor<1x64x64xf32>
  %14 = tosa.exp %13 : (tensor<1x64x64xf32>) -> tensor<1x64x64xf32>
  %15 = tosa.reduce_sum %14 {axis = 2 : i32} : (tensor<1x64x64xf32>) -> tensor<1x64x1xf32>
  %16 = tosa.reciprocal %15 : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %17 = tosa.mul %14, %16, %2 : (tensor<1x64x64xf32>, tensor<1x64x1xf32>, tensor<1xi8>) -> tensor<1x64x64xf32>
  %18 = tosa.cast %17 : (tensor<1x64x64xf32>) -> tensor<1x64x64xf16>
  %19 = tosa.matmul %18, %expanded, %1, %1 {acc_type = f32} : (tensor<1x64x64xf16>, tensor<1x64x32xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x64x32xf16>
  %collapsed = tensor.collapse_shape %19 [[0, 1, 2]] : tensor<1x64x32xf16> into tensor<2048xf16>
  return %collapsed : tensor<2048xf16>
}
