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

