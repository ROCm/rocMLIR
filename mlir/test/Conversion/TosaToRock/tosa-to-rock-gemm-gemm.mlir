// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics -o -| FileCheck %s

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x384x64xf32>) -> tensor<1x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x384xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %2 = "tosa.mul"(%1, %arg3, %shift) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>, tensor<1xi8>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%2, %arg2, %a_zp, %b_zp) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_no_scale(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x384x64xf32>) -> tensor<1x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%1, %arg2, %a_zp, %b_zp) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_bias_only(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x384x64xf32>) -> tensor<1x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x384xf32>
  %2 = "tosa.add"(%1, %arg3) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %9 = "tosa.matmul"(%2, %arg2, %a_zp, %b_zp) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x64xf32>
  return %9 : tensor<1x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_scale_and_bias(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>, %arg4: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x384x64xf32>) -> tensor<1x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x384xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %2 = "tosa.mul"(%1, %arg3, %shift) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>, tensor<1xi8>) -> tensor<1x384x384xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %10 = "tosa.matmul"(%3, %arg2, %a_zp, %b_zp) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x64xf32>
  return %10 : tensor<1x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_scale_bias_exp(%arg0: tensor<1x384x64xf32>, %arg1: tensor<1x384x64xf32>, %arg2: tensor<1x384x64xf32>, %arg3: tensor<1x384x384xf32>, %arg4: tensor<1x384x384xf32>) -> tensor<1x384x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x384x64xf32>) -> tensor<1x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x384x64xf32>, tensor<1x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x384xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %2 = "tosa.mul"(%1, %arg3, %shift) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>, tensor<1xi8>) -> tensor<1x384x384xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x384x384xf32>, tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %exp = "tosa.exp"(%3) : (tensor<1x384x384xf32>) -> tensor<1x384x384xf32>
  %10 = "tosa.matmul"(%exp, %arg2, %a_zp, %b_zp) : (tensor<1x384x384xf32>, tensor<1x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x384x64xf32>
  return %10 : tensor<1x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_reshapes(%arg0: tensor<1x12x384x64xf32>, %arg1: tensor<1x12x64x384xf32>, %arg2: tensor<1x12x384x64xf32>) -> (tensor<1x12x384x64xf32>) attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32>
  %collapsed_0 = tensor.collapse_shape %arg1 [[0, 1], [2], [3]] : tensor<1x12x64x384xf32> into tensor<12x64x384xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = "tosa.matmul"(%collapsed, %collapsed_0, %a_zp, %b_zp) : (tensor<12x384x64xf32>, tensor<12x64x384xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x384x384xf32>
  %expanded = tensor.expand_shape %0 [[0, 1], [2], [3]] output_shape [1, 12, 384, 384] : tensor<12x384x384xf32> into tensor<1x12x384x384xf32>
  %collapsed_1 = tensor.collapse_shape %expanded [[0, 1], [2], [3]] : tensor<1x12x384x384xf32> into tensor<12x384x384xf32>
  %collapsed_2 = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<1x12x384x64xf32> into tensor<12x384x64xf32>
  %7 = "tosa.matmul"(%collapsed_1, %collapsed_2, %a_zp, %b_zp) : (tensor<12x384x384xf32>, tensor<12x384x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x384x64xf32>
  %expanded_3 = tensor.expand_shape %7 [[0, 1], [2], [3]] output_shape [1, 12, 384, 64] : tensor<12x384x64xf32> into tensor<1x12x384x64xf32>
  return %expanded_3 : tensor<1x12x384x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_4d_scale(%arg0: tensor<1x12x256x256xf32> , %arg1: tensor<1x12x256x256xf32>, %arg2: tensor<1x12x256x256xf32>, %arg3: tensor<1x12x256x256xf32>) -> (tensor<1x12x256x256xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
  %0 = "tosa.transpose"(%arg3) {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf32>) -> tensor<1x12x256x256xf32>
  %collapsed = tensor.collapse_shape %arg2 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %collapsed_0 = tensor.collapse_shape %0 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%collapsed, %collapsed_0, %a_zp, %b_zp) : (tensor<12x256x256xf32>, tensor<12x256x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x256x256xf32>
  %expanded = tensor.expand_shape %1 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf32> into tensor<1x12x256x256xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %2 = "tosa.mul"(%expanded, %arg1, %shift) : (tensor<1x12x256x256xf32>, tensor<1x12x256x256xf32>, tensor<1xi8>) -> tensor<1x12x256x256xf32>
  %collapsed_1 = tensor.collapse_shape %2 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %collapsed_2 = tensor.collapse_shape %arg0 [[0, 1], [2], [3]] : tensor<1x12x256x256xf32> into tensor<12x256x256xf32>
  %9 = "tosa.matmul"(%collapsed_1, %collapsed_2, %a_zp, %b_zp) : (tensor<12x256x256xf32>, tensor<12x256x256xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<12x256x256xf32>
  %expanded_3 = tensor.expand_shape %9 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf32> into tensor<1x12x256x256xf32>
  return %expanded_3 : tensor<1x12x256x256xf32>
}

// CHECK: rock.gemm_elementwise_gemm
func.func @self_gemm_gemm_with_dot_product(%arg0: tensor<1x1x64xf32>, %arg1: tensor<1x1x64xf32>, %arg2: tensor<1x1x64xf32>, %arg3: tensor<1x1x1xf32>, %arg4: tensor<1x1x1xf32>) -> tensor<1x1x64xf32> attributes {kernel, arch = "##TOKEN_ARCH##"} {
  %0 = "tosa.transpose"(%arg1) {perms = array<i32: 0, 2, 1>} : (tensor<1x1x64xf32>) -> tensor<1x64x1xf32>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %1 = "tosa.matmul"(%arg0, %0, %a_zp, %b_zp) : (tensor<1x1x64xf32>, tensor<1x64x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x1xf32>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %2 = "tosa.mul"(%1, %arg3, %shift) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1xf32>
  %3 = "tosa.add"(%2, %arg4) : (tensor<1x1x1xf32>, tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %8 = "tosa.matmul"(%3, %arg2, %a_zp, %b_zp) : (tensor<1x1x1xf32>, tensor<1x1x64xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x1x64xf32>
  return %8 : tensor<1x1x64xf32>
}

// CHECK: rock.gemm_elementwise_gemm
// CHECK: elementwise otherIns(%arg2, %arg3 : tensor<786432xi8>, tensor<786432xf16>)
// CHECK: firstGemmIdx = 1 : i32
func.func @mlir_gemm_gemm_where(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xi8>, %arg3: tensor<786432xf16>, %arg4: tensor<786432xf16>) -> tensor<786432xf16> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
  %expanded = tensor.expand_shape %arg4 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_0 = tensor.expand_shape %arg3 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_1 = tensor.expand_shape %arg2 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xi8> into tensor<1x12x256x256xi8>
  %expanded_2 = tensor.expand_shape %arg0 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %expanded_3 = tensor.expand_shape %arg1 [[0, 1, 2, 3]] output_shape [1, 12, 256, 256] : tensor<786432xf16> into tensor<1x12x256x256xf16>
  %1 = tosa.transpose %expanded_3 {perms = array<i32: 0, 1, 3, 2>} : (tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %expanded_4 = tensor.expand_shape %arg0 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %collapsed = tensor.collapse_shape %1 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf16>}> : () -> tensor<1xf16>
  %2 = tosa.matmul %expanded_4, %collapsed, %a_zp, %b_zp : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %expanded_5 = tensor.expand_shape %2 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %3 = "tosa.const"() <{values = dense<1.250000e-01> : tensor<1x12x256x256xf16>}> : () -> tensor<1x12x256x256xf16>
  %shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8> 
  %4 = tosa.mul %expanded_5, %3, %shift : (tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>, tensor<1xi8>) -> tensor<1x12x256x256xf16>
  %5 = tosa.cast %expanded_1 : (tensor<1x12x256x256xi8>) -> tensor<1x12x256x256xi1>
  %6 = tosa.select %5, %4, %expanded_0 : (tensor<1x12x256x256xi1>, tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16>) -> tensor<1x12x256x256xf16>
  %collapsed_6 = tensor.collapse_shape %6 [[0, 1], [2], [3]] : tensor<1x12x256x256xf16> into tensor<12x256x256xf16>
  %expanded_7 = tensor.expand_shape %arg4 [[0, 1, 2]] output_shape [12, 256, 256] : tensor<786432xf16> into tensor<12x256x256xf16>
  %13 = tosa.matmul %collapsed_6, %expanded_7, %a_zp, %b_zp : (tensor<12x256x256xf16>, tensor<12x256x256xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<12x256x256xf16>
  %expanded_8 = tensor.expand_shape %13 [[0, 1], [2], [3]] output_shape [1, 12, 256, 256] : tensor<12x256x256xf16> into tensor<1x12x256x256xf16>
  %collapsed_9 = tensor.collapse_shape %13 [[0, 1, 2]] : tensor<12x256x256xf16> into tensor<786432xf16>
  return %collapsed_9 : tensor<786432xf16>
}
