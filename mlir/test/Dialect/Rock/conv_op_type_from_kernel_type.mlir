// RUN: mlir-opt %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - Fwd (Conv)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @fwd_conv
// CHECK: %[[T:.*]] = "test.conv_op_type_from_kernel_type"() {kernel_type = "Conv"} : () -> i32
// CHECK: return %[[T]]
func.func @fwd_conv() -> i32 {
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "Conv"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - Fwd (ConvElementwiseGemm)
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @fwd_conv_elemwise
// CHECK: %[[T:.*]] = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvElementwiseGemm"} : () -> i32
// CHECK: return %[[T]]
func.func @fwd_conv_elemwise() -> i32 {
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvElementwiseGemm"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - BwdData
//===----------------------------------------------------------------------===//
// CHECK-LABEL: func @bwd_data
// CHECK: %[[T:.*]] = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvBwdData"} : () -> i32
// CHECK: return %[[T]]
func.func @bwd_data() -> i32 {
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvBwdData"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - BwdWeight
// CHECK-LABEL: func @bwd_weight
// CHECK: %[[T:.*]] = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvBwdWeight"} : () -> i32
// CHECK: return %[[T]]
func.func @bwd_weight() -> i32 {
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "ConvBwdWeight"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - Gemm (should error)
//===----------------------------------------------------------------------===//
func.func @gemm() -> i32 {
  // expected-error @+1 {{GEMM ops shouldn't be in convolution-specific lowering passes}}
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "Gemm"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - Attention (should error)
//===----------------------------------------------------------------------===//
func.func @attention() -> i32 {
  // expected-error @+1 {{Attention ops shouldn't be in convolution-specific lowering passes}}
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "Attention"} : () -> i32
  return %t : i32
}

//===----------------------------------------------------------------------===//
// Test: convOpTypeFromKernelType - GemmElementwiseGemm (should error)
//===----------------------------------------------------------------------===//
func.func @gemm_elemwise() -> i32 {
  // expected-error @+1 {{gemm+gemm ops shouldn't be in convolution-specific lowering passes}}
  %t = "test.conv_op_type_from_kernel_type"() {kernel_type = "GemmElementwiseGemm"} : () -> i32
  return %t : i32
}