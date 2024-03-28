// RUN: mlir-opt --tosa-partition='trailing-only=false' %s | FileCheck %s

// CHECK: forward__part_0
// CHECK: func.func @forward
// CHECK-NEXT: call @forward__part_0
// CHECK-NEXT: return

module attributes {torch.debug_module_name = "BertTinyWrapper"} {
func.func @forward(%arg0: tensor<2x128x128xf32> {func.read_access}, %arg1: tensor<512x128xf32> {func.read_access}, %arg2: tensor<1x1x512xf32> {func.read_access}) -> (tensor<2x128x512xf32> {func.write_access}) {
  %5 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %6 = "tosa.const"() <{value = dense<1.000000e+00> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %7 = "tosa.const"() <{value = dense<0.707106769> : tensor<1x1x1xf32>}> : () -> tensor<1x1x1xf32>
  %8 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
  %9 = "tosa.transpose"(%arg1, %8) : (tensor<512x128xf32>, tensor<2xi32>) -> tensor<128x512xf32>
  %10 = "tosa.reshape"(%9) <{new_shape = array<i64: 1, 128, 512>}> : (tensor<128x512xf32>) -> tensor<1x128x512xf32>
  %11 = "tosa.reshape"(%arg0) <{new_shape = array<i64: 1, 256, 128>}> : (tensor<2x128x128xf32>) -> tensor<1x256x128xf32>
  %12 = "tosa.matmul"(%11, %10) : (tensor<1x256x128xf32>, tensor<1x128x512xf32>) -> tensor<1x256x512xf32>
  %13 = "tosa.reshape"(%12) <{new_shape = array<i64: 2, 128, 512>}> : (tensor<1x256x512xf32>) -> tensor<2x128x512xf32>
  %14 = "tosa.add"(%13, %arg2) : (tensor<2x128x512xf32>, tensor<1x1x512xf32>) -> tensor<2x128x512xf32>
  %15 = "tosa.mul"(%14, %7) <{shift = 0 : i8}> : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xf32>
  %16 = "tosa.abs"(%15) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %19 = "tosa.mul"(%16, %16) <{shift = 0 : i8}> : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %28 = "tosa.reciprocal"(%19) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %31 = "tosa.sub"(%6, %28) : (tensor<1x1x1xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %32 = "tosa.greater_equal"(%15, %5) : (tensor<2x128x512xf32>, tensor<1x1x1xf32>) -> tensor<2x128x512xi1>
  %33 = "tosa.negate"(%31) : (tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %34 = "tosa.select"(%32, %31, %33) : (tensor<2x128x512xi1>, tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  %37 = "tosa.mul"(%14, %34) <{shift = 0 : i8}> : (tensor<2x128x512xf32>, tensor<2x128x512xf32>) -> tensor<2x128x512xf32>
  func.return %37 : tensor<2x128x512xf32>
}
}
