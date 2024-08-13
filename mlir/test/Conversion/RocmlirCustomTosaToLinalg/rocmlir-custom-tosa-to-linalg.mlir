// RUN: rocmlir-opt --rocmlir-custom-tosa-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: @integers
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi8>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.extui %[[in]] : i4 to i8
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi8>
// CHECK-NEXT: return %[[ret]]
func.func @integers(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// -----

// CHECK-LABEL: @floats
// CHECK: linalg.generic
// CHECK: arith.uitofp
func.func @floats(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xf16> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xf16>
  func.return %out : tensor<8x8x2xf16>
}
