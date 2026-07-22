// RUN: rocmlir-opt --rocmlir-custom-tosa-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: @integers_i4_to_i8
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
func.func @integers_i4_to_i8(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// CHECK-LABEL: @integers_i8_to_i4
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi8>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi4>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi8>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi4>)
// CHECK-NEXT: %[[in:.+]]: i8
// CHECK-NEXT: %[[res:.+]] = arith.trunci %[[in]] : i8 to i4
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi4>
// CHECK-NEXT: return %[[ret]]
func.func @integers_i8_to_i4(%arg0: tensor<8x8x2xi8>) -> tensor<8x8x2xi4> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi8>) -> tensor<8x8x2xi4>
  func.return %out : tensor<8x8x2xi4>
}

// -----

// CHECK-LABEL: @floats_i4_to_f16
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xf16>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xf16>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.uitofp %[[in]] : i4 to f16
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xf16>
// CHECK-NEXT: return %[[ret]]
func.func @floats_i4_to_f16(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xf16> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xf16>
  func.return %out : tensor<8x8x2xf16>
}

// CHECK-LABEL: @floats_i4_to_f32
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xi4>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xf32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xi4>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xf32>)
// CHECK-NEXT: %[[in:.+]]: i4
// CHECK-NEXT: %[[res:.+]] = arith.uitofp %[[in]] : i4 to f32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xf32>
// CHECK-NEXT: return %[[ret]]
func.func @floats_i4_to_f32(%arg0: tensor<8x8x2xi4>) -> tensor<8x8x2xf32> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xi4>) -> tensor<8x8x2xf32>
  func.return %out : tensor<8x8x2xf32>
}

// CHECK-LABEL: @floats_f16_to_i8
// CHECK-SAME: (%[[arg0:.+]]: tensor<8x8x2xf16>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<8x8x2xi8>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]] : tensor<8x8x2xf16>)
// CHECK-SAME: outs(%[[empty]] : tensor<8x8x2xi8>)
// CHECK-NEXT: %[[in:.+]]: f16
// CHECK-NEXT: %[[res:.+]] = arith.fptoui %[[in]] : f16 to i8
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<8x8x2xi8>
// CHECK-NEXT: return %[[ret]]
func.func @floats_f16_to_i8(%arg0: tensor<8x8x2xf16>) -> tensor<8x8x2xi8> {
  %out = tosa.custom %arg0 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_cast"} : (tensor<8x8x2xf16>) -> tensor<8x8x2xi8>
  func.return %out : tensor<8x8x2xi8>
}

// -----

// CHECK-LABEL: @unsigned_div
// CHECK-SAME: (%[[arg0:.+]]: tensor<1x36x384x64xi32>, %[[arg1:.+]]: tensor<1x36x384x64xi32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<1x36x384x64xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>)
// CHECK-SAME: outs(%[[empty]] : tensor<1x36x384x64xi32>)
// CHECK-NEXT: %[[in:.+]]: i32, %[[in1:.+]]: i32, %[[out:.+]]: i32
// CHECK-NEXT: %[[res:.+]] = arith.divui %[[in]], %[[in1]] : i32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<1x36x384x64xi32>
// CHECK-NEXT: return %[[ret]]
func.func @unsigned_div(%arg0: tensor<1x36x384x64xi32>, %arg1: tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32> {
  %out = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_div"} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi32>
  func.return %out : tensor<1x36x384x64xi32>
}

// -----

// CHECK-LABEL: @unsigned_max
// CHECK-SAME: (%[[arg0:.+]]: tensor<4xi32>, %[[arg1:.+]]: tensor<4xi32>)
// CHECK: %[[empty:.+]] = tensor.empty() : tensor<4xi32>
// CHECK: %[[ret:.+]] = linalg.generic
// CHECK-SAME: ins(%[[arg0]], %[[arg1]] : tensor<4xi32>, tensor<4xi32>)
// CHECK-SAME: outs(%[[empty]] : tensor<4xi32>)
// CHECK-NEXT: %[[in:.+]]: i32, %[[in1:.+]]: i32, %[[out:.+]]: i32
// CHECK-NEXT: %[[res:.+]] = arith.maxui %[[in]], %[[in1]] : i32
// CHECK-NEXT: linalg.yield %[[res]]
// CHECK-NEXT: -> tensor<4xi32>
// CHECK-NEXT: return %[[ret]]
func.func @unsigned_max(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %out = tosa.custom %arg0, %arg1 {domain_name = "rocmlir", implementation_attrs = "", operator_name = "unsigned_max"} : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %out : tensor<4xi32>
}
