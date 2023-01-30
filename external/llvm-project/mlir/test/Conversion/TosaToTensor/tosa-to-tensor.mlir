// RUN: mlir-opt --split-input-file --tosa-to-tensor %s -o -| FileCheck %s

// CHECK-LABEL: @test_reshape_downrank
func.func @test_reshape_downrank(%arg0: tensor<2x3xf32>) -> tensor<6xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6]} : (tensor<2x3xf32>) -> tensor<6xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<6xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_dyn
func.func @test_reshape_downrank_dyn(%arg0: tensor<2x?xf32>) -> tensor<?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.collapse_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1]} : (tensor<2x?xf32>) -> tensor<?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank
func.func @test_reshape_uprank(%arg0: tensor<6xf32>) -> tensor<2x3xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_uprank_dyn
func.func @test_reshape_uprank_dyn(%arg0: tensor<?xf32>) -> tensor<2x?xf32> {
  // CHECK: [[RESHAPE:%.+]] = tensor.expand_shape %arg0 {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<?xf32>) -> tensor<2x?xf32>
  // CHECK: return [[RESHAPE]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank
func.func @test_reshape_samerank(%arg0: tensor<3x2xf32>) -> tensor<2x3xf32> {
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xf32>)
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, 3]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @test_reshape_samerank_dyn
func.func @test_reshape_samerank_dyn(%arg0: tensor<?x2xf32>) -> tensor<2x?xf32> {
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<?x2xf32>)
  // CHECK-NEXT: %[[RESHAPE1:.*]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1]]
  // CHECK-NEXT: %[[RESHAPE2:.*]] = tensor.expand_shape %[[RESHAPE1]] {{\[}}[0, 1]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [2, -1]} : (tensor<?x2xf32>) -> tensor<2x?xf32>
  // CHECK-NEXT: return %[[RESHAPE2]]
  return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D
func.func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  // CHECK: tensor.collapse_shape %arg0 {{\[}}[0, 1, 2], [3], [4, 5]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [6, 5, 77]} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// -----

// CHECK-LABEL: @test_reshape_downrank_6D_dyn
func.func @test_reshape_downrank_6D_dyn(%arg0: tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32> {
  // CHECK: tensor.collapse_shape %arg0 {{\[}}[0, 1, 2, 3, 4, 5]]
  // CHECK: tensor.expand_shape %0 {{\[}}[0, 1, 2]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1, 5, 77]} : (tensor<1x2x?x5x7x11xf32>) -> tensor<?x5x77xf32>
  return %0 : tensor<?x5x77xf32>
}

// -----

// CHECK-LABLE: func @slice
func.func @slice(%arg0: tensor<6xf32>) ->() {
  // CHECK: [[SLICE:%.+]] = tensor.extract_slice %arg0[2] [1] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [1]} : (tensor<6xf32>)  -> (tensor<1xf32>)
  return
}

// -----

// CHECK-LABLE: func @slice_dyn
func.func @slice_dyn(%arg0: tensor<?xf32>) -> (tensor<?xf32>) {
  // CHECK: %[[C0:.+]] = arith.constant 0 : index
  // CHECK: %[[DIM:.+]] = tensor.dim %arg0, %[[C0]]
  // CHECK: %[[C2:.+]] = arith.constant 2 : index
  // CHECK: %[[SUB:.+]] = arith.subi %[[DIM]], %[[C2]]
  // CHECK: %2 = tensor.extract_slice %arg0[2] [%[[SUB]]] [1]
  %0 = "tosa.slice"(%arg0) {start = [2], size = [-1]} : (tensor<?xf32>)  -> (tensor<?xf32>)
  return %0 : tensor<?xf32>
}
