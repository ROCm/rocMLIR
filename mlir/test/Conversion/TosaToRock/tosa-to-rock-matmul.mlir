// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx906"} {
// CHECK: @test_basic
// CHECK-SAME: (%[[a:.*]]: tensor<2x128x64xf32>, %[[b:.*]]: tensor<2x64x256xf32>)
func.func @test_basic(%a: tensor<2x128x64xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x128x256xf32> attributes {kernel} {
  // CHECK: %[[out:.*]] = bufferization.alloc_tensor{{.*}} tensor<2x128x256xf32>
  // CHECK: %[[res:.*]] = rock.gemm %[[out]] = %[[a]] * %[[b]]
  // CHECK: return %[[res]] : tensor<2x128x256xf32>
  %c = "tosa.matmul"(%a, %b) {} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>

  return %c : tensor<2x128x256xf32>
}

// CHECK: @test_quant
// CHECK-SAME: (%[[a:.*]]: tensor<2x128x64xi8>, %[[b:.*]]: tensor<2x64x256xi8>)
func.func @test_quant(%a: tensor<2x128x64xi8>, %b: tensor<2x64x256xi8>) -> tensor<2x128x256xi32> attributes {kernel} {
  // CHECK: %[[out:.*]] = bufferization.alloc_tensor{{.*}} tensor<2x128x256xi32>
  // CHECK: %[[res:.*]] = rock.gemm %[[out]] = %[[a]] * %[[b]]
  // CHECK: return %[[res]] : tensor<2x128x256xi32>
  %c = "tosa.matmul"(%a, %b) {quantizeation_info =#tosa.matmul_quant<a_zp = 0, b_zp = 0>} : (tensor<2x128x64xi8>, tensor<2x64x256xi8>) -> tensor<2x128x256xi32>
  return %c : tensor<2x128x256xi32>
}

// CHECK: @test_transpose
// CHECK-SAME: (%[[a:.*]]: tensor<2x64x128xf32>, %[[b:.*]]: tensor<2x64x256xf32>)
func.func @test_transpose(%a: tensor<2x64x128xf32>, %b: tensor<2x64x256xf32>) -> tensor<2x256x128xf32> attributes {kernel} {
  // CHECK: %[[out:.*]] = bufferization.alloc_tensor{{.*}} tensor<2x256x128xf32>
  // CHECK: %[[res:.*]] = rock.gemm tr %[[out]] = tr %[[a]] * %[[b]]
  // CHECK: return %[[res]] : tensor<2x256x128xf32>
  %tr = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> (tensor<3xi32>)
  %no_tr = "tosa.const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> (tensor<3xi32>)

  %a_tr = "tosa.transpose"(%a, %tr) : (tensor<2x64x128xf32>, tensor<3xi32>) -> (tensor<2x128x64xf32>)
  %b_tr = "tosa.transpose"(%b, %no_tr) : (tensor<2x64x256xf32>, tensor<3xi32>) -> (tensor<2x64x256xf32>)
  %c_tr = "tosa.matmul"(%a_tr, %b_tr) {} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>
  %c = "tosa.transpose"(%c_tr, %tr) {} : (tensor<2x128x256xf32>, tensor<3xi32>) -> (tensor<2x256x128xf32>)
  return %c : tensor<2x256x128xf32>
}

// CHECK: @test_transpose_b
// CHECK-SAME: (%[[a:.*]]: tensor<2x128x64xf32>, %[[b:.*]]: tensor<2x256x64xf32>)
func.func @test_transpose_b(%a: tensor<2x128x64xf32>, %b: tensor<2x256x64xf32>) -> tensor<2x128x256xf32> attributes {kernel} {
  // CHECK: %[[out:.*]] = bufferization.alloc_tensor{{.*}} tensor<2x128x256xf32>
  // CHECK: %[[res:.*]] = rock.gemm %[[out]] = %[[a]] * tr %[[b]]
  // CHECK: return %[[res]] : tensor<2x128x256xf32>
  %tr = "tosa.const"() {value = dense<[0, 2, 1]> : tensor<3xi32>} : () -> (tensor<3xi32>)

  %b_tr = "tosa.transpose"(%b, %tr) : (tensor<2x256x64xf32>, tensor<3xi32>) -> (tensor<2x64x256xf32>)
  %c = "tosa.matmul"(%a, %b_tr) {} : (tensor<2x128x64xf32>, tensor<2x64x256xf32>) -> tensor<2x128x256xf32>

  return %c : tensor<2x128x256xf32>
}
}
