// RUN: rocmlir-driver -host-pipeline highlevel %s | rocmlir-driver --rock-affix-params --rock-conv-to-gemm --rock-gemm-to-gridwise --rock-regularize --rock-gridwise-gemm-to-blockwise --rock-linalg-align --rock-pipeline | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x64x112x112xf32>, %arg1: tensor<1x3x224x224xf32>, %arg2: tensor<64x3x7x7xf32>) -> tensor<1x64x112x112xf32> attributes {kernel, arch = "amdgcn-amd-amdhsa:gfx908"} {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
    %0 = "tosa.transpose"(%arg1, %cst) {changing_layout_root = false} : (tensor<1x3x224x224xf32>, tensor<4xi32>) -> tensor<1x224x224x3xf32>
    %1 = "tosa.transpose"(%arg2, %cst) {changing_layout_root = false} : (tensor<64x3x7x7xf32>, tensor<4xi32>) -> tensor<64x7x7x3xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %2 = "tosa.conv2d"(%0, %1, %cst_0) {dilation = array<i64: 1, 1>, expected_filter_layout = "kcyx", expected_input_layout = "nchw", expected_output_layout = "nkhw", pad = array<i64: 3, 3, 3, 3>, stride = array<i64: 2, 2>, xdlopsV2 = true} : (tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32>, tensor<1xf32>) -> tensor<1x112x112x64xf32>
    %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi32>
    %3 = "tosa.transpose"(%2, %cst_1) {changing_layout_root = true} : (tensor<1x112x112x64xf32>, tensor<4xi32>) -> tensor<1x64x112x112xf32>
    %4 = "tosa.add"(%3, %arg0) : (tensor<1x64x112x112xf32>, tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    %5 = "tosa.clamp"(%4) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return {kernel, arch = "amdgcn-amd-amdhsa:gfx908"} %5 : tensor<1x64x112x112xf32>
  }
}
// 1. Tracks the beginning of the store loop of gemmv2
// Prologue
//CHECK-COUNT-2: rock.threadwise_read_into
//CHECK-COUNT-2: rock.threadwise_write_all {{.*}} #gpu.address_space<workgroup>
// SW pipelined loop
//CHECK: scf.for
//CHECK-COUNT-2: rock.threadwise_read_into
//CHECK: rock.blockwise_gemm_accel
//CHECK-COUNT-2: rock.threadwise_write_all {{.*}} #gpu.address_space<workgroup>
// Epilogue
//CHECK: rock.blockwise_gemm_accel

// 2. Check if ops are fused and copy_v2 is not present here
//CHECK-NOT: rock.threadwise_write_all

// 3. Check correct sequence of load-linalg-store
//CHECK: rock.threadwise_read_into
//CHECK: linalg.generic
//CHECK: rock.threadwise_write_all

// 4. Check if there is leftover ops.
//CHECK-NOT: memref.copy
