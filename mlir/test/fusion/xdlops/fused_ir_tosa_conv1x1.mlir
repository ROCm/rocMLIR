// RUN: rocmlir-driver -host-pipeline highlevel %s | rocmlir-driver --rock-fold-transpose --rock-affix-params --rock-conv-to-gemm --rock-gemm-to-gridwise --rock-gridwise-gemm-to-blockwise --rock-linalg-align | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x64x56x56xf32>, %arg1: tensor<64x64x1x1xf32>, %arg2: tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32> attributes {kernel, arch = "gfx908"} {
    %cst = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi64>
    %0 = "tosa.transpose"(%arg0, %cst) {changing_layout_root = false} : (tensor<1x64x56x56xf32>, tensor<4xi64>) -> tensor<1x56x56x64xf32>
    %1 = "tosa.transpose"(%arg1, %cst) {changing_layout_root = false} : (tensor<64x64x1x1xf32>, tensor<4xi64>) -> tensor<64x1x1x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %2 = "tosa.conv2d"(%0, %1, %cst_0) {xdlopsV2 = true, dilation = [1, 1], expected_filter_layout = "kcyx", expected_input_layout = "nchw", expected_output_layout = "nkhw", pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x56x56x64xf32>, tensor<64x1x1x64xf32>, tensor<1xf32>) -> tensor<1x56x56x64xf32>
    %cst_1 = arith.constant dense<[0, 3, 1, 2]> : tensor<4xi64>
    %3 = "tosa.transpose"(%2, %cst_1) {changing_layout_root = true} : (tensor<1x56x56x64xf32>, tensor<4xi64>) -> tensor<1x64x56x56xf32>
    %4 = "tosa.add"(%3, %arg2) : (tensor<1x64x56x56xf32>, tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    %5 = "tosa.clamp"(%4) {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
    return {kernel, arch = "gfx908"} %5 : tensor<1x64x56x56xf32>
  }
}
// 1. Tracks the beginning of the store loop of gemmv2
//CHECK: rock.blockwise_gemm_v2
//CHECK: rock.transforming_for

// 2. Check if ops are fused and copy_v2 is not present here
//CHECK-NOT: rock.threadwise_copy_v2

// 3. Check correct sequence of load-linalg-store
//CHECK: rock.transforming_for
//CHECK: rock.yield
//CHECK: linalg.generic
//CHECK: rock.threadwise_copy_v2

// 4. Check if there is leftover ops.
//CHECK: rock.yield
//CHECK-NOT: linalg.generic
