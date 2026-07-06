// RUN: rocmlir-opt --tosa-to-rock %s -o - | FileCheck %s

// COM: Split-K coverage for the tosa.conv3d and tosa.matmul_t_block_scaled
// COM: conversion paths in mlir/lib/Conversion/TosaToRock/TosaToRock.cpp
// COM: (setSplitKAttrs<Conv3DOp> and setSplitKAttrs<MatmulTBlockScaledOp>).
// COM: A split-K perf_config makes setSplitKAttrs tag the kernel output with a
// COM: rock.prefill attribute (zero-init) and mhal.read_access; the existing
// COM: split-k tests only exercised tosa.matmul / tosa.conv2d.

module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {

// CHECK-LABEL: @conv3d_splitk
// CHECK-SAME: rock.prefill = 0.000000e+00 : f32
// CHECK: rock.conv
func.func @conv3d_splitk(%arg0: tensor<2x5x5x5x3xf32>, %arg1: tensor<4x2x2x2x3xf32>, %bias: tensor<4xf32>) -> tensor<2x2x2x2x4xf32> attributes {rock.kernel} {
  %izp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %wzp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
  %0 = tosa.conv3d %arg0, %arg1, %bias, %izp, %wzp {acc_type = f32, dilation = array<i64: 1, 1, 1>, group = 1 : i64, pad = array<i64: 0, 0, 0, 0, 0, 0>, stride = array<i64: 1, 1, 1>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<2x5x5x5x3xf32>, tensor<4x2x2x2x3xf32>, tensor<4xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<2x2x2x2x4xf32>
  return %0 : tensor<2x2x2x2x4xf32>
}

// CHECK-LABEL: @bscaled_splitk
// CHECK-SAME: rock.prefill = 0.000000e+00 : f32
// CHECK: rock.gemm %{{.*}} scaled by
func.func @bscaled_splitk(%a: tensor<1x128x256xf4E2M1FN>, %as: tensor<1x128x8xf8E8M0FNU>, %b: tensor<1x512x256xf4E2M1FN>, %bs: tensor<1x512x8xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {rock.kernel} {
  %r = tosa.matmul_t_block_scaled %a, %as, %b, %bs {block_size = #tosa.block_size<BLOCK_SIZE_32>, perf_config = "v3:16,32,4,16,16,4,4,1,2,1,1"} : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) -> tensor<1x128x512xf32>
  return %r : tensor<1x128x512xf32>
}

}
