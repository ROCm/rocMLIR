// `--verifier=clone` builds the host harness assuming kernels have been
// lowered through the rock kernel pipeline. Feeding it a kernel that still
// contains higher-level dialects (tosa / migraphx) must produce an
// actionable error rather than silently mis-routing outputs.

// RUN: sed -e 's/##ARCH##/%arch/g' %s | not rocmlir-gen -ph -rand 1 -rand_type float -fut mm_fut --verifier clone - 2>&1 | FileCheck %s

module attributes {rock.arch = "##ARCH##"} {
  func.func private @mm_fut_cpu_host(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf16>) {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x768xf32>
    %1 = tosa.add %0, %arg2 : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %2 = tosa.cast %1 : (tensor<1x256x768xf32>) -> tensor<1x256x768xf16>
    return %0, %2 : tensor<1x256x768xf32>, tensor<1x256x768xf16>
  }
  func.func private @mm_fut(%arg0: tensor<1x256x768xf32>, %arg1: tensor<1x768x768xf32>, %arg2: tensor<1x256x768xf32>) -> (tensor<1x256x768xf32>, tensor<1x256x768xf16>) attributes {rock.kernel} {
    %a_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %b_zp = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %0 = tosa.matmul %arg0, %arg1, %a_zp, %b_zp {acc_type = f32} : (tensor<1x256x768xf32>, tensor<1x768x768xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x256x768xf32>
    %1 = tosa.add %0, %arg2 : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %2 = tosa.cast %1 : (tensor<1x256x768xf32>) -> tensor<1x256x768xf16>
    return %0, %2 : tensor<1x256x768xf32>, tensor<1x256x768xf16>
  }
}

// CHECK: error:
// CHECK-SAME: --verifier=clone cannot build a host harness around a kernel that is not at the rock level
// CHECK-SAME: run the kernel pipeline first
