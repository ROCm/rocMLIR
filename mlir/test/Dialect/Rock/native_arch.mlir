// Verify that `rock.arch = "native"` flows through the rock pipeline by
// resolving to the hardware-reported gfxXXX via a delay-loaded HIP runtime
// and producing the same lowered IR as a kernel pinned to the concrete arch.
//
// `--arch native:N` requires:
//   1. an AMD GPU visible to the HIP runtime (skipped otherwise via REQUIRES);
//   2. `libamdhip64` present on the dynamic-loader search path -- any
//      standard ROCm install satisfies this.

// REQUIRES: amd-gpu-present

// RUN: rocmlir-opt -mlir-print-local-scope -rock-affix-params %s | FileCheck %s

// CHECK-LABEL: @rock_conv_native
// CHECK-SAME: rock.arch = "native"
// CHECK: rock.conv
// CHECK-SAME: params = #rock.general_gemm_params
func.func @rock_conv_native(%filter : memref<1x128x8x3x3xf32>,
                            %input : memref<128x1x8x32x32xf32>,
                            %output : memref<128x1x128x30x30xf32>)
    attributes {rock.arch = "native"} {
  rock.conv(%filter, %input, %output) features = none {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["ni", "gi", "ci", "0i", "1i"],
    output_layout = ["no", "go", "ko", "0o", "1o"],
    dilations = [1 : index, 1 : index],
    strides = [1 : index, 1 : index],
    padding = [0 : index, 0 : index, 0 : index, 0 : index]
  } : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>
  return
}
