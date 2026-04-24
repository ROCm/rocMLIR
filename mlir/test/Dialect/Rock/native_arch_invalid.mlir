// Confirm that malformed `rock.arch = "native:..."` strings produce a fatal
// error, rather than silently falling back to device 0. The latter behaviour
// (which we explicitly fixed) would have masked user typos and silently
// targeted the wrong GPU on multi-GPU systems.
//
// `parseArchString` reports the failure via `llvm::report_fatal_error`,
// which `abort()`s the process; we use `not --crash` to match that signal-
// terminated exit code (plain `not` only matches non-signal non-zero exits).
// The fatal error fires before any HIP query, so this test does NOT need a
// real GPU and can run anywhere.

// REQUIRES: linux

// RUN: not --crash rocmlir-opt -mlir-print-local-scope -rock-affix-params %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: LLVM ERROR
// CHECK-SAME: native:foo
// CHECK-SAME: must be a non-negative integer device id

func.func @rock_conv_native_invalid(%filter : memref<1x128x8x3x3xf32>,
                                    %input : memref<128x1x8x32x32xf32>,
                                    %output : memref<128x1x128x30x30xf32>)
    attributes {rock.arch = "native:foo"} {
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
