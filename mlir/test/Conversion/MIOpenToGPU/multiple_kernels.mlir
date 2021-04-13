// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step1,step2,step3,step4" %s | mlir-opt
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step1,step2,step3,step4" %s | FileCheck %s
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step1,step2,step3,step4,step5" %s | FileCheck %s
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step1" %s | FileCheck %s --check-prefix=SUBSET1
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step1,step3" %s | FileCheck %s --check-prefix=SUBSET2
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu="kernel-name=step5" %s | FileCheck %s --check-prefix=NONEXIST
// RUN: mlir-opt -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3 -miopen-lowering-step4 -miopen-lowering-step5 -convert-miopen-to-gpu %s | FileCheck %s

// The last kernel be converted would appear as the first.

// CHECK-NOT: func @step1
// CHECK-NOT: func @step2
// CHECK-NOT: func @step3
// CHECK-NOT: func @step4
// CHECK-LABEL: gpu.func @step4
// CHECK-LABEL: gpu.func @step3
// CHECK-LABEL: gpu.func @step2
// CHECK-LABEL: gpu.func @step1

// SUBSET1-NOT: func @step1
// SUBSET1-LABEL: func @step2
// SUBSET1-LABEL: func @step3
// SUBSET1-LABEL: func @step4
// SUBSET1-NOT: gpu.func @step4
// SUBSET1-NOT: gpu.func @step3
// SUBSET1-NOT: gpu.func @step2
// SUBSET1-LABEL: gpu.func @step1

// SUBSET2-NOT: func @step1
// SUBSET2-LABEL: func @step2
// SUBSET2-NOT: func @step3
// SUBSET2-LABEL: func @step4
// SUBSET2-NOT: gpu.func @step4
// SUBSET2-LABEL: gpu.func @step3
// SUBSET2-NOT: gpu.func @step2
// SUBSET2-LABEL: gpu.func @step1

// NONEXIST-LABEL: func @step1
// NONEXIST-LABEL: func @step2
// NONEXIST-LABEL: func @step3
// NONEXIST-LABEL: func @step4
// NONEXIST-NOT: gpu.func @step4
// NONEXIST-NOT: gpu.func @step3
// NONEXIST-NOT: gpu.func @step2
// NONEXIST-NOT: gpu.func @step1

module  {
  func @step1(%arg0: memref<1216x16x3x3xf32>, %arg1: memref<1216x16x32x32xf32>, %arg2: memref<1216x1216x30x30xf32>) {
    miopen.conv2d(%arg0, %arg1, %arg2) {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = 64 : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1216x16x3x3xf32>, memref<1216x16x32x32xf32>, memref<1216x1216x30x30xf32>
    return
  }

  func @step2(%arg0: memref<64x16x3x3xf32>, %arg1: memref<64x16x32x32xf32>, %arg2: memref<64x64x30x30xf32>) {
    miopen.conv2d(%arg0, %arg1, %arg2) {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = 64 : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<64x16x3x3xf32>, memref<64x16x32x32xf32>, memref<64x64x30x30xf32>
    return
  }

  func @step3(%arg0: memref<32x16x3x3xf32>, %arg1: memref<32x16x32x32xf32>, %arg2: memref<32x32x30x30xf32>) {
    miopen.conv2d(%arg0, %arg1, %arg2) {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = 64 : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<32x16x3x3xf32>, memref<32x16x32x32xf32>, memref<32x32x30x30xf32>
    return
  }

  func @step4(%arg0: memref<32x8x3x3xf32>, %arg1: memref<32x8x32x32xf32>, %arg2: memref<32x32x30x30xf32>) {
    miopen.conv2d(%arg0, %arg1, %arg2) {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = 64 : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<32x8x3x3xf32>, memref<32x8x32x32xf32>, memref<32x32x30x30xf32>
    return
  }
}
