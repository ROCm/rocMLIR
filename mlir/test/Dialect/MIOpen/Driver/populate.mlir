// RUN: mlir-miopen-driver -p | FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf32>, {{.*}}: memref<128x8x32x32xf32>, {{.*}}: memref<128x128x30x30xf32>)
// CHECK-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>
