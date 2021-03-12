// RUN: mlir-miopen-driver -p | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -t bf16 | FileCheck %s --check-prefix=BF16

// F32-LABEL: module
// F32-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf32>, {{.*}}: memref<128x8x32x32xf32>, {{.*}}: memref<128x128x30x30xf32>)
// F32-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>

// F16-LABEL: module
// F16-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf16>, {{.*}}: memref<128x8x32x32xf16>, {{.*}}: memref<128x128x30x30xf16>)
// F16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf16>, memref<128x8x32x32xf16>, memref<128x128x30x30xf16>

// BF16-LABEL: module
// BF16-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xbf16>, {{.*}}: memref<128x8x32x32xbf16>, {{.*}}: memref<128x128x30x30xbf16>)
// BF16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xbf16>, memref<128x8x32x32xbf16>, memref<128x128x30x30xbf16>
