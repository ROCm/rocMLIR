// RUN: mlir-miopen-driver -p | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -t bf16 | FileCheck %s --check-prefix=BF16

// F32-LABEL: module
// F32-NEXT: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xf32>, {{.*}}: memref<128x1x8x32x32xf32>, {{.*}}: memref<128x1x128x30x30xf32>) attributes {kernel = 0 : i32}
// F32-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>

// F16-LABEL: module
// F16-NEXT: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xf16>, {{.*}}: memref<128x1x8x32x32xf16>, {{.*}}: memref<128x1x128x30x30xf16>) attributes {kernel = 0 : i32}
// F16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>

// BF16-LABEL: module
// BF16-NEXT: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xi16>, {{.*}}: memref<128x1x8x32x32xi16>, {{.*}}: memref<128x1x128x30x30xi16>) attributes {kernel = 0 : i32}
// BF16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xi16>, memref<128x1x8x32x32xi16>, memref<128x1x128x30x30xi16>
