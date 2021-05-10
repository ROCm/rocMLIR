// RUN: mlir-miopen-driver -p=false --padding_h=0 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 | FileCheck %s --check-prefix=Padding_One
// RUN: mlir-miopen-driver -p=false --padding_h=3 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 | FileCheck %s --check-prefix=Padding_Two

// Padding_One-LABEL: module
// Padding_One-NEXT: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x256x32x1x1xf32>, {{.*}}: memref<32x1x32x14x14xf32>, {{.*}}: memref<32x1x256x14x17xf32>) attributes {kernel = 0 : i32}
// Padding_One-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 1 : i32, 2 : i32], strides = [1 : i32, 1 : i32]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x14x17xf32>

// Padding_Two-LABEL: module
// Padding_Two-NEXT: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x256x32x1x1xf32>, {{.*}}: memref<32x1x32x14x14xf32>, {{.*}}: memref<32x1x256x20x17xf32>) attributes {kernel = 0 : i32}
// Padding_Two-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [3 : i32, 3 : i32, 1 : i32, 2 : i32], strides = [1 : i32, 1 : i32]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x20x17xf32>


