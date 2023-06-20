// RUN: rocmlir-gen --arch %arch -p -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on | FileCheck %s --check-prefix=F32
// RUN: rocmlir-gen --arch %arch -p -t f16 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on | FileCheck %s --check-prefix=F16
// RUN: rocmlir-gen --arch %arch -p -t bf16 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on | FileCheck %s --check-prefix=BF16
// RUN: rocmlir-gen --arch %arch -p -t i8 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on | FileCheck %s --check-prefix=INT8

// F32-LABEL: module
// F32-NEXT: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xf32>, {{.*}}: memref<128x1x8x32x32xf32>, {{.*}}: memref<128x1x128x30x30xf32>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"}
// F32-NEXT: rock.conv2d({{.*}}, {{.*}}, {{.*}})  features = dot {arch = "{{.*}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>

// F16-LABEL: module
// F16-NEXT: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xf16>, {{.*}}: memref<128x1x8x32x32xf16>, {{.*}}: memref<128x1x128x30x30xf16>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"}
// F16-NEXT: rock.conv2d({{.*}}, {{.*}}, {{.*}})  features = dot {arch = "{{.*}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xf16>, memref<128x1x8x32x32xf16>, memref<128x1x128x30x30xf16>

// BF16-LABEL: module
// BF16-NEXT: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xbf16>, {{.*}}: memref<128x1x8x32x32xbf16>, {{.*}}: memref<128x1x128x30x30xbf16>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"}
// BF16-NEXT: rock.conv2d({{.*}}, {{.*}}, {{.*}}) features = dot {arch = "{{.*}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xbf16>, memref<128x1x8x32x32xbf16>, memref<128x1x128x30x30xbf16>

// INT8-LABEL: module
// INT8-NEXT: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}: memref<1x128x8x3x3xi8>, {{.*}}: memref<128x1x8x32x32xi8>, {{.*}}: memref<128x1x128x30x30xi32>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// INT8-NEXT: rock.conv2d({{.*}}, {{.*}}, {{.*}}) features = dot {arch = "{{.*}}", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x128x8x3x3xi8>, memref<128x1x8x32x32xi8>, memref<128x1x128x30x30xi32>
