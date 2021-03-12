// RUN: mlir-miopen-driver -p -ph | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -ph -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -ph -t bf16 | FileCheck %s --check-prefix=BF16

// F32-LABEL: module
// F32-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf32>, {{.*}}: memref<128x8x32x32xf32>, {{.*}}: memref<128x128x30x30xf32>)
// F32-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>

// F16-LABEL: module
// F16-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf16>, {{.*}}: memref<128x8x32x32xf16>, {{.*}}: memref<128x128x30x30xf16>)
// F16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf16>, memref<128x8x32x32xf16>, memref<128x128x30x30xf16>

// BF16-LABEL: module
// BF16-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xbf16>, {{.*}}: memref<128x8x32x32xbf16>, {{.*}}: memref<128x128x30x30xbf16>)
// BF16-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xbf16>, memref<128x8x32x32xbf16>, memref<128x128x30x30xbf16>

// F32-LABEL: func @main()
// F32-NEXT: alloc() : memref<128x8x3x3xf32>
// F32-NEXT: alloc() : memref<128x8x32x32xf32>
// F32-NEXT: alloc() : memref<128x128x30x30xf32>
// F32-NEXT: alloc() : memref<128x128x30x30xf32>
// F32-NEXT: memref_cast {{.*}} : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
// F32-NEXT: memref_cast {{.*}} : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
// F32-NEXT: memref_cast {{.*}} : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>
// F32-NEXT: constant 1.000000e+00 : f32
// F32-NEXT: constant 0.000000e+00 : f32
// F32-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// F32-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// F32-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// F32-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
// F32-NEXT: dealloc {{.*}} : memref<128x8x3x3xf32>
// F32-NEXT: dealloc {{.*}} : memref<128x8x32x32xf32>
// F32-NEXT: dealloc {{.*}} : memref<128x128x30x30xf32>
// F32-NEXT: dealloc {{.*}} : memref<128x128x30x30xf32>
// F32-NEXT: return

// F16-LABEL: func @main()
// F16-NEXT: alloc() : memref<128x8x3x3xf16>
// F16-NEXT: alloc() : memref<128x8x32x32xf16>
// F16-NEXT: alloc() : memref<128x128x30x30xf16>
// F16-NEXT: alloc() : memref<128x128x30x30xf32>
// F16-NEXT: memref_cast {{.*}} : memref<128x8x3x3xf16> to memref<?x?x?x?xf16>
// F16-NEXT: memref_cast {{.*}} : memref<128x8x32x32xf16> to memref<?x?x?x?xf16>
// F16-NEXT: memref_cast {{.*}} : memref<128x128x30x30xf16> to memref<?x?x?x?xf16>
// F16-NEXT: constant 1.000000e+00 : f16
// F16-NEXT: constant 0.000000e+00 : f16
// F16-NEXT: call @mcpuMemset4DHalf({{.*}}, {{.*}}) : (memref<?x?x?x?xf16>, f16) -> ()
// F16-NEXT: call @mcpuMemset4DHalf({{.*}}, {{.*}}) : (memref<?x?x?x?xf16>, f16) -> ()
// F16-NEXT: call @mcpuMemset4DHalf({{.*}}, {{.*}}) : (memref<?x?x?x?xf16>, f16) -> ()
// F16-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<128x8x3x3xf16>, memref<128x8x32x32xf16>, memref<128x128x30x30xf16>) -> ()
// F16-NEXT: dealloc {{.*}} : memref<128x8x3x3xf16>
// F16-NEXT: dealloc {{.*}} : memref<128x8x32x32xf16>
// F16-NEXT: dealloc {{.*}} : memref<128x128x30x30xf16>
// F16-NEXT: dealloc {{.*}} : memref<128x128x30x30xf32>
// F16-NEXT: return

// BF16-LABEL: func @main()
// BF16-NEXT: alloc() : memref<128x8x3x3xbf16>
// BF16-NEXT: alloc() : memref<128x8x32x32xbf16>
// BF16-NEXT: alloc() : memref<128x128x30x30xbf16>
// BF16-NEXT: alloc() : memref<128x128x30x30xf32>
// BF16-NEXT: memref_cast {{.*}} : memref<128x8x3x3xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: memref_cast {{.*}} : memref<128x8x32x32xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: memref_cast {{.*}} : memref<128x128x30x30xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: constant 1.000000e+00 : bf16
// BF16-NEXT: constant 0.000000e+00 : bf16
// BF16-NEXT: call @mcpuMemset4DBF16({{.*}}, {{.*}}) : (memref<?x?x?x?xbf16>, bf16) -> ()
// BF16-NEXT: call @mcpuMemset4DBF16({{.*}}, {{.*}}) : (memref<?x?x?x?xbf16>, bf16) -> ()
// BF16-NEXT: call @mcpuMemset4DBF16({{.*}}, {{.*}}) : (memref<?x?x?x?xbf16>, bf16) -> ()
// BF16-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<128x8x3x3xbf16>, memref<128x8x32x32xbf16>, memref<128x128x30x30xbf16>) -> ()
// BF16-NEXT: dealloc {{.*}} : memref<128x8x3x3xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x8x32x32xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x128x30x30xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x128x30x30xf32>
// BF16-NEXT: return

// F32-LABEL: func @gpu_conv(%{{.*}}: memref<128x8x3x3xf32>, %{{.*}}: memref<128x8x32x32xf32>, %{{.*}}: memref<128x128x30x30xf32>)
// F32-NEXT: memref_cast %{{.*}} : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
// F32-NEXT: memref_cast %{{.*}} : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
// F32-NEXT: memref_cast %{{.*}} : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>
// F32-NEXT: call @mgpuMemAlloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// F32-NEXT: call @mgpuMemAlloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// F32-NEXT: call @mgpuMemAlloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// F32-NEXT: constant 1 : i32
// F32-NEXT: constant 2 : i32
// F32-NEXT: call @mgpuMemCopy4DFloat(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// F32-NEXT: call @mgpuMemCopy4DFloat(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// F32-NEXT: call @mgpuMemCopy4DFloat(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// F32-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf32> to memref<128x8x3x3xf32>
// F32-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf32> to memref<128x8x32x32xf32>
// F32-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf32> to memref<128x128x30x30xf32>
// F32-NEXT: call @conv2d(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
// F32-NEXT: call @mgpuMemCopy4DFloat(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// F32-NEXT: call @mgpuMemDealloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> ()
// F32-NEXT: call @mgpuMemDealloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> ()
// F32-NEXT: call @mgpuMemDealloc4DFloat(%{{.*}}) : (memref<?x?x?x?xf32>) -> ()
// F32-NEXT: return

// F16-LABEL: func @gpu_conv(%{{.*}}: memref<128x8x3x3xf16>, %{{.*}}: memref<128x8x32x32xf16>, %{{.*}}: memref<128x128x30x30xf16>)
// F16-NEXT: memref_cast %{{.*}} : memref<128x8x3x3xf16> to memref<?x?x?x?xf16>
// F16-NEXT: memref_cast %{{.*}} : memref<128x8x32x32xf16> to memref<?x?x?x?xf16>
// F16-NEXT: memref_cast %{{.*}} : memref<128x128x30x30xf16> to memref<?x?x?x?xf16>
// F16-NEXT: call @mgpuMemAlloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> memref<?x?x?x?xf16>
// F16-NEXT: call @mgpuMemAlloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> memref<?x?x?x?xf16>
// F16-NEXT: call @mgpuMemAlloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> memref<?x?x?x?xf16>
// F16-NEXT: constant 1 : i32
// F16-NEXT: constant 2 : i32
// F16-NEXT: call @mgpuMemCopy4DHalf(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32) -> ()
// F16-NEXT: call @mgpuMemCopy4DHalf(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32) -> ()
// F16-NEXT: call @mgpuMemCopy4DHalf(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32) -> ()
// F16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf16> to memref<128x8x3x3xf16>
// F16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf16> to memref<128x8x32x32xf16>
// F16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xf16> to memref<128x128x30x30xf16>
// F16-NEXT: call @conv2d(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<128x8x3x3xf16>, memref<128x8x32x32xf16>, memref<128x128x30x30xf16>) -> ()
// F16-NEXT: call @mgpuMemCopy4DHalf(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xf16>, memref<?x?x?x?xf16>, i32) -> ()
// F16-NEXT: call @mgpuMemDealloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> ()
// F16-NEXT: call @mgpuMemDealloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> ()
// F16-NEXT: call @mgpuMemDealloc4DHalf(%{{.*}}) : (memref<?x?x?x?xf16>) -> ()
// F16-NEXT: return

// BF16-LABEL: func @gpu_conv(%{{.*}}: memref<128x8x3x3xbf16>, %{{.*}}: memref<128x8x32x32xbf16>, %{{.*}}: memref<128x128x30x30xbf16>)
// BF16-NEXT: memref_cast %{{.*}} : memref<128x8x3x3xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: memref_cast %{{.*}} : memref<128x8x32x32xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: memref_cast %{{.*}} : memref<128x128x30x30xbf16> to memref<?x?x?x?xbf16>
// BF16-NEXT: call @mgpuMemAlloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> memref<?x?x?x?xbf16>
// BF16-NEXT: call @mgpuMemAlloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> memref<?x?x?x?xbf16>
// BF16-NEXT: call @mgpuMemAlloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> memref<?x?x?x?xbf16>
// BF16-NEXT: constant 1 : i32
// BF16-NEXT: constant 2 : i32
// BF16-NEXT: call @mgpuMemCopy4DBF16(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xbf16>, memref<?x?x?x?xbf16>, i32) -> ()
// BF16-NEXT: call @mgpuMemCopy4DBF16(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xbf16>, memref<?x?x?x?xbf16>, i32) -> ()
// BF16-NEXT: call @mgpuMemCopy4DBF16(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xbf16>, memref<?x?x?x?xbf16>, i32) -> ()
// BF16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xbf16> to memref<128x8x3x3xbf16>
// BF16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xbf16> to memref<128x8x32x32xbf16>
// BF16-NEXT: memref_cast %{{.*}} : memref<?x?x?x?xbf16> to memref<128x128x30x30xbf16>
// BF16-NEXT: call @conv2d(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<128x8x3x3xbf16>, memref<128x8x32x32xbf16>, memref<128x128x30x30xbf16>) -> ()
// BF16-NEXT: call @mgpuMemCopy4DBF16(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?xbf16>, memref<?x?x?x?xbf16>, i32) -> ()
// BF16-NEXT: call @mgpuMemDealloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> ()
// BF16-NEXT: call @mgpuMemDealloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> ()
// BF16-NEXT: call @mgpuMemDealloc4DBF16(%{{.*}}) : (memref<?x?x?x?xbf16>) -> ()
// BF16-NEXT: return

// F32-LABEL: func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>)
// F32-NEXT: return

// F16-LABEL: func @conv2d(%arg0: memref<128x8x3x3xf16>, %arg1: memref<128x8x32x32xf16>, %arg2: memref<128x128x30x30xf16>)
// F16-NEXT: return

// BF16-LABEL: func @conv2d(%arg0: memref<128x8x3x3xbf16>, %arg1: memref<128x8x32x32xbf16>, %arg2: memref<128x128x30x30xbf16>)
// BF16-NEXT: return
