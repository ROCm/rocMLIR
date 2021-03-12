// RUN: mlir-miopen-driver -p -ph | FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf32>, {{.*}}: memref<128x8x32x32xf32>, {{.*}}: memref<128x128x30x30xf32>)
// CHECK-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>

// CHECK-LABEL: func @main()
// CHECK-NEXT: %0 = alloc() : memref<128x8x3x3xf32>
// CHECK-NEXT: %1 = alloc() : memref<128x8x32x32xf32>
// CHECK-NEXT: %2 = alloc() : memref<128x128x30x30xf32>
// CHECK-NEXT: %3 = alloc() : memref<128x128x30x30xf32>
// CHECK-NEXT: %4 = memref_cast %0 : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: %5 = memref_cast %1 : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: %6 = memref_cast %2 : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: %cst = constant 1.000000e+00 : f32
// CHECK-NEXT: %cst_0 = constant 0.000000e+00 : f32
// CHECK-NEXT: call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECK-NEXT: call @mcpuMemset4DFloat(%5, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECK-NEXT: call @mcpuMemset4DFloat(%6, %cst_0) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECK-NEXT: %7 = call @mgpuMemAlloc4DFloat(%4) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECK-NEXT: %8 = call @mgpuMemAlloc4DFloat(%5) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECK-NEXT: %9 = call @mgpuMemAlloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECK-NEXT: %c1_i32 = constant 1 : i32
// CHECK-NEXT: %c2_i32 = constant 2 : i32
// CHECK-NEXT: call @mgpuMemCopy4DFloat(%4, %7, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECK-NEXT: call @mgpuMemCopy4DFloat(%5, %8, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECK-NEXT: call @mgpuMemCopy4DFloat(%6, %9, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECK-NEXT: %10 = memref_cast %7 : memref<?x?x?x?xf32> to memref<128x8x3x3xf32>
// CHECK-NEXT: %11 = memref_cast %8 : memref<?x?x?x?xf32> to memref<128x8x32x32xf32>
// CHECK-NEXT: %12 = memref_cast %9 : memref<?x?x?x?xf32> to memref<128x128x30x30xf32>
// CHECK-NEXT: call @conv2d(%10, %11, %12) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
// CHECK-NEXT: call @mgpuMemCopy4DFloat(%9, %6, %c2_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECK-NEXT: call @mgpuMemDealloc4DFloat(%4) : (memref<?x?x?x?xf32>) -> ()
// CHECK-NEXT: call @mgpuMemDealloc4DFloat(%5) : (memref<?x?x?x?xf32>) -> ()
// CHECK-NEXT: call @mgpuMemDealloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> ()
// CHECK-NEXT: dealloc %0 : memref<128x8x3x3xf32>
// CHECK-NEXT: dealloc %1 : memref<128x8x32x32xf32>
// CHECK-NEXT: dealloc %2 : memref<128x128x30x30xf32>
// CHECK-NEXT: dealloc %3 : memref<128x128x30x30xf32>
// CHECK-NEXT: return

// CHECK-LABEL: func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>)
// CHECK-NEXT: return
