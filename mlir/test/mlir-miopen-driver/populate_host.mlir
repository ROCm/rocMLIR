// RUN: mlir-miopen-driver -p -ph | FileCheck %s

// CHECK-LABEL: module
// CHECK-NEXT: func @miopen_conv2d_kcyx_nchw_nkhw({{.*}}: memref<128x8x3x3xf32>, {{.*}}: memref<128x8x32x32xf32>, {{.*}}: memref<128x128x30x30xf32>)
// CHECK-NEXT: miopen.conv2d({{.*}}, {{.*}}, {{.*}})  {arch = "{{gfx[0-9]+}}", dilations = [1 : i32, 1 : i32], filter_layout = ["k", "c", "y", "x"], input_layout = ["ni", "ci", "hi", "wi"], num_cu = {{[0-9]+}} : i32, output_layout = ["no", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>

// CHECK-LABEL: func @main()
// CHECK-NEXT: alloc() : memref<128x8x3x3xf32>
// CHECK-NEXT: alloc() : memref<128x8x32x32xf32>
// CHECK-NEXT: alloc() : memref<128x128x30x30xf32>
// CHECK-NEXT: memref_cast {{.*}} : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: memref_cast {{.*}} : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: memref_cast {{.*}} : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>
// CHECK-NEXT: constant 1.000000e+00 : f32
// CHECK-NEXT: constant 0.000000e+00 : f32
// CHECK-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECk-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECk-NEXT: call @mcpuMemset4DFloat({{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, f32) -> ()
// CHECk-NEXT: call @mgpuMemAlloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECk-NEXT: call @mgpuMemAlloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECk-NEXT: call @mgpuMemAlloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
// CHECk-NEXT: constant 1 : i32
// CHECk-NEXT: constant 2 : i32
// CHECk-NEXT: call @mgpuMemCopy4DFloat({{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECk-NEXT: call @mgpuMemCopy4DFloat({{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECk-NEXT: call @mgpuMemCopy4DFloat({{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECk-NEXT: memref_cast {{.*}} : memref<?x?x?x?xf32> to memref<128x8x3x3xf32>
// CHECk-NEXT: memref_cast {{.*}} : memref<?x?x?x?xf32> to memref<128x8x32x32xf32>
// CHECk-NEXT: memref_cast {{.*}} : memref<?x?x?x?xf32> to memref<128x128x30x30xf32>
// CHECk-NEXT: call @conv2d({{.*}}, {{.*}}, {{.*}}) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
// CHECk-NEXT: call @mgpuMemCopy4DFloat({{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
// CHECk-NEXT: memref_cast {{.*}} : memref<?x?x?x?xf32> to memref<*xf32>
// CHECk-NEXT: call @print_memref_f32({{.*}}) : (memref<*xf32>) -> ()
// CHECk-NEXT: call @mgpuMemDealloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> ()
// CHECk-NEXT: call @mgpuMemDealloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> ()
// CHECk-NEXT: call @mgpuMemDealloc4DFloat({{.*}}) : (memref<?x?x?x?xf32>) -> ()
// CHECk-NEXT: dealloc %0 : memref<128x8x3x3xf32>
// CHECk-NEXT: dealloc %1 : memref<128x8x32x32xf32>
// CHECk-NEXT: dealloc %2 : memref<128x128x30x30xf32>
// CHECk-NEXT: return

// CHECK-LABEL: func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>)
// CHECK-NEXT: return
