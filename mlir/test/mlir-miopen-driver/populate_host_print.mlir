// RUN: mlir-miopen-driver -p -ph -pr | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16

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
// F32-NEXT: constant 0 : index
// F32-NEXT: constant 1 : index
// F32-NEXT: constant 128 : index
// F32-NEXT: constant 128 : index
// F32-NEXT: constant 30 : index
// F32-NEXT: constant 30 : index
// F32-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F32-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F32-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F32-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F32-NEXT:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xf32>
// F32-NEXT:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xf32>
// F32-NEXT:       }
// F32-NEXT:     }
// F32-NEXT:   }
// F32-NEXT: }
// F32-NEXT: memref_cast %{{.*}} : memref<128x128x30x30xf32> to memref<*xf32>
// F32-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*xf32>) -> ()
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
// F16-NEXT: constant 0 : index
// F16-NEXT: constant 1 : index
// F16-NEXT: constant 128 : index
// F16-NEXT: constant 128 : index
// F16-NEXT: constant 30 : index
// F16-NEXT: constant 30 : index
// F16-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F16-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F16-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F16-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// F16-NEXT:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xf16>
// F16-NEXT:         %{{.*}} = fpext %{{.*}} : f16 to f32
// F16-NEXT:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xf32>
// F16-NEXT:       }
// F16-NEXT:     }
// F16-NEXT:   }
// F16-NEXT: }
// F16-NEXT: %{{.*}} = memref_cast %{{.*}} : memref<128x128x30x30xf32> to memref<*xf32>
// F16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*xf32>) -> ()
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
// BF16-NEXT: constant 0 : index
// BF16-NEXT: constant 1 : index
// BF16-NEXT: constant 128 : index
// BF16-NEXT: constant 128 : index
// BF16-NEXT: constant 30 : index
// BF16-NEXT: constant 30 : index
// BF16-NEXT: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// BF16-NEXT:   scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// BF16-NEXT:     scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// BF16-NEXT:       scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// BF16-NEXT:         %{{.*}} = load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xbf16>
// BF16-NEXT:         %{{.*}} = fpext %{{.*}} : bf16 to f32
// BF16-NEXT:         store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : memref<128x128x30x30xf32>
// BF16-NEXT:       }
// BF16-NEXT:     }
// BF16-NEXT:   }
// BF16-NEXT: }
// BF16-NEXT: memref_cast %{{.*}} : memref<128x128x30x30xf32> to memref<*xf32>
// BF16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*xf32>) -> ()
// BF16-NEXT: dealloc {{.*}} : memref<128x8x3x3xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x8x32x32xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x128x30x30xbf16>
// BF16-NEXT: dealloc {{.*}} : memref<128x128x30x30xf32>
// BF16-NEXT: return
