// RUN: miopen-gen -p -ph -pr | FileCheck %s --check-prefix=F32
// RUN: miopen-gen -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: miopen-gen -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16
// RUN: miopen-gen -p -ph -pr -t i8 | FileCheck %s --check-prefix=I8

// F32-NOT: func @_memcpy_

// F16: func @_memcpy_f16_f32(%arg0: memref<?xf16>, %arg1: memref<?xf32>, %arg2: index) {
// F16-NEXT:   %c0 = arith.constant 0 : index
// F16-NEXT:   %c1 = arith.constant 1 : index
// F16-NEXT:   scf.for %arg3 = %c0 to %arg2 step %c1 {
// F16-NEXT:     %0 = memref.load %arg0[%arg3] : memref<?xf16>
// F16-NEXT:     %1 = arith.extf %0 : f16 to f32
// F16-NEXT:     memref.store %1, %arg1[%arg3] : memref<?xf32>
// F16-NEXT:   }
// F16-NEXT:   return
// F16-NEXT: }

// BF16: func @_memcpy_bf16_f32(%arg0: memref<?xbf16>, %arg1: memref<?xf32>, %arg2: index) {
// BF16-NEXT:   %c0 = arith.constant 0 : index
// BF16-NEXT:   %c1 = arith.constant 1 : index
// BF16-NEXT:   scf.for %arg3 = %c0 to %arg2 step %c1 {
// BF16-NEXT:     %0 = memref.load %arg0[%arg3] : memref<?xbf16>
// BF16-NEXT:     %1 = arith.extf %0 : bf16 to f32
// BF16-NEXT:     memref.store %1, %arg1[%arg3] : memref<?xf32>
// BF16-NEXT:   }
// BF16-NEXT:   return
// BF16-NEXT: }

// I8:  func @_memcpy_i32_f32(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: index) {
// I8-NEXT:   %c0 = arith.constant 0 : index
// I8-NEXT:   %c1 = arith.constant 1 : index
// I8-NEXT:   scf.for %arg3 = %c0 to %arg2 step %c1 {
// I8-NEXT:     %0 = memref.load %arg0[%arg3] : memref<?xi32>
// I8-NEXT:     %1 = arith.sitofp %0 : i32 to f32
// I8-NEXT:     memref.store %1, %arg1[%arg3] : memref<?xf32>
// I8-NEXT:   }
// I8-NEXT:   return
// I8-NEXT: }

