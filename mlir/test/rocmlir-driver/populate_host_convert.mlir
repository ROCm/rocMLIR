// RUN: rocmlir-gen --arch %arch -p -ph -pr --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=F32
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t f16 --apply-bufferization-pipeline=false | FileCheck %s --check-prefixes=F16,CHECK
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t bf16 --apply-bufferization-pipeline=false | FileCheck %s --check-prefixes=BF16,CHECK
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t i8 --apply-bufferization-pipeline=false | FileCheck %s --check-prefixes=I8

// F32-NOT: func.func @_memcpy_
// I8-NOT: func.func @_memcpy_

// F16: func.func @_memcpy_[[type:f16]]_f32_[[n:[0-9]+]]
// BF16: func.func @_memcpy_[[type:bf16]]_f32_[[n:[0-9]+]]
// CHECK-SAME: (%[[arg0:.*]]: memref<[[n]]x[[type]]>, %[[arg1:.*]]: memref<[[n]]xf32>)

// CHECK-NEXT: %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[size:.*]] = memref.dim %[[arg0]], %[[c0]]
// CHECK-NEXT: scf.for %[[arg2:.*]] = %[[c0]] to %[[size]] step %[[c1]] {
// CHECK-NEXT:   %[[load:.*]] = memref.load %[[arg0]][%[[arg2]]]
// F16-NEXT:     %[[converted:.*]] = arith.extf %[[load]] : f16 to f32
// BF16-NEXT:    %[[converted:.*]] = arith.extf %[[load]] : bf16 to f32
// CHECK-NEXT:   memref.store %[[converted]], %[[arg1]][%[[arg2]]]
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
