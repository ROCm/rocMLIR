// RUN: rocmlir-gen --arch %arch -p -prc | FileCheck %s --check-prefix=F32

// F32: func.func @main()
// F32:  %[[RES:.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @printMemrefF32(%[[RES]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch %arch -p -prc -t f16 | FileCheck %s --check-prefixes=F16,CHECK
// RUN: rocmlir-gen --arch %arch -p -prc -t bf16 | FileCheck %s --check-prefix=BF16

// F16:    %{{.*}} = memref.alloc() : memref<1x128x8x3x3x[[type:f16]]>
// BF16:   %{{.*}}= memref.alloc() : memref<1x128x8x3x3x[[type:bf16]]>
// CHECK:  call @_memcpy_[[type]]_f32_9216(%{{.*}}, %{{.*}}) : (memref<9216x[[type]]>, memref<9216xf32>) -> ()
// CHECK:  %{{.*}} = memref.alloc() : memref<128x1x8x32x32x[[type]]>
// CHECK:  call @_memcpy_[[type]]_f32_1048576(%{{.*}}, %{{.*}}) : (memref<1048576x[[type]]>, memref<1048576xf32>) -> ()
// CHECK:  %{{.*}} = memref.alloc() : memref<{{.*}}>
// CHECK:  call @_memcpy_[[type]]_f32_{{[0-9]+}}(%{{.*}}, %{{.*}}) : (memref<{{.*}}x[[type]]>, memref<{{.*}}xf32>) -> ()
// CHECK-NEXT: call @conv2d_cpu(%{{.*}}, %{{.*}}, %[[RES1:.*]]) : (memref<1x128x8x3x3xf32>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// CHECK:  %[[RES2:.*]] = memref.cast %[[RES1]] : memref<{{.*}}> to memref<*xf32>
// CHECK:  call @printMemrefF32(%[[RES2]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch %arch -p -prc -t i8 | FileCheck %s --check-prefix=INT8

// INT8: func.func @main()
// INT8:  call @_memcpy_i32_f32_{{[0-9]+}}(%{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// INT8-NEXT:  [[RES:%.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// INT8-NEXT:    call @printMemrefF32([[RES]]) : (memref<*xf32>) -> ()
