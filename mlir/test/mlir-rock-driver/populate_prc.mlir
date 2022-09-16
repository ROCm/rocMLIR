// RUN: rock-gen -p -prc | FileCheck %s --check-prefix=F32

// F32: func.func @main()
// F32:  [[RES:%.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @printMemrefF32([[RES]]) : (memref<*xf32>) -> ()

// RUN: rock-gen -p -prc -t f16 | FileCheck %s --check-prefix=F16

// F16:    %{{.*}} = memref.alloc() : memref<1x128x8x3x3xf16>
// F16:    call @_memcpy_f16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// F16:    %{{.*}} = memref.alloc() : memref<128x1x8x32x32xf16>
// F16:    call @_memcpy_f16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// F16:    %{{.*}} = memref.alloc() : memref<{{.*}}>
// F16:    call @_memcpy_f16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// F16-NEXT:    call @conv2d_cpu(%{{.*}}, %{{.*}}, [[RES1:%.*]]) : (memref<1x128x8x3x3xf32>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16:    [[RES2:%.*]] = memref.cast [[RES1]] : memref<{{.*}}> to memref<*xf32>
// F16:    call @printMemrefF32([[RES2]]) : (memref<*xf32>) -> ()


// RUN: rock-gen -p -prc -t bf16 | FileCheck %s --check-prefix=BF16

// BF16:    %{{.*}} = memref.alloc() : memref<1x128x8x3x3xbf16>
// BF16:    call @_memcpy_bf16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// BF16:    %{{.*}} = memref.alloc() : memref<128x1x8x32x32xbf16>
// BF16:    call @_memcpy_bf16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// BF16:    %{{.*}} = memref.alloc() : memref<{{.*}}>
// BF16:    call @_memcpy_bf16_f32(%{{.*}}, %{{.*}}, %c{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// BF16-NEXT:    call @conv2d_cpu(%{{.*}}, %{{.*}}, [[RES1:%.*]]) : (memref<1x128x8x3x3xf32>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16:    [[RES2:%.*]] = memref.cast [[RES1]] : memref<{{.*}}> to memref<*xf32>
// BF16:    call @printMemrefF32([[RES2]]) : (memref<*xf32>) -> ()

// RUN: rock-gen -p -prc -t i8 | FileCheck %s --check-prefix=INT8

// INT8: func.func @main()
// INT8:  call @_memcpy_i32_f32(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>, index) -> ()
// INT8-NEXT:  [[RES:%.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// INT8-NEXT:    call @printMemrefF32([[RES]]) : (memref<*xf32>) -> ()
