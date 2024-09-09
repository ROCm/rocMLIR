// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc | FileCheck %s --check-prefix=F32

// F32: func.func @main()
// F32:  %[[RES:.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @printMemrefF32(%[[RES]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t f16 | FileCheck %s --check-prefixes=CHECK -D\$TYPE=f16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t bf16 | FileCheck %s --check-prefix=CHECK -D\$TYPE=bf16

// CHECK-LABEL: func.func @conv_cpu
// CHECK-SAME: ([[arg0:%.+]]: memref<9216x[[$TYPE]]>, [[arg1:%.+]]: memref<1048576x[[$TYPE]]>, [[arg2:%.+]]: memref<14745600x[[$TYPE]]>)
// CHECK:  [[alloc0:%.+]] = memref.alloc() : memref<9216xf32>
// CHECK:  call @_memcpy_[[$TYPE]]_f32_9216([[arg0]], [[alloc0]]) : (memref<9216x[[$TYPE]]>, memref<9216xf32>) -> ()
// CHECK:  [[alloc1:%.+]] = memref.alloc() : memref<1048576xf32>
// CHECK:  call @_memcpy_[[$TYPE]]_f32_1048576([[arg1]], [[alloc1]]) : (memref<1048576x[[$TYPE]]>, memref<1048576xf32>) -> ()
// CHECK:  [[alloc2:%.+]] = memref.alloc() : memref<14745600xf32>
// CHECK:  call @_memcpy_[[$TYPE]]_f32_14745600([[arg2]], [[alloc2]]) : (memref<14745600x[[$TYPE]]>, memref<14745600xf32>) -> ()
// CHECK-LABEL: func.func @main
// CHECK: call @conv_cpu(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<9216x[[$TYPE]]>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// CHECK:  %[[RES1:.*]] = memref.alloc() : memref<{{.*}}xf32>
// CHECK:  call @_memcpy_[[$TYPE]]_f32_{{[0-9]+}}(%{{.*}}, %{{.*}}) : (memref<{{.*}}x[[$TYPE]]>, memref<{{.*}}xf32>) -> ()
// CHECK:  %[[RES2:.*]] = memref.cast %[[RES1]] : memref<{{.*}}> to memref<*xf32>
// CHECK:  call @printMemrefF32(%[[RES2]]) : (memref<*xf32>) -> ()

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -prc -t i8 | FileCheck %s --check-prefix=INT8

// INT8: func.func @main()
// INT8:  [[RES:%.*]] = memref.cast {{.*}} : memref<{{.*}}> to memref<*xi32>
// INT8-NEXT:    call @printMemrefI32([[RES]]) : (memref<*xi32>) -> ()
