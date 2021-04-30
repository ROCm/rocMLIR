// RUN: mlir-miopen-driver -p -prc | FileCheck %s --check-prefix=F32

// F32:  [[RES:%.*]] = memref_cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> () 

// RUN: mlir-miopen-driver -p -prc -t f16 | FileCheck %s --check-prefix=F16

// F16:   call @convert_tensor[[TYPE:[a-zA-Z0-9]+]]({{.*}}, [[FLT:%.*]]) : (memref<[[TYPE]]xf16>, memref<[[TYPE]]xf32>) -> ()
// F16-NEXT:    [[RES:%.*]] = memref_cast [[FLT]] : memref<[[TYPE]]xf32> to memref<*xf32>
// F16-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> ()


// RUN: mlir-miopen-driver -p -prc -t bf16 | FileCheck %s --check-prefix=BF16

// BF16:  call @mcpuMem5DBF16ConvertFloat({{.*}}, {{.*}}) : (memref<?x?x?x?x?xi16>, memref<?x?x?x?x?xf32>) -> ()
// BF16-NEXT:    [[RES:%.*]] = memref_cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// BF16-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> ()

