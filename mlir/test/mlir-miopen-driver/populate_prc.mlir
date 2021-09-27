// RUN: mlir-miopen-driver -p -prc | FileCheck %s --check-prefix=F32

// F32:  [[RES:%.*]] = memref_cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// F32-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> () 

// RUN: mlir-miopen-driver -p -prc -t f16 | FileCheck %s --check-prefix=F16

// F16:    call @convert_tensor{{.*}}(%{{.*}}, [[FILTER:%.*]]) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16-NEXT:    [[IN:%.*]] = alloc() : memref<{{.*}}>
// F16-NEXT:    call @convert_tensor{{.*}}(%{{.*}}, [[IN]]) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16-NEXT:    [[OUT:%.*]] = alloc() : memref<{{.*}}>
// F16-NEXT:    call @convert_tensor{{.*}}(%{{.*}}, [[OUT]]) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16-NEXT:    call @conv2d_host([[FILTER]], [[IN]], [[OUT]]) : (memref<{{.*}}>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16-NEXT:    [[OUT_F16:%.*]] = alloc() : memref<128x1x128x30x30xf16>
// F16-NEXT:    call @convert_tensor{{.*}}([[OUT]], [[OUT_F16]]) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// F16-NEXT:    [[OUT_PRINT:%.*]] = alloc() : memref<[[TYPE:[a-zA-Z0-9]+]]xf32>
// F16-NEXT:    call @convert_tensor[[TYPE]]xf16([[OUT_F16]], [[OUT_PRINT]]) : (memref<[[TYPE]]xf16>, memref<[[TYPE]]xf32>) -> ()
// F16-NEXT:    [[RES:%.*]] = memref_cast [[OUT_PRINT]] : memref<[[TYPE]]xf32> to memref<*xf32>
// F16-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> ()


// RUN: mlir-miopen-driver -p -prc -t bf16 | FileCheck %s --check-prefix=BF16

// BF16:    call @mcpuMem5DBF16ConvertFloat(%{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    %{{.*}} = alloc() : memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<128x1x8x32x32xi16> to memref<{{.*}}>
// BF16-NEXT:    call @mcpuMem5DBF16ConvertFloat(%{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    %{{.*}} = alloc() : memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}}12 : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}}4 : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    call @mcpuMem5DBF16ConvertFloat(%{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    call @conv2d_host(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<1x128x8x3x3xf32>, memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    %{{.*}} = alloc() : memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    call @mcpuMem5DFloatConvertBF16(%{{.*}}, %{{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    %{{.*}} = alloc() : memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    %{{.*}} = memref_cast %{{.*}} : memref<{{.*}}> to memref<{{.*}}>
// BF16-NEXT:    call @mcpuMem5DBF16ConvertFloat({{.*}}, {{.*}}) : (memref<{{.*}}>, memref<{{.*}}>) -> ()
// BF16-NEXT:    [[RES:%.*]] = memref_cast {{.*}} : memref<{{.*}}> to memref<*xf32>
// BF16-NEXT:    call @print_memref_f32([[RES]]) : (memref<*xf32>) -> ()

