// RUN: mlir-miopen-driver -p -ph -pr | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16

// F32: func @main()
// F32-NEXT: alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// F32-NEXT: alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// F32-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// F32-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// F32-NEXT: memref_cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F32-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F32-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F32-NEXT: constant 0 : i16
// F32-NEXT: constant {{.*}} : i16
// F32-NEXT: constant {{.*}} : i16
// F32-NEXT: constant {{.*}} : i32
// F32-NEXT: call @mcpuMemset5DFloatRand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F32-NEXT: call @mcpuMemset5DFloatRand{{.*}}({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F32-NEXT: call @mcpuMemset5DFloatRand{{.*}}({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F32-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// F32-NEXT: call @convert_tensor[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]](%{{.*}}, %{{.*}}) : (memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>) -> ()
// F32-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// F32-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
// F32-NEXT: dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F32-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F32-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F32-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// F32-NEXT: return

// F16: func @main()
// F16-NEXT: alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// F16-NEXT: alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// F16-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// F16-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// F16-NEXT: memref_cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F16-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F16-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// F16-NEXT: constant 0 : i16
// F16-NEXT: constant {{.*}} : i16 
// F16-NEXT: constant {{.*}} : i16
// F16-NEXT: constant {{.*}} : i32
// F16-NEXT: call @mcpuMemset5DHalfRand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F16-NEXT: call @mcpuMemset5DHalfRand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F16-NEXT: call @mcpuMemset5DHalfRand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// F16-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// F16-NEXT: call @convert_tensor[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]](%{{.*}}, %{{.*}}) : (memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>) -> ()
// F16-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// F16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
// F16-NEXT: dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// F16-NEXT: return

// BF16: func @main()
// BF16-NEXT: alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// BF16-NEXT: alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// BF16-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// BF16-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// BF16-NEXT: memref_cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// BF16-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// BF16-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// BF16-NEXT: constant 0 : i16 
// BF16-NEXT: constant {{.*}} : i16 
// BF16-NEXT: constant {{.*}} : i16
// BF16-NEXT: constant {{.*}} : i32
// BF16-NEXT: call @mcpuMemset5DBF16Rand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// BF16-NEXT: call @mcpuMemset5DBF16Rand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// BF16-NEXT: call @mcpuMemset5DBF16Rand({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// BF16-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// BF16-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<?x?x?x?x?x[[PRINT_TYPE]]>
// BF16-NEXT: call @mcpuMemBF16ConvertFloat({{.*}}, {{.*}}) : (memref<?x?x?x?x?xi16>, memref<?x?x?x?x?xf32>) -> ()
// BF16-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// BF16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
// BF16-NEXT: dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// BF16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// BF16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// BF16-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// BF16-NEXT: return
