// RUN: mlir-miopen-driver -p -ph -pr | FileCheck %s
// RUN: mlir-miopen-driver -p -ph -pr -t f16 | FileCheck %s
// RUN: mlir-miopen-driver -p -ph -pr -t bf16 | FileCheck %s

// CHECK-LABEL: func @main()
// CHECK-NEXT: alloc() : memref<[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: alloc() : memref<[[N:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: alloc() : memref<[[N]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: alloc() : memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[N]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x[[TYPE]]>
// CHECK-NEXT: constant {{.*}} : [[TYPE]]
// CHECK-NEXT: constant {{.*}} : [[TYPE]]
// CHECK-NEXT: call @mcpuMemset4D{{.*}}({{.*}}, {{.*}}) : (memref<?x?x?x?x[[TYPE]]>, [[TYPE]]) -> ()
// CHECK-NEXT: call @mcpuMemset4D{{.*}}({{.*}}, {{.*}}) : (memref<?x?x?x?x[[TYPE]]>, [[TYPE]]) -> ()
// CHECK-NEXT: call @mcpuMemset4D{{.*}}({{.*}}, {{.*}}) : (memref<?x?x?x?x[[TYPE]]>, [[TYPE]]) -> ()
// CHECK-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: call @convert_result(%{{.*}}, %{{.*}}) : (memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>, memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>) -> ()
// CHECK-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// CHECK-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
// CHECK-NEXT: dealloc {{.*}} : memref<[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: dealloc {{.*}} : memref<[[N]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: dealloc {{.*}} : memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: dealloc {{.*}} : memref<[[N]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// CHECK-NEXT: return
