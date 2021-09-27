// RUN: mlir-miopen-driver -p -ph | FileCheck %s
// RUN: mlir-miopen-driver -p -ph -t f16 | FileCheck %s
// RUN: mlir-miopen-driver -p -ph -t bf16 | FileCheck %s

// CHECK-LABEL: func @main()
// CHECK-NEXT: alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: constant 0 : i16
// CHECK-NEXT: constant {{.*}} : i16
// CHECK-NEXT: constant {{.*}} : i16
// CHECK-NEXT: constant {{.*}} : i32
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: call @gpu_conv({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return

// CHECK: func @gpu_conv(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>)
// CHECK-NEXT: memref_cast %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: memref_cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: constant 1 : i32
// CHECK-NEXT: constant 2 : i32
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: memref_cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref_cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: memref_cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: call @conv2d(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: return

// CHECK: func @conv2d(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>)
// CHECK-NEXT: return
