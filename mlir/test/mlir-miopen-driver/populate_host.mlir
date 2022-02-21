// RUN: miopen-gen -p -ph | FileCheck %s
// RUN: miopen-gen -p -ph -t f16 | FileCheck %s
// RUN: miopen-gen -p -ph -t bf16 | FileCheck %s

// CHECK-LABEL: func @main()
// CHECK-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: memref.cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: constant {{.*}} : i16
// CHECK-NEXT: constant {{.*}} : i32
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: memref.cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: memref.cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: constant 0 : i16
// CHECK-NEXT: call @mcpuMemset5D{{.*}}RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// CHECK-NEXT: call @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return

// CHECK: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>)
// CHECK-NEXT: constant 1 : i32
// CHECK-NEXT: constant 2 : i32
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemAlloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: call @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// CHECK-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// CHECK-NEXT: return

// RUN: miopen-gen -p -ph -t i8 | FileCheck %s --check-prefix=INT8

// INT8-LABEL: func @main()
// INT8-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:i8]]>
// INT8-NEXT: memref.cast {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: constant {{.*}} : i16
// INT8-NEXT: constant {{.*}} : i32
// INT8-NEXT: call @mcpuMemset5DInt8RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// INT8-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// INT8-NEXT: memref.cast {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: call @mcpuMemset5DInt8RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, i16, i16, i32) -> ()
// INT8-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPEI32:i32]]>
// INT8-NEXT: memref.cast {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]> to memref<?x?x?x?x?x[[TYPEI32]]>
// INT8-NEXT: constant 0 : i16
// INT8-NEXT: call @mcpuMemset5DInt32RandInt({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (memref<?x?x?x?x?x[[TYPEI32:i32]]>, i16, i16, i32) -> ()
// INT8-NEXT: call @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>) -> ()
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: return

// INT8: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>)
// INT8-NEXT: constant 1 : i32
// INT8-NEXT: constant 2 : i32
// INT8-NEXT: memref.cast %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: call @mgpuMemAlloc5DInt8(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: call @mgpuMemCopy5DInt8(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// INT8-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> to memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: call @mgpuMemAlloc5DInt8(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> memref<?x?x?x?x?x[[TYPE]]>
// INT8-NEXT: call @mgpuMemCopy5DInt8(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// INT8-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPE]]> to memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]> to memref<?x?x?x?x?x[[TYPEI32]]>
// INT8-NEXT: call @mgpuMemAlloc5DInt32(%{{.*}}) : (memref<?x?x?x?x?x[[TYPEI32]]>) -> memref<?x?x?x?x?x[[TYPEI32]]>
// INT8-NEXT: call @mgpuMemCopy5DInt32(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPEI32]]>, memref<?x?x?x?x?x[[TYPEI32]]>, i32) -> ()
// INT8-NEXT: memref.cast %{{.*}} : memref<?x?x?x?x?x[[TYPEI32]]> to memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: call @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>) -> ()
// INT8-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// INT8-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// INT8-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>, memref<?x?x?x?x?x[[TYPE]]>, i32) -> ()
// INT8-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPE]]>) -> ()
// INT8-NEXT: call @mgpuMemCopy5D{{.*}}(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<?x?x?x?x?x[[TYPEI32]]>, memref<?x?x?x?x?x[[TYPEI32]]>, i32) -> ()
// INT8-NEXT: call @mgpuMemDealloc5D{{.*}}(%{{.*}}) : (memref<?x?x?x?x?x[[TYPEI32]]>) -> ()
// INT8-NEXT: return
