// RUN: miopen-gen -p -ph | FileCheck %s
// RUN: miopen-gen -p -ph -t f16 | FileCheck %s

// CHECK-LABEL: func @main()
// CHECK-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// CHECK-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// CHECK-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// CHECK-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// CHECK-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// CHECK-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// CHECK-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[TYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// CHECK-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// CHECK-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// CHECK-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
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
// INT8-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// INT8-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// INT8-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// INT8-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// INT8-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// INT8-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// INT8-NEXT: vector.extractelement
// INT8-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// INT8-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPE]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPE]]>
// INT8-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// INT8-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// INT8-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// INT8-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// INT8-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// INT8-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// INT8-NEXT: vector.extractelement
// INT8-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPEI32:i32]]>
// INT8-NEXT: arith.constant dense{{.*}} : vector<2x[[TYPEI32]]>
// INT8-NEXT: arith.constant 0 : [[TYPEI32]]
// INT8-NEXT: arith.constant 0 : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<2x[[TYPEI32]]>
// INT8-NEXT: arith.constant 0 : [[TYPEI32]]
// INT8-NEXT: arith.constant 1 : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<2x[[TYPEI32]]>
// INT8-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// INT8-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// INT8-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// INT8-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// INT8-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// INT8-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// INT8-NEXT: vector.extractelement
// INT8-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
// INT8-NEXT: }
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
