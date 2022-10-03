// RUN: rocmlir-gen --arch %targetChip -p -ph | FileCheck %s
// RUN: rocmlir-gen --arch %targetChip -p -ph -t f16 | FileCheck %s

// CHECK-LABEL: func.func @main()
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
// CHECK-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return

// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>)
// CHECK-NEXT: gpu.alloc  () : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return

// RUN: rocmlir-gen --arch %targetChip -p -ph -t i8 | FileCheck %s --check-prefix=INT8

// INT8-LABEL: func.func @main()
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
// INT8-NEXT: arith.constant dense{{.*}} : vector<3x[[TYPEI32]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPEI32]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPEI32]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPEI32]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPEI32]]>
// INT8-NEXT: arith.constant {{.*}} : [[TYPEI32]]
// INT8-NEXT: arith.constant {{.*}} : index
// INT8-NEXT: vector.insertelement {{.*}} : vector<3x[[TYPEI32]]>
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
// INT8-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>) -> ()
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: return

// INT8: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>)
// INT8-NEXT: gpu.alloc  () : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>) -> ()
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: gpu.dealloc  %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// INT8-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPEI32]]>
// INT8-NEXT: return
