// RUN: rocmlir-gen --arch %arch -p -ph --apply-bufferization-pipeline=false | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -ph -t f16 --apply-bufferization-pipeline=false | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -ph -t i8 --apply-bufferization-pipeline=false | FileCheck %s

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
// CHECK-NEXT: %[[flatFilter:.*]] = memref.collapse_shape {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> into memref<[[GKCYX:.*]]x[[TYPE]]>
// CHECK-NEXT: affine.for %[[if:.*]] = 0 to [[GKCYX]]
// CHECK-NEXT: affine.apply {{.*}}(%[[if]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatFilter]][%[[if]]] : memref<[[GKCYX]]x[[TYPE]]>
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
// CHECK-NEXT: %[[flatInput:.*]] = memref.collapse_shape {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> into memref<[[NGCHIWI:.*]]x[[TYPE]]>
// CHECK-NEXT: affine.for %[[ii:.*]] = 0 to [[NGCHIWI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[ii]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatInput]][%[[ii]]] : memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[OTYPE:.*]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x[[OTYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[OTYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[OTYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[OTYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[OTYPE]]>
// CHECK-NEXT: arith.constant {{.*}} : [[OTYPE]]
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x[[OTYPE]]>
// CHECK-NEXT: %[[flatOutput:.*]] = memref.collapse_shape {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]> into memref<[[NGKHOWO:.*]]x[[OTYPE]]>
// CHECK-NEXT: affine.for %[[io:.*]] = 0 to [[NGKHOWO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[io]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatOutput]][%[[io]]] : memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>
// CHECK-NEXT: return

// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%{{.*}}: memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, %{{.*}}: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>)
// CHECK-NEXT: gpu.alloc  () : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>
// CHECK-NEXT: call @rock_conv2d_gkcyx_ngchw_ngkhw_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>) -> ()
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>,  memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>,  memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>,  memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[OTYPE]]>
// CHECK-NEXT: return
