// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph --apply-bufferization-pipeline=false | FileCheck %s
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -t f16 --apply-bufferization-pipeline=false | FileCheck %s
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -t i8 --apply-bufferization-pipeline=false | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK-NEXT: %[[filter:.*]] = memref.alloc() : memref<[[GKCYX:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
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
// CHECK-NEXT: affine.for %[[if:.*]] = 0 to [[GKCYX]]
// CHECK-NEXT: affine.apply {{.*}}(%[[if]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[filter]][%[[if]]] : memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: %[[input:.*]] = memref.alloc() : memref<[[NGCHIWI:[0-9]+]]x[[TYPE]]>
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
// CHECK-NEXT: affine.for %[[ii:.*]] = 0 to [[NGCHIWI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[ii]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[input]][%[[ii]]] : memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: %[[output:.*]] = memref.alloc() : memref<[[NGKHOWO:[0-9]+]]x[[OTYPE:.*]]>
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
// CHECK-NEXT: affine.for %[[io:.*]] = 0 to [[NGKHOWO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[io]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[output]][%[[io]]] : memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: }
// CHECK-NEXT: call @rock_conv_gkc01_ngc01_ngk01_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[GKCYX]]x[[TYPE]]>, memref<[[NGCHIWI]]x[[TYPE]]>, memref<[[NGKHOWO]]x[[OTYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc %[[filter]]
// CHECK-NEXT: memref.dealloc %[[input]]
// CHECK-NEXT: memref.dealloc %[[output]]
// CHECK-NEXT: return

// CHECK: func.func @rock_conv_gkc01_ngc01_ngk01_0_gpu(%{{.*}}: memref<[[GKCYX]]x[[TYPE]]>, %{{.*}}: memref<[[NGCHIWI]]x[[TYPE]]>, %{{.*}}: memref<[[NGKHOWO]]x[[OTYPE]]>)
// CHECK-NEXT: gpu.alloc  () : memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[GKCYX]]x[[TYPE]]>,  memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[NGCHIWI]]x[[TYPE]]>,  memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: gpu.alloc  () : memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[NGKHOWO]]x[[OTYPE]]>,  memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: call @rock_conv_gkc01_ngc01_ngk01_0(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<[[GKCYX]]x[[TYPE]]>, memref<[[NGCHIWI]]x[[TYPE]]>, memref<[[NGKHOWO]]x[[OTYPE]]>) -> ()
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[GKCYX]]x[[TYPE]]>,  memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[NGCHIWI]]x[[TYPE]]>,  memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[NGCHIWI]]x[[TYPE]]>
// CHECK-NEXT: gpu.memcpy  %{{.*}}, %{{.*}} : memref<[[NGKHOWO]]x[[OTYPE]]>,  memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: gpu.dealloc  %{{.*}} : memref<[[NGKHOWO]]x[[OTYPE]]>
// CHECK-NEXT: return
