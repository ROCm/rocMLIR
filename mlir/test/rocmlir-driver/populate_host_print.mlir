// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -pr --apply-bufferization-pipeline=false | FileCheck %s
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -pr -t f16 --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=F16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -pr -t bf16 --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=BF16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -ph -pr -t i8 --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=I8

// CHECK-LABEL: func.func @main()
// CHECK-NEXT: %[[filter:.*]] = memref.alloc() : memref<[[GKCYX:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: affine.for %[[if:.*]] = 0 to [[GKCYX]]
// CHECK-NEXT: affine.apply {{.*}}(%[[if]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[filter]][%[[if]]] : memref<[[GKCYX]]x[[TYPE]]>

// CHECK: %[[input:.*]] = memref.alloc() : memref<[[NGCHIWI:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: affine.for %[[ii:.*]] = 0 to [[NGCHIWI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[ii]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[input]][%[[ii]]] : memref<[[NGCHIWI]]x[[TYPE]]>

// CHECK: %[[output:.*]] = memref.alloc() : memref<[[NGKHOWO:[0-9]+]]x[[TYPE]]>
// CHECK-NEXT: arith.constant dense{{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: arith.constant {{.*}} : f32
// CHECK-NEXT: arith.constant {{.*}} : index
// CHECK-NEXT: vector.insertelement {{.*}} : vector<3x{{.*}}>
// CHECK-NEXT: affine.for %[[io:.*]] = 0 to [[NGKHOWO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[io]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[output]][%[[io]]] : memref<[[NGKHOWO]]x[[TYPE]]>

// CHECK: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[GKCYX]]x[[TYPE]]>, memref<[[NGCHIWI]]x[[TYPE]]>, memref<[[NGKHOWO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[NGKHOWO]]x[[TYPE]]> to memref<*x[[TYPE]]>
// CHECK-NEXT: call @printMemrefF32(%{{.*}}) : (memref<*x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[GKCYX]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[NGCHIWI]]x[[TYPE]]>

// F16: memref.alloc() : memref<{{.*}}>
// F16: call @_memcpy_f16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// BF16: memref.alloc() : memref<{{.*}}>
// BF16: call @_memcpy_bf16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// I8: memref.cast %{{.*}} : memref<{{.*}}> to memref<*xi32>
// I8: call @printMemrefI32(%{{.*}}) : (memref<*xi32>) -> ()
// F16: memref.dealloc {{.*}} : memref<{{.*}}>
// BF16: memref.dealloc {{.*}} : memref<{{.*}}>
// I8: memref.dealloc {{.*}} : memref<{{.*}}>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[NGKHOWO]]x[[TYPE]]>
// CHECK-NEXT: return
