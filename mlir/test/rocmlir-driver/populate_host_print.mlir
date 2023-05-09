// RUN: rocmlir-gen --arch %arch -p -ph -pr | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16
// RUN: rocmlir-gen --arch %arch -p -ph -pr -t i8 | FileCheck %s --check-prefix=I8

// CHECK-LABEL: func.func @main()
// CHECK-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
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
// CHECK-NEXT: %[[flatFilter:.*]] = memref.collapse_shape {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]> into memref<[[GKCYX:.*]]x[[TYPE]]>
// CHECK-NEXT: affine.for %[[if:.*]] = 0 to [[GKCYX]]
// CHECK-NEXT: affine.apply {{.*}}(%[[if]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatFilter]][%[[if]]] : memref<[[GKCYX]]x[[TYPE]]>

// CHECK: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
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
// CHECK-NEXT: %[[flatInput:.*]] =  memref.collapse_shape {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]> into memref<[[NGCHIWI:.*]]x[[TYPE]]>
// CHECK-NEXT: affine.for %[[ii:.*]] = 0 to [[NGCHIWI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[ii]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatInput]][%[[ii]]] : memref<[[NGCHIWI]]x[[TYPE]]>

// CHECK: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
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
// CHECK-NEXT: %[[flatOutput:.*]] =  memref.collapse_shape {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> into memref<[[NGKHOWO:.*]]x[[TYPE]]>
// CHECK-NEXT: affine.for %[[io:.*]] = 0 to [[NGKHOWO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[io]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store %{{.*}}, %[[flatOutput]][%[[io]]] : memref<[[NGKHOWO]]x[[TYPE]]>

// CHECK: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<*x[[TYPE]]>
// CHECK-NEXT: call @printMemrefF32(%{{.*}}) : (memref<*x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>

// F16: memref.alloc() : memref<{{.*}}>
// F16: call @_memcpy_f16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// BF16: memref.alloc() : memref<{{.*}}>
// BF16: call @_memcpy_bf16_f32_{{[0-9]+}}(%{{.*}}, %{{.*}})
// I8: memref.cast %{{.*}} : memref<{{.*}}> to memref<*xi32>
// I8: call @printMemrefI32(%{{.*}}) : (memref<*xi32>) -> ()
// F16: memref.dealloc {{.*}} : memref<{{.*}}>
// BF16: memref.dealloc {{.*}} : memref<{{.*}}>
// I8: memref.dealloc {{.*}} : memref<{{.*}}>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return
