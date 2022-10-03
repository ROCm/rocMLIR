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
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// CHECK-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// CHECK-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// CHECK-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// CHECK-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>

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
// CHECK-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// CHECK-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// CHECK-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// CHECK-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>

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
// CHECK-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// CHECK-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// CHECK-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// CHECK-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// CHECK-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// CHECK-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// CHECK-NEXT: vector.extractelement
// CHECK-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>

// CHECK: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>

// F16: memref.alloc() : memref<{{.*}}>
// F16: call @_memcpy_f16_f32(%{{.*}}, %{{.*}}, %{{.*}}) : ({{.*}} -> ()
// BF16: memref.alloc() : memref<{{.*}}>
// BF16: call @_memcpy_bf16_f32(%{{.*}}, %{{.*}}, %{{.*}}) : ({{.*}} -> ()
// I8: memref.alloc() : memref<{{.*}}>
// I8: call @_memcpy_i32_f32(%{{.*}}, %{{.*}}, %{{.*}}) : ({{.*}} -> ()
// CHECK-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<*x[[TYPE]]>
// CHECK-NEXT: call @printMemrefF32(%{{.*}}) : (memref<*x[[TYPE]]>) -> ()
// F16: memref.dealloc {{.*}} : memref<{{.*}}>
// BF16: memref.dealloc {{.*}} : memref<{{.*}}>
// I8: memref.dealloc {{.*}} : memref<{{.*}}>
// CHECK-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// CHECK-NEXT: return
