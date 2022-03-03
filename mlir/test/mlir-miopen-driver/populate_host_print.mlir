// RUN: miopen-gen -p -ph -pr | FileCheck %s --check-prefix=F32
// RUN: miopen-gen -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: miopen-gen -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16

// F32: func @main()
// F32-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// F32-NEXT: arith.constant dense{{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F32-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// F32-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// F32-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// F32-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// F32-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// F32-NEXT: vector.extractelement
// F32-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// F32-NEXT: arith.constant dense{{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: arith.constant {{.*}} : f32
// F32-NEXT: arith.constant {{.*}} : index
// F32-NEXT: vector.insertelement {{.*}} : vector<3xf32>
// F32-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// F32-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F32-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// F32-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// F32-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// F32-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// F32-NEXT: vector.extractelement
// F32-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// F32-NEXT: arith.constant dense{{.*}} : vector<2xf32>
// F32-NEXT: arith.constant 0.000000e+00 : f32
// F32-NEXT: arith.constant 0 : index
// F32-NEXT: vector.insertelement {{.*}} : vector<2xf32>
// F32-NEXT: arith.constant 0.000000e+00 : f32
// F32-NEXT: arith.constant 1 : index
// F32-NEXT: vector.insertelement {{.*}} : vector<2xf32>
// F32-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// F32-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F32-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// F32-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// F32-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// F32-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// F32-NEXT: vector.extractelement
// F32-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: }
// F32-NEXT: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// F32-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F32-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F32-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]> to memref<*x[[TYPE]]>
// F32-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[TYPE]]>) -> ()
// F32-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F32-NEXT: return

// F16: func @main()
// F16-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// F16-NEXT: arith.constant dense{{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F16-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// F16-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// F16-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// F16-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// F16-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// F16-NEXT: vector.extractelement
// F16-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// F16-NEXT: arith.constant dense{{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: arith.constant {{.*}} : f16
// F16-NEXT: arith.constant {{.*}} : index
// F16-NEXT: vector.insertelement {{.*}} : vector<3xf16>
// F16-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// F16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F16-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// F16-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// F16-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// F16-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// F16-NEXT: vector.extractelement
// F16-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// F16-NEXT: arith.constant dense{{.*}} : vector<2xf16>
// F16-NEXT: arith.constant 0.000000e+00 : f16
// F16-NEXT: arith.constant 0 : index
// F16-NEXT: vector.insertelement {{.*}} : vector<2xf16>
// F16-NEXT: arith.constant 0.000000e+00 : f16
// F16-NEXT: arith.constant 1 : index
// F16-NEXT: vector.insertelement {{.*}} : vector<2xf16>
// F16-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// F16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// F16-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// F16-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// F16-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// F16-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// F16-NEXT: vector.extractelement
// F16-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: }
// F16-NEXT: call @{{.*}}({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// F16-NEXT: memref.dealloc {{.*}} : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// F16-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// F16-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// F16: call @_memcpy_f16_f32(%{{.*}}, %{{.*}}, %{{.*}}) : (memref<{{.*}}x[[TYPE]]>, memref<{{.*}}x[[PRINT_TYPE]]>, index) -> ()
// F16-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// F16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
// F16-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// F16-NEXT: memref.dealloc {{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F16-NEXT: return

// BF16: func @main()
// BF16-NEXT: memref.alloc() : memref<[[G:[0-9]+]]x[[K:[0-9]+]]x[[C:[0-9]+]]x[[Y:[0-9]+]]x[[X:[0-9]+]]x[[TYPE:[a-zA-Z0-9]+]]>
// BF16-NEXT: arith.constant dense{{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// BF16-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// BF16-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// BF16-NEXT: affine.for %[[y:.*]] = 0 to [[Y]]
// BF16-NEXT: affine.for %[[x:.*]] = 0 to [[X]]
// BF16-NEXT: affine.apply {{.*}}(%[[g]], %[[k]], %[[c]], %[[y]], %[[x]])
// BF16-NEXT: vector.extractelement
// BF16-NEXT: arith.bitcast
// BF16-NEXT: memref.store {{.*}}[%[[g]], %[[k]], %[[c]], %[[y]], %[[x]]] : memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: memref.alloc() : memref<[[N:[0-9]+]]x[[G:[0-9]+]]x[[C]]x[[HI:[0-9]+]]x[[WI:[0-9]+]]x[[TYPE]]>
// BF16-NEXT: arith.constant dense{{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: arith.constant {{.*}} : i16
// BF16-NEXT: arith.constant {{.*}} : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<3xi16>
// BF16-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// BF16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// BF16-NEXT: affine.for %[[c:.*]] = 0 to [[C]]
// BF16-NEXT: affine.for %[[hi:.*]] = 0 to [[HI]]
// BF16-NEXT: affine.for %[[wi:.*]] = 0 to [[WI]]
// BF16-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]])
// BF16-NEXT: vector.extractelement
// BF16-NEXT: arith.bitcast
// BF16-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[c]], %[[hi]], %[[wi]]] : memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]x[[TYPE]]>
// BF16-NEXT: arith.constant dense{{.*}} : vector<2xi16>
// BF16-NEXT: arith.constant 0 : i16
// BF16-NEXT: arith.constant 0 : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<2xi16>
// BF16-NEXT: arith.constant 0 : i16
// BF16-NEXT: arith.constant 1 : index
// BF16-NEXT: vector.insertelement {{.*}} : vector<2xi16>
// BF16-NEXT: affine.for %[[n:.*]] = 0 to [[N]]
// BF16-NEXT: affine.for %[[g:.*]] = 0 to [[G]]
// BF16-NEXT: affine.for %[[k:.*]] = 0 to [[K]]
// BF16-NEXT: affine.for %[[ho:.*]] = 0 to [[HO]]
// BF16-NEXT: affine.for %[[wo:.*]] = 0 to [[WO]]
// BF16-NEXT: affine.apply {{.*}}(%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]])
// BF16-NEXT: vector.extractelement
// BF16-NEXT: arith.bitcast
// BF16-NEXT: memref.store {{.*}}[%[[n]], %[[g]], %[[k]], %[[ho]], %[[wo]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: }
// BF16-NEXT: call @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}, {{.*}}, {{.*}}) : (memref<[[G]]x[[K]]x[[C]]x[[Y]]x[[X]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[C]]x[[HI]]x[[WI]]x[[TYPE]]>, memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>) -> ()
// BF16: memref.alloc() : memref<[[N]]x[[G:[0-9]+]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>
// BF16: call @_memcpy_bf16_f32({{.*}}, {{.*}}, {{.*}}) : (memref<?xbf16>, memref<?xf32>, index) -> ()
// BF16-NEXT: memref.cast %{{.*}} : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]> to memref<*x[[PRINT_TYPE]]>
// BF16-NEXT: call @print_memref_f32(%{{.*}}) : (memref<*x[[PRINT_TYPE]]>) -> ()
