// RUN: mlir-miopen-driver -p -ph -pr | FileCheck %s --check-prefix=F32
// RUN: mlir-miopen-driver -p -ph -pr -t f16 | FileCheck %s --check-prefix=F16
// RUN: mlir-miopen-driver -p -ph -pr -t bf16 | FileCheck %s --check-prefix=BF16

// F32: func @convert_tensor[[N:[0-9]+]]x[[G:[0-9]+]]x[[K:[0-9]+]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]xf32([[SOURCE:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE:[a-zA-Z0-9]+]]>, [[DEST:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>)
// F32-NEXT: [[ZERO:%[a-zA-Z_0-9]+]] = constant 0 : index
// F32-NEXT: [[ONE:%[a-zA-Z_0-9]+]] = constant 1 : index
// F32-NEXT: [[BOUND_N:%[a-zA-Z_0-9]+]] = constant [[N]] : index
// F32-NEXT: [[BOUND_G:%[a-zA-Z_0-9]+]] = constant [[G]] : index
// F32-NEXT: [[BOUND_K:%[a-zA-Z_0-9]+]] = constant [[K]] : index
// F32-NEXT: [[BOUND_HO:%[a-zA-Z_0-9]+]] = constant [[HO]] : index
// F32-NEXT: [[BOUND_WO:%[a-zA-Z_0-9]+]] = constant [[WO]] : index
// F32-NEXT: scf.for [[IV_N:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_N]] step [[ONE]] {
// F32-NEXT:   scf.for [[IV_G:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_G]] step [[ONE]] {
// F32-NEXT:     scf.for [[IV_K:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_K]] step [[ONE]] {
// F32-NEXT:       scf.for [[IV_HO:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_HO]] step [[ONE]] {
// F32-NEXT:         scf.for [[IV_WO:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_WO]] step [[ONE]] {
// F32-NEXT:           [[VALUE:%[a-zA-Z_0-9]+]] = memref.load [[SOURCE]]{{\[}}[[IV_N]], [[IV_G]], [[IV_K]], [[IV_HO]], [[IV_WO]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F32-NEXT:           memref.store [[VALUE]], [[DEST]]{{\[}}[[IV_N]], [[IV_G]], [[IV_K]], [[IV_HO]], [[IV_WO]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// F32-NEXT:         }
// F32-NEXT:       }
// F32-NEXT:     }
// F32-NEXT:   }
// F32-NEXT: }
// F32-NEXT: return

// F16: func @convert_tensor[[N:[0-9]+]]x[[G:[0-9]+]]x[[K:[0-9]+]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]xf16([[SOURCE:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE:[a-zA-Z0-9]+]]>, [[DEST:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>)
// F16-NEXT: [[ZERO:%[a-zA-Z_0-9]+]] = constant 0 : index
// F16-NEXT: [[ONE:%[a-zA-Z_0-9]+]] = constant 1 : index
// F16-NEXT: [[BOUND_N:%[a-zA-Z_0-9]+]] = constant [[N]] : index
// F16-NEXT: [[BOUND_G:%[a-zA-Z_0-9]+]] = constant [[G]] : index
// F16-NEXT: [[BOUND_K:%[a-zA-Z_0-9]+]] = constant [[K]] : index
// F16-NEXT: [[BOUND_HO:%[a-zA-Z_0-9]+]] = constant [[HO]] : index
// F16-NEXT: [[BOUND_WO:%[a-zA-Z_0-9]+]] = constant [[WO]] : index
// F16-NEXT: scf.for [[IV_N:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_N]] step [[ONE]] {
// F16-NEXT:   scf.for [[IV_G:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_G]] step [[ONE]] {
// F16-NEXT:     scf.for [[IV_K:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_K]] step [[ONE]] {
// F16-NEXT:       scf.for [[IV_HO:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_HO]] step [[ONE]] {
// F16-NEXT:         scf.for [[IV_WO:%[a-zA-Z_0-9]+]] = [[ZERO]] to [[BOUND_WO]] step [[ONE]] {
// F16-NEXT:           [[VALUE:%[a-zA-Z_0-9]+]] = memref.load [[SOURCE]]{{\[}}[[IV_N]], [[IV_G]], [[IV_K]], [[IV_HO]], [[IV_WO]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE]]>
// F16-NEXT:           [[CONVERTED_VALUE:%[a-zA-Z_0-9]+]] = fpext [[VALUE]] : [[TYPE]] to [[PRINT_TYPE]]
// F16-NEXT:           memref.store [[CONVERTED_VALUE]], [[DEST]]{{\[}}[[IV_N]], [[IV_G]], [[IV_K]], [[IV_HO]], [[IV_WO]]] : memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE]]>
// F16-NEXT:         }
// F16-NEXT:       }
// F16-NEXT:     }
// F16-NEXT:   }
// F16-NEXT: }
// F16-NEXT: return

//BF16-NOT: func @convert_tensor[[N:[0-9]+]]x[[G:[0-9]+]]x[[K:[0-9]+]]x[[HO:[0-9]+]]x[[WO:[0-9]+]]([[SOURCE:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[TYPE:[a-zA-Z0-9]+]]>, [[DEST:%[a-zA-Z_0-9]+]]: memref<[[N]]x[[G]]x[[K]]x[[HO]]x[[WO]]x[[PRINT_TYPE:[a-zA-Z0-9]+]]>)
