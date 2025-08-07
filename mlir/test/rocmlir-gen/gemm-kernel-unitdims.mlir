// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f32 -D\$OTYPE=f32 --check-prefixes=ALLUNIT
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 2 -m 1 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f32 -D\$OTYPE=f32 --check-prefixes=ONLYG
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 2 -k 1 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f32 -D\$OTYPE=f32 --check-prefixes=ONLYM
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 2 -n 1 | rocmlir-opt --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f32 -D\$OTYPE=f32 --check-prefixes=ONLYK
// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm -g 1 -m 1 -k 1 -n 2 | rocmlir-opt --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f32 -D\$OTYPE=f32 --check-prefixes=ONLYN

// ALLUNIT-LABEL: module
// ALLUNIT-NEXT: func.func @rock_gemm
// ALLUNIT-SAME: ([[arg0:%.+]]: memref<1x[[$ITYPE]]>, [[arg1:%.+]]: memref<1x[[$ITYPE]]>, [[arg2:%.+]]: memref<1x[[$OTYPE]]>)
// ALLUNIT-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "{{.*}}", num_cu = {{.*}}}
// ALLUNIT-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ALLUNIT-SAME: <Unmerge{1} ["k"]
// ALLUNIT-SAME: <AddDim{1} ["g"]
// ALLUNIT-SAME: <AddDim{1} ["m"]
// ALLUNIT-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ALLUNIT-SAME: <Unmerge{1} ["n"]
// ALLUNIT-SAME: <AddDim{1} ["g"]
// ALLUNIT-SAME: <AddDim{1} ["k"]
// ALLUNIT-NEXT: [[gemmOut:%.+]] = rock.transform [[arg2]]
// ALLUNIT-SAME: <Unmerge{1} ["n"]
// ALLUNIT-SAME: <AddDim{1} ["g"]
// ALLUNIT-SAME: <AddDim{1} ["m"]
// ALLUNIT-NEXT: rock.gemm [[gemmOut]] = [[gemmA]] * [[gemmB]] features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x1x1x[[$OTYPE]]> = memref<1x1x1x[[$ITYPE]]> * memref<1x1x1x[[$ITYPE]]>

// ONLYG-LABEL: module
// ONLYG-NEXT: func.func @rock_gemm
// ONLYG-SAME: ([[arg0:%.+]]: memref<2x[[$ITYPE]]>, [[arg1:%.+]]: memref<2x[[$ITYPE]]>, [[arg2:%.+]]: memref<2x[[$OTYPE]]>)
// ONLYG-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "{{.*}}", num_cu = {{.*}}}
// ONLYG-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYG-SAME: <Unmerge{2} ["g"]
// ONLYG-SAME: <AddDim{1} ["m"]
// ONLYG-SAME: <AddDim{1} ["k"]
// ONLYG-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYG-SAME: <Unmerge{2} ["g"]
// ONLYG-SAME: <AddDim{1} ["k"]
// ONLYG-SAME: <AddDim{1} ["n"]
// ONLYG-NEXT: [[gemmOut:%.+]] = rock.transform [[arg2]]
// ONLYG-SAME: <Unmerge{2} ["g"]
// ONLYG-SAME: <AddDim{1} ["m"]
// ONLYG-SAME: <AddDim{1} ["n"]
// ONLYG-NEXT: rock.gemm [[gemmOut]] = [[gemmA]] * [[gemmB]] features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<2x1x1x[[$OTYPE]]> = memref<2x1x1x[[$ITYPE]]> * memref<2x1x1x[[$ITYPE]]>

// ONLYM-LABEL: module
// ONLYM-NEXT: func.func @rock_gemm
// ONLYM-SAME: ([[arg0:%.+]]: memref<2x[[$ITYPE]]>, [[arg1:%.+]]: memref<1x[[$ITYPE]]>, [[arg2:%.+]]: memref<2x[[$OTYPE]]>)
// ONLYM-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "{{.*}}", num_cu = {{.*}}}
// ONLYM-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYM-SAME: <Unmerge{2} ["m"]
// ONLYM-SAME: <AddDim{1} ["g"]
// ONLYM-SAME: <AddDim{1} ["k"]
// ONLYM-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYM-SAME: <Unmerge{1} ["n"]
// ONLYM-SAME: <AddDim{1} ["g"]
// ONLYM-SAME: <AddDim{1} ["k"]
// ONLYM-NEXT: [[gemmOut:%.+]] = rock.transform [[arg2]]
// ONLYM-SAME: <Unmerge{2} ["m"]
// ONLYM-SAME: <AddDim{1} ["g"]
// ONLYM-SAME: <AddDim{1} ["n"]
// ONLYM-NEXT: rock.gemm [[gemmOut]] = [[gemmA]] * [[gemmB]] features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x2x1x[[$OTYPE]]> = memref<1x2x1x[[$ITYPE]]> * memref<1x1x1x[[$ITYPE]]>

// ONLYK-LABEL: module
// ONLYK-NEXT: func.func @rock_gemm
// ONLYK-SAME: ([[arg0:%.+]]: memref<2x[[$ITYPE]]>, [[arg1:%.+]]: memref<2x[[$ITYPE]]>, [[arg2:%.+]]: memref<1x[[$OTYPE]]>)
// ONLYK-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "{{.*}}", num_cu = {{.*}}}
// ONLYK-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYK-SAME: <Unmerge{2} ["k"]
// ONLYK-SAME: <AddDim{1} ["g"]
// ONLYK-SAME: <AddDim{1} ["m"]
// ONLYK-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYK-SAME: <Unmerge{2} ["k"]
// ONLYK-SAME: <AddDim{1} ["g"]
// ONLYK-SAME: <AddDim{1} ["n"]
// ONLYK-NEXT: [[gemmOut:%.+]] = rock.transform [[arg2]]
// ONLYK-SAME: <Unmerge{1} ["n"]
// ONLYK-SAME: <AddDim{1} ["g"]
// ONLYK-SAME: <AddDim{1} ["m"]
// ONLYK-NEXT: rock.gemm [[gemmOut]] = [[gemmA]] * [[gemmB]] features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x1x1x[[$OTYPE]]> = memref<1x1x2x[[$ITYPE]]> * memref<1x2x1x[[$ITYPE]]>

// ONLYN-LABEL: module
// ONLYN-NEXT: func.func @rock_gemm
// ONLYN-SAME: ([[arg0:%.+]]: memref<1x[[$ITYPE]]>, [[arg1:%.+]]: memref<2x[[$ITYPE]]>, [[arg2:%.+]]: memref<2x[[$OTYPE]]>)
// ONLYN-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "{{.*}}", num_cu = {{.*}}}
// ONLYN-NEXT: [[gemmA:%.+]] = rock.transform [[arg0]]
// ONLYN-SAME: <Unmerge{1} ["k"]
// ONLYN-SAME: <AddDim{1} ["g"]
// ONLYN-SAME: <AddDim{1} ["m"]
// ONLYN-NEXT: [[gemmB:%.+]] = rock.transform [[arg1]]
// ONLYN-SAME: <Unmerge{2} ["n"]
// ONLYN-SAME: <AddDim{1} ["g"]
// ONLYN-SAME: <AddDim{1} ["k"]
// ONLYN-NEXT: [[gemmOut:%.+]] = rock.transform [[arg2]]
// ONLYN-SAME: <Unmerge{2} ["n"]
// ONLYN-SAME: <AddDim{1} ["g"]
// ONLYN-SAME: <AddDim{1} ["m"]
// ONLYN-NEXT: rock.gemm [[gemmOut]] = [[gemmA]] * [[gemmB]] features = mfma|dot|atomic_add|atomic_add_f16 storeMethod = set : memref<1x1x2x[[$OTYPE]]> = memref<1x1x1x[[$ITYPE]]> * memref<1x1x2x[[$ITYPE]]>
