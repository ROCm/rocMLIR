// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,TRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,NOTRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,TRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,TRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,NOTRC
// RUN: rocmlir-gen --arch %arch --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,TRC

// NOTRA-DAG: #[[$mapA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// TRA-DAG:   #[[$mapA:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// NOTRB-DAG: #[[$mapB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// TRB-DAG:   #[[$mapB:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// NOTRC-DAG: #[[$mapC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// TRC-DAG:   #[[$mapC:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// CHECK-LABEL: func.func @rock_gemm
// NOTRA-SAME: (%[[a:.*]]: memref<3x1024x769xf32>,
// TRA-SAME:   (%[[a:.*]]: memref<3x769x1024xf32>,
// NOTRB-SAME: %[[b:.*]]: memref<3x769x512xf32>,
// TRB-SAME:   %[[b:.*]]: memref<3x512x769xf32>,
// NOTRC-SAME: %[[c:.*]]: memref<3x1024x512xf32>)
// TRC-SAME:   %[[c:.*]]: memref<3x512x1024xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: rock.gemm
// NOTRC-SAME: %[[c]] =
// TRC-SAME:   tr %[[c]] =
// NOTRA-SAME: %[[a]] *
// TRA-SAME:   tr %[[a]] *
// NOTRB-SAME: %[[b]] features = {{.*}} storeMethod = set
// TRB-SAME:   tr %[[b]] features = {{.*}} storeMethod = set
// CHECK-SAME arch = "[[$ARCH]]"
// CHECK-NEXT: return

// CHECK-LABEL: func.func @host_naive_gemm
// NOTRA-SAME: (%[[a:.*]]: memref<3x1024x769xf32>,
// TRA-SAME:   (%[[a:.*]]: memref<3x769x1024xf32>,
// NOTRB-SAME: %[[b:.*]]: memref<3x769x512xf32>,
// TRB-SAME:   %[[b:.*]]: memref<3x512x769xf32>,
// NOTRC-SAME: %[[c:.*]]: memref<3x1024x512xf32>)
// TRC-SAME:   %[[c:.*]]: memref<3x512x1024xf32>)
// CHECK-NEXT: %[[cst:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK-NEXT: linalg.fill ins(%[[cst]] : f32) outs(%[[c]] : {{.*}})
// CHECK-NEXT: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$mapA]], #[[$mapB]], #[[$mapC]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[a]], %[[b]] : memref<{{.*}}>, memref<{{.*}}>) outs(%[[c]] : memref<{{.*}}>)
// CHECK-NEXT: (%[[aElem:.*]]: f32, %[[bElem:.*]]: f32, %[[cElem:.*]]: f32)
// CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[aElem]], %[[bElem]]
// CHECK-NEXT: %[[add:.*]] = arith.addf %[[mul]], %[[cElem]]
// CHECK-NEXT: linalg.yield %[[add]]
