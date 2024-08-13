// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,TRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,NOTRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,TRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,TRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,NOTRC
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 769 -n 512 -pv -transA -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,TRC

// NOTRA-DAG: #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 769 + d2)>
// TRA-DAG:   #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 1024 + d2)>
// NOTRB-DAG: #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 769 + d1) * 512 + d2)>
// TRB-DAG:   #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 769 + d2)>
// NOTRC-DAG: #[[mapCUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 512 + d2)>
// TRC-DAG:   #[[mapCUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 1024 + d2)>
// NOTRA-DAG: #[[$mapAHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// TRA-DAG:   #[[$mapAHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>
// NOTRB-DAG: #[[$mapBHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// TRB-DAG:   #[[$mapBHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// NOTRC-DAG: #[[$mapCHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// TRC-DAG:   #[[$mapCHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>
// NOTRA-DAG: #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 1024, 769} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 769] -> [2362368]>
// TRA-DAG:   #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 769, 1024} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 1024] -> [2362368]>
// NOTRB-DAG: #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 769, 512} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 769, 512] -> [1181184]>
// TRB-DAG:   #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 512, 769} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 512, 769] -> [1181184]>
// NOTRC-DAG: #[[$trMapCUnmerge:.*]] = #rock.transform_map<#[[mapCUnmerge]] by [<Unmerge{3, 1024, 512} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 512] -> [1572864]>
// TRC-DAG:   #[[$trMapCUnmerge:.*]] = #rock.transform_map<#[[mapCUnmerge]] by [<Unmerge{3, 512, 1024} ["g", "n", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 512, 1024] -> [1572864]>

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-SAME: (%[[aRaw:.*]]: memref<2362368xf32>, %[[bRaw:.*]]: memref<1181184xf32>, %[[cRaw:.*]]: memref<1572864xf32>)
// CHECK-SAME: attributes {enable_splitk_for_tuning, kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[a:.*]] = rock.transform %[[aRaw]] by #[[$trMapAUnmerge]]
// CHECK-NEXT: %[[b:.*]] = rock.transform %[[bRaw]] by #[[$trMapBUnmerge]]
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] by #[[$trMapCUnmerge]]
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
// CHECK-SAME: (%[[aRaw:.*]]: memref<2362368xf32>, %[[bRaw:.*]]: memref<1181184xf32>, %[[cRaw:.*]]: memref<1572864xf32>)
// CHECK-NEXT: %[[cst:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK-NEXT: linalg.fill ins(%[[cst]] : f32) outs(%[[cRaw]] : {{.*}})
// CHECK-NEXT: %[[a:.*]] = memref.expand_shape %[[aRaw]] [{{\s*}}[0, 1, 2]]
// NOTRA-SAME: into memref<3x1024x769xf32>
// TRA-SAME:   into memref<3x769x1024xf32>
// CHECK-NEXT: %[[b:.*]] = memref.expand_shape %[[bRaw]] [{{\s*}}[0, 1, 2]]
// NOTRB-SAME: into memref<3x769x512xf32>
// TRB-SAME:   into memref<3x512x769xf32>
// CHECK-NEXT: %[[c:.*]] = memref.expand_shape %[[cRaw]] [{{\s*}}[0, 1, 2]]
// NOTRC-SAME: into memref<3x1024x512xf32>
// TRC-SAME:   into memref<3x512x1024xf32>
// CHECK-NEXT: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$mapAHost]], #[[$mapBHost]], #[[$mapCHost]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[a]], %[[b]] : memref<{{.*}}>, memref<{{.*}}>) outs(%[[c]] : memref<{{.*}}>)
// CHECK-NEXT: (%[[aElem:.*]]: f32, %[[bElem:.*]]: f32, %[[cElem:.*]]: f32)
// CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[aElem]], %[[bElem]]
// CHECK-NEXT: %[[add:.*]] = arith.addf %[[mul]], %[[cElem]]
// CHECK-NEXT: linalg.yield %[[add]]
