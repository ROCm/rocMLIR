// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx950:sramecc+:xnack- --operation gemm -g 3 -m 1024 -k 1024 -n 512 -transA=False -transB=True --accelLayoutA=True --accelLayoutB=True -pv  --perf_config=v3:32,32,2,32,32,8,1,2,2,1,1 | rocmlir-opt | FileCheck %s

// CHECK-DAG: #[[mapCUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 512 + d2)>
// CHECK-DAG: #[[$mapAHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-DAG: #[[$mapBHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$mapCHost:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[$trMapCUnmerge:.*]] = #rock.transform_map<#[[mapCUnmerge]] by [<Unmerge{3, 1024, 512} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 512] -> [1572864]>

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-SAME: (%[[aRaw:.*]]: memref<3145728xf32>, %[[bRaw:.*]]: memref<1572864xf32>, %[[cRaw:.*]]: memref<1572864xf32>)
// CHECK-SAME: attributes {enable_splitk_for_tuning, kernel 
// CHECK-SAME: mhal.arch = "[[$ARCH]]"
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] by #[[$trMapCUnmerge]]
// CHECK-NEXT: %[[a:.*]] = rock.accel_layout_transform %[[aRaw]] {isA} : memref<3145728xf32> to memref<3x1024x1024xf32>
// CHECK-NEXT: %[[b:.*]] = rock.accel_layout_transform %[[bRaw]] {transposed} : memref<1572864xf32> to memref<3x512x1024xf32>
// CHECK-NEXT: rock.gemm
// CHECK-SAME: %[[c]] =
// CHECK-SAME: %[[a]] *
// CHECK-SAME:   tr %[[b]] features = {{.*}} storeMethod = set
// CHECK-SAME: aAccelLayout
// CHECK-SAME: bAccelLayout
// CHECK-NEXT: return

// CHECK-LABEL: func.func @host_naive_gemm
// CHECK-SAME: (%[[aRaw:.*]]: memref<3145728xf32>, %[[bRaw:.*]]: memref<1572864xf32>, %[[cRaw:.*]]: memref<1572864xf32>)
// CHECK-NEXT: %[[cst:.*]] = arith.constant 0.0{{.*}} : f32
// CHECK-NEXT: linalg.fill ins(%[[cst]] : f32) outs(%[[cRaw]] : {{.*}})
// CHECK-NEXT: %[[a:.*]] = memref.expand_shape %[[aRaw]] [{{\s*}}[0, 1, 2, 3, 4, 5]]
// CHECK-SAME: into memref<3x32x64x2x32x8xf32>
// CHECK-NEXT: %[[b:.*]] = memref.expand_shape %[[bRaw]] [{{\s*}}[0, 1, 2, 3, 4, 5]]
// CHECK-SAME: into memref<3x16x64x2x32x8xf32>
// CHECK-NEXT: %[[c:.*]] = memref.expand_shape %[[cRaw]] [{{\s*}}[0, 1, 2]]
// CHECK-SAME: into memref<3x1024x512xf32>
// CHECK-NEXT: %[[trA:.*]] = memref.alloc() : memref<3x32x32x64x2x8xf32>
// CHECK-NEXT: linalg.transpose ins(%[[a]] : memref<3x32x64x2x32x8xf32>) outs(%[[trA]] : memref<3x32x32x64x2x8xf32>) permutation = [0, 1, 4, 2, 3, 5] 
// CHECK-NEXT: %[[collapsedA:.*]] = memref.collapse_shape %[[trA]] {{.*}} : memref<3x32x32x64x2x8xf32> into memref<3x1024x1024xf32>
// CHECK-NEXT: %[[trB:.*]] = memref.alloc() : memref<3x16x32x64x2x8xf32>
// CHECK-NEXT: linalg.transpose ins(%[[b]] : memref<3x16x64x2x32x8xf32>) outs(%[[trB]] : memref<3x16x32x64x2x8xf32>) permutation = [0, 1, 4, 2, 3, 5] 
// CHECK-NEXT: %[[collapsedB:.*]] = memref.collapse_shape %[[trB]] {{.*}} : memref<3x16x32x64x2x8xf32> into memref<3x512x1024xf32>
// CHECK-NEXT: linalg.generic
// CHECK-SAME: indexing_maps = [#[[$mapAHost]], #[[$mapBHost]], #[[$mapCHost]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[collapsedA]], %[[collapsedB]] : memref<{{.*}}>, memref<{{.*}}>) outs(%[[c]] : memref<{{.*}}>)
// CHECK-NEXT: (%[[aElem:.*]]: f32, %[[bElem:.*]]: f32, %[[cElem:.*]]: f32)
// CHECK-NEXT: %[[mul:.*]] = arith.mulf %[[aElem]], %[[bElem]]
// CHECK-NEXT: %[[add:.*]] = arith.addf %[[mul]], %[[cElem]]
// CHECK-NEXT: linalg.yield %[[add]]
