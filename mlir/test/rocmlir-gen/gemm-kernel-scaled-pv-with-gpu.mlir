// The extra rocmlir-opt calls check IR validity

// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm --schedule_version 2 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,NOTRC,SCHEDV2
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,NOTRB,TRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,NOTRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,NOTRA,TRB,TRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transA | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,NOTRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transA -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,NOTRB,TRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transA -transB | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,NOTRC
// RUN: rocmlir-gen --arch gfx950 --operation gemm -g 3 -m 1024 -k 768 -n 512 -pv_with_gpu -t f4E2M1FN -out_dtype f32 --scaledGemm -transA -transB -transC | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK,TRA,TRB,TRC

// NOTRA-DAG: #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 768 + d2)>
// TRA-DAG:   #[[mapAUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 1024 + d2)>
// NOTRB-DAG: #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 768 + d1) * 512 + d2)>
// TRB-DAG:   #[[mapBUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 768 + d2)>
// NOTRC-DAG: #[[mapCUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 1024 + d1) * 512 + d2)>
// TRC-DAG:   #[[mapCUnmerge:.*]] = affine_map<(d0, d1, d2) -> ((d0 * 512 + d1) * 1024 + d2)>
// NOTRA-DAG: #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 1024, 768} ["g", "m", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 768] -> [2359296]>
// TRA-DAG:   #[[$trMapAUnmerge:.*]] = #rock.transform_map<#[[mapAUnmerge]] by [<Unmerge{3, 768, 1024} ["g", "k", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 1024] -> [2359296]>
// NOTRB-DAG: #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 768, 512} ["g", "k", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 768, 512] -> [1179648]>
// TRB-DAG:   #[[$trMapBUnmerge:.*]] = #rock.transform_map<#[[mapBUnmerge]] by [<Unmerge{3, 512, 768} ["g", "n", "k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 512, 768] -> [1179648]>
// NOTRC-DAG: #[[$trMapCUnmerge:.*]] = #rock.transform_map<#[[mapCUnmerge]] by [<Unmerge{3, 1024, 512} ["g", "m", "n"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 1024, 512] -> [1572864]>
// TRC-DAG:   #[[$trMapCUnmerge:.*]] = #rock.transform_map<#[[mapCUnmerge]] by [<Unmerge{3, 512, 1024} ["g", "n", "m"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [3, 512, 1024] -> [1572864]>

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// CHECK-LABEL: func.func @rock_gemm
// CHECK-SAME: (%[[aRaw:.*]]: memref<2359296xf4E2M1FN>, %[[bRaw:.*]]: memref<1179648xf4E2M1FN>, %[[cRaw:.*]]: memref<1572864xf32>, %[[scaleARaw:.*]]: memref<73728xf8E8M0FNU>, %[[scaleBRaw:.*]]: memref<36864xf8E8M0FNU>)
// CHECK-SAME: attributes {enable_splitk_for_tuning, kernel 
// CHECK-SAME: mhal.arch = "[[$ARCH]]"
// SCHEDV2-SAME: schedule_version = #rock.schedule_version<2>
// CHECK: rock.gemm
// CHECK-SAME: scaled by
// CHECK-SAME: scaled by
// CHECK-SAME: storeMethod = set
// CHECK-SAME: memref<{{.*}}xf32> = memref<{{.*}}xf4E2M1FN> scaled by memref<{{.*}}xf8E8M0FNU> * memref<{{.*}}xf4E2M1FN> scaled by memref<{{.*}}xf8E8M0FNU>
// CHECK: return

// CHECK-LABEL: func.func @main
// CHECK: call @rock_gemm_gpu
// CHECK: call @rock_gemm_ver_gpu
// CHECK: call @rock_gemm_verify2
// CHECK: return

// CHECK-LABEL: func.func @rock_gemm_ver
// CHECK-SAME: (%[[aRawVer:.*]]: memref<2359296xf32>, %[[bRawVer:.*]]: memref<1179648xf32>, %[[cRawVer:.*]]: memref<1572864xf32>, %[[scaleARawVer:.*]]: memref<73728xf32>, %[[scaleBRawVer:.*]]: memref<36864xf32>)
// CHECK-SAME: attributes {enable_splitk_for_tuning, kernel 
// CHECK-SAME: mhal.arch = "[[$ARCH]]"
// CHECK-SAME: num_cu = {{[0-9]+}} : i64
// CHECK: %[[aVer:.*]] = rock.transform %[[aRawVer]] by
// NOTRA-SAME: memref<2359296xf32> to memref<3x1024x768xf32>
// TRA-SAME:   memref<2359296xf32> to memref<3x768x1024xf32>
// CHECK: %[[bVer:.*]] = rock.transform %[[bRawVer]] by
// NOTRB-SAME: memref<1179648xf32> to memref<3x768x512xf32>
// TRB-SAME:   memref<1179648xf32> to memref<3x512x768xf32>
// CHECK: %[[cVer:.*]] = rock.transform %[[cRawVer]] by
// NOTRC-SAME: memref<1572864xf32> to memref<3x1024x512xf32>
// TRC-SAME:   memref<1572864xf32> to memref<3x512x1024xf32>
// CHECK: %[[scaleAVer:.*]] = rock.transform %[[scaleARawVer]] by
// CHECK-SAME: memref<73728xf32> to memref<3x1024x24xf32>
// CHECK: %[[scaleBVer:.*]] = rock.transform %[[scaleBRawVer]] by
// CHECK-SAME: memref<36864xf32> to memref<3x24x512xf32>
// CHECK: memref.alloc() : memref<3x{{.*}}x{{.*}}xf32>
// CHECK: memref.alloc() : memref<3x{{.*}}x{{.*}}xf32>
// CHECK: linalg.generic {indexing_maps = [{{.*}}, {{.*}}, {{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins({{.*}}, {{.*}} : memref<3x{{.*}}x{{.*}}xf32>, memref<3x{{.*}}x{{.*}}xf32>) outs({{.*}} : memref<3x{{.*}}x{{.*}}xf32>) {
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT: %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT: linalg.yield %{{.*}} : f32
// CHECK-NEXT: }
// CHECK: linalg.generic {indexing_maps = [{{.*}}, {{.*}}, {{.*}}], iterator_types = ["parallel", "parallel", "parallel"]} ins({{.*}}, {{.*}} : memref<3x{{.*}}x{{.*}}xf32>, memref<3x{{.*}}x{{.*}}xf32>) outs({{.*}} : memref<3x{{.*}}x{{.*}}xf32>) {
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT: %{{.*}} = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT: linalg.yield %{{.*}} : f32
// CHECK-NEXT: }
// CHECK: rock.gemm
// CHECK-SAME: features = {{.*}} storeMethod = set
// CHECK: return

// CHECK-LABEL: func.func @rock_gemm_ver_gpu
// CHECK-SAME: (%[[aHost:.*]]: memref<2359296xf32>, %[[bHost:.*]]: memref<1179648xf32>, %[[cHost:.*]]: memref<1572864xf32>, %[[scaleAHost:.*]]: memref<73728xf32>, %[[scaleBHost:.*]]: memref<36864xf32>)
// CHECK-NEXT: %[[aGpu:.*]] = gpu.alloc  () : memref<2359296xf32>
// CHECK-NEXT: gpu.memcpy  %[[aGpu]], %[[aHost]] : memref<2359296xf32>, memref<2359296xf32>
// CHECK-NEXT: %[[bGpu:.*]] = gpu.alloc  () : memref<1179648xf32>
// CHECK-NEXT: gpu.memcpy  %[[bGpu]], %[[bHost]] : memref<1179648xf32>, memref<1179648xf32>
// CHECK-NEXT: %[[cGpu:.*]] = gpu.alloc  () : memref<1572864xf32>
// CHECK-NEXT: gpu.memcpy  %[[cGpu]], %[[cHost]] : memref<1572864xf32>, memref<1572864xf32>
// CHECK-NEXT: %[[scaleAGpu:.*]] = gpu.alloc  () : memref<73728xf32>
// CHECK-NEXT: gpu.memcpy  %[[scaleAGpu]], %[[scaleAHost]] : memref<73728xf32>, memref<73728xf32>
// CHECK-NEXT: %[[scaleBGpu:.*]] = gpu.alloc  () : memref<36864xf32>
// CHECK-NEXT: gpu.memcpy  %[[scaleBGpu]], %[[scaleBHost]] : memref<36864xf32>, memref<36864xf32>
// CHECK-NEXT: call @rock_gemm_ver(%[[aGpu]], %[[bGpu]], %[[cGpu]], %[[scaleAGpu]], %[[scaleBGpu]]) : (memref<2359296xf32>, memref<1179648xf32>, memref<1572864xf32>, memref<73728xf32>, memref<36864xf32>) -> ()
// CHECK-NEXT: gpu.memcpy  %[[aHost]], %[[aGpu]] : memref<2359296xf32>, memref<2359296xf32>
// CHECK-NEXT: gpu.dealloc  %[[aGpu]] : memref<2359296xf32>
// CHECK-NEXT: gpu.memcpy  %[[bHost]], %[[bGpu]] : memref<1179648xf32>, memref<1179648xf32>
// CHECK-NEXT: gpu.dealloc  %[[bGpu]] : memref<1179648xf32>
// CHECK-NEXT: gpu.memcpy  %[[cHost]], %[[cGpu]] : memref<1572864xf32>, memref<1572864xf32>
// CHECK-NEXT: gpu.dealloc  %[[cGpu]] : memref<1572864xf32>
// CHECK-NEXT: gpu.memcpy  %[[scaleAHost]], %[[scaleAGpu]] : memref<73728xf32>, memref<73728xf32>
// CHECK-NEXT: gpu.dealloc  %[[scaleAGpu]] : memref<73728xf32>
// CHECK-NEXT: gpu.memcpy  %[[scaleBHost]], %[[scaleBGpu]] : memref<36864xf32>, memref<36864xf32>
// CHECK-NEXT: gpu.dealloc  %[[scaleBGpu]] : memref<36864xf32>
// CHECK-NEXT: return

// CHECK-LABEL: func.func @rock_gemm_verify2
// CHECK-SAME: (%[[result:.*]]: memref<1572864xf32>, %[[expected:.*]]: memref<1572864xf32>)
// CHECK-NEXT: %[[false:.*]] = arith.constant false
// CHECK-NEXT: %[[true:.*]] = arith.constant true
// CHECK-NEXT: %[[epsilon:.*]] = arith.constant {{.*}} : f32
// CHECK-NEXT: %[[maxval:.*]] = arith.constant {{.*}} : f32
// CHECK-NEXT: %[[tolerance:.*]] = arith.constant {{.*}} : f32
// CHECK-NEXT: %[[flag:.*]] = arith.constant 1 : i8
// CHECK-NEXT: %[[castResult:.*]] = memref.cast %[[result]] : memref<1572864xf32> to memref<?xf32>
// CHECK-NEXT: %[[castExpected:.*]] = memref.cast %[[expected]] : memref<1572864xf32> to memref<?xf32>
// CHECK-NEXT: call @mcpuVerifyFloat(%[[castResult]], %[[castExpected]], %[[tolerance]], %[[maxval]], %[[epsilon]], %[[flag]], %[[true]], %[[false]]) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1) -> ()
// CHECK-NEXT: return

// CHECK: func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1, i1)

// CHECK-LABEL: func.func @rock_gemm_gpu
// CHECK-SAME: (%[[aHostMain:.*]]: memref<2359296xf4E2M1FN>, %[[bHostMain:.*]]: memref<1179648xf4E2M1FN>, %[[cHostMain:.*]]: memref<1572864xf32>, %[[scaleAHostMain:.*]]: memref<73728xf8E8M0FNU>, %[[scaleBHostMain:.*]]: memref<36864xf8E8M0FNU>)
// CHECK-NEXT: %[[aGpuMain:.*]] = gpu.alloc  () : memref<2359296xf4E2M1FN>
// CHECK-NEXT: gpu.memcpy  %[[aGpuMain]], %[[aHostMain]] : memref<2359296xf4E2M1FN>, memref<2359296xf4E2M1FN>
// CHECK-NEXT: %[[bGpuMain:.*]] = gpu.alloc  () : memref<1179648xf4E2M1FN>
// CHECK-NEXT: gpu.memcpy  %[[bGpuMain]], %[[bHostMain]] : memref<1179648xf4E2M1FN>, memref<1179648xf4E2M1FN>
// CHECK-NEXT: %[[cGpuMain:.*]] = gpu.alloc  () : memref<1572864xf32>
// CHECK-NEXT: gpu.memcpy  %[[cGpuMain]], %[[cHostMain]] : memref<1572864xf32>, memref<1572864xf32>
// CHECK-NEXT: %[[scaleAGpuMain:.*]] = gpu.alloc  () : memref<73728xf8E8M0FNU>
// CHECK-NEXT: gpu.memcpy  %[[scaleAGpuMain]], %[[scaleAHostMain]] : memref<73728xf8E8M0FNU>, memref<73728xf8E8M0FNU>
// CHECK-NEXT: %[[scaleBGpuMain:.*]] = gpu.alloc  () : memref<36864xf8E8M0FNU>
// CHECK-NEXT: gpu.memcpy  %[[scaleBGpuMain]], %[[scaleBHostMain]] : memref<36864xf8E8M0FNU>, memref<36864xf8E8M0FNU>
// CHECK-NEXT: call @rock_gemm(%[[aGpuMain]], %[[bGpuMain]], %[[cGpuMain]], %[[scaleAGpuMain]], %[[scaleBGpuMain]]) : (memref<2359296xf4E2M1FN>, memref<1179648xf4E2M1FN>, memref<1572864xf32>, memref<73728xf8E8M0FNU>, memref<36864xf8E8M0FNU>) -> ()
// CHECK-NEXT: gpu.memcpy  %[[aHostMain]], %[[aGpuMain]] : memref<2359296xf4E2M1FN>, memref<2359296xf4E2M1FN>
// CHECK-NEXT: gpu.dealloc  %[[aGpuMain]] : memref<2359296xf4E2M1FN>
// CHECK-NEXT: gpu.memcpy  %[[bHostMain]], %[[bGpuMain]] : memref<1179648xf4E2M1FN>, memref<1179648xf4E2M1FN>
// CHECK-NEXT: gpu.dealloc  %[[bGpuMain]] : memref<1179648xf4E2M1FN>
// CHECK-NEXT: gpu.memcpy  %[[cHostMain]], %[[cGpuMain]] : memref<1572864xf32>, memref<1572864xf32>
// CHECK-NEXT: gpu.dealloc  %[[cGpuMain]] : memref<1572864xf32>
// CHECK-NEXT: gpu.memcpy  %[[scaleAHostMain]], %[[scaleAGpuMain]] : memref<73728xf8E8M0FNU>, memref<73728xf8E8M0FNU>
// CHECK-NEXT: gpu.dealloc  %[[scaleAGpuMain]] : memref<73728xf8E8M0FNU>
// CHECK-NEXT: gpu.memcpy  %[[scaleBHostMain]], %[[scaleBGpuMain]] : memref<36864xf8E8M0FNU>, memref<36864xf8E8M0FNU>
// CHECK-NEXT: gpu.dealloc  %[[scaleBGpuMain]] : memref<36864xf8E8M0FNU>
// CHECK-NEXT: return

