// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t f16 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,F16,FLOAT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t i8 -c_dtype i8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,I8-I8,INT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8,FLOAT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv | FileCheck %s --check-prefixes=CHECK,FP8-BF8,FLOAT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -ta fp8 -tb bf8 -tc f32 -g 3 -m 1024 -k 769 -n 512 -pv --force-f8-types=ocp | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP,FLOAT
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- --operation gemm -t fp8_bf8 -g 3 -m 1024 -k 769 -n 512 -pv --force-f8-types=ocp | FileCheck %s --check-prefixes=CHECK,FP8-BF8-OCP,FLOAT

// CHECK-LABEL: func @rock_gemm
// F16-SAME: (%{{.*}}: memref<2362368xf16>, %{{.*}}: memref<1181184xf16>, %{{.*}}: memref<1572864xf16>)
// I8-SAME: (%{{.*}}: memref<2362368xi8>, %{{.*}}: memref<1181184xi8>, %{{.*}}: memref<1572864xi32>)
// I8-I8-SAME: (%{{.*}}: memref<2362368xi8>, %{{.*}}: memref<1181184xi8>, %{{.*}}: memref<1572864xi8>)
// FP8-BF8-SAME: (%{{.*}}: memref<2362368xf8E4M3FNUZ>, %{{.*}}: memref<1181184xf8E5M2FNUZ>, %{{.*}}: memref<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: memref<2362368xf8E4M3FN>, %{{.*}}: memref<1181184xf8E5M2>, %{{.*}}: memref<1572864xf32>)

// CHECK-LABEL: func @host_naive_gemm
// F16-SAME: (%{{.*}}: memref<2362368xf16>, %{{.*}}: memref<1181184xf16>, %{{.*}}: memref<1572864xf16>)
// I8-SAME: (%{{.*}}: memref<2362368xi8>, %{{.*}}: memref<1181184xi8>, %{{.*}}: memref<1572864xi64>)
// I8-I8-SAME: (%{{.*}}: memref<2362368xi8>, %{{.*}}: memref<1181184xi8>, %{{.*}}: memref<1572864xi64>)
// FP8-BF8-SAME: (%{{.*}}: memref<2362368xf8E4M3FNUZ>, %{{.*}}: memref<1181184xf8E5M2FNUZ>, %{{.*}}: memref<1572864xf32>)
// FP8-BF8-OCP-SAME: (%{{.*}}: memref<2362368xf8E4M3FN>, %{{.*}}: memref<1181184xf8E5M2>, %{{.*}}: memref<1572864xf32>)

// FLOAT: arith.mulf
// FLOAT-NEXT: arith.addf
// INT: arith.extsi
// INT-NEXT: arith.extsi
// INT-NEXT: arith.muli
// INT-NEXT: arith.addi
// CHECK-NEXT: linalg.yield
