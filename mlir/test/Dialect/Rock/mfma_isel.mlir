// 32x32x2xf32 mfma

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:128,128,4,64,3202,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1
// CHECK_1: amdgpu.mfma
// CHECK_1-SAME: k = 2 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:128,64,4,64,3202,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2
// CHECK_2: amdgpu.mfma
// CHECK_2-SAME: k = 2 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:64,128,4,32,3202,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3
// CHECK_3: amdgpu.mfma
// CHECK_3-SAME: k = 2 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:64,64,4,32,3202,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4
// CHECK_4: amdgpu.mfma
// CHECK_4-SAME: k = 2 : i32, m = 32 : i32, n = 32

// 16x16x4xf32 mfma

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:128,128,4,64,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5
// CHECK_5: amdgpu.mfma
// CHECK_5-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:128,64,4,64,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6
// CHECK_6: amdgpu.mfma
// CHECK_6-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:128,32,4,64,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7
// CHECK_7: amdgpu.mfma
// CHECK_7-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:64,128,4,32,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8
// CHECK_8: amdgpu.mfma
// CHECK_8-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:64,64,4,32,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9
// CHECK_9: amdgpu.mfma
// CHECK_9-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:64,32,4,32,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10
// CHECK_10: amdgpu.mfma
// CHECK_10-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:32,128,4,16,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11
// CHECK_11: amdgpu.mfma
// CHECK_11-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:32,64,4,16,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12
// CHECK_12: amdgpu.mfma
// CHECK_12-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:32,32,4,16,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13
// CHECK_13: amdgpu.mfma
// CHECK_13-SAME: k = 4 : i32, m = 16 : i32, n = 16

// 4x4x1xf32 mfma

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:8,128,4,4,401,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_14
// CHECK_14: amdgpu.mfma
// CHECK_14-SAME: k = 1 : i32, m = 4 : i32, n = 4

// RUN: rocmlir-gen -operation gemm -t f32 --perf_config=v2:16,128,4,8,401,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_15
// CHECK_15: amdgpu.mfma
// CHECK_15-SAME: k = 1 : i32, m = 4 : i32, n = 4

// 32x32x8xf16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,128,4,64,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_16
// CHECK_16: amdgpu.mfma
// CHECK_16-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,64,4,64,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_17
// CHECK_17: amdgpu.mfma
// CHECK_17-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,128,4,32,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_18
// CHECK_18: amdgpu.mfma
// CHECK_18-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,64,4,32,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_19
// CHECK_19: amdgpu.mfma
// CHECK_19-SAME: k = 8 : i32, m = 32 : i32, n = 32

// 4x4x4xf16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:8,128,4,4,404,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_20
// CHECK_20: amdgpu.mfma
// CHECK_20-SAME: k = 4 : i32, m = 4 : i32, n = 4

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:16,128,4,8,404,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_21
// CHECK_21: amdgpu.mfma
// CHECK_21-SAME: k = 4 : i32, m = 4 : i32, n = 4

// 16x16x16xf16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,128,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_22
// CHECK_22: amdgpu.mfma
// CHECK_22-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,64,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_23
// CHECK_23: amdgpu.mfma
// CHECK_23-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,32,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_24
// CHECK_24: amdgpu.mfma
// CHECK_24-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,128,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_25
// CHECK_25: amdgpu.mfma
// CHECK_25-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,64,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_26
// CHECK_26: amdgpu.mfma
// CHECK_26-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,32,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_27
// CHECK_27: amdgpu.mfma
// CHECK_27-SAME: k = 16 : i32, m = 16 : i32, n = 16

// 16x16x4xf16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:128,128,4,64,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_28
// CHECK_28: amdgpu.mfma
// CHECK_28-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:64,128,4,32,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_29
// CHECK_29: amdgpu.mfma
// CHECK_29-SAME: k = 4 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t f16 --perf_config=v2:32,128,4,16,1604,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_30
// CHECK_30: amdgpu.mfma
// CHECK_30-SAME: k = 4 : i32, m = 16 : i32, n = 16

// 32x32x8xi8

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,128,4,64,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_31
// CHECK_31: amdgpu.mfma
// CHECK_31-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,64,4,64,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_32
// CHECK_32: amdgpu.mfma
// CHECK_32-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,128,4,32,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_33
// CHECK_33: amdgpu.mfma
// CHECK_33-SAME: k = 8 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,64,4,32,3208,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_34
// CHECK_34: amdgpu.mfma
// CHECK_34-SAME: k = 8 : i32, m = 32 : i32, n = 32

// 16x16x16xi8

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,128,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_35
// CHECK_35: amdgpu.mfma
// CHECK_35-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,64,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_36
// CHECK_36: amdgpu.mfma
// CHECK_36-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,32,4,64,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_37
// CHECK_37: amdgpu.mfma
// CHECK_37-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,128,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_38
// CHECK_38: amdgpu.mfma
// CHECK_38-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,64,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_39
// CHECK_39: amdgpu.mfma
// CHECK_39-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,32,4,32,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_40
// CHECK_40: amdgpu.mfma
// CHECK_40-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,128,4,16,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_41
// CHECK_41: amdgpu.mfma
// CHECK_41-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,64,4,16,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_42
// CHECK_42: amdgpu.mfma
// CHECK_42-SAME: k = 16 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,32,4,16,1616,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx90a | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_43
// CHECK_43: amdgpu.mfma
// CHECK_43-SAME: k = 16 : i32, m = 16 : i32, n = 16
