// 32x32x16xi8 (gfx940+)

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,128,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:128,128,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:128,128,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:128,128,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:128,128,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_1_BF8_BF8
// CHECK_1_I8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_1_F8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_1_F8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_1_BF8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_1_BF8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,64,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:128,64,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:128,64,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:128,64,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:128,64,4,64,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_2_BF8_BF8
// CHECK_2_I8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_2_F8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_2_F8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_2_BF8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_2_BF8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,128,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:64,128,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:64,128,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:64,128,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:64,128,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_3_BF8_BF8
// CHECK_3_I8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_3_F8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_3_F8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_3_BF8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_3_BF8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,64,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:64,64,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:64,64,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:64,64,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:64,64,4,32,3216,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_4_BF8_BF8
// CHECK_4_I8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_4_F8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_4_F8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_4_BF8_F8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32
// CHECK_4_BF8_BF8: amdgpu.mfma {{.*}} k = 16 : i32, m = 32 : i32, n = 32

// 16x16x32xi8 (gfx940+)

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,128,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:128,128,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:128,128,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:128,128,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:128,128,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_5_BF8_BF8
// CHECK_5_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_5_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_5_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_5_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_5_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,64,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:128,64,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:128,64,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:128,64,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:128,64,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_6_BF8_BF8
// CHECK_6_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_6_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_6_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_6_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_6_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:128,32,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:128,32,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:128,32,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:128,32,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:128,32,4,64,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_7_BF8_BF8
// CHECK_7_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_7_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_7_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_7_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_7_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,128,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:64,128,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:64,128,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:64,128,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:64,128,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_8_BF8_BF8
// CHECK_8_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_8_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_8_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_8_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_8_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,64,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:64,64,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:64,64,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:64,64,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:64,64,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_9_BF8_BF8
// CHECK_9_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_9_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_9_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_9_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_9_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:64,32,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:64,32,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:64,32,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:64,32,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:64,32,4,32,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_10_BF8_BF8
// CHECK_10_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_10_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_10_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_10_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_10_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,128,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:32,128,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:32,128,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:32,128,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:32,128,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_11_BF8_BF8
// CHECK_11_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_11_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_11_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_11_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_11_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,64,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:32,64,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:32,64,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:32,64,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:32,64,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_12_BF8_BF8
// CHECK_12_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_12_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_12_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_12_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_12_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16

// RUN: rocmlir-gen -operation gemm -t i8 --perf_config=v2:32,32,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13_I8
// RUN: rocmlir-gen -operation gemm -t fp8_fp8 --perf_config=v2:32,32,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13_F8_F8
// RUN: rocmlir-gen -operation gemm -t fp8_bf8 --perf_config=v2:32,32,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13_F8_BF8
// RUN: rocmlir-gen -operation gemm -t bf8_fp8 --perf_config=v2:32,32,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13_BF8_F8
// RUN: rocmlir-gen -operation gemm -t bf8_bf8 --perf_config=v2:32,32,4,16,1632,16,1,1,1 -g 1 -m 128 -k 64 -n 256 --arch gfx942 | rocmlir-driver -kernel-pipeline=gpu | FileCheck %s --check-prefix=CHECK_13_BF8_BF8
// CHECK_13_I8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_13_F8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_13_F8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_13_BF8_F8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
// CHECK_13_BF8_BF8: amdgpu.mfma {{.*}} k = 32 : i32, m = 16 : i32, n = 16
