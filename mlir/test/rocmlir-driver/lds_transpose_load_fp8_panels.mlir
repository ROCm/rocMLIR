// Verify the number and shape of amdgpu.transpose_load instructions emitted
// by the LDS-transpose fast path for the FP8/BF8 MFMA geometries on gfx950.
// Guards against panel-loop unrolling regressions (missing or duplicated
// loads).

// Unscaled 16x32 FP8 (mfma_f32_16x16x32_fp8_fp8): 8 panels.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8_fp8 \
// RUN:   -g 1 -m 64 -k 64 -n 64 --transA=true --transB=false \
// RUN:   --perf_config="v3:64,64,16,16,16,8,1,3,2,1,1" -p \
// RUN: | rocmlir-driver --kernel-pipeline=gpu --arch gfx950 \
// RUN: | FileCheck %s --check-prefix=UNSCALED_16x32

// UNSCALED_16x32-COUNT-8: amdgpu.transpose_load {{.*}} -> vector<8xf8E4M3FN>
// UNSCALED_16x32-NOT: amdgpu.transpose_load
// UNSCALED_16x32: amdgpu.mfma 16x16x32 {{.*}} : vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<4xf32>

// Unscaled 32x16 FP8 (mfma_f32_32x32x16_fp8_fp8): 4 panels.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8_fp8 \
// RUN:   -g 1 -m 64 -k 32 -n 64 --transA=true --transB=false \
// RUN:   --perf_config="v3:64,64,4,32,32,8,1,3,2,1,1" -p \
// RUN: | rocmlir-driver --kernel-pipeline=gpu --arch gfx950 \
// RUN: | FileCheck %s --check-prefix=UNSCALED_32x16

// UNSCALED_32x16-COUNT-4: amdgpu.transpose_load {{.*}} -> vector<8xf8E4M3FN>
// UNSCALED_32x16-NOT: amdgpu.transpose_load
// UNSCALED_32x16: amdgpu.mfma 32x32x16 {{.*}} : vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<16xf32>

// Scaled 16x128 FP8 (4 chained mfma 16x16x32 per scaled tile): 16 panels.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8_fp8 \
// RUN:   -g 1 -m 64 -k 128 -n 64 --transA=true --transB=false \
// RUN:   --perf_config="v3:64,64,32,16,16,8,1,3,2,1,1" -p \
// RUN: | rocmlir-driver --kernel-pipeline=gpu --arch gfx950 \
// RUN: | FileCheck %s --check-prefix=SCALED_16x128

// SCALED_16x128-COUNT-16: amdgpu.transpose_load {{.*}} -> vector<8xf8E4M3FN>
// SCALED_16x128-NOT: amdgpu.transpose_load
// SCALED_16x128: amdgpu.mfma 16x16x32 {{.*}} : vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<4xf32>

// Scaled 32x64 FP8 (4 chained mfma 32x32x16 per scaled tile): 32 panels.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t fp8_fp8 \
// RUN:   -g 1 -m 256 -k 128 -n 128 --transA=true --transB=false \
// RUN:   --perf_config="v3:256,128,32,64,32,8,1,3,2,1,1" -p \
// RUN: | rocmlir-driver --kernel-pipeline=gpu --arch gfx950 \
// RUN: | FileCheck %s --check-prefix=SCALED_32x64

// SCALED_32x64-COUNT-32: amdgpu.transpose_load {{.*}} -> vector<8xf8E4M3FN>
// SCALED_32x64-NOT: amdgpu.transpose_load
// SCALED_32x64: amdgpu.mfma 32x32x16 {{.*}} : vector<8xf8E4M3FN>, vector<8xf8E4M3FN>, vector<16xf32>

// BF8 variant of unscaled 32x16: same topology, f8E5M2 lanes.
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t bf8_bf8 \
// RUN:   -g 1 -m 64 -k 32 -n 64 --transA=true --transB=false \
// RUN:   --perf_config="v3:64,64,4,32,32,8,1,3,2,1,1" -p \
// RUN: | rocmlir-driver --kernel-pipeline=gpu --arch gfx950 \
// RUN: | FileCheck %s --check-prefix=UNSCALED_32x16_BF8

// UNSCALED_32x16_BF8-COUNT-4: amdgpu.transpose_load {{.*}} -> vector<8xf8E5M2>
// UNSCALED_32x16_BF8-NOT: amdgpu.transpose_load
// UNSCALED_32x16_BF8: amdgpu.mfma 32x32x16 {{.*}} : vector<8xf8E5M2>, vector<8xf8E5M2>, vector<16xf32>
