// Checks that an i4 (iu4) GEMM on gfx1201 lowers through rock to the
// 16x16x32 iu4 WMMA instruction.
// RUN: rocmlir-gen --operation gemm --arch gfx1201 -t i4 -wmma=on -g 1 -m 256 -n 256 -k 256 | rocmlir-driver --arch gfx1201 --kernel-pipeline=gpu | FileCheck %s --check-prefix=AMDGPU
// RUN: rocmlir-gen --operation gemm --arch gfx1201 -t i4 -wmma=on -g 1 -m 256 -n 256 -k 256 | rocmlir-driver --arch gfx1201 --kernel-pipeline=gpu,rocdl | FileCheck %s --check-prefix=ROCDL
// iu4 WMMA is gfx12-only: gfx11 must reject it.
// RUN: not rocmlir-gen --operation gemm --arch gfx1100 -t i4 -wmma=on -g 1 -m 128 -n 128 -k 128 2>&1 | FileCheck %s --check-prefix=GFX11

// AMDGPU: amdgpu.wmma 16x16x32 {{.*}} : vector<16xi4>, vector<16xi4>, vector<8xi32>
// ROCDL: rocdl.wmma.i32.16x16x32.iu4 {{.*}} : (vector<2xi32>, vector<2xi32>, vector<8xi32>) -> vector<8xi32>
// GFX11: Wmma supports only F16/BF16/int8
