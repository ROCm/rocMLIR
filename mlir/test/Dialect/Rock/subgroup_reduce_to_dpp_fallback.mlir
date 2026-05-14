// Verify that rock-subgroup-reduce-to-dpp lowers all gpu.subgroup_reduce ops:
// supported cases via DPP, unsupported (strided) via shuffle fallback.

// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx942" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx950" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx1100" %s | FileCheck %s
// RUN: rocmlir-opt --rock-subgroup-reduce-to-dpp="chip=gfx1201" %s | FileCheck %s

// CHECK-LABEL: gpu.module @test_module
gpu.module @test_module {

  // Contiguous cluster (stride=1) -- handled by DPP.
  // CHECK-LABEL: gpu.func @test_dpp_path
  // CHECK-NOT: gpu.subgroup_reduce
  // CHECK: amdgpu.dpp
  gpu.func @test_dpp_path(%val : f32) -> f32 {
    %res = gpu.subgroup_reduce add %val cluster(size = 4) : (f32) -> f32
    gpu.return %res : f32
  }

  // Strided cluster (stride=2) -- DPP rejects non-contiguous lanes,
  // so this must fall through to the shuffle-based fallback.
  // CHECK-LABEL: gpu.func @test_shuffle_fallback
  // CHECK-NOT: gpu.subgroup_reduce
  // CHECK-NOT: amdgpu.dpp
  // CHECK: gpu.shuffle xor
  gpu.func @test_shuffle_fallback(%val : f32) -> f32 {
    %res = gpu.subgroup_reduce add %val cluster(size = 4, stride = 2) : (f32) -> f32
    gpu.return %res : f32
  }
}
