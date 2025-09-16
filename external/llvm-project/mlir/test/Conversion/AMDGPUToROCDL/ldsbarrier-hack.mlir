// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx942 | FileCheck %s --check-prefixes=CHECK,GFX942
// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx950 | FileCheck %s --check-prefixes=CHECK,GFX950

// Note: #gpu.address_space<global> is hardcoded to `1` here because the
// test pass doesn't set up the GPU address space conversions.

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3
#amdgpu_fat_buffer_addrspace = 7

// CHECK-LABEL: func @lds_barrier_workaround
func.func @lds_barrier_workaround(%mem: memref<192xf32, #amdgpu_fat_buffer_addrspace>) {
  %c0 = arith.constant 0 : index
  %lds = memref.alloc() : memref<4xf32, #gpu_lds_addrspace>
  amdgpu.gather_to_lds %mem[%c0], %lds[%c0] : f32, memref<192xf32, #amdgpu_fat_buffer_addrspace>, memref<4xf32, #gpu_lds_addrspace>
  // GFX942: rocdl.load.to.lds
  // GFX942-NEXT: rocdl.s.waitcnt -49168
  // GFX942-NEXT: rocdl.s.waitcnt -7937
  // GFX942-NEXT: rocdl.s.barrier
  // GFX950: rocdl.load.to.lds
  // GFX950-NEXT: rocdl.s.waitcnt -49168
  // GFX950-NEXT: rocdl.s.waitcnt -7937
  // GFX950-NEXT: rocdl.s.barrier
  amdgpu.lds_barrier
  func.return
}
