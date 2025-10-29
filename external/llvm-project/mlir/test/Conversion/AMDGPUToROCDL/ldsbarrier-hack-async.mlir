// RUN: mlir-opt %s -convert-amdgpu-to-rocdl=chipset=gfx1250 | FileCheck %s

#gpu_global_addrspace = 1
#gpu_lds_addrspace = 3

// CHECK-LABEL: func @lds_barrier_workaround_gfx1250
func.func @lds_barrier_workaround_gfx1250(%mem: memref<192xf32, #gpu_global_addrspace>) {
  %c0 = arith.constant 0 : index
  %lds = memref.alloc() : memref<4xf32, #gpu_lds_addrspace>
  amdgpu.async_load_to_lds %mem[%c0], %lds[%c0] : f32, memref<192xf32, #gpu_global_addrspace>, memref<4xf32, #gpu_lds_addrspace>
  // GFX1250: rocdl.load.to.lds
  // GFX1250-NEXT: rocdl.s.waitasynccnt
  // GFX1250-NEXT: rocdl.s.waitasynccnt
  // GFX1250-NEXT: rocdl.s.barrier
  amdgpu.lds_barrier
  func.return
}
