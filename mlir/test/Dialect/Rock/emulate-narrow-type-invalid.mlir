// RUN: not rocmlir-opt -rock-emulate-narrow-type -split-input-file %s 2>&1 | FileCheck %s

// CHECK: failed to legalize operation 'amdgpu.gather_to_lds'
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tShared = memref<32xi4, #gpu.address_space<workgroup>>
func.func @gather_odd_src(%src: !tGlobal, %dst: !tShared) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %src[%c1], %dst[%c0] : f32, !tGlobal, !tShared
  func.return
}

// -----

// CHECK: failed to legalize operation 'amdgpu.gather_to_lds'
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tShared = memref<32xi4, #gpu.address_space<workgroup>>
func.func @gather_odd_dst(%src: !tGlobal, %dst: !tShared) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %src[%c0], %dst[%c1] : f32, !tGlobal, !tShared
  func.return
}
