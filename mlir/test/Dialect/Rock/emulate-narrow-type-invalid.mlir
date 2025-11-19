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

// -----

// Test that invalid transfer types are rejected. 
// CHECK: Transfer type must be f128 or f32 for GatherToLDS
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tShared = memref<32xi4, #gpu.address_space<workgroup>>
func.func @gather_invalid_transfer_type_i32(%src: !tGlobal, %dst: !tShared) {
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %src[%c0], %dst[%c0] : i32, !tGlobal, !tShared
  func.return
}

// -----

// CHECK: Transfer type must be f128 or f32 for GatherToLDS
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tShared = memref<32xi4, #gpu.address_space<workgroup>>
func.func @gather_invalid_transfer_type_vector(%src: !tGlobal, %dst: !tShared) {
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %src[%c0], %dst[%c0] : vector<3xf32>, !tGlobal, !tShared
  func.return
}

// -----

// CHECK: Transfer type must be f128 or f32 for GatherToLDS
!tGlobalF4 = memref<32xf4E2M1FN, #gpu.address_space<global>>
!tSharedF4 = memref<32xf4E2M1FN, #gpu.address_space<workgroup>>
func.func @gather_invalid_transfer_type_f16(%src: !tGlobalF4, %dst: !tSharedF4) {
  %c0 = arith.constant 0 : index
  amdgpu.gather_to_lds %src[%c0], %dst[%c0] : f16, !tGlobalF4, !tSharedF4
  func.return
}
