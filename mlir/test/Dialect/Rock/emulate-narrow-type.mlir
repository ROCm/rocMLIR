// RUN: rocmlir-opt -rock-emulate-narrow-type -split-input-file -o - %s | FileCheck %s

!tFlat = memref<32xi4>
!tGlobal = memref<32xi4, #gpu.address_space<global>>
// CHECK-LABEL: func.func @global
// CHECK-SAME: ([[arg0:%.+]]: memref<16xi8>, [[arg1:%.+]]: memref<16xi8>)
// CHECK: [[cast0:%.+]] = memref.memory_space_cast [[arg0]]
// CHECK: [[cast1:%.+]] = memref.memory_space_cast [[arg1]]
// CHECK: [[read:%.+]] = vector.load [[cast0]]
// CHECK-SAME: vector<8xi8>
// CHECK: vector.store [[read]], [[cast1]]
func.func @global(%arg0: !tFlat, %arg1: !tFlat) {
  %c0 = arith.constant 0 : index
  %c0_i4 = arith.constant 0 : i4
  %cast0 = memref.memory_space_cast %arg0 : !tFlat to !tGlobal
  %cast1 = memref.memory_space_cast %arg1 : !tFlat to !tGlobal
  %read = vector.transfer_read %cast0[%c0], %c0_i4 {in_bounds = [true]} : !tGlobal, vector<16xi4>
  vector.transfer_write %read, %cast1[%c0] : vector<16xi4>, !tGlobal
  func.return
}

// -----

// CHECK-LABEL: func.func @buffer
// CHECK-SAME: ([[arg0:%.+]]: memref<16xi8>, [[arg1:%.+]]: memref<16xi8>)
// CHECK: [[read:%.+]] = amdgpu.raw_buffer_load [[arg0]]
// CHECK-SAME: vector<8xi8>
// CHECK: amdgpu.raw_buffer_store [[read]] -> [[arg1]]

!tFlat = memref<32xi4>
func.func @buffer(%arg0: !tFlat, %arg1: !tFlat) {
  %c0 = arith.constant 0 : i32
  %read = amdgpu.raw_buffer_load %arg0[%c0] : !tFlat, i32 -> vector<16xi4>
  amdgpu.raw_buffer_store %read -> %arg1[%c0] : vector<16xi4> -> !tFlat, i32
  func.return
}

// -----

// CHECK-LABEL: func.func @extui
// CHECK-SAME: ([[arg0:%.+]]: memref<16xi8>, [[arg1:%.+]]: memref<32xi8>)
// CHECK-DAG: [[shiftLen:%.+]] = arith.constant dense<4> : vector<8xi8>
// CHECK-DAG: [[mask:%.+]] = arith.constant dense<15> : vector<8xi8>
// CHECK: [[load:%.+]] = vector.load [[arg0]]
// CHECK: [[and:%.+]] = arith.andi [[load]], [[mask]]
// CHECK: [[shift:%.+]] = arith.shrui [[load]], [[shiftLen]]
// CHECK: [[bytes:%.+]] = vector.interleave [[and]], [[shift]]
// CHECK: vector.store [[bytes]], [[arg1]]

!tIn = memref<32xi4>
!tOut = memref<32xi8>
func.func @extui(%arg0: !tIn, %arg1: !tOut) {
  %c0 = arith.constant 0 : index
  %c0_i4 = arith.constant 0 : i4
  %read = vector.transfer_read %arg0[%c0], %c0_i4 {in_bounds = [true]} : !tIn, vector<16xi4>
  %ext = arith.extui %read : vector<16xi4> to vector<16xi8>
  vector.transfer_write %ext, %arg1[%c0] : vector<16xi8>, !tOut
  func.return
}
