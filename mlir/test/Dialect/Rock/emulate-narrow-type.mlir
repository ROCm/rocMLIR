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

// CHECK-LABEL: func.func @odd_nibble_clamp
// CHECK-SAME: ([[arg0:%.+]]: memref<2xi8>, [[arg1:%.+]]: memref<2xi8>, [[idx:%.+]]: i32)
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : i32
// CHECK: %[[DIV:.*]] = arith.divui [[idx]], %[[C2]]
// CHECK: %[[CMP:.*]] = arith.cmpi uge, [[idx]], %[[C3]]
// CHECK: %[[SEL:.*]] = arith.select %[[CMP]], %[[C2]], %[[DIV]]
// CHECK: %[[LOAD:.*]] = amdgpu.raw_buffer_load [[arg0]][%[[SEL]]] : memref<2xi8>, i32 -> vector<1xi8>
func.func @odd_nibble_clamp(%arg0: memref<3xi4>, %arg1: memref<3xi4>, %idx: i32) {
  %val = amdgpu.raw_buffer_load %arg0[%idx] : memref<3xi4>, i32 -> vector<2xi4>
  amdgpu.raw_buffer_store %val -> %arg1[%idx] : vector<2xi4> -> memref<3xi4>, i32
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

// -----

// CHECK-LABEL: func.func @extract_metadata_view
// CHECK-SAME: (%[[ARG0:.*]]: memref<128xi8>)
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BASE:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [], strides: [] : memref<128xi8> to memref<i8>
// CHECK: return %[[BASE]], %[[C0]], %[[C256]], %[[C1]]
func.func @extract_metadata_view(%arg0: memref<128xi8>) -> (memref<i4>, index, index, index) {
  %c4 = arith.constant 4 : index
  %view = memref.view %arg0[%c4][] : memref<128xi8> to memref<256xi4>
  %base, %off, %size, %stride = memref.extract_strided_metadata %view : memref<256xi4> -> memref<i4>, index, index, index
  return %base, %off, %size, %stride : memref<i4>, index, index, index
}

// -----

// CHECK-LABEL: func.func @fat_buffer_cast
// CHECK-SAME: (%[[ARG0:.*]]: memref<16xi8, #gpu.address_space<global>>) -> memref<16xi8, #amdgpu.address_space<fat_raw_buffer>>
// CHECK: %[[RET:.*]] = amdgpu.fat_raw_buffer_cast %[[ARG0]] : memref<16xi8, #gpu.address_space<global>> to memref<16xi8, #amdgpu.address_space<fat_raw_buffer>>
// CHECK: return %[[RET]]
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tFat = memref<32xi4, #amdgpu.address_space<fat_raw_buffer>>

func.func @fat_buffer_cast(%arg0: !tGlobal) -> !tFat {
  %0 = amdgpu.fat_raw_buffer_cast %arg0 : !tGlobal to !tFat
  return %0 : !tFat
}

// -----

// CHECK-LABEL: func.func @memref_view_i4
// CHECK-SAME: (%[[ARG0:.*]]: memref<132xi8>) -> memref<128xi8>
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[VIEW:.*]] = memref.view %[[ARG0]][%[[C4]]][] : memref<132xi8> to memref<128xi8>
// CHECK: return %[[VIEW]] : memref<128xi8>
func.func @memref_view_i4(%arg0: memref<132xi8>) -> memref<256xi4> {
  %c4 = arith.constant 4 : index
  %view = memref.view %arg0[%c4][] : memref<132xi8> to memref<256xi4>
  return %view : memref<256xi4>
}

// -----

// CHECK-LABEL: func.func @gather_i4
// CHECK-SAME: (%[[SRC:.*]]: memref<16xi8, #gpu.address_space<global>>, %[[DST:.*]]: memref<16xi8, #gpu.address_space<workgroup>>, %[[IDX:.*]]: index)
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[DIV:.*]] = arith.divui %[[IDX]], %[[C2]] : index
// CHECK: %[[OOB:.*]] = arith.cmpi uge, %[[IDX]], %[[C32]] : index
// CHECK: %[[SEL:.*]] = arith.select %[[OOB]], %[[C16]], %[[DIV]] : index
// CHECK: %[[DIV_DST:.*]] = arith.divui %[[IDX]], %[[C2]] : index
// CHECK: %[[OOB_DST:.*]] = arith.cmpi uge, %[[IDX]], %[[C32]] : index
// CHECK: %[[SEL_DST:.*]] = arith.select %[[OOB_DST]], %[[C16]], %[[DIV_DST]] : index
// CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SEL]]], %[[DST]][%[[SEL_DST]]] : f32, memref<16xi8, #gpu.address_space<global>>, memref<16xi8, #gpu.address_space<workgroup>>
!tGlobal = memref<32xi4, #gpu.address_space<global>>
!tShared = memref<32xi4, #gpu.address_space<workgroup>>
func.func @gather_i4(%src: !tGlobal, %dst: !tShared, %idx: index) {
  amdgpu.gather_to_lds %src[%idx], %dst[%idx] : f32, !tGlobal, !tShared
  func.return
}

// -----

// CHECK-LABEL: func.func @extract_metadata_view_f4
// CHECK-SAME: (%[[ARG0:.*]]: memref<128xi8>)
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[BASE:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [], strides: [] : memref<128xi8> to memref<i8>
// CHECK: return %[[BASE]], %[[C0]], %[[C256]], %[[C1]]
func.func @extract_metadata_view_f4(%arg0: memref<128xi8>) -> (memref<f4E2M1FN>, index, index, index) {
  %c4 = arith.constant 4 : index
  %view = memref.view %arg0[%c4][] : memref<128xi8> to memref<256xf4E2M1FN>
  %base, %off, %size, %stride = memref.extract_strided_metadata %view : memref<256xf4E2M1FN> -> memref<f4E2M1FN>, index, index, index
  return %base, %off, %size, %stride : memref<f4E2M1FN>, index, index, index
}

// -----

// CHECK-LABEL: func.func @fat_buffer_cast_f4
// CHECK-SAME: (%[[ARG0:.*]]: memref<16xi8, #gpu.address_space<global>>) -> memref<16xi8, #amdgpu.address_space<fat_raw_buffer>>
// CHECK: %[[RET:.*]] = amdgpu.fat_raw_buffer_cast %[[ARG0]] : memref<16xi8, #gpu.address_space<global>> to memref<16xi8, #amdgpu.address_space<fat_raw_buffer>>
// CHECK: return %[[RET]]
!tGlobalF4 = memref<32xf4E2M1FN, #gpu.address_space<global>>
!tFatF4 = memref<32xf4E2M1FN, #amdgpu.address_space<fat_raw_buffer>>

func.func @fat_buffer_cast_f4(%arg0: !tGlobalF4) -> !tFatF4 {
  %0 = amdgpu.fat_raw_buffer_cast %arg0 : !tGlobalF4 to !tFatF4
  return %0 : !tFatF4
}

// -----

// CHECK-LABEL: func.func @memref_view_f4
// CHECK-SAME: (%[[ARG0:.*]]: memref<132xi8>) -> memref<128xi8>
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[VIEW:.*]] = memref.view %[[ARG0]][%[[C4]]][] : memref<132xi8> to memref<128xi8>
// CHECK: return %[[VIEW]] : memref<128xi8>
func.func @memref_view_f4(%arg0: memref<132xi8>) -> memref<256xf4E2M1FN> {
  %c4 = arith.constant 4 : index
  %view = memref.view %arg0[%c4][] : memref<132xi8> to memref<256xf4E2M1FN>
  return %view : memref<256xf4E2M1FN>
}

// -----

// CHECK-LABEL: func.func @gather_f4
// CHECK-SAME: (%[[SRC:.*]]: memref<16xi8, #gpu.address_space<global>>, %[[DST:.*]]: memref<16xi8, #gpu.address_space<workgroup>>, %[[IDX:.*]]: index)
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[DIV:.*]] = arith.divui %[[IDX]], %[[C2]] : index
// CHECK: %[[OOB:.*]] = arith.cmpi uge, %[[IDX]], %[[C32]] : index
// CHECK: %[[SEL:.*]] = arith.select %[[OOB]], %[[C16]], %[[DIV]] : index
// CHECK: %[[DIV_DST:.*]] = arith.divui %[[IDX]], %[[C2]] : index
// CHECK: %[[OOB_DST:.*]] = arith.cmpi uge, %[[IDX]], %[[C32]] : index
// CHECK: %[[SEL_DST:.*]] = arith.select %[[OOB_DST]], %[[C16]], %[[DIV_DST]] : index
// CHECK: amdgpu.gather_to_lds %[[SRC]][%[[SEL]]], %[[DST]][%[[SEL_DST]]] : f32, memref<16xi8, #gpu.address_space<global>>, memref<16xi8, #gpu.address_space<workgroup>>
!tGlobalF4 = memref<32xf4E2M1FN, #gpu.address_space<global>>
!tSharedF4 = memref<32xf4E2M1FN, #gpu.address_space<workgroup>>
func.func @gather_f4(%src: !tGlobalF4, %dst: !tSharedF4, %idx: index) {
  amdgpu.gather_to_lds %src[%idx], %dst[%idx] : f32, !tGlobalF4, !tSharedF4
  func.return
}
