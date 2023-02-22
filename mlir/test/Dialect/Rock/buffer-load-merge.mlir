// RUN: rocmlir-opt %s -split-input-file -rock-buffer-load-merge | FileCheck %s

// CHECK-LABEL: @basic_merge
func.func @basic_merge(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[single:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK-NOT: amdgpu.raw_buffer_load
  %0 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %0 -> %arg1[%c0] : f32 -> memref<2xf32>, i32
  %1 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %1 -> %arg1[%c1] : f32 -> memref<2xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: @basic_merge_2
func.func @basic_merge_2(%arg0: memref<2xf32>, %arg1: memref<2xf32>, %arg2: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[single:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK-NOT: amdgpu.raw_buffer_load
  %0 = amdgpu.raw_buffer_load %arg0[%arg2] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %0 -> %arg1[%c0] : f32 -> memref<2xf32>, i32
  %1 = amdgpu.raw_buffer_load %arg0[%arg2] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %1 -> %arg1[%c1] : f32 -> memref<2xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: @alias
func.func @alias(%arg0: memref<2xf32>, %arg1: memref<2xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[first:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[first]]
  // CHECK: %[[second:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[second]]
  %0 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %0 -> %arg0[%c0] : f32 -> memref<2xf32>, i32
  %1 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %1 -> %arg1[%c1] : f32 -> memref<2xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: @danger_is_allowed
func.func @danger_is_allowed(%arg0: memref<2xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[single:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK: amdgpu.raw_buffer_store %[[single]]
  // CHECK-NOT: amdgpu.raw_buffer_load
  %view = memref.cast %arg0 : memref<2xf32> to memref<2xf32>
  %0 = amdgpu.raw_buffer_load %view[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %0 -> %arg0[%c0] : f32 -> memref<2xf32>, i32
  %1 = amdgpu.raw_buffer_load %view[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %1 -> %arg0[%c1] : f32 -> memref<2xf32>, i32
  func.return
}

// -----

// CHECK-LABEL: @some_alias_danger_detected
func.func @some_alias_danger_detected(%arg0: memref<2xf32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // CHECK: %[[first:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[first]]
  // CHECK: %[[second:.*]] = amdgpu.raw_buffer_load
  // CHECK: amdgpu.raw_buffer_store %[[second]]
  %view = memref.cast %arg0 : memref<2xf32> to memref<2xf32>
  %0 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %0 -> %view[%c0] : f32 -> memref<2xf32>, i32
  %1 = amdgpu.raw_buffer_load %arg0[%c0] : memref<2xf32>, i32 -> f32
  amdgpu.raw_buffer_store %1 -> %view[%c1] : f32 -> memref<2xf32>, i32
  func.return
}

