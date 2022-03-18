// RUN: miopen-opt -miopen-lowering-step4 %s | FileCheck %s

module {
// CHECK-LABEL: func @load_scalar_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func @load_scalar_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = memref.load %[[mem]]
    %ret = miopen.buffer_load %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func @load_vector_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func @load_vector_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = gpu.buffer_load(%[[mem]]
    %ret = miopen.buffer_load %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> vector<4xf32>
    // CHECK: return %[[ret]]
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func @load_vector_oob
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index)
func @load_vector_oob(%mem: memref<1x2x3x4x8xf32>, %idx: index) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[c8:.*]] = arith.constant 8
    // CHECK: arith.cmpi sge, %[[idx]], %[[c8]]
    %ret = miopen.buffer_load %mem[%c0, %c0, %c0, %c0, %idx]
        {leftOobDims = [], rightOobDims = [4 : i32]}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> vector<4xf32>
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func @store_scalar_in_bounds
// CHECK-SAME: (%[[val:.*]]: f32, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func @store_scalar_in_bounds(%val: f32, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: gpu.raw_buffer_store(%[[val]], %[[mem]]
    miopen.buffer_store %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : f32 -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func @store_vector_in_bounds
// CHECK-SAME: (%[[val:.*]]: vector<4xf32>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func @store_vector_in_bounds(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: gpu.raw_buffer_store(%[[val]], %[[mem]]
    miopen.buffer_store %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func @store_vector_oob
// CHECK-SAME: (%[[val:.*]]: vector<4xf32>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index)
func @store_vector_oob(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>, %idx: index) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[coob:.*]] = arith.constant 2147483647
    // CHECK-DAG: %[[c8:.*]] = arith.constant 8
    // CHECK: arith.cmpi sge, %[[idx]], %[[c8]]
    // CHECK: gpu.raw_buffer_store(%[[val]], %[[mem]],
    miopen.buffer_store %val -> %mem[%c0, %c0, %c0, %c0, %idx]
        {leftOobDims = [], rightOobDims = [4 : i32]}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func @add_scalar_in_bounds
// CHECK-SAME: (%[[val:.*]]: f32, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func @add_scalar_in_bounds(%val: f32, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: gpu.atomic_fadd(%[[val]], %[[mem]]
    miopen.buffer_store %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = [], storeMethod = 1 : i32}
        : f32 -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}
}
