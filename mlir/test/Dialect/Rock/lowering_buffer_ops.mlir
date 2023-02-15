// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func.func @load_scalar_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @load_scalar_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = amdgpu.raw_buffer_load %[[mem]]
    %ret = rock.buffer_load %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @load_vector_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @load_vector_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = amdgpu.raw_buffer_load %[[mem]]
    %ret = rock.buffer_load %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> vector<4xf32>
    // CHECK: return %[[ret]]
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func.func @load_vector_in_bounds_offset
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @load_vector_in_bounds_offset(%mem: memref<1x2x3x4x8xf32>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = amdgpu.raw_buffer_load {indexOffset = 4 : i32} %[[mem]]
    %ret = rock.buffer_load %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = [], offset = 4 : index}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> vector<4xf32>
    // CHECK: return %[[ret]]
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func.func @load_vector_oob
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index)
func.func @load_vector_oob(%mem: memref<1x2x3x4x8xf32>, %idx: index) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[c8:.*]] = arith.constant 8
    // CHECK: arith.cmpi sge, %[[idx]], %[[c8]]
    %ret = rock.buffer_load %mem[%c0, %c0, %c0, %c0, %idx]
        {leftOobDims = [], rightOobDims = [4 : i32]}
        : memref<1x2x3x4x8xf32>, index, index, index, index, index -> vector<4xf32>
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func.func @store_scalar_in_bounds
// CHECK-SAME: (%[[val:.*]]: f32, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @store_scalar_in_bounds(%val: f32, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: amdgpu.raw_buffer_store %[[val]] -> %[[mem]]
    rock.buffer_store set %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : f32 -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func.func @store_vector_in_bounds
// CHECK-SAME: (%[[val:.*]]: vector<4xf32>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @store_vector_in_bounds(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: amdgpu.raw_buffer_store %[[val]] -> %[[mem]]
    rock.buffer_store set %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func.func @store_vector_in_bounds_offset
// CHECK-SAME: (%[[val:.*]]: vector<4xf32>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @store_vector_in_bounds_offset(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: amdgpu.raw_buffer_store {indexOffset = 4 : i32} %[[val]] -> %[[mem]]
    rock.buffer_store set %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = [], offset = 4 : index}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func.func @store_vector_oob
// CHECK-SAME: (%[[val:.*]]: vector<4xf32>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index)
func.func @store_vector_oob(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>, %idx: index) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c8:.*]] = arith.constant 8
    // CHECK: arith.cmpi sge, %[[idx]], %[[c8]]
    // CHECK: amdgpu.raw_buffer_store %[[val]] -> %[[mem]]
    rock.buffer_store set %val -> %mem[%c0, %c0, %c0, %c0, %idx]
        {leftOobDims = [], rightOobDims = [4 : i32]}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func.func @add_scalar_in_bounds
// CHECK-SAME: (%[[val:.*]]: f32, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @add_scalar_in_bounds(%val: f32, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: amdgpu.raw_buffer_atomic_fadd %[[val]] -> %[[mem]]
    rock.buffer_store atomic_add %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : f32 -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}

// CHECK-LABEL: func.func @add_vector_in_bounds
func.func @add_vector_in_bounds(%val: vector<4xf32>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK-4: amdgpu.raw_buffer_atomic_fadd
    rock.buffer_store atomic_add %val -> %mem[%c0, %c0, %c0, %c0, %c0]
        {leftOobDims = [], rightOobDims = []}
        : vector<4xf32> -> memref<1x2x3x4x8xf32>, index, index, index, index, index
    return
}
}
