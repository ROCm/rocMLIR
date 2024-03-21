// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func.func @load_scalar_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @load_scalar_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: %[[ret:.*]] = memref.load %[[cast]]
    %ret = rock.global_load %mem[%c0, %c0, %c0, %c0, %c0] if %true
        : memref<1x2x3x4x8xf32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @load_scalar_in_bounds_force_oob
// CHECK-SAME: (%[[mem:.*]]: memref<8xf32>)
func.func @load_scalar_in_bounds_force_oob(%mem: memref<8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[ret:.*]] = amdgpu.raw_buffer_load %[[mem]]
    %ret = rock.global_load %mem[%c0] if %true
        {canReadOffEnd}
        : memref<8xf32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}


// CHECK-LABEL: func.func @load_vector_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @load_vector_in_bounds(%mem: memref<1x2x3x4x8xf32>) -> vector<5xf32> {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: %[[ret:.*]] = vector.load %[[cast]]
    %ret = rock.global_load %mem[%c0, %c0, %c0, %c0, %c0] if %true
        : memref<1x2x3x4x8xf32>  -> vector<5xf32>
    // CHECK: return %[[ret]]
    return %ret : vector<5xf32>
}

// CHECK-LABEL: func.func @load_vector_oob
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @load_vector_oob(%mem: memref<1x2x3x4x8xf32>, %idx: index, %valid: i1) -> vector<5xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[c192:.*]] = arith.constant 192
    // CHECK: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-COUNT-2: amdgpu.raw_buffer_load %[[mem]]
    %ret = rock.global_load %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        : memref<1x2x3x4x8xf32> -> vector<5xf32>
    return %ret : vector<5xf32>
}

// CHECK-LABEL: func.func @load_scalar
// CHECK-SAME: (%[[mem:.*]]: memref<f32>, %[[idx:.*]]: index)
func.func @load_scalar_empty_mem(%mem: memref<f32>, %idx: index) -> f32 {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: %[[ret:.*]] = memref.load %[[cast]][] : memref<f32, #gpu.address_space<global>>
    %ret = rock.global_load %mem[] if %true
        : memref<f32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// Ensure that offsets that can be in [2 GB, 4 GB) go to buffer loads.
// CHECK-LABEL: func.func @load_scalar_in_bounds_semi_large
// CHECK-SAME: (%[[mem:.*]]: memref<32769x16384xf32>)
func.func @load_scalar_in_bounds_semi_large(%mem: memref<32769x16384xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[ret:.*]] = amdgpu.raw_buffer_load {boundsCheck = false} %[[mem]]
    %ret = rock.global_load %mem[%c0, %c0] if %true
        : memref<32769x16384xf32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @load_scalar_in_bounds_large
// CHECK-SAME: (%[[mem:.*]]: memref<1073741825xf32>)
func.func @load_scalar_in_bounds_large(%mem: memref<1073741825xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: %[[ret:.*]] = memref.load %[[cast]]
    %ret = rock.global_load %mem[%c0] if %true {needs64BitIdx}
        : memref<1073741825xf32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @load_scalar_oob_large
// CHECK-SAME: (%[[mem:.*]]: memref<1073741825xf32>, %[[valid:.*]]: i1)
func.func @load_scalar_oob_large(%mem: memref<1073741825xf32>, %valid: i1) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: %[[ret:.*]] = scf.if %[[valid]]
    // CHECK: %[[load:.*]] = memref.load %[[cast]]
    // CHECK: scf.yield %[[load]]
    %ret = rock.global_load %mem[%c0] if %valid {needs64BitIdx}
        : memref<1073741825xf32> -> f32
    // CHECK: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @store_scalar_in_bounds
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @store_scalar_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.store %[[val]], %[[cast]]
    rock.global_store set %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @store_vector_in_bounds
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @store_vector_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = vector.transfer_read %[[source]]
    // CHECK: vector.store %[[val]], %[[cast]]
    rock.global_store set %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
        features = none {length = 5 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @store_vector_oob
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @store_vector_oob(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c192:.*]] = arith.constant 192
    // CHECK: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-COUNT-2: amdgpu.raw_buffer_store %{{.*}} -> %[[mem]]
    // CHECK-NEXT: return
    rock.global_store set %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = none {length = 5 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @store_scalar_in_scalar
// CHECK-SAME: (%[[source:.*]]: memref<1xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<f32>)
func.func @store_scalar_in_scalar(%source: memref<1xf32, #gpu.address_space<private>>, %mem: memref<f32>) {
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.store %[[val]], %[[cast]]
    rock.global_store set %source[%c0] -> %mem[] if %true
        features = none {length = 1 : index}
        : memref<1xf32, #gpu.address_space<private>> -> memref<f32>
    return
}

// Some nightly tests failed if this wasn't a buffer store
// CHECK-LABEL: func.func @store_in_scalar_maybe_oob
// CHECK-SAME: (%[[source:.*]]: memref<1xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<f32>, %[[valid:.*]]: i1)
func.func @store_in_scalar_maybe_oob(%source: memref<1xf32, #gpu.address_space<private>>, %mem: memref<f32>, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK-DAG: %[[exp:.*]] = memref.expand_shape %[[mem]] []
    // CHECK: amdgpu.raw_buffer_store %[[val]] -> %[[exp]]
    rock.global_store set %source[%c0] -> %mem[] if %valid
        features = none {length = 1 : index}
        : memref<1xf32, #gpu.address_space<private>> -> memref<f32>
    return
}

// CHECK-LABEL: func.func @store_scalar_in_bounds_semi_large
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<32769x16384xf32>)
func.func @store_scalar_in_bounds_semi_large(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<32769x16384xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: amdgpu.raw_buffer_store {boundsCheck = false} %[[val]] -> %[[mem]]
    rock.global_store set %source[%c0] -> %mem[%c0, %c0] if %true
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<32769x16384xf32>
    return
}

// CHECK-LABEL: func.func @store_scalar_in_bounds_large
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1073741825xf32>)
func.func @store_scalar_in_bounds_large(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1073741825xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.store %[[val]], %[[cast]]
    rock.global_store set %source[%c0] -> %mem[%c0] if %true
        features = none {length = 1 : index, needs64BitIdx}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1073741825xf32>
    return
}

// CHECK-LABEL: func.func @store_scalar_oob_large
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1073741825xf32>, %[[valid:.*]]: i1)
func.func @store_scalar_oob_large(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1073741825xf32>, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: scf.if %[[valid]]
    // CHECK: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.store %[[val]], %[[cast]]
    rock.global_store set %source[%c0] -> %mem[%c0] if %valid
        features = none {length = 1 : index, needs64BitIdx}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1073741825xf32>
    return
}

// CHECK-LABEL: func.func @add_scalar_in_bounds
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @add_scalar_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.atomic_rmw addf %[[val]], %[[cast]]
    rock.global_store atomic_add %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @add_scalar_oob
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @add_scalar_oob(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c192:.*]] = arith.constant 192
    // CHECK-DAG: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: amdgpu.raw_buffer_atomic_fadd %[[val]] -> %[[mem]]
    rock.global_store atomic_add %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @add_scalar_oob_fp16
// CHECK-SAME: (%[[source:.*]]: memref<5xf16, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf16>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @add_scalar_oob_fp16(%source: memref<5xf16, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf16>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK: %[[mod:.*]] = arith.remui
    // CHECK: %[[cmp:.*]] = arith.cmpi ne, %[[mod]]
    // CHECK: %[[val:.*]] = arith.select %[[cmp]], {{.*}} : vector<2xf16>
    // CHECK: amdgpu.raw_buffer_atomic_fadd %[[val]] -> %[[mem]][{{.*}}] : vector<2xf16>
    rock.global_store atomic_add %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = none {length = 1 : index}
        : memref<5xf16, #gpu.address_space<private>> -> memref<1x2x3x4x8xf16>
    return
}

// CHECK-LABEL: func.func @add_packed_oob_fp16
// CHECK-SAME: (%[[source:.*]]: memref<5xf16, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf16>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @add_packed_oob_fp16(%source: memref<5xf16, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf16>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c192:.*]] = arith.constant 192
    // CHECK-DAG: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-DAG: %[[val:.*]] = vector.transfer_read %[[source]]
    // CHECK: amdgpu.raw_buffer_atomic_fadd %[[val]] -> %[[mem]][{{.*}}] : vector<2xf16>
    rock.global_store atomic_add %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = none {length = 2 : index}
        : memref<5xf16, #gpu.address_space<private>> -> memref<1x2x3x4x8xf16>
    return
}

// CHECK-LABEL: func.func @add_vector_in_bounds
func.func @add_vector_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-5: amdgpu.raw_buffer_atomic_fadd
    rock.global_store atomic_add %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
        features = none {length = 5 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @add_scalar_to_scalar_valid
// CHECK-SAME: (%[[source:.*]]: memref<1xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<f32>)
func.func @add_scalar_to_scalar_valid(%source: memref<1xf32, #gpu.address_space<private>>, %mem: memref<f32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK-NOT: scf.if
    // CHECK: memref.atomic_rmw addf %[[val]], %[[cast]]
    rock.global_store atomic_add %source[%c0] -> %mem[] if %true
        features = none {length = 1 : index}
        : memref<1xf32, #gpu.address_space<private>> -> memref<f32>
    return
}

// CHECK-LABEL: func.func @add_scalar_to_scalar_maybe_valid
// CHECK-SAME: (%[[source:.*]]: memref<1xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<f32>, %[[valid:.*]]: i1)
func.func @add_scalar_to_scalar_maybe_valid(%source: memref<1xf32, #gpu.address_space<private>>, %mem: memref<f32>, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK-DAG: %[[exp:.*]] = memref.expand_shape %[[mem]] []
    // CHECK: amdgpu.raw_buffer_atomic_fadd %[[val]] -> %[[exp]]
    rock.global_store atomic_add %source[%c0] -> %mem[] if %valid
        features = none {length = 1 : index}
        : memref<1xf32, #gpu.address_space<private>> -> memref<f32>
    return
}

// CHECK-LABEL: func.func @native_fmax_scalar_in_bounds
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @native_fmax_scalar_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.atomic_rmw maxf %[[val]], %[[cast]]
    rock.global_store atomic_max %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
         features = atomic_fmax_f32 {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @native_fmax_scalar_oob
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @native_fmax_scalar_oob(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c192:.*]] = arith.constant 192
    // CHECK-DAG: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: amdgpu.raw_buffer_atomic_fmax %[[val]] -> %[[mem]]
    rock.global_store atomic_max %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = atomic_fmax_f32 {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @emulated_fmax_scalar_in_bounds
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @emulated_fmax_scalar_in_bounds(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    %true = arith.constant true
    // CHECK-DAG: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: memref.atomic_rmw maxf %[[val]], %[[cast]]
    rock.global_store atomic_max %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %c0] if %true
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @emulated_fmax_scalar_oob
// CHECK-SAME: (%[[source:.*]]: memref<5xf32, #gpu.address_space<private>>, %[[mem:.*]]: memref<1x2x3x4x8xf32>, %[[idx:.*]]: index, %[[valid:.*]]: i1)
func.func @emulated_fmax_scalar_oob(%source: memref<5xf32, #gpu.address_space<private>>, %mem: memref<1x2x3x4x8xf32>, %idx: index, %valid: i1) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c192:.*]] = arith.constant 192
    // CHECK-DAG: arith.select %[[valid]], %[[idx]], %[[c192]]
    // CHECK-DAG: %[[val:.*]] = memref.load %[[source]]
    // CHECK: amdgpu.raw_buffer_atomic_fmax %[[val]] -> %[[mem]]
    rock.global_store atomic_max %source[%c0] -> %mem[%c0, %c0, %c0, %c0, %idx] if %valid
        features = none {length = 1 : index}
        : memref<5xf32, #gpu.address_space<private>> -> memref<1x2x3x4x8xf32>
    return
}
}
