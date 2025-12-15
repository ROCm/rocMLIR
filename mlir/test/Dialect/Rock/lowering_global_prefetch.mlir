// RUN: rocmlir-opt --rock-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func.func @prefetch_scalar_in_bounds
// CHECK-SAME: (%[[mem:.*]]: memref<1x2x3x4x8xf32>)
func.func @prefetch_scalar_in_bounds(%mem: memref<1x2x3x4x8xf32>) {
    %c0 = arith.constant 0 : index
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: memref.prefetch %[[cast]]
    // CHECK-SAME: read, locality<3>, data : memref<1x2x3x4x8xf32, #gpu.address_space<global>>
    rock.global_prefetch %mem[%c0, %c0, %c0, %c0, %c0] : memref<1x2x3x4x8xf32>
    return
}

// CHECK-LABEL: func.func @prefetch_scalar_in_bounds_oob
// CHECK-SAME: (%[[mem:.*]]: memref<8xf32>)
func.func @prefetch_scalar_in_bounds_oob(%mem: memref<8xf32>) {
    %c9 = arith.constant 9 : index
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: memref.prefetch %[[cast]]
    // CHECK-SAME: read, locality<3>, data : memref<8xf32, #gpu.address_space<global>>
    rock.global_prefetch %mem[%c9] : memref<8xf32> 
    return
}

// CHECK-LABEL: func.func @prefetch_scalar_large_i4
// CHECK-SAME: (%[[mem:.*]]: memref<1073741825xi4>)
func.func @prefetch_scalar_large_i4(%mem: memref<1073741825xi4>) {
    %c0 = arith.constant 0 : index
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: memref.prefetch %[[cast]]
    // CHECK-SAME: read, locality<3>, data : memref<1073741825xi4, #gpu.address_space<global>>
    rock.global_prefetch %mem[%c0] : memref<1073741825xi4>
    return
}

// CHECK-LABEL: func.func @prefetch_scalar
// CHECK-SAME: (%[[mem:.*]]: memref<f32>)
func.func @prefetch_scalar(%mem: memref<f32>) {
    // CHECK: %[[cast:.*]] = memref.memory_space_cast %[[mem]]
    // CHECK-SAME: #gpu.address_space<global>
    // CHECK: memref.prefetch %[[cast]][]
    // CHECK-SAME: read, locality<3>, data : memref<f32, #gpu.address_space<global>>
    rock.global_prefetch %mem[] : memref<f32>
    return
}
}
