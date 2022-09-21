// RUN: rocmlir-opt -rock-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func.func @in_bounds_load_scalar
// CHECK-SAME: (%[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_load_scalar(%buf: memref<2x2xf32, 3>) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = memref.load %[[buf]]
    // CHECK-NEXT: return %[[ret]]
    %ret = rock.in_bounds_load %buf[%c0, %c0] : memref<2x2xf32, 3>, index, index -> f32
    return %ret : f32
}

// CHECK-LABEL: func.func @in_bounds_load_vector
// CHECK-SAME: (%[[buf:.*]]: memref<4x4xf32, 3>)
func.func @in_bounds_load_vector(%buf: memref<4x4xf32, 3>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = vector.transfer_read %[[buf]]{{.*}} {in_bounds = [true]}
    // CHECK-NEXT: return %[[ret]] : vector<4xf32>
    %ret = rock.in_bounds_load %buf[%c0, %c0] : memref<4x4xf32, 3>, index, index -> vector<4xf32>
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func.func @in_bounds_store_scalar
// CHECK-SAME: (%[[data:.*]]: f32, %[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_store_scalar(%data: f32, %buf: memref<2x2xf32, 3>) {
    %c0 = arith.constant 0 : index
    // CHECK: memref.store %[[data]], %[[buf]]
    rock.in_bounds_store %data -> %buf[%c0, %c0] : f32 -> memref<2x2xf32, 3>, index, index
    return
}

// CHECK-LABEL: func.func @in_bounds_store_vector
// CHECK-SAME: (%[[data:.*]]: vector<4xf32>, %[[buf:.*]]: memref<4x4xf32, 3>)
func.func @in_bounds_store_vector(%data: vector<4xf32>, %buf: memref<4x4xf32, 3>) {
    %c0 = arith.constant 0 : index
    // CHECK: vector.transfer_write %[[data]], %[[buf]]{{.*}}{in_bounds = [true]}
    rock.in_bounds_store %data -> %buf[%c0, %c0] : vector<4xf32> -> memref<4x4xf32, 3>, index, index
    return
}
}
