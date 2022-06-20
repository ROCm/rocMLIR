// RUN: miopen-opt -miopen-expand-shorthand %s | FileCheck %s

module {
// CHECK-LABEL: func.func @in_bounds_load_scalar
// CHECK-SAME: (%[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_load_scalar(%buf: memref<2x2xf32, 3>) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK: %[[ret:.*]] = memref.load %[[buf]]
    // CHECK-NEXT: return %[[ret]]
    %ret = miopen.in_bounds_load %buf[%c0, %c0] : memref<2x2xf32, 3>, index, index -> f32
    return %ret : f32
}

// CHECK-LABEL: func.func @in_bounds_load_vector
// CHECK-SAME: (%[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_load_vector(%buf: memref<2x2xf32, 3>) -> vector<4xf32> {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK-DAG: %[[cvec:.*]] = arith.constant {{.*}} : vector<4xf32>
    // CHECK: %[[v1:.*]] = memref.load %[[buf]][%[[c0]], %[[c0]]]
    // CHECK: %[[r1:.*]] = vector.insertelement %[[v1]], %[[cvec]][%[[c0]]
    // CHECK: %[[v2:.*]] = memref.load %[[buf]][%[[c0]], %[[c1]]]
    // CHECK: %[[r2:.*]] = vector.insertelement %[[v2]], %[[r1]][%[[c1]]
    // CHECK: %[[v3:.*]] = memref.load %[[buf]][%[[c0]], %[[c2]]]
    // CHECK: %[[r3:.*]] = vector.insertelement %[[v3]], %[[r2]][%[[c2]]
    // CHECK: %[[v4:.*]] = memref.load %[[buf]][%[[c0]], %[[c3]]]
    // CHECK: %[[r4:.*]] = vector.insertelement %[[v4]], %[[r3]][%[[c3]]
    // CHECK-NEXT: return %[[r4]]
    %ret = miopen.in_bounds_load %buf[%c0, %c0] : memref<2x2xf32, 3>, index, index -> vector<4xf32>
    return %ret : vector<4xf32>
}

// CHECK-LABEL: func.func @in_bounds_store_scalar
// CHECK-SAME: (%[[data:.*]]: f32, %[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_store_scalar(%data: f32, %buf: memref<2x2xf32, 3>) {
    %c0 = arith.constant 0 : index
    // CHECK: memref.store %[[data]], %[[buf]]
    miopen.in_bounds_store %data -> %buf[%c0, %c0] : f32 -> memref<2x2xf32, 3>, index, index
    return
}

// CHECK-LABEL: func.func @in_bounds_store_vector
// CHECK-SAME: (%[[data:.*]]: vector<4xf32>, %[[buf:.*]]: memref<2x2xf32, 3>)
func.func @in_bounds_store_vector(%data: vector<4xf32>, %buf: memref<2x2xf32, 3>) {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[c3:.*]] = arith.constant 3 : index
    // CHECK: %[[v0:.*]] = vector.extractelement %[[data]][%[[c0]]
    // CHECK: memref.store %[[v0]], %[[buf]][%[[c0]], %[[c0]]]
    // CHECK: %[[v1:.*]] = vector.extractelement %[[data]][%[[c1]]
    // CHECK: memref.store %[[v1]], %[[buf]][%[[c0]], %[[c1]]]
    // CHECK: %[[v2:.*]] = vector.extractelement %[[data]][%[[c2]]
    // CHECK: memref.store %[[v2]], %[[buf]][%[[c0]], %[[c2]]]
    // CHECK: %[[v3:.*]] = vector.extractelement %[[data]][%[[c3]]
    // CHECK: memref.store %[[v3]], %[[buf]][%[[c0]], %[[c3]]]
    miopen.in_bounds_store %data -> %buf[%c0, %c0] : vector<4xf32> -> memref<2x2xf32, 3>, index, index
    return
}
}
