// RUN: miopen-opt -miopen-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func @extract_slice_scalar
// CHECK-SAME: (%[[vec:.*]]: vector<8xf32>)
func @extract_slice_scalar(%vec : vector<8xf32>) -> f32 {
    // CHECK-NEXT: %[[const:.*]] = arith.constant
    %c0 = arith.constant 0 : index
    // CHECK-NEXT: %[[ret:.*]] = vector.extractelement %[[vec]][%[[const]] : index]
    %ret = miopen.extract_slice %vec[%c0] : vector<8xf32> -> f32
    // CHECK-NEXT: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func @extract_slice_vector
// CHECK-SAME: (%[[vec:.*]]: vector<8xf32>)
func @extract_slice_vector(%vec: vector<8xf32>) -> vector<2xf32> {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[r0:.*]] = arith.constant {{.*}} : vector<2xf32>
    // CHECK-DAG: %[[v0:.*]] = vector.extractelement %[[vec]]{{.*}} : vector<8xf32>
    // CHECK-DAG: %[[r1:.*]] = vector.insertelement %[[v0]], %[[r0]]{{.*}} : vector<2xf32>
    // CHECK-DAG: %[[v1:.*]] = vector.extractelement %[[vec]]{{.*}} : vector<8xf32>
    // CHECK-DAG: %[[r2:.*]] = vector.insertelement %[[v1]], %[[r1]]{{.*}} : vector<2xf32>
    %ret = miopen.extract_slice %vec[%c0] : vector<8xf32> -> vector<2xf32>
    // CHECK: return %[[r2]] : vector<2xf32>
    return %ret : vector<2xf32>
}

// CHECK-LABEL: func @extract_slice_noop
// CHECK-SAME: (%[[v:.*]]: vector<8xf32>)
func @extract_slice_noop(%v: vector<8xf32>) -> vector<8xf32> {
    // CHECK: return %[[v]]
    %c0 = arith.constant 0 : index
    %w = miopen.extract_slice %v[%c0] : vector<8xf32> -> vector<8xf32>
    return %w : vector<8xf32>
}

// CHECK-LABEL: func @insert_slice_scalar
// CHECK-SAME: (%[[v:.*]]: f32, %[[vec:.*]]: vector<8xf32>)
func @insert_slice_scalar(%v : f32, %vec : vector<8xf32>) -> vector<8xf32> {
    // CHECK-NEXT: %[[c0:.*]] = arith.constant
    %c0 = arith.constant 0 : index
    // CHECK-NEXT: %[[ret:.*]] = vector.insertelement %[[v]], %[[vec]][%[[c0]] : index]
    %ret = miopen.insert_slice %v -> %vec[%c0] : f32 -> vector<8xf32>
    // CHECK-NEXT: %[[ret]] : vector<8xf32>
    return %ret : vector<8xf32>
}

//CHECK-LABEL: func @insert_slice_vector
//CHECK-SAME: (%[[v:.*]]: vector<2xf32>, %[[vec:.*]]: vector<8xf32>)
func @insert_slice_vector(%v: vector<2xf32>, %vec: vector<8xf32>) -> vector<8xf32> {
    %c2 = arith.constant 2 : index
    // CHECK-DAG: %[[c0:.*]] = arith.constant 0
    // CHECK-DAG: %[[c1:.*]] = arith.constant 1
    // CHECK-DAG: %[[v0:.*]] = vector.extractelement %[[v]][%[[c0]] : index] : vector<2xf32>
    // CHECK-DAG: %[[r0:.*]] = vector.insertelement %[[v0]], %[[vec]]{{.*}} : vector<8xf32>
    // CHECK-DAG: %[[v1:.*]] = vector.extractelement %[[v]][%[[c1]] : index] : vector<2xf32>
    // CHECK-DAG: %[[ret:.*]] = vector.insertelement %[[v1]], %[[r0]]{{.*}} : vector<8xf32>
    %ret = miopen.insert_slice %v -> %vec[%c2] : vector<2xf32> -> vector<8xf32>
    // CHECK: return %[[ret]] : vector<8xf32>
    return %ret : vector<8xf32>
}

// CHECK-LABEL: func @insert_slice_noop
// CHECK-SAME: (%[[v:.*]]: vector<8xf32>, %[[w:.*]]: vector<8xf32>)
func @insert_slice_noop(%v: vector<8xf32>, %w: vector<8xf32>) -> vector<8xf32> {
    // CHECK: return %[[w]]
    %c0 = arith.constant 0 : index
    %r = miopen.insert_slice %w -> %v[%c0] : vector<8xf32> -> vector<8xf32>
    return %r : vector<8xf32>
}

}
