// RUN: rocmlir-opt -rock-sugar-to-loops %s | FileCheck %s

module {
// CHECK-LABEL: func.func @extract_slice_scalar
// CHECK-SAME: (%[[vec:.*]]: vector<8xf32>)
func.func @extract_slice_scalar(%vec : vector<8xf32>) -> f32 {
    %c0 = arith.constant 0 : index
    // CHECK-NEXT: %[[ret:.*]] = vector.extract %[[vec]][0] : f32 from vector<8xf32>
    %ret = rock.extract_slice %vec[%c0] : vector<8xf32> -> f32
    // CHECK-NEXT: return %[[ret]]
    return %ret : f32
}

// CHECK-LABEL: func.func @extract_slice_vector
// CHECK-SAME: (%[[vec:.*]]: vector<8xf32>)
func.func @extract_slice_vector(%vec: vector<8xf32>) -> vector<2xf32> {
    %c0 = arith.constant 0 : index
    // CHECK-DAG: %[[r0:.*]] = arith.constant {{.*}} : vector<2xf32>
    // CHECK-DAG: %[[v0:.*]] = vector.extract %[[vec]][0] : f32 from vector<8xf32>
    // CHECK-DAG: %[[r1:.*]] = vector.insert %[[v0]], %[[r0]] [0] : f32 into vector<2xf32>
    // CHECK-DAG: %[[v1:.*]] = vector.extract %[[vec]][1] : f32 from vector<8xf32>
    // CHECK-DAG: %[[r2:.*]] = vector.insert %[[v1]], %[[r1]] [1] : f32 into vector<2xf32>
    %ret = rock.extract_slice %vec[%c0] : vector<8xf32> -> vector<2xf32>
    // CHECK: return %[[r2]] : vector<2xf32>
    return %ret : vector<2xf32>
}

// CHECK-LABEL: func.func @extract_slice_noop
// CHECK-SAME: (%[[v:.*]]: vector<8xf32>)
func.func @extract_slice_noop(%v: vector<8xf32>) -> vector<8xf32> {
    // CHECK: return %[[v]]
    %c0 = arith.constant 0 : index
    %w = rock.extract_slice %v[%c0] : vector<8xf32> -> vector<8xf32>
    return %w : vector<8xf32>
}

// CHECK-LABEL: func.func @insert_slice_scalar
// CHECK-SAME: (%[[v:.*]]: f32, %[[vec:.*]]: vector<8xf32>)
func.func @insert_slice_scalar(%v : f32, %vec : vector<8xf32>) -> vector<8xf32> {
    %c0 = arith.constant 0 : index
    // CHECK-NEXT: %[[ret:.*]] = vector.insert %[[v]], %[[vec]] [0] : f32 into vector<8xf32>
    %ret = rock.insert_slice %v -> %vec[%c0] : f32 -> vector<8xf32>
    // CHECK-NEXT: %[[ret]] : vector<8xf32>
    return %ret : vector<8xf32>
}

//CHECK-LABEL: func.func @insert_slice_vector
//CHECK-SAME: (%[[v:.*]]: vector<2xf32>, %[[vec:.*]]: vector<8xf32>)
func.func @insert_slice_vector(%v: vector<2xf32>, %vec: vector<8xf32>) -> vector<8xf32> {
    %c2 = arith.constant 2 : index
    // CHECK-DAG: %[[v0:.*]] = vector.extract %[[v]][0] : f32 from vector<2xf32>
    // CHECK-DAG: %[[r0:.*]] = vector.insert %[[v0]], %[[vec]] [2] : f32 into vector<8xf32>
    // CHECK-DAG: %[[v1:.*]] = vector.extract %[[v]][1] : f32 from vector<2xf32>
    // CHECK-DAG: %[[ret:.*]] = vector.insert %[[v1]], %[[r0]] [3] : f32 into vector<8xf32>
    %ret = rock.insert_slice %v -> %vec[%c2] : vector<2xf32> -> vector<8xf32>
    // CHECK: return %[[ret]] : vector<8xf32>
    return %ret : vector<8xf32>
}

// CHECK-LABEL: func.func @insert_slice_noop
// CHECK-SAME: (%[[v:.*]]: vector<8xf32>, %[[w:.*]]: vector<8xf32>)
func.func @insert_slice_noop(%v: vector<8xf32>, %w: vector<8xf32>) -> vector<8xf32> {
    // CHECK: return %[[w]]
    %c0 = arith.constant 0 : index
    %r = rock.insert_slice %w -> %v[%c0] : vector<8xf32> -> vector<8xf32>
    return %r : vector<8xf32>
}

}
