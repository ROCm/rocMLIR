// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise -split-input-file %s | FileCheck %s

// CHECK-DAG: [[MAP:.*]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-DAG: [[TMAP:.*]] = #rock.transform_map<[[MAP]]{{.*}}
// CHECK-DAG: [[MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[TMAP1:.*]] = #rock.transform_map<[[MAP1]]{{.*}}[<Pad{0, 0} ["dim0"] at [0] -> ["dim0"] at [0]>]
// CHECK: func @rock_blockwise_fill_scalar_case1
func.func @rock_blockwise_fill_scalar_case1(%ldsbuf : memref<256xf32, #gpu.address_space<workgroup>>){
    %c1 = arith.constant 1.0 : f32
    // CHECK: rock.transforming_for {{.*}} [[[TMAP]], [[TMAP1]]]
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<256xf32, #gpu.address_space<workgroup>>, f32
    return
}

// -----

// CHECK-DAG: [[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-DAG: [[TMAP:.*]] = #rock.transform_map<[[MAP]]{{.*}}
// CHECK-DAG: [[MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[TMAP1:.*]] = #rock.transform_map<[[MAP1]]{{.*}}[<Pad{0, 212} ["dim0"] at [0] -> ["dim0"] at [0]>]
// CHECK: func @rock_blockwise_fill_scalar_case2
func.func @rock_blockwise_fill_scalar_case2(%ldsbuf : memref<300xf32, #gpu.address_space<workgroup>>){
    %c1 = arith.constant 1.0 : f32
    // CHECK: rock.transforming_for {{.*}} [[[TMAP]], [[TMAP1]]]
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<300xf32, #gpu.address_space<workgroup>>, f32
    return
}

// -----

// CHECK-DAG: [[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: [[TMAP:.*]] = #rock.transform_map<[[MAP]]{{.*}}
// CHECK-DAG: [[MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[TMAP1:.*]] = #rock.transform_map<[[MAP1]]{{.*}}[<Pad{0, 960} ["dim0"] at [0] -> ["dim0"] at [0]>]
// CHECK: func @rock_blockwise_fill_vec_case1
func.func @rock_blockwise_fill_vec_case1(%ldsbuf : memref<64xf32, #gpu.address_space<workgroup>>){
    %c1 = arith.constant dense<1.0> : vector<4xf32>
    // CHECK: rock.transforming_for {{.*}} [[[TMAP]], [[TMAP1]]]
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<64xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    return
}

// -----

// CHECK-DAG: [[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: [[TMAP:.*]] = #rock.transform_map<[[MAP]]{{.*}}
// CHECK-DAG: [[MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[TMAP1:.*]] = #rock.transform_map<[[MAP1]]{{.*}}[<Pad{0, 768} ["dim0"] at [0] -> ["dim0"] at [0]>]
// CHECK: func @rock_blockwise_fill_vec_case2
func.func @rock_blockwise_fill_vec_case2(%ldsbuf : memref<256xf32, #gpu.address_space<workgroup>>){
    %c1 = arith.constant dense<1.0> : vector<4xf32>
    // CHECK: rock.transforming_for {{.*}} [[[TMAP]], [[TMAP1]]]
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<256xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    return
}

// -----

// CHECK-DAG: [[MAP:.*]] = affine_map<(d0, d1) -> (d0 * 4 + d1)>
// CHECK-DAG: [[TMAP:.*]] = #rock.transform_map<[[MAP]]{{.*}}
// CHECK-DAG: [[MAP1:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[TMAP1:.*]] = #rock.transform_map<[[MAP1]]{{.*}}[<Pad{0, 0} ["dim0"] at [0] -> ["dim0"] at [0]>]
// CHECK: func @rock_blockwise_fill_vec_case3
func.func @rock_blockwise_fill_vec_case3(%ldsbuf : memref<1024xf32, #gpu.address_space<workgroup>>){
    %c1 = arith.constant dense<1.0> : vector<4xf32>
    // CHECK: rock.transforming_for {{.*}} [[[TMAP]], [[TMAP1]]]
    rock.blockwise_fill(%ldsbuf, %c1) {blockSize = 256 : i32} : memref<1024xf32, #gpu.address_space<workgroup>>, vector<4xf32>
    return
}
