// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise %s | FileCheck %s

// CHECK-LABEL: func.func @rock_global_store
func.func @rock_global_store(%source : memref<32xf32, 5>,
                                %dest2D : memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: %[[slice:.*]] = rock.in_bounds_load{{.*}}: memref<32xf32, 5>, index -> vector<4xf32>
  // CHECK: rock.buffer_store set %[[slice]]{{.*}}: vector<4xf32> -> memref<32x32xf32>
  rock.global_store %source[%c0] -> %dest2D[%c0, %c0] if %true
      storeMethod(set) { length = 4 : index }
    : memref<32xf32, 5> -> memref<32x32xf32>, index, index
  func.return
}

// CHECK-LABEL: func @rock_global_store_long
func.func @rock_global_store_long(%source : memref<32xf32, 5>,
                                %dest2D : memref<32x32xf32>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}[%[[coords:[^{]*]] if{{.*}}: vector<4xf32> -> memref<32x32xf32>
  // CHECK: rock.buffer_store set {{.*}}%[[coords]] if{{.*}}offset = 4 : index{{.*}}: vector<4xf32> -> memref<32x32xf32>
  rock.global_store %source[%c0] -> %dest2D[%c0, %c0] if %true
      storeMethod(set) { length = 8 : index }
    : memref<32xf32, 5> -> memref<32x32xf32>, index, index
  func.return
}

// CHECK-LABEL: func @rock_global_store_8xf16
func.func @rock_global_store_8xf16(%source : memref<32xf16, 5>,
                                           %dest : memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_store
  // CHECK-NOT: rock.buffer_store
  rock.global_store %source[%c0] -> %dest[%c0] if %true
      storeMethod(set) { length = 8 : index }
    : memref<32xf16, 5> -> memref<64xf16>, index
  func.return
}

// CHECK-LABEL: func @rock_global_store_7xf16
func.func @rock_global_store_7xf16(%source : memref<32xf16, 5>,
                                           %dest : memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_store {{.*}} : vector<4xf16>
  // CHECK: rock.buffer_store {{.*}} : vector<2xf16>
  // CHECK: rock.buffer_store {{.*}} : f16
  // CHECK-NOT: rock.buffer_store
  rock.global_store %source[%c0] -> %dest[%c0] if %true
      storeMethod(set) { length = 7 : index }
    : memref<32xf16, 5> -> memref<64xf16>, index
  func.return
}

// CHECK-LABEL: func.func @rock_global_load
func.func @rock_global_load(%source2D : memref<32x32xf32>) -> vector<4xf32> {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: %[[loaded:.*]] = rock.buffer_load {{.*}}: memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret:.*]] = rock.insert_slice %[[loaded]]
  %loaded = rock.global_load %source2D[%c0, %c0] if %true
    : memref<32x32xf32> -> vector<4xf32>
  // CHECK: return %[[ret]]
  func.return %loaded : vector<4xf32>
}

// CHECK-LABEL: func.func @rock_global_load_long
func.func @rock_global_load_long(%source2D : memref<32x32xf32>) -> vector<8xf32> {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: %[[ret_0:.*]] = arith.constant {{.*}} : vector<8xf32>
  // CHECK: %[[loaded_0:.*]] = rock.buffer_load {{.*}}: memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret_1:.*]] = rock.insert_slice %[[loaded_0]] -> %[[ret_0]]
  // CHECK: %[[loaded_1:.*]] = rock.buffer_load {{.*}} {offset = 4 : index} : memref<32x32xf32>, index, index -> vector<4xf32>
  // CHECK: %[[ret:.*]] = rock.insert_slice %[[loaded_1]] -> %[[ret_1]]
  %loaded = rock.global_load %source2D[%c0, %c0] if %true
    : memref<32x32xf32> -> vector<8xf32>
  // CHECK: return %[[ret]]
  func.return %loaded : vector<8xf32>
}

// CHECK-LABEL: func.func @rock_global_load_8xf16
func.func @rock_global_load_8xf16(%source2D : memref<32x32xf16>) -> vector<8xf16> {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_load
  // CHECK-NOT: rock.buffer_load
  %loaded = rock.global_load %source2D[%c0, %c0] if %true
    : memref<32x32xf16> -> vector<8xf16>
  func.return %loaded : vector<8xf16>
}

// CHECK-LABEL: func.func @rock_global_load_7xf16
func.func @rock_global_load_7xf16(%source2D : memref<32x32xf16>) -> vector<7xf16> {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_load
  // CHECK: rock.buffer_load
  // CHECK: rock.buffer_load
  // CHECK-NOT: rock.buffer_load
  %loaded = rock.global_load %source2D[%c0, %c0] if %true
    : memref<32x32xf16> -> vector<7xf16>
  func.return %loaded : vector<7xf16>
}

// CHECK-LABEL: func.func @rock_global_load_7xi8
func.func @rock_global_load_7xi8(%source2D : memref<32x32xi8>) -> vector<7xi8> {
  %c0 = arith.constant 0 : index
  %true = arith.constant true
  // CHECK: rock.buffer_load {{.*}} -> vector<4xi8>
  // CHECK: rock.buffer_load {{.*}} -> i8
  // CHECK: rock.buffer_load {{.*}} -> i8
  // CHECK: rock.buffer_load {{.*}} -> i8
  // CHECK-NOT: rock.buffer_load
  %loaded = rock.global_load %source2D[%c0, %c0] if %true
    : memref<32x32xi8> -> vector<7xi8>
  func.return %loaded : vector<7xi8>
}

