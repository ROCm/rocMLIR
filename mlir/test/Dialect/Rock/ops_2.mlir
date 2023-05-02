// RUN: rocmlir-opt %s | FileCheck %s
// RUN: rocmlir-opt %s | rocmlir-opt | FileCheck %s
// Run: rocmlir-opt -mlir-print-op-generic %s | rocmlir-opt | FileCheck %s

func.func @rock_alloc() {
  // allocation on global.
  %buffer_global = rock.alloc() : memref<1024xi8>

  // allocation on LDS.
  %buffer_lds = rock.alloc() : memref<1024xi8, 3>

  // allocation on register (VGPR).
  %buffer_register = rock.alloc() : memref<1024xi8, 5>

  return
}

// CHECK-LABEL: func.func @rock_alloc
//   CHECK: rock.alloc
//   CHECK-NEXT: rock.alloc
//   CHECK-NEXT: rock.alloc


func.func @rock_fill(%buffer_f32 : memref<1024xf32, 5>, %buffer_i32 : memref<2xi32, 5>, %buffer_f16 : memref<1024xf16, 5>) {
  %cst = arith.constant 0.0 : f32
  rock.fill(%buffer_f32, %cst) : memref<1024xf32, 5>, f32

  %cst_f16 = arith.constant 0.0 : f16
  rock.fill(%buffer_f16, %cst_f16) : memref<1024xf16, 5>, f16

  %c0 = arith.constant 0 : i32
  rock.fill(%buffer_i32, %c0) : memref<2xi32, 5>, i32
  return
}

// CHECK-LABEL: func.func @rock_fill
//   CHECK: rock.fill
//   CHECK: rock.fill
//   CHECK: rock.fill

func.func @rock_workgroup_barrier() {
  rock.workgroup_barrier
  return
}

// CHECK-LABEL: func.func @rock_workgroup_barrier
//   CHECK-NEXT: rock.workgroup_barrier

func.func @rock_lds_barrier() {
  rock.lds_barrier
  return
}

// CHECK-LABEL: func.func @rock_lds_barrier
//   CHECK-NEXT: rock.lds_barrier

func.func @rock_indexing() {
  %0 = rock.workgroup_id : index
  %1 = rock.workitem_id : index
  return
}

// CHECK-LABEL: func.func @rock_indexing
//   CHECK-NEXT: rock.workgroup_id
//   CHECK-NEXT: rock.workitem_id

func.func @rock_blockwise_gemm(%A : memref<8x128x1xf32, 3>, %B : memref<8x128x1xf32, 3>, %C : memref<8x8xf32, 5>) {
  rock.blockwise_gemm %C += %A * %B {
    params = #rock.general_gemm_params<
    blockSize = 256,
    kPerBlock = 8,
    mPerBlock = 128,
    nPerBlock = 128,
    kpack = 1,
    kPerThread = 1,
    mPerThread = 4,
    nPerThread = 4>
  } :  memref<8x8xf32, 5> += memref<8x128x1xf32, 3> * memref<8x128x1xf32, 3>
  return
}

// --------------------------
// buffer_load tests.

// f32 tests.

func.func @rock_buffer_load_f32(%source : memref<?x?x?xf32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> f32  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf32>, index, index, index -> f32
  return %result : f32
}

// CHECK-LABEL: func.func @rock_buffer_load_f32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf32>, index, index, index -> f32

func.func @rock_buffer_load_2xf32(%source : memref<?x?x?xf32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf32>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xf32>, index, index, index -> vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func.func @rock_buffer_load_2xf32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf32>, index, index, index -> vector<2xf32>

func.func @rock_buffer_load_4xf32(%source : memref<?x?x?xf32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf32>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf32>, index, index, index -> vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func.func @rock_buffer_load_4xf32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf32>, index, index, index -> vector<4xf32>

// f16 tests.

func.func @rock_buffer_load_f16(%source : memref<?x?x?xf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> f16  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf16>, index, index, index -> f16
  return %result : f16
}

// CHECK-LABEL: func.func @rock_buffer_load_f16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf16>, index, index, index -> f16

func.func @rock_buffer_load_2xf16(%source : memref<?x?x?xf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf16>, index, index, index -> vector<2xf16>
  return %result : vector<2xf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_2xf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf16>, index, index, index -> vector<2xf16>

func.func @rock_buffer_load_4xf16(%source : memref<?x?x?xf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf16>, index, index, index -> vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_4xf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf16>, index, index, index -> vector<4xf16>

func.func @rock_buffer_load_8xf16(%source : memref<?x?x?xf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
    : memref<?x?x?xf16>, index, index, index -> vector<8xf16>
  return %result : vector<8xf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_8xf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xf16>, index, index, index -> vector<8xf16>

// bf16 tests.

func.func @rock_buffer_load_bf16(%source : memref<?x?x?xbf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> bf16  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xbf16>, index, index, index -> bf16
  return %result : bf16
}

// CHECK-LABEL: func.func @rock_buffer_load_bf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xbf16>, index, index, index -> bf16

func.func @rock_buffer_load_2xbf16(%source : memref<?x?x?xbf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xbf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>
  return %result : vector<2xbf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_2xbf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>

func.func @rock_buffer_load_4xbf16(%source : memref<?x?x?xbf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xbf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>
  return %result : vector<4xbf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_4xbf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>

func.func @rock_buffer_load_8xbf16(%source : memref<?x?x?xbf16>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xbf16>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2]  if %valid
   : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>
  return %result : vector<8xbf16>
}

// CHECK-LABEL: func.func @rock_buffer_load_8xbf16
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>

// i8 tests.

func.func @rock_buffer_load_i8(%source : memref<?x?x?xi8>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> i8  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xi8>, index, index, index -> i8
  return %result : i8
}

// CHECK-LABEL: func.func @rock_buffer_load_i8
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xi8>, index, index, index -> i8

func.func @rock_buffer_load_4xi8(%source : memref<?x?x?xi8>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xi8>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xi8>, index, index, index -> vector<4xi8>
  return %result : vector<4xi8>
}

// CHECK-LABEL: func.func @rock_buffer_load_4xi8
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xi8>, index, index, index -> vector<4xi8>

// i32 tests.

func.func @rock_buffer_load_i32(%source : memref<?x?x?xi32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> i32  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xi32>, index, index, index -> i32
  return %result : i32
}

// CHECK-LABEL: func.func @rock_buffer_load_i32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xi32>, index, index, index -> i32

func.func @rock_buffer_load_2xi32(%source : memref<?x?x?xi32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xi32>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xi32>, index, index, index -> vector<2xi32>
  return %result : vector<2xi32>
}

// CHECK-LABEL: func.func @rock_buffer_load_2xi32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xi32>, index, index, index -> vector<2xi32>

func.func @rock_buffer_load_4xi32(%source : memref<?x?x?xi32>, %valid : i1, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xi32>  {
  %result = rock.buffer_load %source[%sc0, %sc1, %sc2] if %valid
   : memref<?x?x?xi32>, index, index, index -> vector<4xi32>
  return %result : vector<4xi32>
}

// CHECK-LABEL: func.func @rock_buffer_load_4xi32
// CHECK: %{{.*}} = rock.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] if %{{.*}} : memref<?x?x?xi32>, index, index, index -> vector<4xi32>

// --------------------------
// buffer_store tests

// f32 tests.

func.func @rock_buffer_store_f32(%data: f32, %dest: memref<1x1x1x1x16xf32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : f32 -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_f32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_2xf32(%data: vector<2xf32>, %dest: memref<1x1x1x1x16xf32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<2xf32> -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_2xf32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_4xf32(%data: vector<4xf32>, %dest: memref<1x1x1x1x16xf32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<4xf32> -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_4xf32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

// f16 tests.

func.func @rock_buffer_store_f16(%data: f16, %dest: memref<1x1x1x1x16xf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : f16 -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_f16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_2xf16(%data: vector<2xf16>, %dest: memref<1x1x1x1x16xf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<2xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_2xf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_4xf16(%data: vector<4xf16>, %dest: memref<1x1x1x1x16xf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<4xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_4xf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_8xf16(%data: vector<8xf16>, %dest: memref<1x1x1x1x16xf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<8xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_8xf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

// bf16 tests.

func.func @rock_buffer_store_bf16(%data: bf16, %dest: memref<1x1x1x1x16xbf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : bf16 -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_bf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_2xbf16(%data: vector<2xbf16>, %dest: memref<1x1x1x1x16xbf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<2xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_2xbf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_4xbf16(%data: vector<4xbf16>, %dest: memref<1x1x1x1x16xbf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<4xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_4xbf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_8xbf16(%data: vector<8xbf16>, %dest: memref<1x1x1x1x16xbf16>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<8xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_8xbf16
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

// i8 tests.

func.func @rock_buffer_store_i8(%data: i8, %dest: memref<1x1x1x1x16xi8>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : i8 -> memref<1x1x1x1x16xi8>, index, index, index, index, index
          return
}

// CHECK-LABEL: func.func @rock_buffer_store_i8
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_4xi8(%data: vector<4xi8>, %dest: memref<1x1x1x1x16xi8>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<4xi8> -> memref<1x1x1x1x16xi8>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_4xi8
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

// i32 tests.
func.func @rock_buffer_store_i32(%data: i32, %dest: memref<1x1x1x1x16xi32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : i32 -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_i32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_2xi32(%data: vector<2xi32>, %dest: memref<1x1x1x1x16xi32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<2xi32> -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_2xi32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

func.func @rock_buffer_store_4xi32(%data: vector<4xi32>, %dest: memref<1x1x1x1x16xi32>, %valid: i1, %idx: index) {
  %c0 = arith.constant 0 : index
  rock.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    if %valid features = none
   : vector<4xi32> -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func.func @rock_buffer_store_4xi32
// CHECK: rock.buffer_store set %{{.*}} -> %{{.*}}

// --------------------------
// global_load tests.

func.func @rock_global_load(%source : memref<?x?x?x?x?xf32>, %valid : i1) -> vector<8xf32> {
  %c1 = arith.constant 1 : index
  // check source and destination with coordinate transforms.
  %loaded = rock.global_load
    %source[%c1, %c1, %c1, %c1, %c1] if %valid
    : memref<?x?x?x?x?xf32> -> vector<8xf32>

  return %loaded : vector<8xf32>
}

// CHECK-LABEL: func.func @rock_global_load
// CHECK: rock.global_load

// --------------------------
// global_store tests.

func.func @rock_global_store(%source : memref<32xf32, 5>,
                                %dest : memref<?x?x?x?x?xf32>, %valid : i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // check source and destination with coordinate transforms.
  rock.global_store
    %source[%c0] ->
    %dest[%c1, %c1, %c1, %c1, %c1]
    if %valid
    storeMethod(set)
    features = none
    {length = 1 : index}
    : memref<32xf32, 5>
    -> memref<?x?x?x?x?xf32>, index, index, index, index, index

  return
}

// CHECK-LABEL: func.func @rock_global_store
// CHECK: rock.global_store

func.func @rock_threadwise_gemm(%lhs : memref<4x8x1xf32, 5>, %rhs : memref<4x8x1xf32, 5>, %output : memref<8x8xf32, 5>) {
  rock.threadwise_gemm %output += %lhs * %rhs
  : memref<8x8xf32, 5> += memref<4x8x1xf32, 5> * memref<4x8x1xf32, 5>
  return
}

// CHECK-LABEL: func.func @rock_threadwise_gemm
// CHECK: rock.threadwise_gemm

// ----

func.func @rock_accel_gemm_one_result(%matrixA : memref<16xf32, 5>,
                                            %matrixB : memref<16xf32, 5>,
                                            %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.accel_gemm %matrixC += %matrixA[%c0] * %matrixB[%c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> += memref<16xf32, 5> * memref<16xf32, 5>
  return
}

// CHECK-LABEL: func.func @rock_accel_gemm_one_result
// CHECK: rock.accel_gemm

// ----

func.func @rock_accel_gemm_two_results(%matrixA : memref<16xf32, 5>,
                                             %matrixB : memref<16xf32, 5>,
                                             %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.accel_gemm %matrixC += %matrixA[%c0] * %matrixB[%c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> += memref<16xf32, 5> * memref<16xf32, 5>
  return
}

// CHECK-LABEL: func.func @rock_accel_gemm_two_results
// CHECK: rock.accel_gemm

// ----

func.func @rock_blockwise_gemm_accel_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                              %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>,
                                              %matrixC : memref<1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1xvector<32xf32>, 5> += memref<32xf32, 5> from memref<12288xf32, 3> * memref<16xf32, 5> from memref<12288xf32, 3>
  return
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_accel_one_result
// CHECK: rock.blockwise_gemm_accel

// ----

func.func @rock_blockwise_gemm_accel_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                                %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>,
                                                %matrixC : memref<2xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA[%c0] * %bufferB from %matrixB[%c0] features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<2xvector<32xf32>, 5> += memref<32xf32, 5> from memref<12288xf32, 3> * memref<16xf32, 5> from memref<12288xf32, 3>
  return
}

// CHECK-LABEL: func.func @rock_blockwise_gemm_accel_two_results
// CHECK: rock.blockwise_gemm_accel
