// RUN: miopen-opt %s | FileCheck %s
// RUN: miopen-opt %s | miopen-opt | FileCheck %s
// Run: miopen-opt -mlir-print-op-generic %s | miopen-opt | FileCheck %s

func @miopen_alloc() {
  // allocation on global.
  %buffer_global = miopen.alloc() : memref<1024xi8>

  // allocation on LDS.
  %buffer_lds = miopen.alloc() : memref<1024xi8, 3>

  // allocation on register (VGPR).
  %buffer_register = miopen.alloc() : memref<1024xi8, 5>

  return
}

// CHECK-LABEL: func @miopen_alloc
//   CHECK: miopen.alloc
//   CHECK-NEXT: miopen.alloc
//   CHECK-NEXT: miopen.alloc


func @miopen_fill(%buffer_f32 : memref<1024xf32, 5>, %buffer_i32 : memref<2xi32, 5>, %buffer_f16 : memref<1024xf16, 5>) {
  %cst = arith.constant 0.0 : f32
  miopen.fill(%buffer_f32, %cst) : memref<1024xf32, 5>, f32

  %cst_f16 = arith.constant 0.0 : f16
  miopen.fill(%buffer_f16, %cst_f16) : memref<1024xf16, 5>, f16

  %c0 = arith.constant 0 : i32
  miopen.fill(%buffer_i32, %c0) : memref<2xi32, 5>, i32
  return
}

// CHECK-LABEL: func @miopen_fill
//   CHECK: miopen.fill
//   CHECK: miopen.fill
//   CHECK: miopen.fill

func @miopen_workgroup_barrier() {
  miopen.workgroup_barrier
  return
}

// CHECK-LABEL: func @miopen_workgroup_barrier
//   CHECK-NEXT: miopen.workgroup_barrier

func @miopen_lds_barrier() {
  miopen.lds_barrier
  return
}

// CHECK-LABEL: func @miopen_lds_barrier
//   CHECK-NEXT: miopen.lds_barrier

func @miopen_indexing() {
  %0 = miopen.workgroup_id : index
  %1 = miopen.workitem_id : index
  return
}

#gemm_padding0 = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0>

// CHECK-LABEL: func @miopen_indexing
//   CHECK-NEXT: miopen.workgroup_id
//   CHECK-NEXT: miopen.workitem_id

func @miopen_blockwise_gemm(%A : memref<?x?x?xf32, 3>, %B : memref<?x?x?xf32, 3>, %C : memref<?x?x?xf32, 5>) {
  %c0 = arith.constant 0 : index
  miopen.blockwise_gemm(%A, %B, %C, %c0, %c0) {
    transforms = [[], []],

    m_per_thread = 64,
    n_per_thread = 64,
    k_per_thread = 16,

    m_level0_cluster = 16,
    n_level0_cluster = 16,
    m_level1_cluster = 16,
    n_level1_cluster = 16,

    matrix_a_source_data_per_read = 4,
    matrix_b_source_data_per_read = 4
  } : memref<?x?x?xf32, 3>, memref<?x?x?xf32, 3>, memref<?x?x?xf32, 5>, index, index
  return
}

// --------------------------
// buffer_load tests.

// f32 tests.

func @miopen_buffer_load_f32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> f32  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf32>, index, index, index -> f32
  return %result : f32
}

// CHECK-LABEL: func @miopen_buffer_load_f32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf32>, index, index, index -> f32

func @miopen_buffer_load_2xf32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf32>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf32>, index, index, index -> vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @miopen_buffer_load_2xf32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf32>, index, index, index -> vector<2xf32>

func @miopen_buffer_load_4xf32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf32>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf32>, index, index, index -> vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func @miopen_buffer_load_4xf32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf32>, index, index, index -> vector<4xf32>

// f16 tests.

func @miopen_buffer_load_f16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> f16  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf16>, index, index, index -> f16
  return %result : f16
}

// CHECK-LABEL: func @miopen_buffer_load_f16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf16>, index, index, index -> f16

func @miopen_buffer_load_2xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf16>, index, index, index -> vector<2xf16>
  return %result : vector<2xf16>
}

// CHECK-LABEL: func @miopen_buffer_load_2xf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<2xf16>

func @miopen_buffer_load_4xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf16>, index, index, index -> vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func @miopen_buffer_load_4xf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<4xf16>

func @miopen_buffer_load_8xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xf16>, index, index, index -> vector<8xf16>
  return %result : vector<8xf16>
}

// CHECK-LABEL: func @miopen_buffer_load_8xf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<8xf16>

// bf16 tests.

func @miopen_buffer_load_bf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> bf16  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xbf16>, index, index, index -> bf16
  return %result : bf16
}

// CHECK-LABEL: func @miopen_buffer_load_bf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xbf16>, index, index, index -> bf16

func @miopen_buffer_load_2xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xbf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>
  return %result : vector<2xbf16>
}

// CHECK-LABEL: func @miopen_buffer_load_2xbf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>

func @miopen_buffer_load_4xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xbf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>
  return %result : vector<4xbf16>
}

// CHECK-LABEL: func @miopen_buffer_load_4xbf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>

func @miopen_buffer_load_8xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xbf16>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2]  {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>
  return %result : vector<8xbf16>
}

// CHECK-LABEL: func @miopen_buffer_load_8xbf16
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>

// i8 tests.

func @miopen_buffer_load_i8(%source : memref<?x?x?xi8>, %sc0 : index, %sc1 : index, %sc2 : index) -> i8  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xi8>, index, index, index -> i8
  return %result : i8
}

// CHECK-LABEL: func @miopen_buffer_load_i8
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xi8>, index, index, index -> i8

func @miopen_buffer_load_4xi8(%source : memref<?x?x?xi8>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xi8>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xi8>, index, index, index -> vector<4xi8>
  return %result : vector<4xi8>
}

// CHECK-LABEL: func @miopen_buffer_load_4xi8
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xi8>, index, index, index -> vector<4xi8>

// i32 tests.

func @miopen_buffer_load_i32(%source : memref<?x?x?xi32>, %sc0 : index, %sc1 : index, %sc2 : index) -> i32  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xi32>, index, index, index -> i32
  return %result : i32
}

// CHECK-LABEL: func @miopen_buffer_load_i32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xi32>, index, index, index -> i32

func @miopen_buffer_load_2xi32(%source : memref<?x?x?xi32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xi32>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xi32>, index, index, index -> vector<2xi32>
  return %result : vector<2xi32>
}

// CHECK-LABEL: func @miopen_buffer_load_2xi32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xi32>, index, index, index -> vector<2xi32>

func @miopen_buffer_load_4xi32(%source : memref<?x?x?xi32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xi32>  {
  %result = miopen.buffer_load %source[%sc0, %sc1, %sc2] {
    leftOobDims = [], rightOobDims = []
  } : memref<?x?x?xi32>, index, index, index -> vector<4xi32>
  return %result : vector<4xi32>
}

// CHECK-LABEL: func @miopen_buffer_load_4xi32
// CHECK: %{{.*}} = miopen.buffer_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xi32>, index, index, index -> vector<4xi32>

// --------------------------
// buffer_store tests

// f32 tests.

func @miopen_buffer_store_f32(%data: f32, %dest: memref<1x1x1x1x16xf32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : f32 -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_f32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_2xf32(%data: vector<2xf32>, %dest: memref<1x1x1x1x16xf32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<2xf32> -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_2xf32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_4xf32(%data: vector<4xf32>, %dest: memref<1x1x1x1x16xf32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<4xf32> -> memref<1x1x1x1x16xf32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_4xf32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

// f16 tests.

func @miopen_buffer_store_f16(%data: f16, %dest: memref<1x1x1x1x16xf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : f16 -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_f16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_2xf16(%data: vector<2xf16>, %dest: memref<1x1x1x1x16xf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<2xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_2xf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_4xf16(%data: vector<4xf16>, %dest: memref<1x1x1x1x16xf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<4xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_4xf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_8xf16(%data: vector<8xf16>, %dest: memref<1x1x1x1x16xf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<8xf16> -> memref<1x1x1x1x16xf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_8xf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

// bf16 tests.

func @miopen_buffer_store_bf16(%data: bf16, %dest: memref<1x1x1x1x16xbf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : bf16 -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_bf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_2xbf16(%data: vector<2xbf16>, %dest: memref<1x1x1x1x16xbf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<2xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_2xbf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_4xbf16(%data: vector<4xbf16>, %dest: memref<1x1x1x1x16xbf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<4xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_4xbf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_8xbf16(%data: vector<8xbf16>, %dest: memref<1x1x1x1x16xbf16>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<8xbf16> -> memref<1x1x1x1x16xbf16>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_8xbf16
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

// i8 tests.

func @miopen_buffer_store_i8(%data: i8, %dest: memref<1x1x1x1x16xi8>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : i8 -> memref<1x1x1x1x16xi8>, index, index, index, index, index
          return
}

// CHECK-LABEL: func @miopen_buffer_store_i8
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_4xi8(%data: vector<4xi8>, %dest: memref<1x1x1x1x16xi8>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<4xi8> -> memref<1x1x1x1x16xi8>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_4xi8
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

// i32 tests.
func @miopen_buffer_store_i32(%data: i32, %dest: memref<1x1x1x1x16xi32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : i32 -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_i32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_2xi32(%data: vector<2xi32>, %dest: memref<1x1x1x1x16xi32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<2xi32> -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_2xi32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

func @miopen_buffer_store_4xi32(%data: vector<4xi32>, %dest: memref<1x1x1x1x16xi32>, %idx: index) {
  %c0 = arith.constant 0 : index
  miopen.buffer_store set %data -> %dest[%c0, %c0, %c0, %c0, %idx]
    {leftOobDims = [], rightOobDims = []}
    : vector<4xi32> -> memref<1x1x1x1x16xi32>, index, index, index, index, index
  return
}

// CHECK-LABEL: func @miopen_buffer_store_4xi32
// CHECK: miopen.buffer_store set %{{.*}} -> %{{.*}}

// --------------------------
// threadwise_copy tests.

#transform_map0 = #miopen.transform_map<
  affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)> by
  [#miopen.transform<PassThrough ["b"] at [1] -> ["w"] at [0]>,
   #miopen.transform<Merge{3, 3, 3} ["a"] at [0] -> ["x", "y", "z"] at [1, 2, 3]>
  ] bounds = [1, 27] -> [1, 3, 3, 3]>

#transform_map1 = #miopen.transform_map<
  affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)> by [
    #miopen.transform<PassThrough ["w", "x", "y", "z"] at [0, 1, 2, 3]
      -> ["x", "w", "y", "z"] at [1, 0, 2, 3]>
    ] bounds = [1, 3, 3, 3] -> [3, 1, 3, 3]>

func @miopen_threadwise_copy(%source_coord : memref<2xindex, 5>, %dest_coord : memref<2xindex, 5>,
                             %source : memref<?x?xf32, 5>, %dest : memref<?x?xf32, 5>,
                             %source_with_transform_maps : memref<?x?x?x?xf32>,
                             %dest_with_transform_maps : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_coord_y = memref.load %source_coord[%c0] : memref<2xindex, 5>
  %source_coord_x = memref.load %source_coord[%c0] : memref<2xindex, 5>
  %dest_coord_y = memref.load %dest_coord[%c0] : memref<2xindex, 5>
  %dest_coord_x = memref.load %dest_coord[%c0] : memref<2xindex, 5>

  // check source and dest as vanilla memrefs.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[], []]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index]}
    : memref<?x?xf32, 5>, index, index -> memref<?x?xf32, 5>, index, index

  // -----

  // check source with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y] with [[#transform_map0], []]
    { paddingInfo = #gemm_padding0,
     bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?xf32, 5>, index, index

  // check source with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], []]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?xf32, 5>, index, index

  // check destination with one coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0]]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index] }
    : memref<?x?xf32, 5>, index, index -> memref<?x?x?x?xf32>, index, index

  // check destination with multiple coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index] }
    : memref<?x?xf32, 5>, index, index -> memref<?x?x?x?xf32>, index, index

  // -----

  // check source and destination with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0], [#transform_map0]]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?x?x?xf32>, index, index

  // check source and destination with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0,
      bounds = [1 : index, 1 : index] }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?x?x?xf32>, index, index

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy
// CHECK: miopen.threadwise_copy

// --------------------------
// threadwise_copy_v2 tests.

func @miopen_threadwise_copy_v2(%source : memref<32xf32, 5>,
                                %dest : memref<?x?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // check source and destination with coordinate transforms.
  miopen.threadwise_copy_v2
    %source[%c0] ->
    %dest[%c1, %c1, %c1, %c1, %c1]
    storeMethod(set)
    {length = 1 : index, leftOobDims = [], rightOobDims = []}
    : memref<32xf32, 5>
    -> memref<?x?x?x?x?xf32>, index, index, index, index, index

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2
// CHECK: miopen.threadwise_copy_v2

func @miopen_threadwise_gemm(%lhs : memref<1x4x8xf32>, %rhs : memref<1x4x8xf32>, %output : memref<1x8x8xf32>) {
  miopen.threadwise_gemm(%lhs, %rhs, %output) : memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm
// CHECK: miopen.threadwise_gemm

// ----

func @miopen_xdlops_gemm_v2_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                       %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = vector.splat %c0f : vector<32xf32>
  %vectorD0 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32> -> vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result
// CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_xdlops_gemm_v2_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                        %bufferA : memref<32xf32, 5>, %bufferB: memref<16xf32, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = vector.splat %c0f : vector<32xf32>
  %vectorC1 = vector.splat %c0f : vector<32xf32>
  %vectorD0, %vectorD1 = miopen.xdlops_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results
// CHECK: miopen.xdlops_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                          %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> vector<32xf32> {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = vector.splat %c0f : vector<32xf32>
  %vectorD0 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 0 : index
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32> -> vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result
// CHECK: miopen.blockwise_gemm_v2

// ----

func @miopen_blockwise_gemm_v2_two_results(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                           %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>) -> (vector<32xf32>, vector<32xf32>) {
  %c0 = arith.constant 0 : index
  %c0f = arith.constant 0.0 : f32
  %vectorC0 = vector.splat %c0f : vector<32xf32>
  %vectorC1 = vector.splat %c0f : vector<32xf32>
  %vectorD0, %vectorD1 = miopen.blockwise_gemm_v2(%matrixA, %matrixB, %c0, %c0, %bufferA, %bufferB, %vectorC0, %vectorC1) {
    m = 256,
    n = 256,
    k = 16,
    m_per_wave = 128,
    n_per_wave = 64,
    ldsBufferOffsetA = 0 : index,
    ldsBufferOffsetB = 8192 : index
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results
// CHECK: miopen.blockwise_gemm_v2
