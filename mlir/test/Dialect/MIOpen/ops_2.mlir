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

#gemm_padding0 = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">

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
// blockwise_load tests.

// f32 tests.

func @miopen_blockwise_load_f32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> f32  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf32>, index, index, index -> f32
  return %result : f32
}

// CHECK-LABEL: func @miopen_blockwise_load_f32
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : memref<?x?x?xf32>, index, index, index -> f32

func @miopen_blockwise_load_2xf32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf32>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf32>, index, index, index -> vector<2xf32>
  return %result : vector<2xf32>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xf32
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf32>, index, index, index -> vector<2xf32>

func @miopen_blockwise_load_4xf32(%source : memref<?x?x?xf32>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf32>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf32>, index, index, index -> vector<4xf32>
  return %result : vector<4xf32>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xf32
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf32>, index, index, index -> vector<4xf32>

// f16 tests.

func @miopen_blockwise_load_f16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> f16  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf16>, index, index, index -> f16
  return %result : f16
}

// CHECK-LABEL: func @miopen_blockwise_load_f16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf16>, index, index, index -> f16

func @miopen_blockwise_load_2xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf16>, index, index, index -> vector<2xf16>
  return %result : vector<2xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<2xf16>

func @miopen_blockwise_load_4xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf16>, index, index, index -> vector<4xf16>
  return %result : vector<4xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<4xf16>

func @miopen_blockwise_load_8xf16(%source : memref<?x?x?xf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xf16>, index, index, index -> vector<8xf16>
  return %result : vector<8xf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_8xf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xf16>, index, index, index -> vector<8xf16>

// bf16 tests.

func @miopen_blockwise_load_bf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> bf16  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xbf16>, index, index, index -> bf16
  return %result : bf16
}

// CHECK-LABEL: func @miopen_blockwise_load_bf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xbf16>, index, index, index -> bf16

func @miopen_blockwise_load_2xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<2xbf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>
  return %result : vector<2xbf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_2xbf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<2xbf16>

func @miopen_blockwise_load_4xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<4xbf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>
  return %result : vector<4xbf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_4xbf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<4xbf16>

func @miopen_blockwise_load_8xbf16(%source : memref<?x?x?xbf16>, %sc0 : index, %sc1 : index, %sc2 : index) -> vector<8xbf16>  {
  %result = miopen.blockwise_load %source[%sc0, %sc1, %sc2] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>
  return %result : vector<8xbf16>
}

// CHECK-LABEL: func @miopen_blockwise_load_8xbf16
//  CHECK: %{{.*}} = miopen.blockwise_load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : memref<?x?x?xbf16>, index, index, index -> vector<8xbf16>

// --------------------------
// blockwise_store tests.

// f32 tests.

func @miopen_blockwise_store_f32(%data : f32, %dest : memref<?x?x?xf32, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : f32 -> memref<?x?x?xf32, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_f32
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : f32 -> memref<?x?x?xf32, 3>, index, index, index

func @miopen_blockwise_store_2xf32(%data : vector<2xf32>, %dest : memref<?x?x?xf32, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<2xf32> -> memref<?x?x?xf32, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xf32
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<2xf32> -> memref<?x?x?xf32, 3>, index, index, index

func @miopen_blockwise_store_4xf32(%data : vector<4xf32>, %dest : memref<?x?x?xf32, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<4xf32> -> memref<?x?x?xf32, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xf32
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<4xf32> -> memref<?x?x?xf32, 3>, index, index, index

// f16 tests.

func @miopen_blockwise_store_f16(%data : f16, %dest : memref<?x?x?xf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : f16 -> memref<?x?x?xf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_f16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : f16 -> memref<?x?x?xf16, 3>, index, index, index

func @miopen_blockwise_store_2xf16(%data : vector<2xf16>, %dest : memref<?x?x?xf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<2xf16> -> memref<?x?x?xf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}}  : vector<2xf16> -> memref<?x?x?xf16, 3>, index, index, index

func @miopen_blockwise_store_4xf16(%data : vector<4xf16>, %dest : memref<?x?x?xf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<4xf16> -> memref<?x?x?xf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<4xf16> -> memref<?x?x?xf16, 3>, index, index, index

func @miopen_blockwise_store_8xf16(%data : vector<8xf16>, %dest : memref<?x?x?xf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<8xf16> -> memref<?x?x?xf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_8xf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<8xf16> -> memref<?x?x?xf16, 3>, index, index, index

// bf16 tests.

func @miopen_blockwise_store_bf16(%data : bf16, %dest : memref<?x?x?xbf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : bf16 -> memref<?x?x?xbf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_bf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : bf16 -> memref<?x?x?xbf16, 3>, index, index, index

func @miopen_blockwise_store_2xbf16(%data : vector<2xbf16>, %dest : memref<?x?x?xbf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<2xbf16> -> memref<?x?x?xbf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_2xbf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<2xbf16> -> memref<?x?x?xbf16, 3>, index, index, index

func @miopen_blockwise_store_4xbf16(%data : vector<4xbf16>, %dest : memref<?x?x?xbf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<4xbf16> -> memref<?x?x?xbf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_4xbf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<4xbf16> -> memref<?x?x?xbf16, 3>, index, index, index

func @miopen_blockwise_store_8xbf16(%data : vector<8xbf16>, %dest : memref<?x?x?xbf16, 3>, %dc0 : index, %dc1 : index, %dc2 : index) {
  miopen.blockwise_store %data -> %dest[%dc0, %dc1, %dc2]
  with [[]] { bounds = [1 : index, 1 : index, 1 : index ]}
  : vector<8xbf16> -> memref<?x?x?xbf16, 3>, index, index, index
  return
}

// CHECK-LABEL: func @miopen_blockwise_store_8xbf16
//  CHECK: miopen.blockwise_store %{{.*}} ->  %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] with {{.*}} : vector<8xbf16> -> memref<?x?x?xbf16, 3>, index, index, index

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
    {paddingInfo = #gemm_padding0, oobDims=[], globalArg = -1 : index}
    : memref<?x?xf32, 5>, index, index -> memref<?x?xf32, 5>, index, index

  // -----

  // check source with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0], []]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 0 : index }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?xf32, 5>, index, index

  // check source with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], []]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 0 : index }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?xf32, 5>, index, index

  // check destination with one coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 1 : index }
    : memref<?x?xf32, 5>, index, index -> memref<?x?x?x?xf32>, index, index

  // check destination with multiple coordinate transform.
  miopen.threadwise_copy
    %source[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 1 : index }
    : memref<?x?xf32, 5>, index, index -> memref<?x?x?x?xf32>, index, index

  // -----

  // check source and destination with one coordinate transform.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0], [#transform_map0]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 0 : index }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?x?x?xf32>, index, index

  // check source and destination with multiple coordinate transforms.
  miopen.threadwise_copy
    %source_with_transform_maps[%source_coord_x, %source_coord_y] ->
    %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
    with [[#transform_map0, #transform_map1], [#transform_map0, #transform_map1]]
    { paddingInfo = #gemm_padding0, oobDims=[false, false, false, false],
      globalArg = 0 : index }
    : memref<?x?x?x?xf32>, index, index -> memref<?x?x?x?xf32>, index, index

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy
//  CHECK: miopen.threadwise_copy

// --------------------------
// threadwise_load tests.

// CHECK-LABEL: func @miopen_threadwise_load
func @miopen_threadwise_load(%source_coord : memref<2xindex, 5>,
                             %source : memref<?x?xf32>,
                             %source_with_transform_maps : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %source_coord_y = memref.load %source_coord[%c0] : memref<2xindex, 5>
  %source_coord_x = memref.load %source_coord[%c0] : memref<2xindex, 5>

  // check source as vanilla memref, dest as scalar.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}} : memref<?x?xf32>, index, index -> f32
  %v0 = miopen.threadwise_load %source[%source_coord_x, %source_coord_y] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?xf32>, index, index -> f32

  // check source as vanilla memref, dest as vector.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}} : memref<?x?xf32>, index, index -> vector<4xf32>
  %v1 = miopen.threadwise_load %source[%source_coord_x, %source_coord_y] with [[]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?xf32>, index, index -> vector<4xf32>

  // -----

  // check source with one coordinate transform, dest as scalar.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}]
  %v2 = miopen.threadwise_load %source_with_transform_maps[%source_coord_x, %source_coord_y]
  with [[#transform_map0]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?x?xf32>, index, index -> f32

  // check source with one coordinate transform, dest as vector.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}]
  %v3 = miopen.threadwise_load %source_with_transform_maps[%source_coord_x, %source_coord_y]
  with [[#transform_map0]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?x?xf32>, index, index -> vector<4xf32>

  // check source with multiple coordinate transforms, dest as scalar.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}]
  %v4 = miopen.threadwise_load %source_with_transform_maps[%source_coord_x, %source_coord_y]
  with [[#transform_map0, #transform_map1]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  } : memref<?x?x?x?xf32>, index, index -> f32

  // check source with multiple coordinate transforms, dest as vector.
  // CHECK: %{{.*}} = miopen.threadwise_load %{{.*}}[%{{.*}}, %{{.*}}]
  %v5 = miopen.threadwise_load %source_with_transform_maps[%source_coord_x, %source_coord_y]
  with [[#transform_map0, #transform_map1]] {
    paddingInfo = #gemm_padding0,
    bounds = [1 : index, 1 : index],
    oobDims = [false, false, false, false, false]
  }  : memref<?x?x?x?xf32>, index, index -> vector<4xf32>

  return
}

// --------------------------
// threadwise_store tests.

// CHECK-LABEL: func @miopen_threadwise_store
func @miopen_threadwise_store(%data_scalar : f32,
                              %data_vector : vector<4xf32>,
                              %dest_coord : memref<2xindex, 5>,
                              %dest : memref<?x?xf32>,
                              %dest_with_transform_maps : memref<?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dest_coord_y = memref.load %dest_coord[%c0] : memref<2xindex, 5>
  %dest_coord_x = memref.load %dest_coord[%c0] : memref<2xindex, 5>

  // check dest as vanilla memrefs, data as scalar.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}} : f32 -> memref<?x?xf32>, index, index
  miopen.threadwise_store %data_scalar -> %dest[%dest_coord_x, %dest_coord_y]
  with [[]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : f32 -> memref<?x?xf32>, index, index

  // check dest as vanilla memrefs, data as vector.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}} : vector<4xf32> -> memref<?x?xf32>, index, index
  miopen.threadwise_store %data_vector -> %dest[%dest_coord_x, %dest_coord_y]
  with [[]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : vector<4xf32> -> memref<?x?xf32>, index, index

  // -----

  // check destination with one coordinate transform, data as scalar.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}]
  miopen.threadwise_store %data_scalar -> %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
  with [[#transform_map0]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : f32 -> memref<?x?x?x?xf32>, index, index

  // check destination with one coordinate transform, data as vector.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}}
  miopen.threadwise_store %data_vector -> %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
  with [[#transform_map0]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : vector<4xf32> -> memref<?x?x?x?xf32>, index, index

  // check destination with multiple coordinate transform, data as scalar.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}}
  miopen.threadwise_store %data_scalar -> %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
  with [[#transform_map0, #transform_map1]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : f32 -> memref<?x?x?x?xf32>, index, index

  // check destination with multiple coordinate transform, data as vector.
  // CHECK: miopen.threadwise_store %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}] with {{.*}}
  miopen.threadwise_store %data_vector -> %dest_with_transform_maps[%dest_coord_x, %dest_coord_y]
  with [[#transform_map0, #transform_map1]] {
    bounds = [1 : index, 1 : index, 1 : index]
  } : vector<4xf32> -> memref<?x?x?x?xf32>, index, index

  return
}

// --------------------------
// threadwise_copy_v2 tests.

#transform_map2 = #miopen.transform_map<
  affine_map<(d0, d1, d2, d3, d4) -> (d1 * 4 + d3)> by [
    #miopen.transform<Embed{0, 4, 0, 1, 0} ["g", "m0", "m1", "m2", "n"] at [0, 1, 2, 3, 4]
    -> ["raw"] at [1]>
  ] bounds = [1, 4, 1, 4, 1] -> [16]>
#transform_map3 = #miopen.transform_map<
  affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by [
    #miopen.transform<PassThrough ["g", "m0", "m1", "m2", "n"] at [0, 1, 2, 3, 4] ->
      ["g", "n", "c", "h", "w"] at [0, 1, 2, 3, 4]>
  ] bounds = [1, 4, 1, 4, 1] -> [1, 4, 1, 4, 1]>

func @miopen_threadwise_copy_v2(%source : vector<32xf32>,
                                %dest : memref<?x?x?x?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // check source and destination with coordinate transforms.
  miopen.threadwise_copy_v2
    %source[%c0, %c0, %c0, %c0, %c0] ->
    %dest[%c1, %c1, %c1, %c1, %c1]
    with [[#transform_map2], [#transform_map3]]
    {
      sourceOffset = 0 : index,
      paddingInfo = #gemm_padding0,
      destOobDims = [false, false, false, false, false],
      bounds = [1 : index, 4 : index, 1 : index, 4 : index, 1 : index],
      dataOperation = 0 : i32
    } : vector<32xf32>, index, index, index, index, index
    -> memref<?x?x?x?x?xf32>, index, index, index, index, index

  return
}

// CHECK-LABEL: func @miopen_threadwise_copy_v2
//  CHECK: miopen.threadwise_copy_v2

func @miopen_threadwise_gemm(%lhs : memref<1x4x8xf32>, %rhs : memref<1x4x8xf32>, %output : memref<1x8x8xf32>) {
  miopen.threadwise_gemm(%lhs, %rhs, %output) : memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>
  return
}

// CHECK-LABEL: func @miopen_threadwise_gemm
//  CHECK: miopen.threadwise_gemm

// ----

func @miopen_mfma_v2_f32(%a : f32, %b : f32, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x1f32", imm = [1, 0, 0] } : f32, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_f32
//   CHECK: miopen.mfma_v2

func @miopen_mfma_v2_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x4f16", imm = [1, 0, 0] } : vector<4xf16>, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_f16
//   CHECK: miopen.mfma_v2

func @miopen_mfma_v2_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<32xf32>) -> vector<32xf32> {
  %d = miopen.mfma_v2(%a, %b, %c) { instr = "mfma_f32_32x32x2bf16", imm = [1, 0, 0] } : vector<2xbf16>, vector<32xf32>
  return %d : vector<32xf32>
}

// CHECK-LABEL: func @miopen_mfma_v2_bf16
//   CHECK: miopen.mfma_v2

// ----

#transform_map4 = #miopen.transform_map<
  affine_map<(d0) -> (d0)> by [
    #miopen.transform<Slice{0, 4096} ["x"] at [0] -> ["x"] at [0]>
  ] bounds = [4096] -> [12288]>
#transform_map5 = #miopen.transform_map<
  affine_map<(d0) -> (d0 + 8192)> by [
    #miopen.transform<Slice{8192, 12288} ["x"] at [0] -> ["x"] at [0]>
  ] bounds = [4096] -> [12288]>


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
    transforms = [[#transform_map4], [#transform_map5]]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32> -> vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_one_result
//  CHECK: miopen.xdlops_gemm_v2

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
    transforms = [[#transform_map4], [#transform_map5]]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_xdlops_gemm_v2_two_results
//  CHECK: miopen.xdlops_gemm_v2

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
    transforms = [[#transform_map4], [#transform_map5]]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32> -> vector<32xf32>
  return %vectorD0 : vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_one_result
//  CHECK: miopen.blockwise_gemm_v2

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
    transforms = [[#transform_map4], [#transform_map5]]
  } : memref<12288xf32, 3>, memref<12288xf32, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
  return %vectorD0, %vectorD1 : vector<32xf32>, vector<32xf32>
}

// CHECK-LABEL: func @miopen_blockwise_gemm_v2_two_results
//  CHECK: miopen.blockwise_gemm_v2
