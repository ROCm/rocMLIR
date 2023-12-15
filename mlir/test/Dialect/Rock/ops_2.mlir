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
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
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

func.func @rock_global_store(%source : memref<32xf32, #gpu.address_space<private>>,
                                %dest : memref<?x?x?x?x?xf32>, %valid : i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // check source and destination with coordinate transforms.
  rock.global_store
    set
    %source[%c0] ->
    %dest[%c1, %c1, %c1, %c1, %c1]
    if %valid
    features = none
    {length = 1 : index}
    : memref<32xf32, #gpu.address_space<private>> -> memref<?x?x?x?x?xf32>

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

func.func @rock_accel_gemm_one_result(%matrixA : memref<1x16xf32, 5>,
                                            %matrixB : memref<1x16xf32, 5>,
                                            %matrixC : memref<1x1xvector<32xf32>, 5>) {
  %c0 = arith.constant 0 : index
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<1x1xvector<32xf32>, 5> += memref<1x16xf32, 5> * memref<1x16xf32, 5>
  return
}

// CHECK-LABEL: func.func @rock_accel_gemm_one_result
// CHECK: rock.threadwise_accel_gemm

// ----

#transform_map0 = #rock.transform_map<affine_map<(d0, d1, d2, d3) -> (2*d0 + d1)> by [<AddDim{1} ["i"] at [2] -> [] at []>, <AddDim{1} ["j"] at [3] -> [] at []>, <Unmerge{2, 2} ["ci", "cj"] at [0, 1] -> ["offset"] at [0]>] bounds = [2, 2, 1, 1] -> [4]>

func.func @rock_accel_gemm_two_results(%matrixA : memref<1x16xf32, 5>,
                                             %matrixB : memref<1x16xf32, 5>,
                                             %matrixC : memref<4xvector<32xf32>, 5>) {
  %c1 = arith.constant 1 : index
  %matrixCView = rock.transform %matrixC by #transform_map0: memref<4xvector<32xf32>, 5> to memref<2x2x1x1xvector<32xf32>, 5>

  rock.threadwise_accel_gemm %matrixCView[%c1, %c1] += %matrixA * %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    params = #rock.xdlops_gemm_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      kpack = 1,
      forceUnroll = true>
  } : memref<2x2x1x1xvector<32xf32>, 5> += memref<1x16xf32, 5> * memref<1x16xf32, 5>
  return
}

// CHECK-LABEL: func.func @rock_accel_gemm_two_results
// CHECK: rock.threadwise_accel_gemm

// ----

func.func @rock_blockwise_gemm_accel_one_result(%matrixA : memref<12288xf32, 3>, %matrixB : memref<12288xf32, 3>,
                                              %bufferA : memref<32xf32, 5>, %bufferB : memref<16xf32, 5>,
                                              %matrixC : memref<1xvector<32xf32>, 5>) {
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
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
  rock.blockwise_gemm_accel %matrixC += %bufferA from %matrixA * %bufferB from %matrixB features = mfma {
    arch = "amdgcn-amd-amdhsa:gfx90a",
    blockSize = 256 : i32,
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
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
