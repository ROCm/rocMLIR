// RUN: rocmlir-opt -split-input-file -rock-gridwise-gemm-to-blockwise -rock-blockwise-load-tile-to-threadwise -rock-prepare-pipeline -rock-linalg-align -rock-blockwise-gemm-to-threadwise -rock-pipeline -canonicalize -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_issue_1811
func.func @gridwise_attn_barriers_before_lds_write_issue_1811(%arg0: memref<4096xi8>, %arg1: memref<4096xi8>, %arg2: memref<4096xf16>, %arg3: memref<1xi8>, %arg4: memref<1xf16>, %arg5: memref<4096xf16>) attributes {block_size = 64 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  // CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index

  // CHECK: affine.for %{{.*}} = 0 to 2
  // CHECK: rock.threadwise_read_into
  // CHECK: scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]] {
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<64xi8, #gpu.address_space<private>> -> memref<64x64xvector<8xi8>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_gemm_accel
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<16xf16, #gpu.address_space<private>> -> memref<64x16xvector<8xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_gemm_accel
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<16xf16, #gpu.address_space<private>> -> memref<64x16xvector<8xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_gemm_accel
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_q", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xi8> to memref<1x64x64xi8>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_k", "head_qk"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xi8> to memref<1x64x64xi8>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["head_v", "seq_k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d2)> by [<Unmerge{1} ["seq_k"] at [2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [1, 1, 1] -> [1]> : memref<1xi8> to memref<1x1x1xi8>
  %4 = rock.transform %arg4 by <affine_map<(d0, d1, d2) -> (d2)> by [<Unmerge{1} ["seq_k"] at [2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [1, 1, 1] -> [1]> : memref<1xf16> to memref<1x1x1xf16>
  %5 = rock.transform %arg5 by <affine_map<(d0, d1, d2) -> (d1 * 64 + d2)> by [<Unmerge{64, 64} ["seq_q", "head_v"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 64, 64] -> [4096]> : memref<4096xf16> to memref<1x64x64xf16>
  %6 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x64x64xi8>
  %7 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x64x64xi8>
  %8 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm1K", "gemm1N"] at [1, 2] -> ["gemm1K", "gemm1N"] at [2, 1]>] bounds = [1, 64, 64] -> [1, 64, 64]> : memref<1x64x64xf16> to memref<1x64x64xf16>
  %9 = rock.transform %6 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 64} ["gemm0KPad"] at [1] -> ["gemm0K"] at [1]>, <PassThrough ["gemm0N"] at [2] -> ["gemm0N"] at [2]>] bounds = [1, 128, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x128x64xi8>
  %10 = rock.transform %7 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 64} ["gemm0KPad"] at [1] -> ["gemm0K"] at [1]>, <PassThrough ["gemm0M"] at [2] -> ["gemm0M"] at [2]>] bounds = [1, 128, 64] -> [1, 64, 64]> : memref<1x64x64xi8> to memref<1x128x64xi8>
  rock.gridwise_attention_accel(%9, %10, %8, %3, %4, %5) preSoftmaxOps = {
  ^bb0(%arg6: memref<1x64x64xi32>, %arg7: memref<1x1x1xi8>, %arg8: memref<1x1x1xf16>, %arg9: memref<1x64x64xf16>):
    %11 = rock.transform %arg6 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 64] -> [1, 64, 64]> : memref<1x64x64xi32> to memref<64x64xi32>
    %12 = rock.transform %arg7 by <affine_map<() -> (0, 0, 0)> by [<ConstDim{0, 1} [] at [] -> ["const0"] at [0]>, <ConstDim{0, 1} [] at [] -> ["const1"] at [1]>, <ConstDim{0, 1} [] at [] -> ["const2"] at [2]>] bounds = [] -> [1, 1, 1]> : memref<1x1x1xi8> to memref<i8>
    %13 = rock.transform %arg8 by <affine_map<() -> (0, 0, 0)> by [<ConstDim{0, 1} [] at [] -> ["const0"] at [0]>, <ConstDim{0, 1} [] at [] -> ["const1"] at [1]>, <ConstDim{0, 1} [] at [] -> ["const2"] at [2]>] bounds = [] -> [1, 1, 1]> : memref<1x1x1xf16> to memref<f16>
    %alloc = memref.alloc() : memref<1x64x64xf16>
    %14 = rock.transform %alloc by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{64} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [64, 64] -> [1, 64, 64]> : memref<1x64x64xf16> to memref<64x64xf16>
    %15 = rock.transform %12 by <affine_map<(d0, d1) -> ()> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>] bounds = [1, 1] -> []> : memref<i8> to memref<1x1xi8>
    %16 = rock.transform %15 by <affine_map<(d0, d1) -> (0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 64] -> [1, 1]> : memref<1x1xi8> to memref<64x64xi8>
    %17 = rock.transform %13 by <affine_map<(d0, d1) -> ()> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <AddDim{1} ["exp1"] at [1] -> [] at []>] bounds = [1, 1] -> []> : memref<f16> to memref<1x1xf16>
    %18 = rock.transform %17 by <affine_map<(d0, d1) -> (0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [64, 64] -> [1, 1]> : memref<1x1xf16> to memref<64x64xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%11, %16, %18 : memref<64x64xi32>, memref<64x64xi8>, memref<64x64xf16>) outs(%14 : memref<64x64xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: i32, %in_0: i8, %in_1: f16, %out: f16):
      %19 = arith.extsi %in_0 : i8 to i32
      %20 = arith.subi %in, %19 : i32
      %21 = arith.sitofp %20 : i32 to f16
      %22 = arith.mulf %21, %in_1 : f16
      linalg.yield %22 : f16
    }
    memref.copy %alloc, %arg9 : memref<1x64x64xf16> to memref<1x64x64xf16>
    rock.yield
  } {arch = "amdgcn-amd-amdhsa:gfx1100", blockSize = 64 : i32, firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>, splitKV = 1 : i32, gridSize = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 2, 0, 0, 1, 0>, params0 = #rock.accel_gemm_params<kpackPerBlock = 16, mPerBlock = 32, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, params1 = #rock.accel_gemm_params<kpackPerBlock = 4, mPerBlock = 32, nPerBlock = 64, kpack = 8, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x64xi8>, memref<1x128x64xi8>, memref<1x64x64xf16>, memref<1x1x1xi8>, memref<1x1x1xf16>, memref<1x64x64xf16>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_issue_1844
func.func @gridwise_attn_barriers_before_lds_write_issue_1844(%arg0: memref<32768xf16>, %arg1: memref<32768xf16>, %arg2: memref<32768xf16>, %arg3: memref<32768xf16>) attributes {block_size = 256 : i32, grid_size = 2 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
  // CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
  // CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index

  // CHECK: affine.for %{{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK: scf.for %{{.*}} = %[[c0]] to %[[c2]] step %[[c1]]
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<64xf16, #gpu.address_space<private>> -> memref<256x64xvector<4xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<64xf16, #gpu.address_space<private>> -> memref<256x64xvector<4xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["n", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{256, 128} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]> : memref<32768xf16> to memref<1x256x128xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 128, 256] -> [1, 256, 128]> : memref<1x256x128xf16> to memref<1x128x256xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 128, 256] -> [1, 256, 128]> : memref<1x256x128xf16> to memref<1x128x256xf16>
  rock.gridwise_attention_accel(%4, %5, %2, %3) preSoftmaxOps = {
  } {blockSize = 256 : i32, enableSoftmax = false, firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>, splitKV = 1 : i32, gridSize = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>, params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x256xf16>, memref<1x128x256xf16>, memref<1x256x128xf16>, memref<1x256x128xf16>
  return
}

// -----

// CHECK-LABEL: @gridwise_attn_barriers_before_lds_write_nobarriers
func.func @gridwise_attn_barriers_before_lds_write_nobarriers(%arg0: memref<16384xf16>, %arg1: memref<16384xf16>, %arg2: memref<16384xf16>, %arg3: memref<16384xf16>) attributes {block_size = 256 : i32, grid_size = 1 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  // CHECK: affine.for %{{.*}} = 0 to 1
  // CHECK: rock.threadwise_read_into
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<64xf16, #gpu.address_space<private>> -> memref<256x64xvector<4xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_gemm_accel
  // CHECK-NOT: rock.lds_barrier
  // CHECK: rock.threadwise_write_all {{.*}} : memref<64xf16, #gpu.address_space<private>> -> memref<256x64xvector<4xf16>, #gpu.address_space<workgroup>>
  // CHECK: rock.lds_barrier
  // CHECK: rock.threadwise_gemm_accel
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["n", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %3 = rock.transform %arg3 by <affine_map<(d0, d1, d2) -> (d1 * 128 + d2)> by [<Unmerge{128, 128} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 128, 128] -> [16384]> : memref<16384xf16> to memref<1x128x128xf16>
  %4 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 128, 128] -> [1, 128, 128]> : memref<1x128x128xf16> to memref<1x128x128xf16>
  %5 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [1, 128, 128] -> [1, 128, 128]> : memref<1x128x128xf16> to memref<1x128x128xf16>
  rock.gridwise_attention_accel(%4, %5, %2, %3) preSoftmaxOps = {
  } {blockSize = 256 : i32, enableSoftmax = false, firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>, splitKV = 1 : i32, gridSize = 1 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0>, params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 128, nPerBlock = 128, kpack = 4, mPerWave = 128, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>} : memref<1x128x128xf16>, memref<1x128x128xf16>, memref<1x128x128xf16>, memref<1x128x128xf16>
  return
}
