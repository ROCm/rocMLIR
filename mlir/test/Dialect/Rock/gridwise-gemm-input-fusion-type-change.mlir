// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise -verify-diagnostics %s | FileCheck %s

#accel_gemm_params = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>
#map = affine_map<(d0, d1, d2, d3) -> ((d1 * 16 + d2) * 64 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map3 = affine_map<(d0, d1, d2) -> (d1 * 3072 + d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2 + 2048)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 * 64 + d3)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (((d0 * 1500 + d1) * 16 + d2) * 64 + d3)>
#map10 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
#map12 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d2 * 1500 + d3)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#transform_map = #rock.transform_map<#map by [<Unmerge{1500, 16, 64} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 1500, 16, 64] -> [1536000]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0", "dim2", "dim3", "dim1"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim3", "dim1"] at [0, 2, 3, 1]>] bounds = [1, 16, 64, 1500] -> [1, 1500, 16, 64]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["dim0", "dim2", "dim1", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim1", "dim3"] at [0, 2, 1, 3]>] bounds = [1, 16, 1500, 64] -> [1, 1500, 16, 64]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{1500, 3072} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 1500, 3072] -> [4608000]>
#transform_map4 = #rock.transform_map<#map4 by [<Slice{0, 1, 0, 1500, 2048, 3072} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [1, 1500, 1024] -> [1, 1500, 3072]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{16, 64} ["exp2", "exp3"] at [2, 3] -> ["dim2"] at [2]>] bounds = [1, 1500, 16, 64] -> [1, 1500, 1024]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{1, 16} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [16, 1500, 64] -> [1, 16, 1500, 64]>
#transform_map7 = #rock.transform_map<#map8 by [<Unmerge{16} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 16, 1500, 64] -> [16, 1500, 64]>
#transform_map8 = #rock.transform_map<#map6 by [<Merge{1, 16} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [16, 64, 1500] -> [1, 16, 64, 1500]>
#transform_map9 = #rock.transform_map<#map9 by [<Unmerge{1, 1500, 16, 64} ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [1, 1500, 16, 64] -> [1536000]>
#transform_map10 = #rock.transform_map<#map2 by [<PassThrough ["dim0", "dim2", "dim1", "dim3"] at [0, 2, 1, 3] -> ["dim0", "dim2", "dim1", "dim3"] at [0, 1, 2, 3]>] bounds = [1, 16, 1500, 64] -> [1, 1500, 16, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Merge{16} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [16, 1500, 64] -> [1, 16, 1500, 64]>
#transform_map12 = #rock.transform_map<#map10 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [1500, 16, 64] -> [16, 1500, 64]>
#transform_map13 = #rock.transform_map<#map10 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [16, 1500, 64] -> [1500, 16, 64]>
#transform_map14 = #rock.transform_map<#map11 by [<PassThrough ["dim2", "dim0", "dim1"] at [0, 1, 2] -> ["dim2", "dim0", "dim1"] at [2, 0, 1]>] bounds = [1500, 16, 64] -> [16, 64, 1500]>
#transform_map15 = #rock.transform_map<#map12 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [16, 64, 1500] -> [16, 1500, 64]>
#transform_map16 = #rock.transform_map<#map12 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0N"] at [1, 2] -> ["gemm0K", "gemm0N"] at [2, 1]>] bounds = [16, 64, 1500] -> [16, 1500, 64]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 4} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [16, 64, 1504] -> [16, 64, 1500]>
#transform_map18 = #rock.transform_map<#map7 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 4} ["gemm0MPad"] at [2] -> ["gemm0M"] at [2]>] bounds = [16, 64, 1504] -> [16, 64, 1500]>
#transform_map19 = #rock.transform_map<#map7 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 4} ["gemm1KPad"] at [1] -> ["gemm1K"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [16, 1504, 64] -> [16, 1500, 64]>
#transform_map20 = #rock.transform_map<#map7 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 4} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [16, 1504, 64] -> [16, 1500, 64]>
#transform_map21 = #rock.transform_map<#map8 by [<Unmerge{16} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 16, 1500, 1500] -> [16, 1500, 1500]>
#transform_map22 = #rock.transform_map<#map13 by [<Unmerge{1500, 1500} ["exp2", "exp3"] at [2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>] bounds = [1, 1, 1500, 1500] -> [2250000]>
#transform_map23 = #rock.transform_map<#map14 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 16, 1500, 1500] -> [1, 1, 1500, 1500]>
#transform_map24 = #rock.transform_map<#map6 by [<Merge{1, 16} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [16, 1500, 1500] -> [1, 16, 1500, 1500]>
#transform_map25 = #rock.transform_map<#map6 by [<Merge{16} ["dim0"] at [0] -> ["exp1"] at [1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <ConstDim{0, 1} [] at [] -> ["unit0"] at [0]>] bounds = [16, 1500, 1500] -> [1, 16, 1500, 1500]>
module {
  // CHECK-LABEL: @mlir_slice_reshape_transpose_convert_dot_convert_add
  func.func @mlir_slice_reshape_transpose_convert_dot_convert_add_reshape_reduce_max_reshape_sub_exp_reshape_reduce_sum_reshape_div_dot_transpose_reshape(%arg0: memref<4608000xf16>, %arg1: memref<1536000xf16>, %arg2: memref<1536000xf16>, %arg3: memref<2250000xf32>, %arg4: memref<1536000xf32>) attributes {arch = "gfx942:sramecc+:xnack-", block_size = 64 : i32, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, grid_size = 752 : i32, kernel = "mixr", num_chiplets = 4 : i64, num_cu = 80 : i64} {
    %0 = rock.transform %arg2 by #transform_map : memref<1536000xf16> to memref<1x1500x16x64xf16>
    %1 = rock.transform %0 by #transform_map1 : memref<1x1500x16x64xf16> to memref<1x16x64x1500xf16>
    %2 = rock.transform %arg1 by #transform_map : memref<1536000xf16> to memref<1x1500x16x64xf16>
    %3 = rock.transform %2 by #transform_map2 : memref<1x1500x16x64xf16> to memref<1x16x1500x64xf16>
    %4 = rock.transform %arg0 by #transform_map3 : memref<4608000xf16> to memref<1x1500x3072xf16>
    %5 = rock.transform %4 by #transform_map4 : memref<1x1500x3072xf16> to memref<1x1500x1024xf16>
    %6 = rock.transform %5 by #transform_map5 : memref<1x1500x1024xf16> to memref<1x1500x16x64xf16>
    %7 = rock.transform %6 by #transform_map2 : memref<1x1500x16x64xf16> to memref<1x16x1500x64xf16>
    %8 = rock.transform %7 by #transform_map6 : memref<1x16x1500x64xf16> to memref<16x1500x64xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x1500x64xf32>
    linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : memref<16x1500x64xf16>) outs(%alloc : memref<16x1500x64xf32>) attrs =  {rock.majorTensorNumber = 0 : index} {
    ^bb0(%in: f16, %out: f32):
      %28 = arith.extf %in : f16 to f32
      linalg.yield %28 : f32
    }
    %9 = rock.transform %alloc by #transform_map7 : memref<16x1500x64xf32> to memref<1x16x1500x64xf32>
    %10 = rock.transform %3 by #transform_map6 : memref<1x16x1500x64xf16> to memref<16x1500x64xf16>
    %11 = rock.transform %1 by #transform_map8 : memref<1x16x64x1500xf16> to memref<16x64x1500xf16>
    %12 = rock.transform %9 by #transform_map6 : memref<1x16x1500x64xf32> to memref<16x1500x64xf32>
    %alloc_0 = memref.alloc() : memref<1536000xf32>
    %13 = rock.transform %alloc_0 by #transform_map9 : memref<1536000xf32> to memref<1x1500x16x64xf32>
    %14 = rock.transform %13 by #transform_map10 : memref<1x1500x16x64xf32> to memref<1x16x1500x64xf32>
    %15 = rock.transform %14 by #transform_map11 : memref<1x16x1500x64xf32> to memref<16x1500x64xf32>
    %16 = rock.transform %10 by #transform_map12 : memref<16x1500x64xf16> to memref<1500x16x64xf16>
    %17 = rock.transform %16 by #transform_map13 : memref<1500x16x64xf16> to memref<16x1500x64xf16>
    %18 = rock.transform %11 by #transform_map14 : memref<16x64x1500xf16> to memref<1500x16x64xf16>
    %19 = rock.transform %18 by #transform_map13 : memref<1500x16x64xf16> to memref<16x1500x64xf16>
    %20 = rock.transform %12 by #transform_map12 : memref<16x1500x64xf32> to memref<1500x16x64xf32>
    %21 = rock.transform %20 by #transform_map13 : memref<1500x16x64xf32> to memref<16x1500x64xf32>
    %22 = rock.transform %17 by #transform_map15 : memref<16x1500x64xf16> to memref<16x64x1500xf16>
    %23 = rock.transform %19 by #transform_map16 : memref<16x1500x64xf16> to memref<16x64x1500xf16>
    %24 = rock.transform %22 by #transform_map17 : memref<16x64x1500xf16> to memref<16x64x1504xf16>
    %25 = rock.transform %23 by #transform_map18 : memref<16x64x1500xf16> to memref<16x64x1504xf16>
    %26 = rock.transform %21 by #transform_map19 : memref<16x1500x64xf32> to memref<16x1504x64xf32>
    %27 = rock.transform %15 by #transform_map20 : memref<16x1500x64xf32> to memref<16x1504x64xf32>
    rock.gridwise_attention_accel(%24, %25, %26, %arg3, %27) preSoftmaxOps = {
    ^bb0(%arg5: memref<16x1500x1500xf16>, %arg6: memref<2250000xf32>, %arg7: memref<1x16x1500x1500xf32>):
      %28 = rock.transform %arg5 by #transform_map21 : memref<16x1500x1500xf16> to memref<1x16x1500x1500xf16>
      %29 = rock.transform %arg6 by #transform_map22 : memref<2250000xf32> to memref<1x1x1500x1500xf32>
      %30 = rock.transform %29 by #transform_map23 : memref<1x1x1500x1500xf32> to memref<1x16x1500x1500xf32>
      %31 = rock.transform %28 by #transform_map24 : memref<1x16x1500x1500xf16> to memref<16x1500x1500xf16>
      %32 = rock.transform %30 by #transform_map24 : memref<1x16x1500x1500xf32> to memref<16x1500x1500xf32>
      %alloc_1 = memref.alloc() : memref<1x16x1500x1500xf32>
      %33 = rock.transform %alloc_1 by #transform_map25 : memref<1x16x1500x1500xf32> to memref<16x1500x1500xf32>

       // CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}} : memref<16xf16, #gpu.address_space<private>>, memref<16xf32, #gpu.address_space<private>>)
      // CHECK-SAME: outs(%{{.*}} : memref<16xf32, #gpu.address_space<private>>)
      // CHECK: ^bb0(%[[ARG0:.*]]: f16, %[[ARG1:.*]]: f32, %[[ARG2:.*]]: f32):
      // CHECK: arith.extf %[[ARG0]] : f16 to f32
      linalg.generic {indexing_maps = [#map7, #map7, #map7], iterator_types = ["parallel", "parallel", "parallel"]} ins(%31, %32 : memref<16x1500x1500xf16>, memref<16x1500x1500xf32>) outs(%33 : memref<16x1500x1500xf32>) attrs =  {rock.majorTensorNumber = 0 : index} {
      ^bb0(%in: f16, %in_2: f32, %out: f32):
        %34 = arith.extf %in : f16 to f32
        %35 = arith.addf %34, %in_2 : f32
        linalg.yield %35 : f32
      }
      memref.copy %alloc_1, %arg7 : memref<1x16x1500x1500xf32> to memref<1x16x1500x1500xf32>
      rock.yield
    } {blockSize = 64 : i32, firstGemmIndices = array<i64: 0>, gridSize = 752 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 1, 0>, params0 = #accel_gemm_params, params1 = #accel_gemm_params, prePadG0M = 1500 : index, prePadG0N = 1500 : index, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} : memref<16x64x1504xf16>, memref<16x64x1504xf16>, memref<16x1504x64xf32>, memref<2250000xf32>, memref<16x1504x64xf32>
    memref.copy %alloc_0, %arg4 : memref<1536000xf32> to memref<1536000xf32>
    return
  }
}

