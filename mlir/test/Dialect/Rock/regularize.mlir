// RUN: rocmlir-opt --rock-regularize %s | FileCheck %s

// -----// IR Dump After RockGemmToGridwisePass (rock-gemm-to-gridwise) //----- //
func.func private @bert_part_11__part_0(%arg0: memref<1x12x12x32xf32> {func.read_access}, %arg1: memref<1x12x32x12xf32> {func.read_access}, %arg2: memref<1x1x1x1xf32> {func.read_access}, %arg3: memref<1x1x1x12xf32> {func.read_access}, %arg4: memref<1x12x12x12xf32> {func.write_access}) attributes {block_size = 64 : i32, grid_size = 12 : i32, kernel, original_func = @bert_part_11__part_0} {
  %0 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> (d0 floordiv 12, d0 mod 12, d1, d2)> by [<Merge{1, 12} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [12, 32, 12] -> [1, 12, 32, 12]> : memref<1x12x32x12xf32> to memref<12x32x12xf32>
  %1 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0 floordiv 12, d0 mod 12, d1, d2)> by [<Merge{1, 12} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [12, 12, 32] -> [1, 12, 12, 32]> : memref<1x12x12x32xf32> to memref<12x12x32xf32>
  %2 = memref.alloc() {alignment = 128 : i64} : memref<12x12x12xf32>
  %3 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [12, 32, 12] -> [12, 12, 32]> : memref<12x12x32xf32> to memref<12x32x12xf32>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 4} ["gemmMPad"] at [2] -> ["gemmM"] at [2]>] bounds = [12, 32, 16] -> [12, 32, 12]> : memref<12x32x12xf32> to memref<12x32x16xf32>
  %5 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK"] at [1] -> ["gemmK"] at [1]>, <Pad{0, 4} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [12, 32, 16] -> [12, 32, 12]> : memref<12x32x12xf32> to memref<12x32x16xf32>
  %6 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 4} ["gemmMPad"] at [1] -> ["gemmM"] at [1]>, <Pad{0, 4} ["gemmNPad"] at [2] -> ["gemmN"] at [2]>] bounds = [12, 16, 16] -> [12, 12, 12]> : memref<12x12x12xf32> to memref<12x16x16xf32>
  rock.gridwise_gemm_accel(%4, %5, %6) storeMethod( set) features = mfma {arch = "amdgcn-amd-amdhsa:gfx90a", blockSize = 64 : i32, gridSize = 12 : i32, params = #rock.xdlops_gemm_params<kpackPerBlock = 16, mPerBlock = 16, nPerBlock = 16, kpack = 1, mPerWave = 16, nPerWave = 16, forceUnroll = true>} : memref<12x32x16xf32>, memref<12x32x16xf32>, memref<12x16x16xf32>
  %7 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0 * 12 + d1, d2, d3)> by [<Unmerge{1, 12} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>] bounds = [1, 12, 12, 12] -> [12, 12, 12]> : memref<12x12x12xf32> to memref<1x12x12x12xf32>
  %8 = memref.collapse_shape %7 [[0, 1], [2], [3]] : memref<1x12x12x12xf32> into memref<12x12x12xf32>
  %9 = memref.collapse_shape %arg2 [] : memref<1x1x1x1xf32> into memref<f32>
  %10 = memref.collapse_shape %arg3 [[0, 1, 2, 3]] : memref<1x1x1x12xf32> into memref<12xf32>
  %11 = memref.alloc() {alignment = 128 : i64} : memref<12x12x12xf32>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> ()>, affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8, %9, %10 : memref<12x12x12xf32>, memref<f32>, memref<12xf32>) outs(%11 : memref<12x12x12xf32>) {
  ^bb0(%arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
    %13 = arith.mulf %arg5, %arg6 : f32
    %14 = arith.addf %13, %arg7 : f32
    linalg.yield %14 : f32
  }
  %12 = memref.expand_shape %11 [[0, 1], [2], [3]] : memref<12x12x12xf32> into memref<1x12x12x12xf32>
  memref.copy %12, %arg4 : memref<1x12x12x12xf32> to memref<1x12x12x12xf32>
  return
}
// CHECK: [[MAP0:#.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func private @bert_part_11__part_0([[ARG0:%arg.*]]: memref<1x12x12x32xf32> {func.read_access}, [[ARG1:%arg.*]]: memref<1x12x32x12xf32> {func.read_access}, [[ARG2:%arg.*]]: memref<1x1x1x1xf32> {func.read_access}, [[ARG3:%arg.*]]: memref<1x1x1x12xf32> {func.read_access}, [[ARG4:%arg.*]]: memref<1x12x12x12xf32> {func.write_access})
// CHECK: [[ALLOC0:%.*]] = memref.alloc() : memref<12x12x12xf32>

// CHECK: [[ARG2_TR0:%.*]] = rock.transform [[ARG2]] by {{.*}} : memref<1x1x1x1xf32> to memref<f32>
// CHECK: [[ARG3_TR0:%.*]] = rock.transform [[ARG3]] by {{.*}} : memref<1x1x1x12xf32> to memref<12xf32>
// CHECK: [[ALLOC1:%.*]] = memref.alloc() : memref<1x12x12x12xf32>
// CHECK: [[ALLOC1_TR0:%.*]] = rock.transform [[ALLOC1]] by {{.*}} : memref<1x12x12x12xf32> to memref<12x12x12xf32>
// CHECK: [[ARG2_TR1:%.*]] = rock.transform [[ARG2_TR0]] by {{.*}} : memref<f32> to memref<1x1x1xf32>
// CHECK: [[ARG2_TR2:%.*]] = rock.transform [[ARG2_TR1]] by {{.*}} : memref<1x1x1xf32> to memref<12x12x12xf32>
// CHECK: [[ARG3_TR1:%.*]] = rock.transform [[ARG3_TR0]] by {{.*}} : memref<12xf32> to memref<1x1x12xf32>
// CHECK: [[ARG3_TR2:%.*]] = rock.transform [[ARG3_TR1]] by {{.*}} : memref<1x1x12xf32> to memref<12x12x12xf32>

// CHECK: linalg.generic {indexing_maps = [[[MAP0]], [[MAP0]], [[MAP0]], [[MAP0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins([[ALLOC0]], [[ARG2_TR2]], [[ARG3_TR2]] : memref<12x12x12xf32>, memref<12x12x12xf32>, memref<12x12x12xf32>) outs([[ARG4_TR:.*]] : memref<12x12x12xf32>)

// CHECK: memref.copy [[ALLOC1]], [[ARG4]] : memref<1x12x12x12xf32> to memref<1x12x12x12xf32>
