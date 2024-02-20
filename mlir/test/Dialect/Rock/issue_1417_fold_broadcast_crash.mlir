// RUN: rocmlir-opt -rock-fold-broadcast %s | FileCheck %s
// Regression test for a crash where the number of Merge{}'s output dimension
// was used as an array index into the Merge parameters instead of the position
// of that output.
//
// While we're here, verify that this IR isn't eligible for broadcast folding
// CHECK: Broadcast{1}
func.func @mlir_reshape_reshape_transpose_dot(%arg0: tensor<1x256x32x64xf32>, %arg1: tensor<1x4x256x64xf32>) -> tensor<1x32x256x256xf32> attributes {arch = "gfx908:sramecc+:xnack-", kernel = "mixr", num_cu = 120 : i64} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)> by [<PassThrough ["dim0", "dim2", "dim1", "dim3"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim1", "dim3"] at [0, 2, 1, 3]>] bounds = [1, 32, 256, 64] -> [1, 256, 32, 64]> : tensor<1x256x32x64xf32> to tensor<1x32x256x64xf32>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{4, 1} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [1, 4, 1, 256, 64] -> [1, 4, 256, 64]> : tensor<1x4x256x64xf32> to tensor<1x4x1x256x64xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 4, 8, 256, 64] -> [1, 4, 1, 256, 64]> : tensor<1x4x1x256x64xf32> to tensor<1x4x8x256x64xf32>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 8, d1 mod 8, d2, d3)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Merge{4, 8} ["dim1"] at [1] -> ["col1", "col2"] at [1, 2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>] bounds = [1, 32, 256, 64] -> [1, 4, 8, 256, 64]> : tensor<1x4x8x256x64xf32> to tensor<1x32x256x64xf32>
  %4 = rock.transform %3 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [32, 256, 64] -> [1, 32, 256, 64]> : tensor<1x32x256x64xf32> to tensor<32x256x64xf32>
  %5 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [32, 256, 64] -> [1, 32, 256, 64]> : tensor<1x32x256x64xf32> to tensor<32x256x64xf32>
  %6 = bufferization.alloc_tensor() : tensor<32x256x256xf32>
  %7 = rock.gemm %6 = %5 * tr %4 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx908:sramecc+:xnack-", numCU = 120 : i32} : tensor<32x256x256xf32> = tensor<32x256x64xf32> * tensor<32x256x64xf32> -> tensor<32x256x256xf32>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2, d3) -> (d0 * 32 + d1, d2, d3)> by [<Unmerge{1, 32} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>] bounds = [1, 32, 256, 256] -> [32, 256, 256]> : tensor<32x256x256xf32> to tensor<1x32x256x256xf32>
  return %8 : tensor<1x32x256x256xf32>
}
