// RUN: rocmlir-gen -emit-module-fusibility-for=v2:16,16,4,16,16,1,5,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:0
// RUN: rocmlir-gen -emit-module-fusibility-for=v2:16,16,4,16,16,1,1,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1
module {
  func.func @mlir_convolution_add_relu(%arg0: memref<64x1x1x1xf32>, %arg1: memref<1x256x56x56xf32>, %arg2: memref<64x256x1x1xf32>, %arg3: memref<1x64x56x56xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-"} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim3", "dim0", "dim1", "dim2"] at [3, 0, 1, 2]>] bounds = [1, 64, 1, 1] -> [64, 1, 1, 1]> : memref<64x1x1x1xf32> to memref<1x64x1x1xf32>
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 64, 56, 56] -> [1, 64, 1, 1]> : memref<1x64x1x1xf32> to memref<1x64x56x56xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x64x56x56xf32>
    %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 256 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 256} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 256, 56, 56] -> [1, 256, 56, 56]> : memref<1x256x56x56xf32> to memref<1x1x256x56x56xf32>
    %3 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 64, 256, 1, 1] -> [64, 256, 1, 1]> : memref<64x256x1x1xf32> to memref<1x64x256x1x1xf32>
    %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [1, 1, 64, 56, 56] -> [1, 64, 56, 56]> : memref<1x64x56x56xf32> to memref<1x1x64x56x56xf32>
    rock.conv2d(%3, %2, %4) features =  mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx90a:sramecc+:xnack-", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x64x256x1x1xf32>, memref<1x1x256x56x56xf32>, memref<1x1x64x56x56xf32>
    %5 = rock.transform %alloc by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [64, 56, 56] -> [1, 64, 56, 56]> : memref<1x64x56x56xf32> to memref<64x56x56xf32>
    %6 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [64, 56, 56] -> [1, 64, 56, 56]> : memref<1x64x56x56xf32> to memref<64x56x56xf32>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64x56x56xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : memref<64x56x56xf32>, memref<64x56x56xf32>) outs(%alloc_0 : memref<64x56x56xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.addf %in, %in_1 : f32
      %9 = arith.maximumf %8, %cst : f32
      linalg.yield %9 : f32
    }
    %7 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2, d3)> by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>] bounds = [1, 64, 56, 56] -> [64, 56, 56]> : memref<64x56x56xf32> to memref<1x64x56x56xf32>
    memref.copy %7, %arg3 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
    return
  }
}
