// RUN: rocmlir-opt -rock-buffer-dependency-analysis-test %s | FileCheck %s --check-prefix=BDA
// RUN: rocmlir-opt -rock-function-fusibility-test %s | FileCheck %s --check-prefix=FUSION

#gemm_map = affine_map<(d0, d1) -> (0, d0, d1)>
#gemm_map1 = affine_map<(d0, d1) -> (d0, d1)>
#gemm_map2 = affine_map<(d0, d1, d2) -> (d0 * 64 + d1, d2)>
#gemm_transform_map = #rock.transform_map<#gemm_map by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [64, 64] -> [1, 64, 64]>
#gemm_transform_map1 = #rock.transform_map<#gemm_map2 by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 64, 64] -> [64, 64]>

// BDA-LABEL: @gemm_test1
// BDA: passed
// FUSION-LABEL: @gemm_test1
// FUSION: fusibile = "no"
func.func @gemm_test1(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x1024x64xf16>, %arg2: memref<1x64x64xf16>)
  attributes {
    kernel,
    expected = [{alloc_name = "alloc_0", writers = ["rock.gemm"], readers = ["linalg.generic"]},
                {alloc_name = "alloc_1", writers = ["linalg.generic"], readers = ["memref.copy"]}]} {
  %alloc_0 = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x64xf16>
  rock.gemm %alloc_0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = ""} : memref<1x64x64xf16> = memref<1x64x1024xf16> * memref<1x1024x64xf16>

  %0 = rock.transform %alloc_0 by #gemm_transform_map : memref<1x64x64xf16> to memref<64x64xf16>
  %alloc_1 = memref.alloc() {alignment = 64 : i64, name = "alloc_1"} : memref<64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  linalg.generic {indexing_maps = [#gemm_map1, #gemm_map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<64x64xf16>) outs(%alloc_1 : memref<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %2 = arith.maximumf %in, %cst : f16
    linalg.yield %2 : f16
  }
  %1 = rock.transform %alloc_1 by #gemm_transform_map1 : memref<64x64xf16> to memref<1x64x64xf16>

  memref.copy %1, %arg2 : memref<1x64x64xf16> to memref<1x64x64xf16>
  return
}

// BDA-LABEL: @gemm_test2
// BDA: passed
// FUSION-LABEL: @gemm_test2
// FUSION: fusibile = "no"
func.func @gemm_test2(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x1024x64xf16>, %arg2: memref<1x64x64xf16>)
  attributes {
    kernel,
    expected = [{alloc_name = "alloc_0", writers = ["rock.gemm", "linalg.generic"], readers = ["memref.copy"]}]} {
  %alloc = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x64xf16>
  rock.gemm %alloc = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = ""} : memref<1x64x64xf16> = memref<1x64x1024xf16> * memref<1x1024x64xf16>

  %0 = rock.transform %alloc by #gemm_transform_map : memref<1x64x64xf16> to memref<64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  linalg.generic {indexing_maps = [#gemm_map1], iterator_types = ["parallel", "parallel"]} outs(%0 : memref<64x64xf16>) {
  ^bb0(%out: f16):
    %2 = arith.maximumf %out, %cst : f16
    linalg.yield %2 : f16
  }
  %1 = rock.transform %0 by #gemm_transform_map1 : memref<64x64xf16> to memref<1x64x64xf16>

  memref.copy %1, %arg2 : memref<1x64x64xf16> to memref<1x64x64xf16>
  return
}

// BDA-LABEL: @gemm_test3
// BDA: passed
// FUSION-LABEL: @gemm_test3
// FUSION: fusibile = "yes"
func.func @gemm_test3(%arg0: memref<1x64x1024xf16>, %arg1: memref<1x1024x64xf16>, %arg2: memref<1x64x64xf16>, %arg3: memref<1x64x64xf16>)
  attributes {
    kernel,
    expected = [{alloc_name = "alloc_0", writers = ["rock.gemm"], readers = ["memref.copy"]},
                {alloc_name = "alloc_1", writers = ["linalg.generic"], readers = ["memref.copy"]}]} {
  %alloc_0 = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x64xf16>
  rock.gemm %alloc_0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = ""} : memref<1x64x64xf16> = memref<1x64x1024xf16> * memref<1x1024x64xf16>

  %0 = rock.transform %arg3 by #gemm_transform_map : memref<1x64x64xf16> to memref<64x64xf16>
  %alloc_1 = memref.alloc() {alignment = 64 : i64, name = "alloc_1"} : memref<64x64xf16>
  %cst = arith.constant 0.000000e+00 : f16
  linalg.generic {indexing_maps = [#gemm_map1, #gemm_map1], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<64x64xf16>) outs(%alloc_1 : memref<64x64xf16>) {
  ^bb0(%in: f16, %out: f16):
    %2 = arith.maximumf %in, %cst : f16
    linalg.yield %2 : f16
  }
  %1 = rock.transform %alloc_1 by #gemm_transform_map1 : memref<64x64xf16> to memref<1x64x64xf16>

  memref.copy %alloc_0, %arg2 : memref<1x64x64xf16> to memref<1x64x64xf16>
  memref.copy %1, %arg3 : memref<1x64x64xf16> to memref<1x64x64xf16>
  return
}


#conv_map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>
#conv_map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)>
#conv_map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d2, d3, d4)>
#conv_map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1, d2, d3, d4)>
#conv_map4 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#conv_map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#conv_map6 = affine_map<(d0, d1, d2, d3) -> (d0 * 64 + d1, d2, d3)>
#conv_transform_map = #rock.transform_map<#conv_map by [<PassThrough ["dim3", "dim0", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim3", "dim0", "dim1", "dim2"] at [3, 0, 1, 2]>] bounds = [1, 64, 1, 1] -> [64, 1, 1, 1]>
#conv_transform_map1 = #rock.transform_map<#conv_map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 64, 56, 56] -> [1, 64, 1, 1]>
#conv_transform_map2 = #rock.transform_map<#conv_map2 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 64} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 64, 56, 56] -> [1, 64, 56, 56]>
#conv_transform_map3 = #rock.transform_map<#conv_map3 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 64, 64, 1, 1] -> [64, 64, 1, 1]>
#conv_transform_map4 = #rock.transform_map<#conv_map2 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [1, 1, 64, 56, 56] -> [1, 64, 56, 56]>
#conv_transform_map5 = #rock.transform_map<#conv_map4 by [<Merge{1, 64} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [64, 56, 56] -> [1, 64, 56, 56]>
#conv_transform_map6 = #rock.transform_map<#conv_map6 by [<Unmerge{1, 64} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>] bounds = [1, 64, 56, 56] -> [64, 56, 56]>


// BDA-LABEL: @conv_test1
// BDA: passed
// FUSION-LABEL: @conv_test1
// FUSION: fusibile = "no"
func.func @conv_test1(%arg0: memref<64x1x1x1xf32>, %arg1: memref<1x64x56x56xf32>, %arg2: memref<64x64x1x1xf32>, %arg3: memref<1x64x56x56xf32>)
  attributes {
    kernel,
    expected = [{alloc_name = "alloc_0", writers = ["rock.conv"], readers = ["linalg.generic"]},
                {alloc_name = "alloc_1", writers = ["linalg.generic"], readers = ["memref.copy"]}]} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = rock.transform %arg0 by #conv_transform_map : memref<64x1x1x1xf32> to memref<1x64x1x1xf32>
  %1 = rock.transform %0 by #conv_transform_map1 : memref<1x64x1x1xf32> to memref<1x64x56x56xf32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64, name = "alloc_0"} : memref<1x64x56x56xf32>
  %2 = rock.transform %arg1 by #conv_transform_map2 : memref<1x64x56x56xf32> to memref<1x1x64x56x56xf32>
  %3 = rock.transform %arg2 by #conv_transform_map3 : memref<64x64x1x1xf32> to memref<1x64x64x1x1xf32>
  %4 = rock.transform %alloc_0 by #conv_transform_map4 : memref<1x64x56x56xf32> to memref<1x1x64x56x56xf32>
  rock.conv(%3, %2, %4) features =  none {arch = "", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x64x64x1x1xf32>, memref<1x1x64x56x56xf32>, memref<1x1x64x56x56xf32>
  %5 = rock.transform %alloc_0 by #conv_transform_map5 : memref<1x64x56x56xf32> to memref<64x56x56xf32>
  %6 = rock.transform %1 by #conv_transform_map5 : memref<1x64x56x56xf32> to memref<64x56x56xf32>
  %alloc_1 = memref.alloc() {alignment = 64 : i64, name = "alloc_1"} : memref<64x56x56xf32>
  linalg.generic {indexing_maps = [#conv_map5, #conv_map5, #conv_map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %6 : memref<64x56x56xf32>, memref<64x56x56xf32>) outs(%alloc_1 : memref<64x56x56xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %8 = arith.addf %in, %in_1 : f32
    %9 = arith.maximumf %8, %cst : f32
    linalg.yield %9 : f32
  }
  %7 = rock.transform %alloc_1 by #conv_transform_map6 : memref<64x56x56xf32> to memref<1x64x56x56xf32>
  memref.copy %7, %arg3 : memref<1x64x56x56xf32> to memref<1x64x56x56xf32>
  return
}
