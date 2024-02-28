// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm %s | FileCheck %s
module  {
  func.func @rock_conv2d_bwd_weight_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x32x32x3x3xf32>, %arg1: memref<32x1x32x7x7xf32>, %arg2: memref<32x1x32x9x9xf32>) attributes {kernel = 0 : i32} {
    rock.conv2d_bwd_weight(%arg0, %arg1, %arg2) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908", dilations = [1, 1], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [2, 2, 2, 2], strides = [1, 1]} : memref<1x32x32x3x3xf32>, memref<32x1x32x7x7xf32>, memref<32x1x32x9x9xf32>
    return
  }
}

// CHECK-DAG: #[[map:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0, d1, d2) -> (0, 0, d1, d2 floordiv 9, (d2 mod 9) floordiv 3, d2 mod 3)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3, d4 - 2, d5 - 2)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4 + d5, d6 + d7)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, d2 floordiv 9, (d2 mod 9) floordiv 3, (d1 mod 81) floordiv 9, d2 mod 3, d1 mod 9)>
// CHECK-DAG: #[[map5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[map6:.*]] = affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, d2, (d1 mod 81) floordiv 9, d1 mod 9)>
// CHECK-DAG: #rock.transform_map<#[[map]] by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <AddDim{1} ["kBlock"] at [1] -> [] at []>, <PassThrough ["k", "c", "y", "x"] at [2, 3, 4, 5] -> ["k", "c", "y", "x"] at [1, 2, 3, 4]>] bounds = [1, 1, 32, 32, 3, 3] -> [1, 32, 32, 3, 3]>
// CHECK-DAG: #rock.transform_map<#[[map1]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["g", "kBlock"] at [0, 1]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [2]>, <Merge{32, 3, 3} ["gemmN"] at [2] -> ["c", "y", "x"] at [3, 4, 5]>] bounds = [1, 32, 288] -> [1, 1, 32, 32, 3, 3]>
// CHECK-DAG: #rock.transform_map<#[[map2]] by [<PassThrough ["gi"] at [2] -> ["gi"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["ni"] at [0]>, <PassThrough ["ci"] at [3] -> ["ci"] at [2]>, <Pad{2, 2, 2, 2} ["hipad", "wipad"] at [4, 5] -> ["hi", "wi"] at [3, 4]>] bounds = [1, 32, 1, 32, 11, 11] -> [32, 1, 32, 7, 7]>
// CHECK-DAG: #rock.transform_map<#[[map3]] by [<PassThrough ["gi", "n0", "n1", "ci"] at [2, 0, 1, 3] -> ["gi", "n0", "n1", "ci"] at [2, 0, 1, 3]>, <Embed{1, 1} ["y", "ho"] at [4, 5] -> ["hipad"] at [4]>, <Embed{1, 1} ["x", "wo"] at [6, 7] -> ["wipad"] at [5]>] bounds = [1, 32, 1, 32, 3, 9, 3, 9] -> [1, 32, 1, 32, 11, 11]>
// CHECK-DAG: #rock.transform_map<#[[map4]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["gi", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "ho", "wo"] at [1, 5, 7]>, <Merge{32, 3, 3} ["gemmN"] at [2] -> ["ci", "y", "x"] at [3, 4, 6]>] bounds = [1, 2592, 288] -> [1, 32, 1, 32, 3, 9, 3, 9]>
// CHECK-DAG: #rock.transform_map<#[[map5]] by [<PassThrough ["go"] at [2] -> ["go"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["no"] at [0]>, <PassThrough ["ko", "ho", "wo"] at [3, 4, 5] -> ["ko", "ho", "wo"] at [2, 3, 4]>] bounds = [1, 32, 1, 32, 9, 9] -> [32, 1, 32, 9, 9]>
// CHECK-DAG: #rock.transform_map<#[[map6]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["go", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "ho", "wo"] at [1, 4, 5]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [3]>] bounds = [1, 2592, 32] -> [1, 32, 1, 32, 9, 9]>
