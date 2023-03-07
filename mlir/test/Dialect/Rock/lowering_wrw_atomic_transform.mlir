// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm %s | FileCheck %s
module  {
  func.func @rock_conv2d_bwd_weight_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x32x32x3x3xf32>, %arg1: memref<32x1x32x7x7xf32>, %arg2: memref<32x1x32x9x9xf32>) attributes {kernel = 0 : i32} {
    rock.conv2d_bwd_weight(%arg0, %arg1, %arg2) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [2 : i32, 2 : i32, 2 : i32, 2 : i32], strides = [1 : i32, 1 : i32]} : memref<1x32x32x3x3xf32>, memref<32x1x32x7x7xf32>, memref<32x1x32x9x9xf32>
    return
  }
}

// CHECK-DAG: #[[map:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 9, (d2 mod 9) floordiv 3, d2 mod 3)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3 - 2, d4 - 2)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0, d1, d2) -> (d1 floordiv 81, d0, d2 floordiv 9, (d2 mod 9) floordiv 3, (d1 mod 81) floordiv 9, d2 mod 3, d1 mod 9)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0, d1, d2) -> (d1 floordiv 81, d0, d2, (d1 mod 81) floordiv 9, d1 mod 9)>
// CHECK-DAG: #rock.transform_map<#[[map]] by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{32, 3, 3} ["gemmN"] at [2] -> ["c", "y", "x"] at [2, 3, 4]>] bounds = [1, 32, 288] -> [1, 32, 32, 3, 3]>
// CHECK-DAG: #rock.transform_map<#[[map1]] by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{2, 2, 2, 2} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [32, 1, 32, 11, 11] -> [32, 1, 32, 7, 7]>
// CHECK-DAG: #rock.transform_map<#[[map2]] by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, <Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [32, 1, 32, 3, 9, 3, 9] -> [32, 1, 32, 11, 11]>
// CHECK-DAG: #rock.transform_map<#[[map3]] by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["ni", "ho", "wo"] at [0, 4, 6]>, <Merge{32, 3, 3} ["gemmN"] at [2] -> ["ci", "y", "x"] at [2, 3, 5]>] bounds = [1, 2592, 288] -> [32, 1, 32, 3, 9, 3, 9]>
// CHECK-DAG: #rock.transform_map<#[[map4]] by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["no", "ho", "wo"] at [0, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [2]>] bounds = [1, 2592, 32] -> [32, 1, 32, 9, 9]>
