// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm %s | FileCheck %s
module  {
  func.func @rock_conv2d_bwd_weight_gkyxc_nghwc_nghwk_0(%arg0: memref<1x32x3x3x32xf32>, %arg1: memref<32x1x7x7x32xf32>, %arg2: memref<32x1x9x9x32xf32>) attributes {kernel = 0 : i32} {
    rock.conv2d_bwd_weight(%arg0, %arg1, %arg2) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "gi", "hi", "wi", "ci"], numCU = 120 : i32, output_layout = ["no", "go", "ho", "wo", "ko"], padding = [2 : i32, 2 : i32, 2 : i32, 2 : i32], strides = [1 : i32, 1 : i32]} : memref<1x32x3x3x32xf32>, memref<32x1x7x7x32xf32>, memref<32x1x9x9x32xf32>
    return
  }
}

// CHECK-DAG: #[[map:.*]] =  affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 96, (d2 mod 96) floordiv 32, d2 mod 32)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 - 2, d3 - 2, d4)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2 + d3, d4 + d5, d6)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0, d1, d2) -> (d1 floordiv 81, d0, d2 floordiv 96, (d1 mod 81) floordiv 9, (d2 mod 96) floordiv 32, d1 mod 9, d2 mod 32)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0, d1, d2) -> (d1 floordiv 81, d0, (d1 mod 81) floordiv 9, d1 mod 9, d2)>
// CHECK-DAG: #rock.transform_map<#[[map]]  by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [1]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["y", "x", "c"] at [2, 3, 4]>] bounds = [1, 32, 288] -> [1, 32, 3, 3, 32]>
// CHECK-DAG: #rock.transform_map<#[[map1]] by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [4] -> ["ci"] at [4]>, <Pad{2, 2, 2, 2} ["hipad", "wipad"] at [2, 3] -> ["hi", "wi"] at [2, 3]>] bounds = [32, 1, 11, 11, 32] -> [32, 1, 7, 7, 32]>
// CHECK-DAG: #rock.transform_map<#[[map2]] by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 6] -> ["ni", "gi", "ci"] at [0, 1, 4]>, <Embed{1, 1} ["y", "ho"] at [2, 3] -> ["hipad"] at [2]>, <Embed{1, 1} ["x", "wo"] at [4, 5] -> ["wipad"] at [3]>] bounds = [32, 1, 3, 9, 3, 9, 32] -> [32, 1, 11, 11, 32]>
// CHECK-DAG: #rock.transform_map<#[[map3]] by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["ni", "ho", "wo"] at [0, 3, 5]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["y", "x", "ci"] at [2, 4, 6]>] bounds = [1, 2592, 288] -> [32, 1, 3, 9, 3, 9, 32]>
// CHECK-DAG: #rock.transform_map<#[[map4]] by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["no", "ho", "wo"] at [0, 2, 3]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [4]>] bounds = [1, 2592, 32] -> [32, 1, 9, 9, 32]>
