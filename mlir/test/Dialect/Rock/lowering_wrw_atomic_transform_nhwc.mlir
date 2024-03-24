// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm %s | FileCheck %s
module  {
  func.func @rock_conv_bwd_weight_gkyxc_nghwc_nghwk_0(%arg0: memref<1x32x3x3x32xf32>, %arg1: memref<32x1x7x7x32xf32>, %arg2: memref<32x1x9x9x32xf32>) attributes {kernel = 0 : i32} {
    rock.conv_bwd_weight(%arg0, %arg1, %arg2) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "gi", "hi", "wi", "ci"], numCU = 120 : i32, output_layout = ["no", "go", "ho", "wo", "ko"], padding = [2 : index, 2 : index, 2 : index, 2 : index], strides = [1 : index, 1 : index]} : memref<1x32x3x3x32xf32>, memref<32x1x7x7x32xf32>, memref<32x1x9x9x32xf32>
    return
  }
}

// CHECK-DAG: #[[map:.*]] =  affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d4, d5)>
// CHECK-DAG: #[[map1:.*]] = affine_map<(d0, d1, d2) -> (0, 0, d1, d2 floordiv 96, (d2 mod 96) floordiv 32, d2 mod 32)>
// CHECK-DAG: #[[map2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3 - 2, d4 - 2, d5)>
// CHECK-DAG: #[[map3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3 + d4, d5 + d6, d7)>
// CHECK-DAG: #[[map4:.*]] = affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, d2 floordiv 96, (d1 mod 81) floordiv 9, (d2 mod 96) floordiv 32, d1 mod 9, d2 mod 32)>
// CHECK-DAG: #[[map5:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d1, d2, d3, d4, d5)>
// CHECK-DAG: #[[map6:.*]] = affine_map<(d0, d1, d2) -> (0, d1 floordiv 81, 0, (d1 mod 81) floordiv 9, d1 mod 9, d2)>
// CHECK-DAG: #rock.transform_map<#[[map]]  by [<PassThrough ["g"] at [0] -> ["g"] at [0]>, <AddDim{1} ["kBlock"] at [1] -> [] at []>, <PassThrough ["k", "c", "0", "1"] at [2, 5, 3, 4] -> ["k", "c", "0", "1"] at [1, 4, 2, 3]>] bounds = [1, 1, 32, 3, 3, 32] -> [1, 32, 3, 3, 32]>
// CHECK-DAG: #rock.transform_map<#[[map1]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["g", "kBlock"] at [0, 1]>, <PassThrough ["gemmM"] at [1] -> ["k"] at [2]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["0", "1", "c"] at [3, 4, 5]>] bounds = [1, 32, 288] -> [1, 1, 32, 3, 3, 32]>
// CHECK-DAG: #rock.transform_map<#[[map2]] by [<PassThrough ["gi"] at [2] -> ["gi"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["ni"] at [0]>, <PassThrough ["ci"] at [5] -> ["ci"] at [4]>, <Pad{2, 2, 2, 2} ["hipad", "wipad"] at [3, 4] -> ["0i", "1i"] at [2, 3]>] bounds = [1, 32, 1, 11, 11, 32] -> [32, 1, 7, 7, 32]>
// CHECK-DAG: #rock.transform_map<#[[map3]] by [<PassThrough ["gi", "n0", "n1", "ci"] at [2, 0, 1, 7] -> ["gi", "n0", "n1", "ci"] at [2, 0, 1, 5]>, <Embed{1, 1} ["0", "0o"] at [3, 4] -> ["hipad"] at [3]>, <Embed{1, 1} ["1", "1o"] at [5, 6] -> ["wipad"] at [4]>] bounds = [1, 32, 1, 3, 9, 3, 9, 32] -> [1, 32, 1, 11, 11, 32]>
// CHECK-DAG: #rock.transform_map<#[[map4]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["gi", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "0o", "1o"] at [1, 4, 6]>, <Merge{3, 3, 32} ["gemmN"] at [2] -> ["0", "1", "ci"] at [3, 5, 7]>] bounds = [1, 2592, 288] -> [1, 32, 1, 3, 9, 3, 9, 32]>
// CHECK-DAG: #rock.transform_map<#[[map5]] by [<PassThrough ["go"] at [2] -> ["go"] at [1]>, <Unmerge{1, 32} ["n0", "n1"] at [0, 1] -> ["no"] at [0]>, <PassThrough ["ko", "0o", "1o"] at [5, 3, 4] -> ["ko", "0o", "1o"] at [4, 2, 3]>] bounds = [1, 32, 1, 9, 9, 32] -> [32, 1, 9, 9, 32]>
// CHECK-DAG: #rock.transform_map<#[[map6]] by [<Merge{1, 1} ["gemmG"] at [0] -> ["go", "n0"] at [2, 0]>, <Merge{32, 9, 9} ["gemmK"] at [1] -> ["n1", "0o", "1o"] at [1, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["ko"] at [5]>] bounds = [1, 2592, 32] -> [1, 32, 1, 9, 9, 32]>
