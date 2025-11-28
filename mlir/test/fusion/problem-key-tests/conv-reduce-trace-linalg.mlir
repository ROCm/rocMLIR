// RUN: rocmlir-gen --emit-tuning-key %s | FileCheck %s

// CHECK: 256	convfp16 -F 1 -f GNC01 -I NGC01 -O NGC01 -n 1 -c 128 -H 32 -W 32 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1  -fusion_reduce count=2 sum:rank3:axis2:hasPointwise sum:rank3:axis2:hasPointwise

#map = affine_map<(d0, d1, d2, d3) -> (((d0 * 128 + d1) * 3 + d2) * 3 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ((d1 * 32 + d2) * 32 + d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 128 + d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 256 + d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 256 + d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d2, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d0, d1, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 8 + d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 8 + d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, 0, 0)>
#map10 = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4)>
#map13 = affine_map<(d0) -> (0, d0 floordiv 8192, (d0 mod 8192) floordiv 1024, (d0 mod 1024) floordiv 32, d0 mod 32)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 1024, (d2 mod 1024) floordiv 32, d2 mod 32)>
#map15 = affine_map<(d0) -> (0, d0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [256, 128, 3, 3] -> [294912]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{128, 32, 32} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 32, 32] -> [131072]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 128} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 128, 32, 32] -> [1, 128, 32, 32]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 256} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 256, 128, 3, 3] -> [256, 128, 3, 3]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 256} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [1, 1, 256, 32, 32] -> [1, 256, 32, 32]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim1", "dim2", "dim3", "dim0", "dim4"] at [0, 1, 2, 3, 4] -> ["dim1", "dim2", "dim3", "dim0", "dim4"] at [1, 2, 3, 0, 4]>] bounds = [256, 128, 3, 1, 3] -> [1, 256, 128, 3, 3]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim2", "dim3", "dim0", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim2", "dim3", "dim0", "dim1", "dim4"] at [2, 3, 0, 1, 4]>] bounds = [128, 32, 1, 1, 32] -> [1, 1, 128, 32, 32]>
#transform_map7 = #rock.transform_map<#map7 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{32, 8} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [1, 32, 8, 32, 32] -> [1, 256, 32, 32]>
#transform_map8 = #rock.transform_map<#map8 by [<Unmerge{32, 8} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 32, 8, 1, 1] -> [256]>
#transform_map9 = #rock.transform_map<#map9 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <Broadcast{1} ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 32, 8, 32, 32] -> [1, 32, 8, 1, 1]>
#transform_map10 = #rock.transform_map<#map10 by [<Merge{1, 32} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [4]>] bounds = [32, 8, 32, 32] -> [1, 32, 8, 32, 32]>
#transform_map11 = #rock.transform_map<#map12 by [<Unmerge{32} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 32, 8, 32, 32] -> [32, 8, 32, 32]>
#transform_map12 = #rock.transform_map<#map13 by [<Merge{1, 32, 8, 32, 32} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [262144] -> [1, 32, 8, 32, 32]>
#transform_map13 = #rock.transform_map<#map14 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{8, 32, 32} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [1, 32, 8192] -> [1, 32, 8, 32, 32]>
#transform_map14 = #rock.transform_map<#map15 by [<Merge{1, 32, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [32] -> [1, 32, 1]>
module {
  func.func @mlir_reshape_convolution_reshape_broadcast_add_mul_reshape_reduce_sum_reshape_mul_reshape_reduce_sum_reshape(%arg0: memref<131072xf16>, %arg1: memref<294912xf16>, %arg2: memref<256xf16>, %arg3: memref<32xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16}, %arg4: memref<32xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16}, %arg5: memref<262144xf16>) attributes {arch = "gfx950:sramecc+:xnack-", enable_splitk_for_tuning, kernel = "mixr", num_cu = 256 : i64} {
    %cst = arith.constant 1.220700e-04 : f16
    %0 = rock.transform %arg1 by #transform_map : memref<294912xf16> to memref<256x128x3x3xf16>
    %1 = rock.transform %arg0 by #transform_map1 : memref<131072xf16> to memref<1x128x32x32xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x256x32x32xf16>
    %2 = rock.transform %1 by #transform_map2 : memref<1x128x32x32xf16> to memref<1x1x128x32x32xf16>
    %3 = rock.transform %0 by #transform_map3 : memref<256x128x3x3xf16> to memref<1x256x128x3x3xf16>
    %4 = rock.transform %alloc by #transform_map4 : memref<1x256x32x32xf16> to memref<1x1x256x32x32xf16>
    %5 = rock.transform %3 by #transform_map5 : memref<1x256x128x3x3xf16> to memref<256x128x3x1x3xf16>
    %6 = rock.transform %2 by #transform_map6 : memref<1x1x128x32x32xf16> to memref<128x32x1x1x32xf16>
    rock.conv(%3, %2, %4) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x256x128x3x3xf16>, memref<1x1x128x32x32xf16>, memref<1x1x256x32x32xf16>
    %7 = rock.transform %alloc by #transform_map7 : memref<1x256x32x32xf16> to memref<1x32x8x32x32xf16>
    %8 = rock.transform %arg2 by #transform_map8 : memref<256xf16> to memref<1x32x8x1x1xf16>
    %9 = rock.transform %8 by #transform_map9 : memref<1x32x8x1x1xf16> to memref<1x32x8x32x32xf16>
    %10 = rock.transform %7 by #transform_map10 : memref<1x32x8x32x32xf16> to memref<32x8x32x32xf16>
    %11 = rock.transform %9 by #transform_map10 : memref<1x32x8x32x32xf16> to memref<32x8x32x32xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<32x8x32x32xf16>
    linalg.generic {indexing_maps = [#map11, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%10, %11 : memref<32x8x32x32xf16>, memref<32x8x32x32xf16>) outs(%alloc_0 : memref<32x8x32x32xf16>) {
    ^bb0(%in: f16, %in_5: f16, %out: f16):
      %20 = arith.addf %in, %in_5 : f16
      linalg.yield %20 : f16
    }
    %12 = rock.transform %alloc_0 by #transform_map11 : memref<32x8x32x32xf16> to memref<1x32x8x32x32xf16>
    %13 = rock.transform %12 by #transform_map12 : memref<1x32x8x32x32xf16> to memref<262144xf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<32x8x32x32xf16>
    linalg.generic {indexing_maps = [#map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<32x8x32x32xf16>) outs(%alloc_1 : memref<32x8x32x32xf16>) {
    ^bb0(%in: f16, %out: f16):
      %20 = arith.mulf %in, %cst : f16
      linalg.yield %20 : f16
    }
    %14 = rock.transform %alloc_1 by #transform_map11 : memref<32x8x32x32xf16> to memref<1x32x8x32x32xf16>
    %15 = rock.transform %14 by #transform_map13 : memref<1x32x8x32x32xf16> to memref<1x32x8192xf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1xf16>
    rock.reduce  sum %15 into %alloc_2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 1024 : i32} : memref<1x32x8192xf16> into memref<1x32x1xf16>
    %16 = rock.transform %alloc_2 by #transform_map14 : memref<1x32x1xf16> to memref<32xf16>
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<32x8x32x32xf16>
    linalg.generic {indexing_maps = [#map11, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc_1, %alloc_0 : memref<32x8x32x32xf16>, memref<32x8x32x32xf16>) outs(%alloc_3 : memref<32x8x32x32xf16>) {
    ^bb0(%in: f16, %in_5: f16, %out: f16):
      %20 = arith.mulf %in, %in_5 : f16
      linalg.yield %20 : f16
    }
    %17 = rock.transform %alloc_3 by #transform_map11 : memref<32x8x32x32xf16> to memref<1x32x8x32x32xf16>
    %18 = rock.transform %17 by #transform_map13 : memref<1x32x8x32x32xf16> to memref<1x32x8192xf16>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x32x1xf16>
    rock.reduce  sum %18 into %alloc_4 {axis = 2 : index, blockSize = 256 : i32, gridSize = 1024 : i32} : memref<1x32x8192xf16> into memref<1x32x1xf16>
    %19 = rock.transform %alloc_4 by #transform_map14 : memref<1x32x1xf16> to memref<32xf16>
    memref.copy %16, %arg3 : memref<32xf16> to memref<32xf16>
    memref.copy %19, %arg4 : memref<32xf16> to memref<32xf16>
    memref.copy %13, %arg5 : memref<262144xf16> to memref<262144xf16>
    return
  }
}
