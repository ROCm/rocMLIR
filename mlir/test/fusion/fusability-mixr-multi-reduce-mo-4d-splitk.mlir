// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,5,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-SPLITK
// CHECK-SPLITK: fusible:1
// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,16,4,16,16,1,1,1,2,1,1 - < %s | FileCheck %s --check-prefixes=CHECK-NONSPLITK
// CHECK-NONSPLITK: fusible:1

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 10 + d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d1, d2, d3, d4, d0)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, 0, 0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 64 + d2) * 64 + d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d0, d1, d2, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 10 + d2, d3, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map12 = affine_map<(d0) -> (d0 floordiv 1310720, (d0 mod 1310720) floordiv 40960, (d0 mod 40960) floordiv 4096, (d0 mod 4096) floordiv 64, d0 mod 64)>
#map13 = affine_map<(d0, d1, d2) -> (d0, d1, d2 floordiv 4096, (d2 mod 4096) floordiv 64, d2 mod 64)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map15 = affine_map<(d0) -> (d0 floordiv 32, d0 mod 32, 0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{32, 10} ["exp0", "exp1"] at [0, 1] -> ["dim0"] at [0]>, <AddDim{1} ["unit2"] at [2] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [32, 10, 1, 1, 1] -> [320]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim4", "dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3, 4] -> ["dim4", "dim0", "dim1", "dim2", "dim3"] at [4, 0, 1, 2, 3]>] bounds = [1, 32, 10, 1, 1] -> [32, 10, 1, 1, 1]>
#transform_map2 = #rock.transform_map<#map2 by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>, <Broadcast{1} ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 32, 10, 64, 64] -> [1, 32, 10, 1, 1]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{320, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [320, 4, 3, 3] -> [11520]>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{2, 4, 64, 64} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [2, 4, 64, 64] -> [32768]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 4, 64, 64] -> [2, 4, 64, 64]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 4, 3, 3] -> [320, 4, 3, 3]>
#transform_map7 = #rock.transform_map<#map7 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 64, 64] -> [2, 320, 64, 64]>
#transform_map8 = #rock.transform_map<#map8 by [<PassThrough ["dim1", "dim2", "dim3", "dim0", "dim4"] at [0, 1, 2, 3, 4] -> ["dim1", "dim2", "dim3", "dim0", "dim4"] at [1, 2, 3, 0, 4]>] bounds = [320, 4, 3, 1, 3] -> [1, 320, 4, 3, 3]>
#transform_map9 = #rock.transform_map<#map9 by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [2, 4, 64, 1, 64] -> [2, 1, 4, 64, 64]>
#transform_map10 = #rock.transform_map<#map10 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{32, 10} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [2, 32, 10, 64, 64] -> [2, 320, 64, 64]>
#transform_map11 = #rock.transform_map<#map12 by [<Merge{2, 32, 10, 64, 64} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [2621440] -> [2, 32, 10, 64, 64]>
#transform_map12 = #rock.transform_map<#map13 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Merge{10, 64, 64} ["dim2"] at [2] -> ["col2", "col3", "col4"] at [2, 3, 4]>] bounds = [2, 32, 40960] -> [2, 32, 10, 64, 64]>
#transform_map13 = #rock.transform_map<#map14 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Unmerge{40960} ["exp2"] at [2] -> ["dim2"] at [2]>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [2, 32, 40960, 1] -> [2, 32, 40960]>
#transform_map14 = #rock.transform_map<#map15 by [<Merge{2, 32, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [64] -> [2, 32, 1, 1]>
module {
  func.func @mlir_convolution_multi_reduce(%arg0: memref<320xf16>, %arg1: memref<32768xf16>, %arg2: memref<11520xf16>, %arg3: memref<64xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16}, %arg4: memref<2621440xf16>) attributes {arch = "gfx942:sramecc+:xnack-", enable_splitk_for_tuning, kernel = "mixr", features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>} {
    %cst = arith.constant 2.443790e-05 : f16
    %0 = rock.transform %arg0 by #transform_map : memref<320xf16> to memref<32x10x1x1x1xf16>
    %1 = rock.transform %0 by #transform_map1 : memref<32x10x1x1x1xf16> to memref<1x32x10x1x1xf16>
    %2 = rock.transform %1 by #transform_map2 : memref<1x32x10x1x1xf16> to memref<2x32x10x64x64xf16>
    %3 = rock.transform %arg2 by #transform_map3 : memref<11520xf16> to memref<320x4x3x3xf16>
    %4 = rock.transform %arg1 by #transform_map4 : memref<32768xf16> to memref<2x4x64x64xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x64x64xf16>
    %5 = rock.transform %4 by #transform_map5 : memref<2x4x64x64xf16> to memref<2x1x4x64x64xf16>
    %6 = rock.transform %3 by #transform_map6 : memref<320x4x3x3xf16> to memref<1x320x4x3x3xf16>
    %7 = rock.transform %alloc by #transform_map7 : memref<2x320x64x64xf16> to memref<2x1x320x64x64xf16>
    %8 = rock.transform %6 by #transform_map8 : memref<1x320x4x3x3xf16> to memref<320x4x3x1x3xf16>
    %9 = rock.transform %5 by #transform_map9 : memref<2x1x4x64x64xf16> to memref<2x4x64x1x64xf16>
    rock.conv(%6, %5, %7) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x320x4x3x3xf16>, memref<2x1x4x64x64xf16>, memref<2x1x320x64x64xf16>
    %10 = rock.transform %alloc by #transform_map10 : memref<2x320x64x64xf16> to memref<2x32x10x64x64xf16>
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf16>
    linalg.generic {indexing_maps = [#map11, #map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%10, %2 : memref<2x32x10x64x64xf16>, memref<2x32x10x64x64xf16>) outs(%alloc_0 : memref<2x32x10x64x64xf16>) {
    ^bb0(%in: f16, %in_3: f16, %out: f16):
      %15 = arith.addf %in, %in_3 : f16
      linalg.yield %15 : f16
    }
    %11 = rock.transform %alloc_0 by #transform_map11 : memref<2x32x10x64x64xf16> to memref<2621440xf16>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<2x32x10x64x64xf16>
    linalg.generic {indexing_maps = [#map11, #map11], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc_0 : memref<2x32x10x64x64xf16>) outs(%alloc_1 : memref<2x32x10x64x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %15 = arith.mulf %in, %cst : f16
      linalg.yield %15 : f16
    }
    %12 = rock.transform %alloc_1 by #transform_map12 : memref<2x32x10x64x64xf16> to memref<2x32x40960xf16>
    %13 = rock.transform %12 by #transform_map13 : memref<2x32x40960xf16> to memref<2x32x40960x1xf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<2x32x1x1xf16>
    rock.reduce  sum %13 into %alloc_2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 10240 : i32} : memref<2x32x40960x1xf16> into memref<2x32x1x1xf16>
    %14 = rock.transform %alloc_2 by #transform_map14 : memref<2x32x1x1xf16> to memref<64xf16>
    memref.copy %14, %arg3 : memref<64xf16> to memref<64xf16>
    memref.copy %11, %arg4 : memref<2621440xf16> to memref<2621440xf16>
    return
  }
}
