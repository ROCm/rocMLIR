// This test checks that we are able to remove redundant cast chains. In this
// case, we have a TruncF operation whose result is immediately followed by
// an ExtF operation. 

// RUN: rocmlir-opt %s --rock-remove-redundant-casts | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1 * 3 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#map2 = affine_map<(d0, d1) -> (0, d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map4 = affine_map<(d0) -> (0, d0 floordiv 3, d0 mod 3)>
#map5 = affine_map<(d0, d1) -> (d0, d1)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 3} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 3] -> [12]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{5, 4} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 5, 4] -> [20]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{1, 5} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [5, 3] -> [1, 5, 3]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{5} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 5, 3] -> [5, 3]>
#transform_map4 = #rock.transform_map<#map4 by [<Merge{1, 5, 3} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [15] -> [1, 5, 3]>
module {
  func.func @dot_add(%arg0: tensor<20xf16>, %arg1: tensor<12xf16>) -> tensor<15xf32> attributes {arch = "gfx942", kernel} {
    %0 = rock.transform %arg1 by #transform_map : tensor<12xf16> to tensor<1x4x3xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<20xf16> to tensor<1x5x4xf16>
    %3 = bufferization.alloc_tensor() : tensor<1x5x3xf32>
    %4 = rock.gemm %3 = %1 * %0 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942"} : tensor<1x5x3xf32> = tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf32>
    %5 = rock.transform %4 by #transform_map2 : tensor<1x5x3xf32> to tensor<5x3xf32>

    %temp_alloc = bufferization.alloc_tensor() : tensor<5x3xf16>
    // CHECK-NOT: %downcast
    %downcast = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%5 : tensor<5x3xf32>) outs(%temp_alloc : tensor<5x3xf16>) {
    ^bb0(%in: f32, %out: f16):
      %truncf = arith.truncf %in : f32 to f16
      linalg.yield %truncf : f16
    } -> tensor<5x3xf16>
    %res_alloc = bufferization.alloc_tensor() : tensor<5x3xf32>
    // CHECK-NOT: %upcast
    %upcast = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%downcast : tensor<5x3xf16>) outs(%res_alloc : tensor<5x3xf32>) {
    ^bb0(%in: f16, %out: f32):
      %extf = arith.extf %in : f16 to f32
      linalg.yield %extf : f32
    } -> tensor<5x3xf32>

    %7 = rock.transform %upcast by #transform_map3 : tensor<5x3xf32> to tensor<1x5x3xf32>
    %8 = rock.transform %7 by #transform_map4 : tensor<1x5x3xf32> to tensor<15xf32>
    return %8 : tensor<15xf32>
  }
}
