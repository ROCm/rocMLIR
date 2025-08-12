// This test checks that we are able to remove redundant cast chains. In this
// case, we have a rock.gemm operation that will use MFMA, which accumulates
// in higher precision at F32, the result of which is then casted to F16. This
// instruction then has multiple uses, one of which is an ExtF operation. We
// want to make sure that we remove the ExtF operation, but preserve the
// other uses of the rock.gemm

// RUN: rocmlir-opt %s --rock-remove-redundant-casts | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1 * 3 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 4 + d2)>
#map2 = affine_map<(d0, d1) -> (0, d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map5 = affine_map<(d0) -> (0, d0 floordiv 3, d0 mod 3)>
#map6 = affine_map<(d0) -> (d0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{4, 3} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 4, 3] -> [12]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{5, 4} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 5, 4] -> [20]>
#transform_map2 = #rock.transform_map<#map2 by [<Merge{1, 5} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [5, 3] -> [1, 5, 3]>
#transform_map3 = #rock.transform_map<#map4 by [<Unmerge{5} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 5, 3] -> [5, 3]>
#transform_map4 = #rock.transform_map<#map5 by [<Merge{1, 5, 3} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [15] -> [1, 5, 3]>
module { 
  func.func @dot_add(%arg0: tensor<20xf16>, %arg1: tensor<12xf16>) -> tensor<15xf32> attributes {arch = "gfx942", kernel} {
    %0 = rock.transform %arg1 by #transform_map : tensor<12xf16> to tensor<1x4x3xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<20xf16> to tensor<1x5x4xf16>
    %2 = bufferization.alloc_tensor() : tensor<1x5x3xf16>
    %3 = rock.gemm %2 = %1 * %0 features =  mfma|dot|atomic_add|atomic_add_f16 storeMethod =  set {arch = "gfx942"} : tensor<1x5x3xf16> = tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf16>
    // Check that the original rock.gemm has been deleted and that we have
    // created a new one with the proper type
    // CHECK-NOT: rock.gemm {{.*}} tensor<1x5x3xf16> = tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf16>
    // CHECK: rock.gemm {{.*}} tensor<1x5x3xf32> = tensor<1x5x4xf16> * tensor<1x4x3xf16> -> tensor<1x5x3xf32>

    // Check that we have now created a truncf operation here for the other uses
    // of the rock.gemm
    // CHECK: arith.truncf %in : f32 to f16
    // Use of the rock.gemm directly
    %float_tensor = bufferization.alloc_tensor() : tensor<1x5x3xf16>
    %direct_use = arith.addf %3, %float_tensor : tensor<1x5x3xf16>
    %4 = rock.transform %3 by #transform_map2 : tensor<1x5x3xf16> to tensor<5x3xf16>
    // Use of the rock.transform (using rock.gemm indirectly)
    %float_tensor2 = bufferization.alloc_tensor() : tensor<5x3xf16>
    %indirect_use = arith.addf %4, %float_tensor2 : tensor<5x3xf16>
    %5 = bufferization.alloc_tensor() : tensor<5x3xf32>
    // Check that we have removed the redundant cast
    // CHECK-NOT: arith.extf %in : f16 to f32
    %6 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<5x3xf16>) outs(%5 : tensor<5x3xf32>) {
    ^bb0(%in: f16, %out: f32):
      %9 = arith.extf %in : f16 to f32
      linalg.yield %9 : f32
    } -> tensor<5x3xf32>
    %7 = rock.transform %6 by #transform_map3 : tensor<5x3xf32> to tensor<1x5x3xf32>
    %8 = rock.transform %7 by #transform_map4 : tensor<1x5x3xf32> to tensor<15xf32>
    %t1 = rock.transform %direct_use by #transform_map4 : tensor<1x5x3xf16> to tensor<15xf16>
    %t2 = rock.transform %indirect_use by #transform_map3 : tensor<5x3xf16> to tensor<1x5x3xf16>
    %t3 = rock.transform %t2 by #transform_map4 : tensor<1x5x3xf16> to tensor<15xf16>
    %combine_output = linalg.generic {indexing_maps = [#map6, #map6, #map6], iterator_types = ["parallel"]} ins(%t1, %t3 : tensor<15xf16>, tensor<15xf16>) outs(%8 : tensor<15xf32>) {
    ^bb0(%in1: f16, %in2: f16, %out: f32):
      %sum_f16 = arith.addf %in1, %in2 : f16
      %sum_f32 = arith.extf %sum_f16 : f16 to f32
      %result = arith.addf %sum_f32, %out : f32
      linalg.yield %result : f32
    } -> tensor<15xf32>
    return %combine_output : tensor<15xf32>
  }
}
