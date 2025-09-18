// RUN: rocmlir-opt -rock-remove-output-alloc %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (((d0 * 4 + d1) * 3 + d2) * 3 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> ((d1 * 6 + d2) * 7 + d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 3 + d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 3 + d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 4 + d2, d3, d4)>
#map5 = affine_map<(d0) -> (0, d0 floordiv 209, (d0 mod 209) floordiv 19, d0 mod 19)>
#transform_map = #rock.transform_map<#map by [<Unmerge{3, 4, 3, 3} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [3, 4, 3, 3] -> [108]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{3, 6, 7} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 3, 6, 7] -> [126]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 3} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [1, 1, 3, 6, 7] -> [1, 3, 6, 7]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 3} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 3, 4, 3, 3] -> [3, 4, 3, 3]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 4} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [1, 1, 4, 11, 19] -> [1, 4, 11, 19]>
#transform_map5 = #rock.transform_map<#map5 by [<Merge{1, 4, 11, 19} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [836] -> [1, 4, 11, 19]>
// Check that we have created an inverse transform map to transform_map5
module {
  func.func @mlir_bwd_data_conv(%arg0: memref<126xf32>, %arg1: memref<108xf32>, %arg2: memref<836xf32>) attributes {arch = "gfx942", kernel} {
    %0 = rock.transform %arg1 by #transform_map : memref<108xf32> to memref<3x4x3x3xf32>
    %1 = rock.transform %arg0 by #transform_map1 : memref<126xf32> to memref<1x3x6x7xf32>
    // CHECK-NOT: %alloc
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x4x11x19xf32>
    // CHECK: %[[NEW_T:.*]] = rock.transform %arg2 by #transform_map{{.*}} : memref<836xf32> to memref<1x4x11x19xf32>
    %2 = rock.transform %1 by #transform_map2 : memref<1x3x6x7xf32> to memref<1x1x3x6x7xf32>
    %3 = rock.transform %0 by #transform_map3 : memref<3x4x3x3xf32> to memref<1x3x4x3x3xf32>
    %4 = rock.transform %alloc by #transform_map4 : memref<1x4x11x19xf32> to memref<1x1x4x11x19xf32>
    // CHECK: rock.transform %[[NEW_T]] by #transform_map{{.*}} : memref<1x4x11x19xf32> to memref<1x1x4x11x19xf32>
    rock.conv_bwd_data(%3, %4, %2) {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], kernelId = 0 : index, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 3 : index], usesV4R1 = false} : memref<1x3x4x3x3xf32>, memref<1x1x4x11x19xf32>, memref<1x1x3x6x7xf32>
    %5 = rock.transform %alloc by #transform_map5 : memref<1x4x11x19xf32> to memref<836xf32>
    // CHECK-NOT: memref.copy
    memref.copy %5, %arg2 : memref<836xf32> to memref<836xf32>
    return
  }
}

