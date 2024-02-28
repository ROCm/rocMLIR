// RUN: rocmlir-opt -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-regularize %s -verify-diagnostics

#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (0, 0, 0, 0, d4)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{30} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{30} ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim0"] at [4] -> ["dim0"] at [0]>] bounds = [1, 1, 30, 30, 16] -> [1, 1, 1, 1, 16]>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (0, 0, 0, 0, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>

  func.func @invalid_generic_reduction(%arg0: memref<1x1x32x32x8xf32>, %arg1: memref<1x16x3x3x8xf32>, %arg2: memref<16xf32>, %arg3: memref<1x1x30x30x16xf32>) attributes {kernel, arch = ""} {
    %0 = memref.alloc() : memref<1x1x30x30x16xf32>
    rock.conv2d(%arg1, %arg0, %0) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1, 1], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["gi", "ni", "hi", "wi", "ci"], output_layout = ["go", "no", "ho", "wo", "ko"], padding = [0, 0, 0, 0], strides = [1, 1]} : memref<1x16x3x3x8xf32>, memref<1x1x32x32x8xf32>, memref<1x1x30x30x16xf32>
    %4 = memref.expand_shape %arg2 [[0, 1, 2, 3, 4]] : memref<16xf32> into memref<1x1x1x1x16xf32>
    %5 = rock.transform %4 by #transform_map1 : memref<1x1x1x1x16xf32> to memref<1x1x30x30x16xf32>
    // expected-error@+2 {{Only fully parallel supported}}
    // expected-error@+1 {{explicitly marked illegal}}
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["reduction", "parallel", "parallel", "parallel", "parallel"]} ins(%0, %5 : memref<1x1x30x30x16xf32>, memref<1x1x30x30x16xf32>) outs(%arg3 : memref<1x1x30x30x16xf32>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
      %8 = arith.addf %arg4, %arg5 : f32
      linalg.yield %8 : f32
    }
    return
  }
