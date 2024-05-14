// RUN: not rocmlir-opt -rock-affix-params %s
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 960 + d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 320 + d1, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 320 + d2, d3, d4)>
#transform_map = #rock.transform_map<#map by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 960} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 960, 128, 128] -> [2, 960, 128, 128]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 320, 960, 1, 1] -> [320, 960, 1, 1]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 320} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 320, 128, 128] -> [2, 320, 128, 128]>
module {
  func.func @mlir_convolution(%arg0: memref<2x960x128x128xf16>, %arg1: memref<320x960x1x1xf16>, %arg2: memref<2x320x128x128xf16>) attributes {arch = "gfx1100", kernel = "mixr", num_cu = 48 : i64} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x320x128x128xf16>
    %0 = rock.transform %arg0 by #transform_map : memref<2x960x128x128xf16> to memref<2x1x960x128x128xf16>
    %1 = rock.transform %arg1 by #transform_map1 : memref<320x960x1x1xf16> to memref<1x320x960x1x1xf16>
    %2 = rock.transform %alloc by #transform_map2 : memref<2x320x128x128xf16> to memref<2x1x320x128x128xf16>
    rock.conv(%1, %0, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "gfx1100", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCU = 48 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : index, 0 : index, 0 : index, 0 : index], perf_config = "v2:256,128,4,4,128,16,1,1,1", strides = [1 : index, 1 : index]} : memref<1x320x960x1x1xf16>, memref<2x1x960x128x128xf16>, memref<2x1x320x128x128xf16>
    memref.copy %alloc, %arg2 : memref<2x320x128x128xf16> to memref<2x320x128x128xf16>
    return
  }
}
