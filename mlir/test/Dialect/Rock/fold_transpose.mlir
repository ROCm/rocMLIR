//RUN: rocmlir-opt --rock-fold-transpose %s
#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> ()>
#map2 = affine_map<(d0, d1, d2) -> (d2)>
module {
    func.func private @bert_part_3__part_0(%arg0: memref<1x12x12x32xf32> {func.read_access}, %arg1: memref<1x12x32x12xf32> {func.read_access}, %arg2: memref<1x1x1x1xf32> {func.read_access}, %arg3: memref<1x1x1x12xf32> {func.read_access}, %arg4: memref<1x12x12x12xf32> {func.write_access}) attributes {kernel, original_func = @bert_part_3__part_0} {
      %0 = memref.collapse_shape %arg1 [[0, 1], [2], [3]] : memref<1x12x32x12xf32> into memref<12x32x12xf32>
      %1 = memref.collapse_shape %arg0 [[0, 1], [2], [3]] : memref<1x12x12x32xf32> into memref<12x12x32xf32>
      %2 = memref.alloc() {alignment = 128 : i64} : memref<12x12x12xf32>
      rock.gemm %2 = %1 * %0 features =  mfma|dot|atomic_add storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx90a"} : memref<12x12x12xf32> = memref<12x12x32xf32> * memref<12x32x12xf32>
      %3 = memref.collapse_shape %arg2 [] : memref<1x1x1x1xf32> into memref<f32>
      %4 = memref.collapse_shape %arg3 [[0, 1, 2, 3]] : memref<1x1x1x12xf32> into memref<12xf32>
      %5 = memref.collapse_shape %arg4 [[0, 1], [2], [3]] : memref<1x12x12x12xf32> into memref<12x12x12xf32>
      linalg.generic {indexing_maps = [#map0, #map1, #map2, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2, %3, %4 : memref<12x12x12xf32>, memref<f32>, memref<12xf32>) outs(%5 : memref<12x12x12xf32>) {
      ^bb0(%arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
        %6 = arith.mulf %arg5, %arg6 : f32
        %7 = arith.addf %6, %arg7 : f32
        linalg.yield %7 : f32
      }
      return
    }
}
