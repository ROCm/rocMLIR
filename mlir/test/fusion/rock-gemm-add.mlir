// RUN: rocmlir-opt --rock-fold-transpose -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s | FileCheck %s

// CHECK-DAG: #[[MAP1:.*]] = #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d2)> by [<AddDim{1} ["exp0"] at [0] -> [] at []>, <PassThrough ["dim0"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>] bounds = [1, 1, 1000] -> [1, 1000]>
// CHECK-DAG: #[[MAP2:.*]] = #rock.transform_map<affine_map<(d0, d1) -> (0, d1)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>] bounds = [1, 1000] -> [1, 1000]>
// CHECK:rock.transforming_for{{.*}}#[[MAP1]], #[[MAP2]]
// CHECK:rock.transforming_for{{.*}}#[[MAP2]]
// CHECK-NEXT: %[[ldVal:.*]] = rock.global_load{{.*}}: memref<1x1000xf32> -> vector<4xf32>
// CHECK-NEXT: rock.in_bounds_store %[[ldVal]] ->{{.*}} : vector<4xf32> -> memref<4xf32, 5>, index
#map8 = affine_map<(d0) -> (d0)>
module {
  func.func @test_fusion(%arg0: memref<1x1x1x512xf32>, %arg1: memref<1x512x1000xf32>, %arg2: memref<1x1000xf32>, %arg3: memref<1x1000xf32>) attributes {kernel, arch = ""} {
      %0 = memref.collapse_shape %arg0 [[0], [1], [2, 3]] : memref<1x1x1x512xf32> into memref<1x1x512xf32>
      %1 = memref.alloc() {alignment = 128 : i64} : memref<1x1x1000xf32>
      rock.gemm %1 = %0 * %arg1 features =  none storeMethod =  set {arch = "amdgcn-amd-amdhsa:gfx900"} : memref<1x1x1000xf32> = memref<1x1x512xf32> * memref<1x512x1000xf32>
      %2 = memref.collapse_shape %1 [[0, 1, 2]] : memref<1x1x1000xf32> into memref<1000xf32>
      %3 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x1000xf32> into memref<1000xf32>
      %4 = memref.collapse_shape %arg3 [[0, 1]] : memref<1x1000xf32> into memref<1000xf32>
      linalg.generic {indexing_maps = [#map8, #map8, #map8], iterator_types = ["parallel"]} ins(%2, %3 : memref<1000xf32>, memref<1000xf32>) outs(%4 : memref<1000xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
        %5 = arith.addf %arg4, %arg5 : f32
        linalg.yield %5 : f32
      }
      return
  }
}
