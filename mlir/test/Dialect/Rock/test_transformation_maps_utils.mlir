// RUN: rocmlir-opt -rock-transform-maps-utils-test -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @test_transform_ex1
func.func @test_transform_ex1(%arg0: memref<1x64x1024xf32>) -> memref<4x256x64xf32> attributes {remove_dims_by_indices = [0, 2]} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "D"] at [0, 1]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x256x64xf32>
  return %2 : memref<4x256x64xf32>
}

// CHECK-LABEL: @test_transform_ex2
func.func @test_transform_ex2(%arg0: memref<1x64x1024xf32>) -> memref<4x256x64xf32> attributes {remove_dims_by_names = ["A", "C"]} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "D"] at [0, 1]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x256x64xf32>
  return %2 : memref<4x256x64xf32>
}
