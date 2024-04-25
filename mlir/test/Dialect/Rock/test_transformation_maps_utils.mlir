// RUN: rocmlir-opt -rock-transform-maps-utils-test -allow-unregistered-dialect --mlir-print-local-scope --mlir-disable-threading %s | FileCheck %s

// CHECK-LABEL: test transform_ex1
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]>
func.func @transform_ex1(%arg0: memref<4x1x32xf32>) -> memref<4x32xf32>
  attributes {
    remove_dims_by_names = ["A"]
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : memref<4x1x32xf32> to memref<1x4x32xf32>
  %1= rock.transform %0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  return %1 : memref<4x32xf32>
}

// CHECK-LABEL: test transform_ex2
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>] bounds = [4] -> [1, 4]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["B", "A"] at [0, 1] -> ["A", "B"] at [0, 1]>] bounds = [1, 4] -> [4, 1]>
func.func @transform_ex2(%arg0: memref<4x1x32xf32>) -> memref<4x32xf32>
  attributes {
    remove_dims_by_names = ["B"]
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : memref<4x1x32xf32> to memref<1x4x32xf32>
  %1= rock.transform %0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["B"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  return %1 : memref<4x32xf32>
}

// CHECK-LABEL: test transform_ex3
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["B"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [0, 1, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]>
func.func @transform_ex3(%arg0: memref<4x1x32xf32>) -> memref<4x32xf32>
  attributes {
    remove_dims_by_names = ["X"]
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : memref<4x1x32xf32> to memref<1x4x32xf32>
  %1= rock.transform %0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["B"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  return %1 : memref<4x32xf32>
}

// CHECK-LABEL1: @test_transform_ex6
//func.func @test_transform_ex6(%arg0: memref<1x64x1024xf32>) -> memref<4x256x64xf32> attributes {remove_dims_by_indices = [0, 2]} {
//  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
//  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
//  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "D"] at [0, 1]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x256x64xf32>
//  return %2 : memref<4x256x64xf32>
//}

//// CHECK-LABEL1: @test_transform_ex7
//func.func @test_transform_ex7(%arg0: memref<1x64x1024xf32>) -> memref<4x256x64xf32> attributes {remove_dims_by_names = ["A", "C"]} {
//  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
//  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
//  %2 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "D"] at [0, 1]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 3]>] bounds = [4, 256, 64] -> [1, 4, 256, 64]> : memref<1x4x256x64xf32> to memref<4x256x64xf32>
//  return %2 : memref<4x256x64xf32>
//}
