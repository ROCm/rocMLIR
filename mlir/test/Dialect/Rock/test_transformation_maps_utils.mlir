// RUN: rocmlir-opt -rock-transform-maps-utils-test -allow-unregistered-dialect --mlir-print-local-scope --mlir-disable-threading %s | FileCheck %s

// CHECK-LABEL: test0 transform_ex1
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["C"] at [0] -> ["C"] at [0]>] bounds = [32] -> [32]>

// CHECK-LABEL: test1 transform_ex1
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>] bounds = [4] -> [1, 4]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "A"] at [0, 1] -> ["A", "B"] at [1, 0]>] bounds = [1, 4] -> [4, 1]>

// CHECK-LABEL: test2 transform_ex1
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]>

func.func @transform_ex1(%arg0: memref<4x1x32xf32>) -> memref<4x32xf32>
  attributes {
    remove_dims_by_names = {
      test0 = ["A"],
      test1 = ["C"],
      test2 = ["X"]}
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d1, d0, d2)> by [<PassThrough ["B", "A", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [1, 0, 2]>] bounds = [1, 4, 32] -> [4, 1, 32]> : memref<4x1x32xf32> to memref<1x4x32xf32>
  %1= rock.transform %0 by <affine_map<(d0, d1) -> (0, d0, d1)> by [<Merge{1, 4} ["A"] at [0] -> ["A", "B"] at [0, 1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [4, 32] -> [1, 4, 32]> : memref<1x4x32xf32> to memref<4x32xf32>
  return %1 : memref<4x32xf32>
}

// CHECK-LABEL: test0 transform_ex2
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "C"] at [0, 2] -> ["A", "C"] at [0, 2]>, <Unmerge{256} ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 256, 64] -> [1, 256, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 256, 64] -> [1, 64, 256]>

// CHECK-LABEL: test1 transform_ex2
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "C"] at [0, 2] -> ["A", "C"] at [0, 2]>, <Unmerge{4} ["D"] at [1] -> ["B"] at [1]>] bounds = [1, 4, 64] -> [1, 4, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 4, 64] -> [1, 64, 4]>

// CHECK-LABEL: test2 transform_ex2
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A", "C"] at [0, 1] -> ["A", "C"] at [0, 1]>] bounds = [1, 64] -> [1, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["C"] at [1] -> ["C"] at [1]>] bounds = [1, 64] -> [1, 64]>

// CHECK-LABEL: test3 transform_ex2
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0 * 256 + d1)> by [<Unmerge{4, 256} ["D", "B"] at [0, 1] -> ["B"] at [0]>] bounds = [4, 256] -> [1024]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["B"] at [0] -> ["B"] at [0]>] bounds = [1024] -> [1024]>

func.func @transform_ex2(%arg0: memref<1x64x1024xf32>) -> memref<1x4x256x64xf32>
  attributes {
    remove_dims_by_names = {
      test0 = ["D"],
      test1 = ["B"],
      test2 = ["B", "D"],
      test3 = ["A", "C"]
    }
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1 * 256 + d2, d3)> by [<PassThrough ["A", "C"] at [0, 3] -> ["A", "C"] at [0, 2]>, <Unmerge{4, 256} ["D", "B"] at [1, 2] -> ["B"] at [1]>] bounds = [1, 4, 256, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x4x256x64xf32>
  return %1 : memref<1x4x256x64xf32>
}

// CHECK-LABEL: test0 transform_ex3
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [0, 1]>] bounds = [1024, 64] -> [1024, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [1, 0]>] bounds = [1024, 64] -> [64, 1024]>

// CHECK-LABEL: test1 transform_ex3
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0 mod 32, d1)> by [<Broadcast{32} ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [32, 1024] -> [1, 1024]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 1024] -> [1, 1024]>

func.func @transform_ex3(%arg0: memref<1x64x1024xf32>) -> memref<32x1024x64xf32>
  attributes {
    remove_dims_by_names = {
      test0 = ["A"],
      test1 = ["C"]
    }
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (0, d1, d2)> by [<Broadcast{32} ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [1, 2]>] bounds = [32, 1024, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<32x1024x64xf32>
  return %1 : memref<32x1024x64xf32>
}

// CHECK-LABEL: test0 transform_ex4
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (0, d0)> by [<ConstDim{0, 1} [] at [] -> ["B"] at [0]>, <PassThrough ["C"] at [0] -> ["C"] at [1]>] bounds = [64] -> [1024, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["B", "C"] at [0, 1] -> ["B", "C"] at [1, 0]>] bounds = [1024, 64] -> [64, 1024]>

// CHECK-LABEL: test1 transform_ex4
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0, 0)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <ConstDim{0, 1} [] at [] -> ["B"] at [1]>] bounds = [1] -> [1, 1024]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1) -> (d0, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B"] at [1] -> ["B"] at [1]>] bounds = [1, 1024] -> [1, 1024]>

// CHECK-LABEL: test2 transform_ex4
// CHECK-NEXT: #rock.transform_map<affine_map<() -> (0)> by [<ConstDim{0, 1} [] at [] -> ["B"] at [0]>] bounds = [] -> [1024]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> (d0)> by [<PassThrough ["B"] at [0] -> ["B"] at [0]>] bounds = [1024] -> [1024]>

func.func @transform_ex4(%arg0: memref<1x64x1024xf32>) -> memref<1x64xf32>
  attributes {
    remove_dims_by_names = {
      test0 = ["A"],
      test1 = ["C"],
      test2 = ["A", "C"]
    }
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d2) -> (d0, 0, d2)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <ConstDim{0, 1} [] at [] -> ["B"] at [1]>, <PassThrough ["C"] at [1] -> ["C"] at [2]>] bounds = [1, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x64xf32>
  return %1 : memref<1x64xf32>
}

// CHECK-LABEL: test0 transform_ex5
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<PassThrough ["A", "B", "C"] at [0, 1, 2] -> ["A", "B", "C"] at [0, 1, 2]>] bounds = [1, 1024, 64] -> [1, 1024, 64]>
// CHECK-NEXT: #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]>

// CHECK-LABEL: test1 transform_ex5
// CHECK-NEXT: #rock.transform_map<affine_map<(d0) -> ()> by [<AddDim{8} ["X"] at [0] -> [] at []>] bounds = [8] -> []>

func.func @transform_ex5(%arg0: memref<1x64x1024xf32>) -> memref<1x8x1024x64xf32>
  attributes {
    remove_dims_by_names = {
      test0 = ["X"],
      test1 = ["A", "B", "C"]
    }
  } {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["A"] at [0] -> ["A"] at [0]>, <PassThrough ["B", "C"] at [1, 2] -> ["B", "C"] at [2, 1]>] bounds = [1, 1024, 64] -> [1, 64, 1024]> : memref<1x64x1024xf32> to memref<1x1024x64xf32>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)> by [<AddDim{8} ["X"] at [1] -> [] at []>, <PassThrough ["A", "B", "C"] at [0, 2, 3] -> ["A", "B", "C"] at [0, 1, 2]>] bounds = [1, 8, 1024, 64] -> [1, 1024, 64]> : memref<1x1024x64xf32> to memref<1x8x1024x64xf32>
  return %1 : memref<1x8x1024x64xf32>
}
