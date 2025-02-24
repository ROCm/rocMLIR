// RUN: rocmlir-opt --rock-sort-dimensions-memory-layout %s -verify-diagnostics -o -| FileCheck %s

// CHECK-LABEL: test_conv
func.func @test_conv(%arg0: memref<2304xf16>, %arg1: memref<1638400xf16>, %arg2: memref<16xf16>, %arg3: memref<819200xf16>) attributes {kernel, arch = ""} {
  %cst = arith.constant 1.000000e+00 : f16
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)> by [<Unmerge{16, 1, 1, 1} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [16, 1, 1, 1] -> [16]> : memref<16xf16> to memref<16x1x1x1xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim3", "dim0", "dim1", "dim2"] at [3, 0, 1, 2]>] bounds = [1, 16, 1, 1] -> [16, 1, 1, 1]> : memref<16x1x1x1xf16> to memref<1x16x1x1xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{1} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [2, 16, 160, 160] -> [1, 16, 1, 1]> : memref<1x16x1x1xf16> to memref<2x16x160x160xf16>
  %3 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3) -> (((d0 * 3 + d1) * 3 + d2) * 16 + d3)> by [<Unmerge{16, 3, 3, 16} ["exp0", "exp1", "exp2", "exp3"] at [0, 1, 2, 3] -> ["dim0"] at [0]>] bounds = [16, 3, 3, 16] -> [2304]> : memref<2304xf16> to memref<16x3x3x16xf16>
  %4 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> ((((((d0 * 2 + d1) * 5 + d2) * 5 + d3) * 32 + d4) * 32 + d5) * 16 + d6)> by [<Unmerge{2, 2, 5, 5, 32, 32, 16} ["exp0", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0"] at [0]>] bounds = [2, 2, 5, 5, 32, 32, 16] -> [1638400]> : memref<1638400xf16> to memref<2x2x5x5x32x32x16xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d4, d6, d3, d5, d2)> by [<PassThrough ["dim0", "dim1", "dim6", "dim4", "dim2", "dim5", "dim3"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0", "dim1", "dim6", "dim4", "dim2", "dim5", "dim3"] at [0, 1, 6, 4, 2, 5, 3]>] bounds = [2, 2, 16, 32, 5, 32, 5] -> [2, 2, 5, 5, 32, 32, 16]> : memref<2x2x5x5x32x32x16xf16> to memref<2x2x16x32x5x32x5xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + 1, d2, d3, d4, d5, d6)> by [<Slice{0, 2, 1, 2, 0, 16, 0, 32, 0, 5, 0, 32, 0, 5} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced", "dim5_sliced", "dim6_sliced"] at [0, 1, 2, 3, 4, 5, 6] -> ["dim0", "dim1", "dim2", "dim3", "dim4", "dim5", "dim6"] at [0, 1, 2, 3, 4, 5, 6]>] bounds = [2, 1, 16, 32, 5, 32, 5] -> [2, 2, 16, 32, 5, 32, 5]> : memref<2x2x16x32x5x32x5xf16> to memref<2x1x16x32x5x32x5xf16>
  %7 = rock.transform %6 by <affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2 floordiv 5, d2 mod 5, d3 floordiv 5, d3 mod 5)> by [<Merge{2, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <Merge{32, 5} ["dim2"] at [2] -> ["col3", "col4"] at [3, 4]>, <Merge{32, 5} ["dim3"] at [3] -> ["col5", "col6"] at [5, 6]>] bounds = [2, 16, 160, 160] -> [2, 1, 16, 32, 5, 32, 5]> : memref<2x1x16x32x5x32x5xf16> to memref<2x16x160x160xf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x16x160x160xf16>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 16} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [2, 1, 16, 160, 160] -> [2, 16, 160, 160]> : memref<2x16x160x160xf16> to memref<2x1x16x160x160xf16>
  %9 = rock.transform %3 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 16 + d1, d2, d3, d4)> by [<PassThrough ["y", "x", "c"] at [2, 3, 4] -> ["y", "x", "c"] at [1, 2, 3]>, <Unmerge{1, 16} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 16, 3, 3, 16] -> [16, 3, 3, 16]> : memref<16x3x3x16xf16> to memref<1x16x3x3x16xf16>
  %10 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 16 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 16} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [2, 1, 16, 160, 160] -> [2, 16, 160, 160]> : memref<2x16x160x160xf16> to memref<2x1x16x160x160xf16>

  // CHECK: %[[b:.*]] = rock.transform %{{.*}} memref<2x1x16x160x160xf16> to memref<2x160x1x160x16xf16>
  // CHECK: rock.conv(%{{.*}}, %[[b]], %{{.*}})
  rock.conv(%9, %8, %10) features =  dot|atomic_add|atomic_fmax_f32|atomic_add_f16|wmma {arch = "gfx1200", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "y", "x", "c"], input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [1 : index, 1 : index]} : memref<1x16x3x3x16xf16>, memref<2x1x16x160x160xf16>, memref<2x1x16x160x160xf16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x16x160x160xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloc, %2 : memref<2x16x160x160xf16>, memref<2x16x160x160xf16>) outs(%alloc_0 : memref<2x16x160x160xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %13 = arith.addf %in, %in_1 : f16
    %14 = arith.negf %13 : f16
    %15 = math.exp %14 : f16
    %16 = arith.addf %15, %cst : f16
    %17 = arith.divf %cst, %16 : f16
    %18 = arith.mulf %13, %17 : f16
    linalg.yield %18 : f16
  }
  %11 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)> by [<PassThrough ["dim0", "dim2", "dim3", "dim1"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim3", "dim1"] at [0, 2, 3, 1]>] bounds = [2, 160, 160, 16] -> [2, 16, 160, 160]> : memref<2x16x160x160xf16> to memref<2x160x160x16xf16>
  %12 = rock.transform %11 by <affine_map<(d0) -> (d0 floordiv 409600, (d0 mod 409600) floordiv 2560, (d0 mod 2560) floordiv 16, d0 mod 16)> by [<Merge{2, 160, 160, 16} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [819200] -> [2, 160, 160, 16]> : memref<2x160x160x16xf16> to memref<819200xf16>
  memref.copy %12, %arg3 : memref<819200xf16> to memref<819200xf16>
  return
}

// CHECK-LABEL: test_attention
func.func @test_attention(%arg0: memref<1024xf16>, %arg1: memref<1024xf16>, %arg2: memref<512xf16>, %arg3: memref<256xf16>) attributes {kernel, arch = ""} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2) -> ((d0 * 8 + d1) * 64 + d2)> by [<Unmerge{1, 8, 64} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 8, 64] -> [512]> : memref<512xf16> to memref<1x8x64xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 64, 8] -> [1, 8, 64]> : memref<1x8x64xf16> to memref<1x64x8xf16>
  %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 16 + d2)> by [<Unmerge{1, 64, 16} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 64, 16] -> [1024]> : memref<1024xf16> to memref<1x64x16xf16>
  %3 = rock.transform %2 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 16, 64] -> [1, 64, 16]> : memref<1x64x16xf16> to memref<1x16x64xf16>
  %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 32 + d1) * 32 + d2)> by [<Unmerge{1, 32, 32} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1, 32, 32] -> [1024]> : memref<1024xf16> to memref<1x32x32xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [1, 32, 32] -> [1, 32, 32]> : memref<1x32x32xf16> to memref<1x32x32xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<Slice{0, 1, 0, 32, 0, 16} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [1, 32, 16] -> [1, 32, 32]> : memref<1x32x32xf16> to memref<1x32x16xf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x8xf16>

  // CHECK: %[[a:.*]] = rock.transform %{{.*}} memref<1x32x16xf16> to memref<1x16x32xf16>
  // CHECK: %[[b:.*]] = rock.transform %{{.*}} memref<1x16x64xf16> to memref<1x64x16xf16>
  // CHECK: rock.attention
  // CHECK-NEXT: qk = tr %[[a]] * tr %[[b]]
  rock.attention{
    qk = %6 * %3 : memref<1x32x16xf16>, memref<1x16x64xf16>
    qk = elementwise {
  ^bb0(%arg4: memref<1x32x64xf16>, %arg5: memref<1x32x64xf16>):
    memref.copy %arg4, %arg5 : memref<1x32x64xf16> to memref<1x32x64xf16>
    rock.yield
  }
    %alloc = softmax(qk) * %1 : memref<1x64x8xf16> -> memref<1x32x8xf16>
  } {arch = "gfx1200", features = #rock<GemmFeatures dot|atomic_add|atomic_fmax_f32|atomic_add_f16|wmma>, firstGemmIdx = 0 : i32}
  %7 = rock.transform %alloc by <affine_map<(d0) -> (0, d0 floordiv 8, d0 mod 8)> by [<Merge{1, 32, 8} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [256] -> [1, 32, 8]> : memref<1x32x8xf16> to memref<256xf16>
  memref.copy %7, %arg3 : memref<256xf16> to memref<256xf16>
  return
}

// CHECK-LABEL: test_gemm
func.func @test_gemm(%arg0: memref<5242880xf16>, %arg1: memref<409600xf16>, %arg2: memref<2621440xf16>, %arg3: memref<5242880xf16>) attributes {kernel, arch = ""} {
  %0 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d1) * 64 + d2) * 10 + d3 + d4)> by [<Unmerge{64, 64, 64, 10, 1} ["exp0", "exp1", "exp2", "exp3", "exp4"] at [0, 1, 2, 3, 4] -> ["dim0"] at [0]>] bounds = [64, 64, 64, 10, 1] -> [2621440]> : memref<2621440xf16> to memref<64x64x64x10x1xf16>
  %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d1, d2, d0)> by [<PassThrough ["dim4", "dim2", "dim3", "dim0", "dim1"] at [0, 1, 2, 3, 4] -> ["dim4", "dim2", "dim3", "dim0", "dim1"] at [4, 2, 3, 0, 1]>] bounds = [1, 64, 10, 64, 64] -> [64, 64, 64, 10, 1]> : memref<64x64x64x10x1xf16> to memref<1x64x10x64x64xf16>
  %2 = rock.transform %1 by <affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, d3, d4)> by [<Broadcast{1} ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 64, 10, 64, 64] -> [1, 64, 10, 64, 64]> : memref<1x64x10x64x64xf16> to memref<2x64x10x64x64xf16>
  %3 = rock.transform %arg1 by <affine_map<(d0, d1, d2) -> ((d0 * 320 + d1) * 640 + d2)> by [<Unmerge{2, 320, 640} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 320, 640] -> [409600]> : memref<409600xf16> to memref<2x320x640xf16>
  %4 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> ((d0 * 4096 + d1) * 640 + d2)> by [<Unmerge{2, 4096, 640} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [2, 4096, 640] -> [5242880]> : memref<5242880xf16> to memref<2x4096x640xf16>
  %5 = rock.transform %4 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["dim0", "dim2", "dim1"] at [0, 1, 2] -> ["dim0", "dim2", "dim1"] at [0, 2, 1]>] bounds = [2, 640, 4096] -> [2, 4096, 640]> : memref<2x4096x640xf16> to memref<2x640x4096xf16>
  %6 = rock.transform %5 by <affine_map<(d0, d1, d2) -> (d0, d1, d2)> by [<Slice{0, 2, 0, 320, 0, 4096} ["dim0_sliced", "dim1_sliced", "dim2_sliced"] at [0, 1, 2] -> ["dim0", "dim1", "dim2"] at [0, 1, 2]>] bounds = [2, 320, 4096] -> [2, 640, 4096]> : memref<2x640x4096xf16> to memref<2x320x4096xf16>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4096x640xf16>

  // CHECK: %[[a:.*]] = rock.transform %{{.*}} memref<2x320x4096xf16> to memref<2x4096x320xf16>
  // CHECK: rock.gemm %{{.*}} = %[[a]] * %{{.*}}
  rock.gemm %alloc = tr %6 * %3 features =  dot|atomic_add|atomic_fmax_f32|atomic_add_f16|wmma storeMethod =  set {arch = "gfx1200"} : memref<2x4096x640xf16> = memref<2x320x4096xf16> * memref<2x320x640xf16>
  %7 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d2, d3 * 10 + d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{64, 64} ["exp1", "exp2"] at [1, 2] -> ["dim1"] at [1]>, <Unmerge{64, 10} ["exp3", "exp4"] at [3, 4] -> ["dim2"] at [2]>] bounds = [2, 64, 64, 64, 10] -> [2, 4096, 640]> : memref<2x4096x640xf16> to memref<2x64x64x64x10xf16>
  %8 = rock.transform %7 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)> by [<PassThrough ["dim0", "dim3", "dim4", "dim1", "dim2"] at [0, 1, 2, 3, 4] -> ["dim0", "dim3", "dim4", "dim1", "dim2"] at [0, 3, 4, 1, 2]>] bounds = [2, 64, 10, 64, 64] -> [2, 64, 64, 64, 10]> : memref<2x64x64x64x10xf16> to memref<2x64x10x64x64xf16>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<2x64x10x64x64xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%8, %2 : memref<2x64x10x64x64xf16>, memref<2x64x10x64x64xf16>) outs(%alloc_0 : memref<2x64x10x64x64xf16>) {
  ^bb0(%in: f16, %in_1: f16, %out: f16):
    %11 = arith.addf %in, %in_1 : f16
    linalg.yield %11 : f16
  }
  %9 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d1, d2)> by [<PassThrough ["dim0", "dim3", "dim4", "dim1", "dim2"] at [0, 1, 2, 3, 4] -> ["dim0", "dim3", "dim4", "dim1", "dim2"] at [0, 3, 4, 1, 2]>] bounds = [2, 64, 64, 64, 10] -> [2, 64, 10, 64, 64]> : memref<2x64x10x64x64xf16> to memref<2x64x64x64x10xf16>
  %10 = rock.transform %9 by <affine_map<(d0) -> (d0 floordiv 2621440, (d0 mod 2621440) floordiv 40960, (d0 mod 40960) floordiv 640, (d0 mod 640) floordiv 10, d0 mod 10)> by [<Merge{2, 64, 64, 64, 10} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [5242880] -> [2, 64, 64, 64, 10]> : memref<2x64x64x64x10xf16> to memref<5242880xf16>
  memref.copy %10, %arg3 : memref<5242880xf16> to memref<5242880xf16>
  return
}
