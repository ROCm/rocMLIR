// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --rock-detect-flash-decoding -o - | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 6 + d1) * 2 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 2 + d1) * 2 + d2) * 2 + d3) * 2 + d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d0 floordiv 4, (d0 mod 4) floordiv 2, d0 mod 2, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> ((d0 * 2 + d1) * 2 + d2)>
#map7 = affine_map<(d0, d1, d2) -> (d0)>
#map8 = affine_map<(d0, d1, d2) -> (d0, 0, 0)>
#map9 = affine_map<(d0) -> (d0 floordiv 4, (d0 mod 4) floordiv 2, d0 mod 2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 2 + d1) * 2 + d2, d3, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 2 + d1) * 2 + d2, d4)>
#map12 = affine_map<(d0) -> (d0 floordiv 4, (d0 mod 4) floordiv 2, d0 mod 2, 0, 0)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>
#map14 = affine_map<(d0) -> (d0 floordiv 8, (d0 mod 8) floordiv 4, 0, (d0 mod 4) floordiv 2, d0 mod 2)>
#transform_map = #rock.transform_map<#map by [<Unmerge{2, 6, 2} ["exp0", "exp1", "exp4"] at [0, 1, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit2"] at [2] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [2, 6, 1, 1, 2] -> [24]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [2, 6, 2, 1, 2] -> [2, 6, 1, 1, 2]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{2, 2, 2, 2, 2} ["exp0", "exp1", "exp2", "exp3", "exp4"] at [0, 1, 2, 3, 4] -> ["dim0"] at [0]>] bounds = [2, 2, 2, 2, 2] -> [32]>
#transform_map3 = #rock.transform_map<#map3 by [<Slice{0, 2, 0, 2, 0, 2, 0, 1, 0, 2} ["dim0_sliced", "dim1_sliced", "dim2_sliced", "dim3_sliced", "dim4_sliced"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4]>] bounds = [2, 2, 2, 1, 2] -> [2, 6, 2, 1, 2]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [2, 2, 2, 2, 2] -> [2, 2, 2, 2, 2]>
#transform_map5 = #rock.transform_map<#map5 by [<Merge{2, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [8, 1, 2] -> [2, 2, 2, 1, 2]>
#transform_map6 = #rock.transform_map<#map5 by [<Merge{2, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [8, 2, 2] -> [2, 2, 2, 2, 2]>
#transform_map7 = #rock.transform_map<#map6 by [<Unmerge{8, 2, 2} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [8, 2, 2] -> [32]>
#transform_map8 = #rock.transform_map<#map7 by [<Unmerge{2} ["exp0"] at [0] -> ["dim0"] at [0]>, <AddDim{1} ["unit1"] at [1] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [2, 1, 1] -> [2]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>] bounds = [2, 2, 2] -> [2, 1, 1]>
#transform_map10 = #rock.transform_map<#map9 by [<Merge{2, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [8] -> [2, 2, 2]>
#transform_map11 = #rock.transform_map<#map10 by [<Unmerge{2, 2, 2} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>] bounds = [2, 2, 2, 1, 2] -> [8, 1, 2]>
#transform_map12 = #rock.transform_map<#map11 by [<Unmerge{2, 2, 2} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>, <Unmerge{1} ["exp4"] at [4] -> ["dim1"] at [1]>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [2, 2, 2, 1, 1] -> [8, 1]>
#transform_map13 = #rock.transform_map<#map12 by [<Merge{2, 2, 2, 1, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [8] -> [2, 2, 2, 1, 1]>
#transform_map14 = #rock.transform_map<#map13 by [<PassThrough ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim2", "dim3", "dim1", "dim4"] at [0, 2, 3, 1, 4]>] bounds = [2, 2, 1, 2, 2] -> [2, 2, 2, 1, 2]>
#transform_map15 = #rock.transform_map<#map14 by [<Merge{2, 2, 1, 2, 2} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [16] -> [2, 2, 1, 2, 2]>
module {
  func.func @mlir_attention(%arg0: tensor<24xf16>, %arg1: tensor<32xf16>, %arg2: tensor<2xi32>, %arg3: tensor<32xf16>) -> (tensor<16xf16>, tensor<8xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = "tosa.const"() <{values = dense<5.000000e-01> : tensor<2x2x2x1x2xf16>}> : () -> tensor<2x2x2x1x2xf16>
    %1 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = rock.transform %arg0 by #transform_map : tensor<24xf16> to tensor<2x6x1x1x2xf16>
    %3 = rock.transform %2 by #transform_map1 : tensor<2x6x1x1x2xf16> to tensor<2x6x2x1x2xf16>
    %4 = rock.transform %arg1 by #transform_map2 : tensor<32xf16> to tensor<2x2x2x2x2xf16>
    %5 = rock.transform %3 by #transform_map3 : tensor<2x6x2x1x2xf16> to tensor<2x2x2x1x2xf16>
    %6 = rock.transform %4 by #transform_map4 : tensor<2x2x2x2x2xf16> to tensor<2x2x2x2x2xf16>
    %7 = rock.transform %5 by #transform_map5 : tensor<2x2x2x1x2xf16> to tensor<8x1x2xf16>
    %8 = rock.transform %6 by #transform_map6 : tensor<2x2x2x2x2xf16> to tensor<8x2x2xf16>
    %9 = rock.transform %arg3 by #transform_map7 : tensor<32xf16> to tensor<8x2x2xf16>
    %10 = bufferization.alloc_tensor() : tensor<8x1x2xf16>
    %11 = bufferization.alloc_tensor() : tensor<8x1xf32>
    %12 = rock.transform %arg2 by #transform_map8 : tensor<2xi32> to tensor<2x1x1xi32>
    %13 = rock.transform %12 by #transform_map9 : tensor<2x1x1xi32> to tensor<2x2x2xi32>
    %14 = rock.transform %13 by #transform_map10 : tensor<2x2x2xi32> to tensor<8xi32>

    // CHECK: rock.attention{
    // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<4x1x2xf16>, tensor<4x2x4xf16>
    // CHECK-NEXT: currentSeqLen = (%{{.*}} : tensor<4xi32>)
    // CHECK-NEXT: lse = %11 : tensor<8x1xf32>
    // CHECK: softmax(qk) * %{{.*}} : tensor<4x4x2xf16> -> tensor<8x1x2xf16>
    // CHECK: splitKV = 2

    %result, %lseOut = rock.attention{
     qk = %7 * %8 : tensor<8x1x2xf16>, tensor<8x2x2xf16>
     currentSeqLen = (%14 : tensor<8xi32>)
     lse = %11 : tensor<8x1xf32>
     qk = elementwise {
    ^bb0(%arg4: memref<8x1x2xf16>, %arg5: memref<2x2x2x1x2xf16>):
      %20 = bufferization.to_tensor %arg4 restrict : memref<8x1x2xf16> to tensor<8x1x2xf16>
      %21 = rock.transform %20 by #transform_map11 : tensor<8x1x2xf16> to tensor<2x2x2x1x2xf16>
      %22 = tosa.mul %21, %0, %1 : (tensor<2x2x2x1x2xf16>, tensor<2x2x2x1x2xf16>, tensor<1xi8>) -> tensor<2x2x2x1x2xf16>
      %23 = bufferization.to_buffer %22 : tensor<2x2x2x1x2xf16> to memref<2x2x2x1x2xf16>
      memref.copy %23, %arg5 : memref<2x2x2x1x2xf16> to memref<2x2x2x1x2xf16>
      rock.yield
    }
     %10 = softmax(qk) * %9 : tensor<8x2x2xf16> -> tensor<8x1x2xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<8x1x2xf16>, tensor<8x1xf32>
    %15 = rock.transform %lseOut by #transform_map12 : tensor<8x1xf32> to tensor<2x2x2x1x1xf32>
    %16 = rock.transform %15 by #transform_map13 : tensor<2x2x2x1x1xf32> to tensor<8xf32>
    %17 = rock.transform %result by #transform_map11 : tensor<8x1x2xf16> to tensor<2x2x2x1x2xf16>
    %18 = rock.transform %17 by #transform_map14 : tensor<2x2x2x1x2xf16> to tensor<2x2x1x2x2xf16>
    %19 = rock.transform %18 by #transform_map15 : tensor<2x2x1x2x2xf16> to tensor<16xf16>
    return %19, %16 : tensor<16xf16>, tensor<8xf32>
  }
}

