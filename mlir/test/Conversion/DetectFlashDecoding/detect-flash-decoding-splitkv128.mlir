// RUN: rocmlir-opt --rock-detect-flash-decoding %s | FileCheck %s

// Test with splitKV=128 and different dimensions (8 heads, 128 query seq, 64 head dim)
// This tests the maximum supported splitKV value with smaller attention dimensions
// Q: [1024, 128, 64], K: [1024, 64, 1], V: [1024, 1, 64]

#map = affine_map<(d0, d1, d2, d3) -> ((d1 * 128 + d2) * 64 + d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 128 + d3) * 64 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 1 + d2) * 128 + d3) * 64 + d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 128, d0 mod 128, d1, d2)>
#map7 = affine_map<(d0) -> (0, d0 floordiv 8192, (d0 mod 8192) floordiv 64, d0 mod 64)>
#map8 = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 1 + d2)>
#map9 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 128, d1, d0 mod 128, d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 128 + d3, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 128 + d2, d3)>
#map14 = affine_map<(d0) -> (d0 floordiv 8192, (d0 mod 8192) floordiv 64, d0 mod 64)>
#map12 = affine_map<(d0) -> (0, d0 floordiv 16384, (d0 mod 16384) floordiv 128, d0 mod 128, 0)>
#map13 = affine_map<(d0) -> (d0 floordiv 8192, (d0 mod 8192) floordiv 64, d0 mod 64)>

#transform_map = #rock.transform_map<#map by [<Unmerge{8, 128, 64} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 8, 128, 64] -> [65536]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{8, 128, 64} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 8, 1, 128, 64] -> [65536]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 8, 128, 128, 64] -> [1, 8, 1, 128, 64]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 3, 2]>] bounds = [1, 8, 64, 128] -> [1, 8, 128, 64]>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{8, 1, 128, 64} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 8, 1, 128, 64] -> [65536]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim0", "dim1", "dim2", "dim3", "dim4"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim3", "dim2", "dim4"] at [0, 1, 3, 2, 4]>] bounds = [1, 8, 128, 1, 64] -> [1, 8, 1, 128, 64]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{1, 8, 128} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [1024, 128, 64] -> [1, 8, 128, 128, 64]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{1, 8, 64, 128} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [65536] -> [1, 8, 64, 128]>
#transform_map8 = #rock.transform_map<#map8 by [<Unmerge{1024, 64, 1} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [1024, 64, 1] -> [65536]>
#transform_map9 = #rock.transform_map<#map9 by [<Merge{1, 8, 128} ["dim0"] at [0] -> ["col0", "col1", "col3"] at [0, 1, 3]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [1024, 1, 64] -> [1, 8, 1, 128, 64]>
#transform_map10 = #rock.transform_map<#map10 by [<Unmerge{8, 128} ["exp1", "exp3"] at [1, 3] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 8, 1, 128, 64] -> [1024, 1, 64]>
#transform_map11 = #rock.transform_map<#map11 by [<Unmerge{8, 128} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <Unmerge{128} ["exp3"] at [3] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 8, 128, 128, 1] -> [1024, 128]>
#transform_map12 = #rock.transform_map<#map12 by [<Merge{1, 8, 128, 128, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [131072] -> [1, 8, 128, 128, 1]>
#transform_map13 = #rock.transform_map<#map14 by [<Merge{1024, 128, 64} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [8388608] -> [1024, 128, 64]>

module {
  func.func @flash_decode_splitkv128(%arg0: tensor<65536xf16>, %arg1: tensor<65536xf16>, %arg2: tensor<65536xf16>) -> (tensor<8388608xf16>, tensor<131072xf32>) attributes {arch = "gfx942", kernel = "mixr", num_cu = 304 : i64} {
    // Q tensor transforms
    %0 = rock.transform %arg0 by #transform_map1 : tensor<65536xf16> to tensor<1x8x1x128x64xf16>
    %1 = rock.transform %0 by #transform_map2 : tensor<1x8x1x128x64xf16> to tensor<1x8x128x128x64xf16>
    %2 = rock.transform %1 by #transform_map6 : tensor<1x8x128x128x64xf16> to tensor<1024x128x64xf16>
    
    // K tensor transforms (direct unmerge, no 5D intermediate)
    %3 = rock.transform %arg1 by #transform_map8 : tensor<65536xf16> to tensor<1024x64x1xf16>
    
    // V tensor transforms
    %4 = rock.transform %arg2 by #transform_map4 : tensor<65536xf16> to tensor<1x8x1x128x64xf16>
    %5 = rock.transform %4 by #transform_map5 : tensor<1x8x1x128x64xf16> to tensor<1x8x128x1x64xf16>
    %6 = rock.transform %5 by #transform_map9 : tensor<1x8x128x1x64xf16> to tensor<1024x1x64xf16>
    
    %10 = bufferization.alloc_tensor() : tensor<1024x128x64xf16>
    %11 = bufferization.alloc_tensor() : tensor<1024x128xf32>

    // CHECK: rock.attention{
    // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<8x128x64xf16>, tensor<8x64x128xf16>
    // CHECK-NEXT: lse = %{{.*}} : tensor<1024x128xf32>
    // CHECK: softmax(qk) * %{{.*}} : tensor<8x128x64xf16> -> tensor<1024x128x64xf16>
    // CHECK: splitKV = 128 : i32

    %result, %lseOut = rock.attention{
     qk = %2 * %3 : tensor<1024x128x64xf16>, tensor<1024x64x1xf16>
     lse = %11 : tensor<1024x128xf32>
     qk = elementwise {
    ^bb0(%arg3: memref<1024x1x64xf16>, %arg4: memref<1x8x1x128x64xf32>):
      %15 = bufferization.to_tensor %arg3 restrict : memref<1024x1x64xf16> to tensor<1024x1x64xf16>
      %16 = rock.transform %15 by #transform_map10 : tensor<1024x1x64xf16> to tensor<1x8x1x128x64xf16>
      %17 = tosa.cast %16 : (tensor<1x8x1x128x64xf16>) -> tensor<1x8x1x128x64xf32>
      %18 = bufferization.to_buffer %17 : tensor<1x8x1x128x64xf32> to memref<1x8x1x128x64xf32>
      memref.copy %18, %arg4 : memref<1x8x1x128x64xf32> to memref<1x8x1x128x64xf32>
      rock.yield
    }
     %10 = softmax(qk) * %6 : tensor<1024x1x64xf16> -> tensor<1024x128x64xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<1024x128x64xf16>, tensor<1024x128xf32>
    %12 = rock.transform %lseOut by #transform_map11 : tensor<1024x128xf32> to tensor<1x8x128x128x1xf32>
    %13 = rock.transform %12 by #transform_map12 : tensor<1x8x128x128x1xf32> to tensor<131072xf32>
    %14 = rock.transform %result by #transform_map13 : tensor<1024x128x64xf16> to tensor<8388608xf16>
    return %14, %13 : tensor<8388608xf16>, tensor<131072xf32>
  }
}

