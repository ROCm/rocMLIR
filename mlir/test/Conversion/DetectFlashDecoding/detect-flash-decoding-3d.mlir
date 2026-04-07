// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --rock-detect-flash-decoding --verify-diagnostics -o -| FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> (d2 * 256 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> ((d1 * 128 + d2) * 256 + d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> ((d1 * 2 + d2) * 128 + d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
#map9 = affine_map<(d0) -> (0, d0 floordiv 256, d0 mod 256, 0)>
#map10 = affine_map<(d0) -> (d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 256} ["exp2", "exp3"] at [2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit1"] at [1] -> [] at []>] bounds = [1, 1, 256, 256] -> [65536]>
#transform_map1 = #rock.transform_map<#map1 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Broadcast{1} ["dim1"] at [1] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 2, 256, 256] -> [1, 1, 256, 256]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{2, 128, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 2, 128, 256] -> [65536]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 3, 2]>] bounds = [1, 2, 256, 128] -> [1, 2, 128, 256]>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{256, 2, 128} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 2, 128] -> [65536]>
#transform_map5 = #rock.transform_map<#map5 by [<PassThrough ["dim0", "dim2", "dim3", "dim1"] at [0, 1, 2, 3] -> ["dim0", "dim2", "dim3", "dim1"] at [0, 2, 3, 1]>] bounds = [1, 2, 128, 256] -> [1, 256, 2, 128]>
#transform_map6 = #rock.transform_map<#map6 by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [2, 256, 256] -> [1, 2, 256, 256]>
#transform_map7 = #rock.transform_map<#map6 by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [2, 256, 128] -> [1, 2, 256, 128]>
#transform_map8 = #rock.transform_map<#map6 by [<Merge{1, 2} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [3]>] bounds = [2, 128, 256] -> [1, 2, 128, 256]>
#transform_map9 = #rock.transform_map<#map7 by [<Unmerge{2} ["exp1"] at [1] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 2, 256, 128] -> [2, 256, 128]>
#transform_map10 = #rock.transform_map<#map8 by [<Unmerge{2} ["exp1"] at [1] -> ["dim0"] at [0]>, <Unmerge{256} ["exp2"] at [2] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit3"] at [3] -> [] at []>] bounds = [1, 2, 256, 1] -> [2, 256]>
#transform_map11 = #rock.transform_map<#map9 by [<Merge{1, 2, 256, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [512] -> [1, 2, 256, 1]>
#transform_map12 = #rock.transform_map<#map10 by [<Merge{2, 256, 256} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [131072] -> [2, 256, 256]>
module {
  func.func @mlir_attention(%arg0: tensor<65536xf16>, %arg1: tensor<65536xf16>, %arg2: tensor<65536xf16>) -> (tensor<131072xf16>, tensor<512xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = rock.transform %arg0 by #transform_map : tensor<65536xf16> to tensor<1x1x256x256xf16>
    %1 = rock.transform %0 by #transform_map1 : tensor<1x1x256x256xf16> to tensor<1x2x256x256xf16>
    %2 = rock.transform %arg1 by #transform_map2 : tensor<65536xf16> to tensor<1x2x128x256xf16>
    %3 = rock.transform %2 by #transform_map3 : tensor<1x2x128x256xf16> to tensor<1x2x256x128xf16>
    %4 = rock.transform %arg2 by #transform_map4 : tensor<65536xf16> to tensor<1x256x2x128xf16>
    %5 = rock.transform %4 by #transform_map5 : tensor<1x256x2x128xf16> to tensor<1x2x128x256xf16>
    %6 = rock.transform %1 by #transform_map6 : tensor<1x2x256x256xf16> to tensor<2x256x256xf16>
    %7 = rock.transform %3 by #transform_map7 : tensor<1x2x256x128xf16> to tensor<2x256x128xf16>
    %8 = rock.transform %5 by #transform_map8 : tensor<1x2x128x256xf16> to tensor<2x128x256xf16>
    %9 = bufferization.alloc_tensor() : tensor<2x256x256xf16>
    %10 = bufferization.alloc_tensor() : tensor<2x256xf32>

    // CHECK: rock.attention{
    // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<1x256x256xf16>, tensor<1x256x256xf16>
    // CHECK-NEXT: lse = %{{.*}} : tensor<2x256xf32>
    // CHECK: splitKV = 2

    %result, %lseOut = rock.attention{
     qk = %6 * %7 : tensor<2x256x256xf16>, tensor<2x256x128xf16>
     lse = %10 : tensor<2x256xf32>
     qk = elementwise {
    ^bb0(%arg3: memref<2x256x128xf16>, %arg4: memref<1x2x256x128xf32>):
      %14 = bufferization.to_tensor %arg3 restrict : memref<2x256x128xf16> to tensor<2x256x128xf16>
      %15 = rock.transform %14 by #transform_map9 : tensor<2x256x128xf16> to tensor<1x2x256x128xf16>
      %16 = tosa.cast %15 : (tensor<1x2x256x128xf16>) -> tensor<1x2x256x128xf32>
      %17 = bufferization.to_buffer %16 : tensor<1x2x256x128xf32> to memref<1x2x256x128xf32>
      memref.copy %17, %arg4 : memref<1x2x256x128xf32> to memref<1x2x256x128xf32>
      rock.yield
    }
     %9 = softmax(qk) * %8 : tensor<2x128x256xf16> -> tensor<2x256x256xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<2x256x256xf16>, tensor<2x256xf32>
    %11 = rock.transform %lseOut by #transform_map10 : tensor<2x256xf32> to tensor<1x2x256x1xf32>
    %12 = rock.transform %11 by #transform_map11 : tensor<1x2x256x1xf32> to tensor<512xf32>
    %13 = rock.transform %result by #transform_map12 : tensor<2x256x256xf16> to tensor<131072xf16>
    return %13, %12 : tensor<131072xf16>, tensor<512xf32>
  }
}

