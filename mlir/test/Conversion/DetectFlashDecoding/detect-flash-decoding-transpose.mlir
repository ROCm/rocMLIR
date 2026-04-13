// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --rock-detect-flash-decoding -o - | FileCheck %s

#map = affine_map<(d0, d1, d2, d3) -> ((d1 * 256 + d2) * 256 + d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d1 * 256 + d3) * 256 + d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 256 + d2) * 2 + d3) * 128 + d4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2, d3)>
#map7 = affine_map<(d0, d1, d2) -> (0, d0 floordiv 2, d0 mod 2, d1, d2)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)>
#map9 = affine_map<(d0, d1, d2) -> ((d0 * 256 + d1) * 128 + d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 2 + d2, d3)>
#map12 = affine_map<(d0) -> (0, d0 floordiv 512, (d0 mod 512) floordiv 256, d0 mod 256, 0)>
#map13 = affine_map<(d0) -> (d0 floordiv 65536, (d0 mod 65536) floordiv 256, d0 mod 256)>
#transform_map = #rock.transform_map<#map by [<Unmerge{12, 256, 256} ["exp1", "exp2", "exp3"] at [1, 2, 3] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 12, 256, 256] -> [786432]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{12, 256, 256} ["exp1", "exp3", "exp4"] at [1, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit2"] at [2] -> [] at []>] bounds = [1, 12, 1, 256, 256] -> [786432]>
#transform_map2 = #rock.transform_map<#map2 by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{1} ["dim2"] at [2] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [3] -> ["dim3"] at [3]>, <PassThrough ["dim4"] at [4] -> ["dim4"] at [4]>] bounds = [1, 12, 2, 256, 256] -> [1, 12, 1, 256, 256]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim2", "dim4", "dim3"] at [0, 1, 2, 4, 3]>] bounds = [1, 12, 2, 256, 256] -> [1, 12, 2, 256, 256]>
#transform_map4 = #rock.transform_map<#map4 by [<PassThrough ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 2, 3] -> ["dim0", "dim1", "dim3", "dim2"] at [0, 1, 3, 2]>] bounds = [1, 12, 256, 256] -> [1, 12, 256, 256]>
#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{12, 256, 2, 128} ["exp1", "exp2", "exp3", "exp4"] at [1, 2, 3, 4] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 12, 256, 2, 128] -> [786432]>
#transform_map6 = #rock.transform_map<#map6 by [<PassThrough ["dim0", "dim1", "dim3", "dim4", "dim2"] at [0, 1, 2, 3, 4] -> ["dim0", "dim1", "dim3", "dim4", "dim2"] at [0, 1, 3, 4, 2]>] bounds = [1, 12, 2, 128, 256] -> [1, 12, 256, 2, 128]>
#transform_map7 = #rock.transform_map<#map7 by [<Merge{1, 12, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [24, 256, 256] -> [1, 12, 2, 256, 256]>
#transform_map8 = #rock.transform_map<#map8 by [<Merge{1, 12, 256, 256} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3"] at [0, 1, 2, 3]>] bounds = [786432] -> [1, 12, 256, 256]>
#transform_map9 = #rock.transform_map<#map9 by [<Unmerge{24, 256, 128} ["exp0", "exp1", "exp2"] at [0, 1, 2] -> ["dim0"] at [0]>] bounds = [24, 256, 128] -> [786432]>
#transform_map10 = #rock.transform_map<#map7 by [<Merge{1, 12, 2} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [3]>, <PassThrough ["dim2"] at [2] -> ["dim2"] at [4]>] bounds = [24, 128, 256] -> [1, 12, 2, 128, 256]>
#transform_map11 = #rock.transform_map<#map10 by [<Unmerge{12, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [3] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [4] -> ["dim2"] at [2]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 12, 2, 256, 128] -> [24, 256, 128]>
#transform_map12 = #rock.transform_map<#map11 by [<Unmerge{12, 2} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <Unmerge{256} ["exp3"] at [3] -> ["dim1"] at [1]>, <AddDim{1} ["unit0"] at [0] -> [] at []>, <AddDim{1} ["unit4"] at [4] -> [] at []>] bounds = [1, 12, 2, 256, 1] -> [24, 256]>
#transform_map13 = #rock.transform_map<#map12 by [<Merge{1, 12, 2, 256, 1} ["dim0"] at [0] -> ["col0", "col1", "col2", "col3", "col4"] at [0, 1, 2, 3, 4]>] bounds = [6144] -> [1, 12, 2, 256, 1]>
#transform_map14 = #rock.transform_map<#map13 by [<Merge{24, 256, 256} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [1572864] -> [24, 256, 256]>
module {
  func.func @mlir_attention(%arg0: tensor<786432xf16>, %arg1: tensor<786432xf16>, %arg2: tensor<786432xf16>) -> (tensor<1572864xf16>, tensor<6144xf32>) attributes {rock.arch = "##TOKEN_ARCH##", rock.kernel = "mixr"} {
    %0 = rock.transform %arg1 by #transform_map : tensor<786432xf16> to tensor<1x12x256x256xf16>
    %1 = rock.transform %arg0 by #transform_map1 : tensor<786432xf16> to tensor<1x12x1x256x256xf16>
    %2 = rock.transform %1 by #transform_map2 : tensor<1x12x1x256x256xf16> to tensor<1x12x2x256x256xf16>
    %3 = rock.transform %2 by #transform_map3 : tensor<1x12x2x256x256xf16> to tensor<1x12x2x256x256xf16>
    %4 = rock.transform %0 by #transform_map4 : tensor<1x12x256x256xf16> to tensor<1x12x256x256xf16>
    %5 = rock.transform %arg2 by #transform_map5 : tensor<786432xf16> to tensor<1x12x256x2x128xf16>
    %6 = rock.transform %5 by #transform_map6 : tensor<1x12x256x2x128xf16> to tensor<1x12x2x128x256xf16>
    %7 = rock.transform %3 by #transform_map7 : tensor<1x12x2x256x256xf16> to tensor<24x256x256xf16>
    %8 = rock.transform %4 by #transform_map8 : tensor<1x12x256x256xf16> to tensor<786432xf16>
    %9 = rock.transform %8 by #transform_map9 : tensor<786432xf16> to tensor<24x256x128xf16>
    %10 = rock.transform %6 by #transform_map10 : tensor<1x12x2x128x256xf16> to tensor<24x128x256xf16>
    %11 = bufferization.alloc_tensor() : tensor<24x256x256xf16>
    %12 = bufferization.alloc_tensor() : tensor<24x256xf32>

    // CHECK: rock.attention{
    // CHECK-NEXT: qk = %{{.*}} * %{{.*}} : tensor<12x256x256xf16>, tensor<12x256x256xf16>
    // CHECK-NEXT: lse = %{{.*}} : tensor<24x256xf32>
    // CHECK: softmax(qk) * %{{.*}} : tensor<12x256x256xf16> -> tensor<24x256x256xf16>
    // CHECK: splitKV = 2

    %result, %lseOut = rock.attention{
     qk = %7 * %9 : tensor<24x256x256xf16>, tensor<24x256x128xf16>
     lse = %12 : tensor<24x256xf32>
     qk = elementwise {
    ^bb0(%arg3: memref<24x256x128xf16>, %arg4: memref<1x12x2x256x128xf32>):
      %16 = bufferization.to_tensor %arg3 restrict : memref<24x256x128xf16> to tensor<24x256x128xf16>
      %17 = rock.transform %16 by #transform_map11 : tensor<24x256x128xf16> to tensor<1x12x2x256x128xf16>
      %18 = tosa.cast %17 : (tensor<1x12x2x256x128xf16>) -> tensor<1x12x2x256x128xf32>
      %19 = bufferization.to_buffer %18 : tensor<1x12x2x256x128xf32> to memref<1x12x2x256x128xf32>
      memref.copy %19, %arg4 : memref<1x12x2x256x128xf32> to memref<1x12x2x256x128xf32>
      rock.yield
    }
     %11 = softmax(qk) * %10 : tensor<24x128x256xf16> -> tensor<24x256x256xf16>
    } {firstGemmIndices = array<i64: 0>, numHeadsKV = 1 : i32, numHeadsQ = 1 : i32, softmaxType = f32, splitKV = 1 : i32, storeMethod = #rock<StoreMethod set>} -> tensor<24x256x256xf16>, tensor<24x256xf32>
    %13 = rock.transform %lseOut by #transform_map12 : tensor<24x256xf32> to tensor<1x12x2x256x1xf32>
    %14 = rock.transform %13 by #transform_map13 : tensor<1x12x2x256x1xf32> to tensor<6144xf32>
    %15 = rock.transform %result by #transform_map14 : tensor<24x256x256xf16> to tensor<1572864xf16>
    return %15, %14 : tensor<1572864xf16>, tensor<6144xf32>
  }
}


