// RUN: rocmlir-gen --emit-tuning-key %s | FileCheck %s

// CHECK: gfx942  120     -t f32 -transA false -transB false -transC false -transO false -g 1 -m 128 -n 256 -k 64 -gemmO 128 -fusion_reduce count=1 sum:axis2


#map = affine_map<(d0, d1, d2) -> (d1 * 128 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#map4 = affine_map<(d0) -> (0, d0, 0)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 128} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 256, 128] -> [32768]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 256} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 64, 256] -> [16384]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{128, 64} ["exp1", "exp2"] at [1, 2] -> ["dim0"] at [0]>, <AddDim{1} ["unit0"] at [0] -> [] at []>] bounds = [1, 128, 64] -> [8192]>
#transform_map3 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [128, 1, 64] -> [1, 128, 64]>
#transform_map4 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [1, 128, 64] -> [128, 1, 64]>
#transform_map5 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [64, 1, 256] -> [1, 64, 256]>
#transform_map6 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [1, 64, 256] -> [64, 1, 256]>
#transform_map7 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [256, 1, 128] -> [1, 256, 128]>
#transform_map8 = #rock.transform_map<#map3 by [<PassThrough ["dim1", "dim0", "dim2"] at [0, 1, 2] -> ["dim1", "dim0", "dim2"] at [1, 0, 2]>] bounds = [1, 256, 128] -> [256, 1, 128]>
#transform_map9 = #rock.transform_map<#map4 by [<Merge{1, 128, 1} ["dim0"] at [0] -> ["col0", "col1", "col2"] at [0, 1, 2]>] bounds = [128] -> [1, 128, 1]>
module {
  func.func private @gemm_gemm_reduce_sum_axis2(%arg0: memref<8192xf32> {mhal.read_access}, %arg1: memref<16384xf32> {mhal.read_access}, %arg2: memref<32768xf32> {mhal.read_access}, %arg3: memref<128xf32> {mhal.read_access, mhal.write_access, rock.prefill = 0.000000e+00 : f32}) attributes {arch = "gfx942", kernel, num_cu = 120 : i64} {
    %0 = rock.transform %arg2 by #transform_map : memref<32768xf32> to memref<1x256x128xf32>
    %1 = rock.transform %arg1 by #transform_map1 : memref<16384xf32> to memref<1x64x256xf32>
    %2 = rock.transform %arg0 by #transform_map2 : memref<8192xf32> to memref<1x128x64xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x128x128xf32>
    %3 = rock.transform %2 by #transform_map3 : memref<1x128x64xf32> to memref<128x1x64xf32>
    %4 = rock.transform %3 by #transform_map4 : memref<128x1x64xf32> to memref<1x128x64xf32>
    %5 = rock.transform %1 by #transform_map5 : memref<1x64x256xf32> to memref<64x1x256xf32>
    %6 = rock.transform %5 by #transform_map6 : memref<64x1x256xf32> to memref<1x64x256xf32>
    %7 = rock.transform %0 by #transform_map7 : memref<1x256x128xf32> to memref<256x1x128xf32>
    %8 = rock.transform %7 by #transform_map8 : memref<256x1x128xf32> to memref<1x256x128xf32>
    rock.gemm_elementwise_gemm{
     ab = %2 * %1 : memref<1x128x64xf32>, memref<1x64x256xf32>
     ab = elementwise {
    ^bb0(%arg4: memref<1x128x256xf32>, %arg5: memref<1x128x256xf32>):
      memref.copy %arg4, %arg5 : memref<1x128x256xf32> to memref<1x128x256xf32>
      rock.yield
    }
     %alloc = ab * %0 : memref<1x256x128xf32> -> memref<1x128x128xf32>
    } {firstGemmIndices = array<i64: 0>, storeMethod = #rock<StoreMethod set>}
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<1x128x1xf32>
    rock.reduce  sum %alloc into %alloc_0 {axis = 2 : index, blockSize = 256 : i32, gridSize = 64 : i32} : memref<1x128x128xf32> into memref<1x128x1xf32>
    %9 = rock.transform %alloc_0 by #transform_map9 : memref<1x128x1xf32> to memref<128xf32>
    memref.copy %9, %arg3 : memref<128xf32> to memref<128xf32>
    return
  }
}

