// RUN: rocmlir-gen -emit-module-fusibility-for=v3:16,128,8,16,16,4,5,1,2,1,1 - < %s | FileCheck %s
// CHECK: fusible:0  
module {  
  func.func @mlir_conv_bwd_data_add_relu(%arg0: memref<1x64x3x7x7xf32>, %arg1: memref<256x1x3x230x230xf32>, %arg2: memref<256x1x64x112x112xf32>, %arg3: memref<64x1x1x1xf32>, %arg4: memref<256x1x64x112x112xf32>) attributes {enable_splitk_for_tuning, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-"} {  
    %cst = arith.constant 0.000000e+00 : f32  
    %0 = rock.transform %arg3 by <affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)> by [<PassThrough ["dim3", "dim0", "dim1", "dim2"] at [0, 1, 2, 3] -> ["dim3", "dim0", "dim1", "dim2"] at [3, 0, 1, 2]>] bounds = [1, 64, 1, 1] -> [64, 1, 1, 1]> : memref<64x1x1x1xf32> to memref<1x64x1x1xf32>  
    %1 = rock.transform %0 by <affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [1]>, <Broadcast{112} ["dim2"] at [2] -> ["dim2"] at [2]>, <Broadcast{112} ["dim3"] at [3] -> ["dim3"] at [3]>] bounds = [1, 64, 112, 112] -> [1, 64, 1, 1]> : memref<1x64x1x1xf32> to memref<1x64x112x112xf32>  
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x1x64x112x112xf32>  
    %2 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 3 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 3} ["g", "c"] at [1, 2] -> ["c"] at [1]>] bounds = [256, 1, 3, 230, 230] -> [256, 3, 230, 230]> : memref<256x1x3x230x230xf32> to memref<256x1x3x230x230xf32>  
    %3 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0 * 64 + d1, d2, d3, d4)> by [<PassThrough ["c", "y", "x"] at [2, 3, 4] -> ["c", "y", "x"] at [1, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [0, 1] -> ["k"] at [0]>] bounds = [1, 64, 3, 7, 7] -> [64, 3, 7, 7]> : memref<1x64x3x7x7xf32> to memref<1x64x3x7x7xf32>  
    %4 = rock.transform %alloc by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d2, d3, d4)> by [<PassThrough ["n", "h", "w"] at [0, 3, 4] -> ["n", "h", "w"] at [0, 2, 3]>, <Unmerge{1, 64} ["g", "k"] at [1, 2] -> ["k"] at [1]>] bounds = [256, 1, 64, 112, 112] -> [256, 64, 112, 112]> : memref<256x1x64x112x112xf32> to memref<256x1x64x112x112xf32>  
    rock.conv_bwd_data(%3, %2, %4) features = mfma|dot|atomic_add|atomic_add_f16 {  
      arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-",   
      kernelId = 1 : index,  
      dilations = [1 : index, 1 : index],   
      filter_layout = ["g", "k", "c", "y", "x"],   
      input_layout = ["ni", "gi", "ci", "hi", "wi"],   
      output_layout = ["no", "go", "ko", "ho", "wo"],   
      padding = [0 : index, 0 : index, 0 : index, 0 : index],   
      strides = [2 : index, 2 : index]  
    } : memref<1x64x3x7x7xf32>, memref<256x1x3x230x230xf32>, memref<256x1x64x112x112xf32>  
    %5 = rock.transform %alloc by <affine_map<(d0, d1, d2) -> (d0, 0, d1, d2, 0)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <PassThrough ["dim2"] at [1] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [2] -> ["dim3"] at [3]>, <AddDim{1} ["dim1"] at [3] -> [] at []>, <AddDim{64} ["dim4"] at [4] -> [] at []>] bounds = [256, 112, 112] -> [256, 1, 64, 112, 112]> : memref<256x1x64x112x112xf32> to memref<256x112x112xf32>  
    %6 = rock.transform %1 by <affine_map<(d0, d1, d2) -> (0, d0, d1, d2)> by [<PassThrough ["dim1"] at [0] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [1] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [2] -> ["dim3"] at [3]>, <AddDim{256} ["dim0"] at [3] -> [] at []>] bounds = [64, 112, 112] -> [1, 64, 112, 112]> : memref<1x64x112x112xf32> to memref<64x112x112xf32>  
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<256x64x112x112xf32>  
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%5, %6 : memref<256x112x112xf32>, memref<64x112x112xf32>) outs(%alloc_0 : memref<256x64x112x112xf32>) {  
    ^bb0(%in: f32, %in_1: f32, %out: f32):  
      %8 = arith.addf %in, %in_1 : f32  
      %9 = arith.maximumf %8, %cst : f32  
      linalg.yield %9 : f32  
    }  
    %7 = rock.transform %alloc_0 by <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 64 + d2, d3, d4)> by [<PassThrough ["dim0"] at [0] -> ["dim0"] at [0]>, <Unmerge{1, 64} ["exp0", "exp1"] at [1, 2] -> ["dim1"] at [1]>, <PassThrough ["dim2"] at [3] -> ["dim2"] at [2]>, <PassThrough ["dim3"] at [4] -> ["dim3"] at [3]>] bounds = [256, 1, 64, 112, 112] -> [256, 64, 112, 112]> : memref<256x64x112x112xf32> to memref<256x1x64x112x112xf32>  
    memref.copy %7, %arg4 : memref<256x1x64x112x112xf32> to memref<256x1x64x112x112xf32>  
    return  
  }  
}