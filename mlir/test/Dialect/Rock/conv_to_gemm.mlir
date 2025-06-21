// RUN: rocmlir-opt --rock-conv-to-gemm --mlir-print-local-scope --split-input-file %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 64 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2) * 64 + d4)>
#map2 = affine_map<(d0, d1, d2) -> (d1 * 256 + d2)>
#map4 = affine_map<(d0, d1, d2) -> (d1 * 3136 + d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (((d1 * 3 + d2) * 3 + d3) * 64 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 64} ["k", "c"] at [1, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["0"] at [2] -> [] at []>, <AddDim{1} ["1"] at [3] -> [] at []>] bounds = [1, 256, 1, 1, 64] -> [16384]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{64, 14, 14, 64} ["n", "0", "1", "c"] at [0, 1, 2, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [3] -> [] at []>] bounds = [64, 14, 14, 1, 64] -> [802816]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]>
#transform_map3 = #rock.transform_map<#map2 by [<Unmerge{3136, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 3136, 256] -> [802816]>
#transform_map4 = #rock.transform_map<#map2 by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]>
#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{256, 3, 3, 64} ["k", "0", "1", "c"] at [1, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 3, 3, 64] -> [147456]>
#transform_map6 = #rock.transform_map<#map2 by [<Unmerge{256, 256} ["m", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 256, 256] -> [65536]>
#transform_map7 = #rock.transform_map<#map2 by [<Unmerge{9216, 256} ["n", "gemmO"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 9216, 256] -> [2359296]>


// CHECK-LABEL: @nhwc_1x1
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <PassThrough ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <PassThrough ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK-NOT: Embed
// CHECK: rock.gemm
func.func @nhwc_1x1(%arg0: memref<16384xf16>, %arg1: memref<802816xf16>, %arg2: memref<3211264xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : memref<16384xf16> to memref<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : memref<3211264xf16> to memref<64x14x14x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, strides = [1 : index, 1 : index]} : memref<1x256x1x1x64xf16>, memref<64x14x14x1x64xf16>, memref<64x14x14x1x256xf16>
  return
}

// CHECK-LABEL: @nhwc_1x1_stride_2
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <Embed{2} ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <Embed{2} ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_1x1_stride_2(%arg0: memref<16384xf16>, %arg1: memref<802816xf16>, %arg2: memref<802816xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : memref<16384xf16> to memref<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 7 + d1) * 7 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 7, 7, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 7, 7, 1, 256] -> [802816]> : memref<802816xf16> to memref<64x7x7x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, strides = [2 : index, 2 : index]} : memref<1x256x1x1x64xf16>, memref<64x14x14x1x64xf16>, memref<64x7x7x1x256xf16>
  return
}

// CHECK-LABEL: @nhwc_3x3
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm
func.func @nhwc_3x3(%arg0: memref<147456xf16>, %arg1: memref<802816xf16>, %arg2: memref<2359296xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : memref<147456xf16> to memref<1x256x3x3x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 12 + d1) * 12 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 12, 12, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 12, 12, 1, 256] -> [2359296]> : memref<2359296xf16> to memref<64x12x12x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>, strides = [1 : index, 1 : index]} : memref<1x256x3x3x64xf16>, memref<64x14x14x1x64xf16>, memref<64x12x12x1x256xf16>
  return
}

// CHECK-LABEL: @conv_gemm_nhwc_1x1
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <PassThrough ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <PassThrough ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK-NOT: Embed
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_1x1(%arg0: memref<16384xf32>, %arg1: memref<802816xf32>, %arg2: memref<65536xf32>, %arg3: memref<3211264xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by #transform_map : memref<16384xf32> to memref<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by #transform_map1 : memref<802816xf32> to memref<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by #transform_map2 : memref<65536xf32> to memref<1x256x256xf32>
  %3 = rock.transform %arg3 by #transform_map3 : memref<3211264xf32> to memref<1x12544x256xf32>
  rock.conv_elementwise_gemm{
    ab = conv(%0, %1) : memref<1x256x1x1x64xf32>, memref<64x14x14x1x64xf32>
    ab = elementwise {
  ^bb0(%arg4: memref<1x256x12544xf32>, %arg5: memref<1x256x12544xf32>):
    memref.copy %arg4, %arg5 : memref<1x256x12544xf32> to memref<1x256x12544xf32>
    rock.yield
  }
    %3 = ab * %2 : memref<1x256x256xf32> -> memref<1x12544x256xf32>
  } {arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, filter_layout = ["g", "k", "0", "1", "c"], firstGemmIdx = 0 : i32, input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
  return
}

// CHECK-LABEL: @conv_gemm_nhwc_1x1_stride_2
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <Embed{2} ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <Embed{2} ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_1x1_stride_2(%arg0: memref<16384xf32>, %arg1: memref<802816xf32>, %arg2: memref<65536xf32>, %arg3: memref<802816xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by #transform_map : memref<16384xf32> to memref<1x256x1x1x64xf32>
  %1 = rock.transform %arg1 by #transform_map1 : memref<802816xf32> to memref<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by #transform_map4 : memref<65536xf32> to memref<1x256x256xf32>
  %3 = rock.transform %arg3 by #transform_map3 : memref<802816xf32> to memref<1x3136x256xf32>
  rock.conv_elementwise_gemm{
    ab = conv(%0, %1) : memref<1x256x1x1x64xf32>, memref<64x14x14x1x64xf32>
    ab = elementwise {
  ^bb0(%arg4: memref<1x256x3136xf32>, %arg5: memref<1x256x3136xf32>):
    memref.copy %arg4, %arg5 : memref<1x256x3136xf32> to memref<1x256x3136xf32>
    rock.yield
  }
    %3 = ab * %2 : memref<1x256x256xf32> -> memref<1x3136x256xf32>
  } {arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, filter_layout = ["g", "k", "0", "1", "c"], firstGemmIdx = 0 : i32, input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index]}
  return
}

// CHECK-LABEL: @conv_gemm_nhwc_3x3
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
// CHECK: rock.gemm_elementwise_gemm
func.func @conv_gemm_nhwc_3x3(%arg0: memref<147456xf32>, %arg1: memref<802816xf32>, %arg2: memref<65536xf32>, %arg3: memref<2359296xf32>) attributes {kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  %0 = rock.transform %arg0 by #transform_map5 : memref<147456xf32> to memref<1x256x3x3x64xf32>
  %1 = rock.transform %arg1 by #transform_map1 : memref<802816xf32> to memref<64x14x14x1x64xf32>
  %2 = rock.transform %arg2 by #transform_map6 : memref<65536xf32> to memref<1x256x256xf32>
  %3 = rock.transform %arg3 by #transform_map7 : memref<2359296xf32> to memref<1x9216x256xf32>
  rock.conv_elementwise_gemm{
    ab = conv(%0, %1) : memref<1x256x3x3x64xf32>, memref<64x14x14x1x64xf32>
    ab = elementwise {
  ^bb0(%arg4: memref<1x256x9216xf32>, %arg5: memref<1x256x9216xf32>):
    memref.copy %arg4, %arg5 : memref<1x256x9216xf32> to memref<1x256x9216xf32>
    rock.yield
  }
    %3 = ab * %2 : memref<1x256x256xf32> -> memref<1x9216x256xf32>
  } {arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", dilations = [1 : index, 1 : index], features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16>, filter_layout = ["g", "k", "0", "1", "c"], firstGemmIdx = 0 : i32, input_layout = ["ni", "0i", "1i", "gi", "ci"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
  return
}
