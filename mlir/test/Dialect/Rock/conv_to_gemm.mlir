// RUN: rocmlir-opt --rock-conv-to-gemm --mlir-print-local-scope --split-input-file %s | FileCheck %s

// CHECK-LABEL: @nhwc_1x1
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <PassThrough ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <PassThrough ["1o"] at [4] -> ["1ipad"] at [2]>
// CHECK-NOT: Embed
func.func @nhwc_1x1(%arg0: memref<16384xf16>, %arg1: memref<802816xf16>, %arg2: memref<3211264xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : memref<16384xf16> to memref<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 14, 14, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 256] -> [3211264]> : memref<3211264xf16> to memref<64x14x14x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, forceUnroll = true>, strides = [1 : index, 1 : index]} : memref<1x256x1x1x64xf16>, memref<64x14x14x1x64xf16>, memref<64x14x14x1x256xf16>
  return
}

// CHECK-LABEL: @nhwc_1x1_stride_2
// CHECK: <AddDim{1} ["0"] at [1] -> [] at []>, <Embed{2} ["0o"] at [2] -> ["0ipad"] at [1]>, <AddDim{1} ["1"] at [3] -> [] at []>, <Embed{2} ["1o"] at [4] -> ["1ipad"] at [2]>
func.func @nhwc_1x1_stride_2(%arg0: memref<16384xf16>, %arg1: memref<802816xf16>, %arg2: memref<802816xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 256 + d1 + d2 + d3) * 64 + d4)> by [<Unmerge{1, 256, 1, 1, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 1, 1, 64] -> [16384]> : memref<16384xf16> to memref<1x256x1x1x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 7 + d1) * 7 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 7, 7, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 7, 7, 1, 256] -> [802816]> : memref<802816xf16> to memref<64x7x7x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, forceUnroll = true>, strides = [2 : index, 2 : index]} : memref<1x256x1x1x64xf16>, memref<64x14x14x1x64xf16>, memref<64x7x7x1x256xf16>
  return
}

// CHECK-LABEL: @nhwc_3x3
// CHECK: <Embed{1, 1} ["0", "0o"] at [1, 2] -> ["0ipad"] at [1]>, <Embed{1, 1} ["1", "1o"] at [3, 4] -> ["1ipad"] at [2]>
func.func @nhwc_3x3(%arg0: memref<147456xf16>, %arg1: memref<802816xf16>, %arg2: memref<2359296xf16>) attributes {block_size = 128 : i32, enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "amdgcn-amd-amdhsa:gfx1100"} {
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2, d3, d4) -> ((((d0 * 256 + d1) * 3 + d2) * 3 + d3) * 64 + d4)> by [<Unmerge{1, 256, 3, 3, 64} ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [1, 256, 3, 3, 64] -> [147456]> : memref<147456xf16> to memref<1x256x3x3x64xf16>
  %1 = rock.transform %arg1 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 14 + d1) * 14 + d2 + d3) * 64 + d4)> by [<Unmerge{64, 14, 14, 1, 64} ["ni", "0i", "1i", "gi", "ci"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 14, 14, 1, 64] -> [802816]> : memref<802816xf16> to memref<64x14x14x1x64xf16>
  %2 = rock.transform %arg2 by <affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 12 + d1) * 12 + d2 + d3) * 256 + d4)> by [<Unmerge{64, 12, 12, 1, 256} ["no", "0o", "1o", "go", "ko"] at [0, 1, 2, 3, 4] -> ["raw"] at [0]>] bounds = [64, 12, 12, 1, 256] -> [2359296]> : memref<2359296xf16> to memref<64x12x12x1x256xf16>
  rock.conv(%0, %1, %2) features =  dot|atomic_add|atomic_fmax_f32|wmma {arch = "amdgcn-amd-amdhsa:gfx1100", derivedBlockSize = 128 : i32, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "0i", "1i", "gi", "ci"], numCU = 96 : i32, output_layout = ["no", "0o", "1o", "go", "ko"], padding = [0 : index, 0 : index, 0 : index, 0 : index], params = #rock.wmma_gemm_params<kpackPerBlock = 4, mPerBlock = 256, nPerBlock = 64, kpack = 8, mPerWave = 64, nPerWave = 64, splitKFactor = 1, forceUnroll = true>, strides = [1 : index, 1 : index]} : memref<1x256x3x3x64xf16>, memref<64x14x14x1x64xf16>, memref<64x12x12x1x256xf16>
  return
}
