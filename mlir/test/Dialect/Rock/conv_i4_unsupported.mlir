// i4 is added to the shared GemmInputTypes, so rock.conv accepts i4 operands.
// Since i4 WMMA is wired for GEMM only (no conv tuning/lowering yet), the conv
// verifier must reject i4 cleanly rather than fatal-erroring later in tuning.
// RUN: not rocmlir-opt %s 2>&1 | FileCheck %s

// CHECK: 'rock.conv' op i4 convolution is not yet supported

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 64 + d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (((d0 * 64 + d2) * 8 + d3) * 8 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{64, 64} ["k", "c"] at [1, 4] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["0"] at [2] -> [] at []>, <AddDim{1} ["1"] at [3] -> [] at []>] bounds = [1, 64, 1, 1, 64] -> [4096]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{8, 64, 8, 8} ["ni", "ci", "0i", "1i"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [8, 1, 64, 8, 8] -> [32768]>
#transform_map2 = #rock.transform_map<#map1 by [<Unmerge{8, 64, 8, 8} ["no", "ko", "0o", "1o"] at [0, 2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [8, 1, 64, 8, 8] -> [32768]>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1201"} {
  func.func @rock_conv_gk01c_ngc01_ngk01_0(%arg0: memref<4096xi4>, %arg1: memref<32768xi4>, %arg2: memref<32768xi32>) attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1201", rock.enable_splitk_for_tuning, rock.kernel = 0 : i32, rock.num_chiplets = 1 : i64, rock.num_cu = 12 : i32} {
    %0 = rock.transform %arg0 by #transform_map : memref<4096xi4> to memref<1x64x1x1x64xi4>
    %1 = rock.transform %arg1 by #transform_map1 : memref<32768xi4> to memref<8x1x64x8x8xi4>
    %2 = rock.transform %arg2 by #transform_map2 : memref<32768xi32> to memref<8x1x64x8x8xi32>
    rock.conv(%0, %1, %2) features = wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32 {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "0", "1", "c"], input_layout = ["ni", "gi", "ci", "0i", "1i"], output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x64x1x1x64xi4>, memref<8x1x64x8x8xi4>, memref<8x1x64x8x8xi32>
    return
  }
}
