// RUN: rocmlir-driver --rock-linalg-align %s | FileCheck %s
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d0 * 16 + d5 + d7, (d2 * 16 + d4) * 4 + d6)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 16, d4 mod 16, d5, 0)>

// CHECK-DAG: [[MAP:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 16, 1} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{1, 16, 4} ["m_block", "m_thread", "m_iter"] at [2, 4, 6] -> ["m"] at [2]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [4, 64, 1, 1, 16, 16, 4, 1] -> [64, 64, 64]>
#transform_map = #rock.transform_map<#map2 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 16, 1} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{1, 16, 4} ["m_block", "m_thread", "m_iter"] at [2, 4, 6] -> ["m"] at [2]>, <AddDim{1} ["n_block"] at [3] -> [] at []>] bounds = [4, 64, 1, 1, 16, 16, 4, 1] -> [64, 64, 64]>

// CHECK-DAG: [[MAP1:.*]] = #rock.transform_map<#map{{.*}} by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{16, 16} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{4, 1} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [4, 64, 1, 1, 256, 4] -> [4, 64, 1, 1, 16, 16, 4, 1]>
#transform_map1 = #rock.transform_map<#map3 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{16, 16} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{4, 1} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [4, 64, 1, 1, 256, 4] -> [4, 64, 1, 1, 16, 16, 4, 1]>


module {
  // CHECK-LABEL: @rock_gemm_1
  func.func @rock_gemm_1(%arg0: memref<64x64x64xf16>, %arg1: memref<64x64x64xf32>, %arg2: memref<64x64x64xf32>, %arg3: memref<64x64x64xf32>) attributes {block_size = 256 : i32, grid_size = 64 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", wave_size = 32 : i32} {
    %alloc = memref.alloc() : memref<64x64x64xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg3 : memref<64x64x64xf16>, memref<64x64x64xf32>) outs(%alloc : memref<64x64x64xf32>) {
    ^bb0(%in: f16, %in_10: f32, %out: f32):
      %46 = arith.extf %in : f16 to f32
      %47 = arith.addf %46, %in_10 : f32
      linalg.yield %47 : f32
    }
    %c0 = arith.constant 0 : index
    %1 = rock.workgroup_id : index
    %2 = rock.workitem_id : index
    %3 = rock.transform %alloc by #transform_map : memref<64x64x64xf32> to memref<4x64x1x1x16x16x4x1xf32>
    %4 = rock.transform %3 by #transform_map1 : memref<4x64x1x1x16x16x4x1xf32> to memref<4x64x1x1x256x4xf32>
    // CHECK: [[out_alloc:%.+]] = rock.alloc() : memref<4xf32
    %out = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
    %c1_1 = arith.constant 1 : index
    %5 = arith.divui %1, %c1_1 : index
    // CHECK: [[tiled_arg0:%.+]] = rock.alloc() : memref<4xf16
    // CHECK-NEXT: [[arg0_tr:%.+]] = rock.transform %arg0 by [[MAP]]
    // CHECK-NEXT: [[arg0_view:%.+]] = rock.transform [[arg0_tr]] by [[MAP1]]
    // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg0_view]]) [{{.*}}] -> [[tiled_arg0]]
    // CHECK: [[tiled_arg3:%.+]] = rock.alloc() : memref<4xf32
    // CHECK-NEXT: [[arg3_tr:%.+]] = rock.transform %arg3 by [[MAP]]
    // CHECK-NEXT: [[arg3_view:%.+]] = rock.transform [[arg3_tr]] by [[MAP1]]
    // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg3_view]]) [{{.*}}] -> [[tiled_arg3]]
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: ins([[tiled_arg0]], [[tiled_arg3]]
    // CHECK-SAME: outs([[out_alloc]]
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%c0, %2, %2, %5, %5] -> %out : memref<4x64x1x1x256x4xf32> -> memref<4xf32, #gpu.address_space<private>>
    // CHECK: [[arg0_alloc:%.+]] = rock.alloc() : memref<4xf16
    // CHECK: [[arg3_alloc:%.+]] = rock.alloc() : memref<4xf32
    affine.for %arg4 = 1 to 4 {
      // CHECK: [[arg0_tf:%.+]] = rock.transform %arg0 by [[MAP]]
      // CHECK-NEXT: [[arg0_v:%.+]] = rock.transform [[arg0_tf]] by [[MAP1]]
      // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg0_v]]) [%arg4, {{.*}}] -> [[arg0_alloc]]
      // CHECK: [[arg3_tf:%.+]] = rock.transform %arg3 by [[MAP]]
      // CHECK-NEXT: [[arg3_v:%.+]] = rock.transform [[arg3_tf]] by [[MAP1]]
      // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg3_v]]) [%arg4, {{.*}}] -> [[arg3_alloc]]
      // CHECK: linalg.generic
      // CHECK-SAME: ins([[arg0_alloc]], [[arg3_alloc]]
      // CHECK-SAME: outs([[out_alloc]]
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%arg4,  %2, %2, %5, %5] -> %out : memref<4x64x1x1x256x4xf32> -> memref<4xf32, #gpu.address_space<private>>
    }
    return
  }

  // CHECK-LABEL: @rock_gemm_2
  func.func @rock_gemm_2(%arg0: memref<64x64x64xf16>, %arg1: memref<64x64x64xf16>, %arg2: memref<64x64x64xf32>, %arg3: memref<64x64x64xf32>) attributes {block_size = 256 : i32, grid_size = 64 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1100", wave_size = 32 : i32} {
    %alloc = memref.alloc() : memref<64x64x64xf32>
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg3 : memref<64x64x64xf16>, memref<64x64x64xf32>) outs(%alloc : memref<64x64x64xf32>) {
    ^bb0(%in: f16, %in_10: f32, %out: f32):
      %46 = arith.extf %in : f16 to f32
      %47 = arith.addf %46, %in_10 : f32
      linalg.yield %47 : f32
    }
    %alloc_0 = memref.alloc() : memref<64x64x64xf32>
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : memref<64x64x64xf16>) outs(%alloc_0 : memref<64x64x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %43 = arith.extf %in : f16 to f32
      linalg.yield %43 : f32
    }
    %c0 = arith.constant 0 : index
    %1 = rock.workgroup_id : index
    %2 = rock.workitem_id : index
    %3 = rock.transform %alloc by #transform_map : memref<64x64x64xf32> to memref<4x64x1x1x16x16x4x1xf32>
    %4 = rock.transform %3 by #transform_map1 : memref<4x64x1x1x16x16x4x1xf32> to memref<4x64x1x1x256x4xf32>
    // CHECK: [[tiled_out:%.+]] = rock.alloc() : memref<2xf32
    // CHECK: [[out:%.+]] = rock.alloc() : memref<4xf32
    // CHECK: [[tiled_out_1:%.+]] = rock.alloc() : memref<2xf32
    // CHECK: [[out_1:%.+]] = rock.alloc() : memref<4xf32
    %9 = rock.transform %alloc_0 by #transform_map : memref<64x64x64xf32> to memref<4x64x1x1x16x16x1x4xf32>
    %10 = rock.transform %9 by #transform_map1 : memref<4x64x1x1x16x16x1x4xf32> to memref<4x64x1x1x256x4xf32>
    %out = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
    %out_1 = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
    %c1_1 = arith.constant 1 : index
    %5 = arith.divui %1, %c1_1 : index
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%c0, %2, %2, %5, %5] -> %out : memref<4x64x1x1x256x4xf32> -> memref<4xf32, #gpu.address_space<private>>

    // CHECK: [[tiled_arg1:%.+]] = rock.alloc() : memref<4xf16
    // CHECK-NEXT: [[arg1_tr:%.+]] = rock.transform %arg1 by [[MAP]]
    // CHECK-NEXT: [[arg1_view:%.+]] = rock.transform [[arg1_tr]] by [[MAP1]]
    // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg1_view]]) [{{.*}}] -> [[tiled_arg1]]
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: ins([[tiled_arg1]]
    // CHECK-SAME: outs([[out_1]]
    rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%10) [%c0, %2, %2, %5, %5] -> %out_1 : memref<4x64x1x1x256x4xf32> -> memref<4xf32, #gpu.address_space<private>>

    // CHECK: [[tiled_arg0:%.+]] = rock.alloc() : memref<4xf16
    // CHECK-NEXT: [[arg0_tr:%.+]] = rock.transform %arg0 by [[MAP]]
    // CHECK-NEXT: [[arg0_view:%.+]] = rock.transform [[arg0_tr]] by [[MAP1]]
    // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg0_view]]) [{{.*}}] -> [[tiled_arg0]]
    // CHECK: [[tiled_arg3:%.+]] = rock.alloc() : memref<4xf32
    // CHECK-NEXT: [[arg3_tr:%.+]] = rock.transform %arg3 by [[MAP]]
    // CHECK-NEXT: [[arg3_view:%.+]] = rock.transform [[arg3_tr]] by [[MAP1]]
    // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg3_view]]) [{{.*}}] -> [[tiled_arg3]]
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: ins([[tiled_arg0]], [[tiled_arg3]]
    // CHECK-SAME: outs([[out]]

    %tiled_out = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    %tiled_out_1 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    // CHECK: [[arg1_alloc:%.+]] = rock.alloc() : memref<2xf16
    // CHECK: [[arg0_alloc:%.+]] = rock.alloc() : memref<2xf16
    // CHECK: [[arg3_alloc:%.+]] = rock.alloc() : memref<2xf32
    affine.for %arg4 = 1 to 4 {
      // CHECK: [[arg0_tf:%.+]] = rock.transform %arg0 by [[MAP]]
      // CHECK-NEXT: [[arg0_v:%.+]] = rock.transform [[arg0_tf]] by [[MAP1]]
      // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg0_v]]) [%arg4, {{.*}}] -> [[arg0_alloc]]
      // CHECK: [[arg3_tf:%.+]] = rock.transform %arg3 by [[MAP]]
      // CHECK-NEXT: [[arg3_v:%.+]] = rock.transform [[arg3_tf]] by [[MAP1]]
      // CHECK-NEXT: rock.threadwise_read_into {{.*}} []([[arg3_v]]) [%arg4, {{.*}}] -> [[arg3_alloc]]
      // CHECK: linalg.generic
      // CHECK-SAME: ins([[arg0_alloc]], [[arg3_alloc]]
      // CHECK-SAME: outs([[tiled_out]]
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%4) [%arg4,  %2, %2, %5, %5] -> %tiled_out : memref<4x64x1x1x256x4xf32> -> memref<2xf32, #gpu.address_space<private>>

      // CHECK: [[arg1_tf:%.+]] = rock.transform %arg1 by [[MAP]]
      // CHECK: [[arg1_v:%.+]] = rock.transform [[arg1_tf]] by [[MAP1]]
      // CHECK: rock.threadwise_read_into {{.*}} []([[arg1_v]]) [%arg4, {{.*}}] -> [[arg1_alloc]]
      // CHECK: linalg.generic
      // CHECK-SAME: ins([[arg1_alloc]]
      // CHECK-SAME: outs([[tiled_out_1]]
      rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%10) [%arg4,  %2, %2, %5, %5] -> %tiled_out_1 : memref<4x64x1x1x256x4xf32> -> memref<2xf32, #gpu.address_space<private>>
    }
    return
  }
}
