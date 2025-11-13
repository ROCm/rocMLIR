// RUN: rocmlir-opt -rock-threadwise-gemm-lowering %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1 * 256 + d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> ((d2 * 20 + d3) * 20 + d4)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d6)>
#map5 = affine_map<(d0, d1, d2) -> (0, d0, d1, 0, d2 floordiv 20, 0, d2 mod 20)>
#map6 = affine_map<(d0, d1, d2) -> (0, d0, d1, d2 floordiv 20, d2 mod 20)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d0 * 64 + d5 + d7, d2 * 16 + d6 + d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, 0, d4, d5, 0)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d0 * 64 + d5 + d7, d3 * 16 + d6 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{256, 256} ["k", "c"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>, <AddDim{1} ["0"] at [3] -> [] at []>, <AddDim{1} ["1"] at [4] -> [] at []>] bounds = [1, 256, 256, 1, 1] -> [65536]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{256, 20, 20} ["ci", "0i", "1i"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["ni"] at [0] -> [] at []>, <AddDim{1} ["gi"] at [1] -> [] at []>] bounds = [1, 1, 256, 20, 20] -> [102400]>
#transform_map2 = #rock.transform_map<#map1 by [<Unmerge{256, 20, 20} ["ko", "0o", "1o"] at [2, 3, 4] -> ["raw"] at [0]>, <AddDim{1} ["no"] at [0] -> [] at []>, <AddDim{1} ["go"] at [1] -> [] at []>] bounds = [1, 1, 256, 20, 20] -> [102400]>
#transform_map3 = #rock.transform_map<#map2 by [<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, <Merge{256, 1, 1} ["gemmK"] at [1] -> ["c", "0", "1"] at [2, 3, 4]>, <PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 256, 256] -> [1, 256, 256, 1, 1]>
#transform_map4 = #rock.transform_map<#map3 by [<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, <PassThrough ["gi"] at [1] -> ["gi"] at [1]>, <PassThrough ["ci"] at [2] -> ["ci"] at [2]>, <Pad{0, 0, 0, 0} ["0ipad", "1ipad"] at [3, 4] -> ["0i", "1i"] at [3, 4]>] bounds = [1, 1, 256, 20, 20] -> [1, 1, 256, 20, 20]>
#transform_map5 = #rock.transform_map<#map4 by [<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, <AddDim{1} ["0"] at [3] -> [] at []>, <PassThrough ["0o"] at [4] -> ["0ipad"] at [3]>, <AddDim{1} ["1"] at [5] -> [] at []>, <PassThrough ["1o"] at [6] -> ["1ipad"] at [4]>] bounds = [1, 1, 256, 1, 20, 1, 20] -> [1, 1, 256, 20, 20]>
#transform_map6 = #rock.transform_map<#map5 by [<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, <Merge{256, 1, 1} ["gemmK"] at [1] -> ["ci", "0", "1"] at [2, 3, 5]>, <Merge{1, 20, 20} ["gemmN"] at [2] -> ["ni", "0o", "1o"] at [0, 4, 6]>] bounds = [1, 256, 400] -> [1, 1, 256, 1, 20, 1, 20]>
#transform_map7 = #rock.transform_map<#map6 by [<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, <PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, <Merge{1, 20, 20} ["gemmN"] at [2] -> ["no", "0o", "1o"] at [0, 3, 4]>] bounds = [1, 256, 400] -> [1, 1, 256, 20, 20]>
#transform_map8 = #rock.transform_map<#map7 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64, 1} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{16, 16, 1} ["m_block", "m_iter", "m_thread"] at [2, 6, 4] -> ["m"] at [2]>, <AddDim{25} ["n_block"] at [3] -> [] at []>] bounds = [4, 1, 16, 25, 1, 64, 16, 1] -> [1, 256, 256]>
#transform_map9 = #rock.transform_map<#map8 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{1, 64} ["tid"] at [4] -> ["m_thread", "k_thread"] at [4, 5]>, <Merge{16, 1} ["iter"] at [5] -> ["m_iter", "k_iter"] at [6, 7]>] bounds = [4, 1, 16, 25, 64, 16] -> [4, 1, 16, 25, 1, 64, 16, 1]>
#transform_map10 = #rock.transform_map<#map9 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{4, 64, 1} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{25, 16, 1} ["n_block", "n_iter", "n_thread"] at [3, 6, 4] -> ["n"] at [2]>, <AddDim{16} ["m_block"] at [2] -> [] at []>] bounds = [4, 1, 16, 25, 1, 64, 16, 1] -> [1, 256, 400]>
#transform_map11 = #rock.transform_map<#map8 by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{1, 64} ["tid"] at [4] -> ["n_thread", "k_thread"] at [4, 5]>, <Merge{16, 1} ["iter"] at [5] -> ["n_iter", "k_iter"] at [6, 7]>] bounds = [4, 1, 16, 25, 64, 16] -> [4, 1, 16, 25, 1, 64, 16, 1]>

// CHECK-LABEL: func.func @direct_to_lds_32b_test
// CHECK-SAME: features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-"} {
  func.func @direct_to_lds_32b_test(%arg0: memref<65536xf32>, %arg1: memref<102400xf32>, %arg2: memref<102400xf32>) 
    attributes {block_size = 64 : i32, 
                features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_f16|direct_to_lds_32b>, 
                grid_size = 400 : i32, 
                kernel = 0 : i32, 
                mhal.arch = "amdgcn-amd-amdhsa:gfx942:sramecc+:xnack-", 
                num_cu = 304 : i32} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    
    %0 = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    %1 = rock.alloc() : memref<4096xi8, #gpu.address_space<workgroup>>
    
    %view_lds_a = memref.view %0[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    %view_lds_b = memref.view %1[%c0][] : memref<4096xi8, #gpu.address_space<workgroup>> to memref<1024xf32, #gpu.address_space<workgroup>>
    
    %2 = rock.transform %arg0 by #transform_map : memref<65536xf32> to memref<1x256x256x1x1xf32>
    %3 = rock.transform %arg1 by #transform_map1 : memref<102400xf32> to memref<1x1x256x20x20xf32>
    %4 = rock.transform %2 by #transform_map3 : memref<1x256x256x1x1xf32> to memref<1x256x256xf32>
    %5 = rock.transform %3 by #transform_map4 : memref<1x1x256x20x20xf32> to memref<1x1x256x20x20xf32>
    %6 = rock.transform %5 by #transform_map5 : memref<1x1x256x20x20xf32> to memref<1x1x256x1x20x1x20xf32>
    %7 = rock.transform %6 by #transform_map6 : memref<1x1x256x1x20x1x20xf32> to memref<1x256x400xf32>
    
    %8 = rock.transform %4 by #transform_map8 : memref<1x256x256xf32> to memref<4x1x16x25x1x64x16x1xf32>
    %9 = rock.transform %8 by #transform_map9 : memref<4x1x16x25x1x64x16x1xf32> to memref<4x1x16x25x64x16xf32>
    
    %10 = rock.transform %7 by #transform_map10 : memref<1x256x400xf32> to memref<4x1x16x25x1x64x16x1xf32>
    %11 = rock.transform %10 by #transform_map11 : memref<4x1x16x25x1x64x16x1xf32> to memref<4x1x16x25x64x16xf32>
    
    %bid = rock.workgroup_id : index
    %tid = rock.workitem_id : index
    
    // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
    // CHECK-SAME: bounds [1, 1, 1, 1, 1, 16]
    // CHECK-SAME: strides [1, 1, 1, 1, 1, 1]
    // CHECK: rock.global_load_to_lds
    // CHECK-SAME: {transferType = f32}
    %12 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%9) [%c0, %c0, %c0, %c0, %tid] -> %view_lds_a 
      : memref<4x1x16x25x64x16xf32> -> memref<1024xf32, #gpu.address_space<workgroup>>, vector<1024xi1>
    
    // CHECK: rock.transforming_for {forceUnroll, useIndexDiffs}
    // CHECK-SAME: bounds [1, 1, 1, 1, 1, 16]
    // CHECK-SAME: strides [1, 1, 1, 1, 1, 1]
    // CHECK: rock.global_load_to_lds
    // CHECK-SAME: {transferType = f32}
    %13 = rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%11) [%c1, %c0, %c0, %c0, %tid] -> %view_lds_b 
      : memref<4x1x16x25x64x16xf32> -> memref<1024xf32, #gpu.address_space<workgroup>>, vector<1024xi1>
    
    return
  }
}

