// Manual lowering using TransformToRockptr pass
// Goal: Convert blockwise_load_tile and blockwise_store_tile into transforms_to_ptr + blockwise_load_ptr and blockwise_store_ptr

#map = affine_map<(d0, d1, d2) -> (d1 * 3072 + d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1 * 768 + d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d5, d3 * 64 + d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0 * 64 + d5, d2 * 32 + d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1 * 32 + d3, d2 * 64 + d4)>
#transform_map = #rock.transform_map<#map by [<Unmerge{384, 3072} ["m", "k"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 3072] -> [1179648]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{3072, 768} ["k", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 3072, 768] -> [2359296]>
#transform_map2 = #rock.transform_map<#map1 by [<Unmerge{384, 768} ["m", "n"] at [1, 2] -> ["raw"] at [0]>, <AddDim{1} ["g"] at [0] -> [] at []>] bounds = [1, 384, 768] -> [294912]>
#transform_map3 = #rock.transform_map<#map2 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemmK", "gemmM"] at [1, 2] -> ["gemmK", "gemmM"] at [2, 1]>] bounds = [1, 3072, 384] -> [1, 384, 3072]>
#transform_map4 = #rock.transform_map<#map3 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{48, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [1]>, <Unmerge{12, 64} ["n_block", "n_iter"] at [3, 4] -> ["n"] at [2]>, <AddDim{12} ["m_block"] at [2] -> [] at []>] bounds = [48, 1, 12, 12, 64, 64] -> [1, 3072, 768]>
#transform_map5 = #rock.transform_map<#map4 by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{48, 64} ["k_loop", "k_iter"] at [0, 5] -> ["k"] at [1]>, <Unmerge{12, 32} ["m_block", "m_iter"] at [2, 4] -> ["m"] at [2]>, <AddDim{12} ["n_block"] at [3] -> [] at []>] bounds = [48, 1, 12, 12, 32, 64] -> [1, 3072, 384]>
#transform_map6 = #rock.transform_map<#map5 by [<PassThrough ["g_block"] at [0] -> ["gemmG"] at [0]>, <Unmerge{12, 32} ["m_block", "m_iter"] at [1, 3] -> ["gemmM"] at [1]>, <Unmerge{12, 64} ["n_block", "n_iter"] at [2, 4] -> ["gemmN"] at [2]>] bounds = [1, 12, 12, 32, 64] -> [1, 384, 768]>
module {
  func.func @rock_gemm(%arg0: memref<1179648xf16>, %arg1: memref<2359296xf16>, %arg2: memref<294912xf32>) attributes {arch = "amdgcn-amd-amdhsa:gfx950:sramecc+:xnack-", block_size = 256 : i32, enable_splitk_for_tuning, features = #rock<GemmFeatures mfma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|direct_to_lds_32b|direct_to_lds_128b>, grid_size = 144 : i32, kernel, num_chiplets = 8 : i64, num_cu = 256 : i64, waves_per_eu = 0 : i64} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c48 = arith.constant 48 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c144 = arith.constant 144 : index
    %c12 = arith.constant 12 : index
    %c384 = arith.constant 384 : index
    %c32 = arith.constant 32 : index
    %c143 = arith.constant 143 : index
    %c36 = arith.constant 36 : index
    %c4 = arith.constant 4 : index

    %0 = rock.transform %arg0 by #transform_map : memref<1179648xf16> to memref<1x384x3072xf16>
    %1 = rock.transform %arg1 by #transform_map1 : memref<2359296xf16> to memref<1x3072x768xf16>
    %2 = rock.transform %arg2 by #transform_map2 : memref<294912xf32> to memref<1x384x768xf32>
    %3 = rock.transform %0 by #transform_map3 : memref<1x384x3072xf16> to memref<1x3072x384xf16>

    %4 = rock.workgroup_id : index

    %5 = arith.remui %4, %c4 : index
    %6 = arith.divui %4, %c4 : index
    %7 = arith.muli %5, %c36 : index
    %8 = arith.addi %6, %7 : index
    %9 = arith.cmpi sgt, %4, %c143 : index
    %10 = arith.select %9, %4, %8 : index
    %11 = arith.divui %10, %c144 : index
    %12 = arith.remui %10, %c144 : index
    %13 = arith.divui %12, %c384 : index
    %14 = arith.muli %13, %c32 : index
    %15 = arith.subi %c12, %14 : index
    %16 = arith.minui %15, %c32 : index
    %17 = arith.remui %12, %16 : index
    %18 = arith.addi %14, %17 : index
    %19 = arith.remui %12, %c384 : index
    %20 = arith.divui %19, %16 : index

    %21 = rock.alloc() : memref<32x64xf16, #gpu.address_space<private>>
    %22 = rock.alloc() : memref<64x64xf16, #gpu.address_space<private>>
    %23 = rock.alloc() : memref<32x64xf32, #gpu.address_space<private>>
    rock.fill(%23, %cst) : memref<32x64xf32, #gpu.address_space<private>>, f32
    
    // Step 0. Make sure transforms are hoisted out of the loop.
    %t24 = rock.transform %1 by #transform_map4 : memref<1x3072x768xf16> to memref<48x1x12x12x64x64xf16>
    %t25 = rock.transform %3 by #transform_map5 : memref<1x3072x384xf16> to memref<48x1x12x12x32x64xf16>

    scf.for %arg3 = %c0 to %c48 step %c1 {

      %mask_tensor_a, %24 = rock.transforms_to_ptr %t24[%arg3, %11, %18, %20] : memref<48x1x12x12x64x64xf16> to memref<64x64xindex>, memref<64x64xi1>
      %mask_tensor_b, %25 = rock.transforms_to_ptr %t25[%arg3, %11, %18, %20] : memref<48x1x12x12x32x64xf16> to memref<32x64xindex>, memref<32x64xi1>

      // Step 1. Translate blockwise_load_tile into transforms_to_ptr + blockwise_load_ptr
      // [OLD IR] rock.blockwise_load_tile %24[%arg3, %11, %18, %20] -> %22 : memref<48x1x12x12x64x64xf16> -> memref<64x64xf16, #gpu.address_space<private>>
      // [OLD IR] rock.blockwise_load_tile %25[%arg3, %11, %18, %20] -> %21 : memref<48x1x12x12x32x64xf16> -> memref<32x64xf16, #gpu.address_space<private>>

      rock.blockwise_load_ptr(%24, %mask_tensor_a) -> %22 : (memref<64x64xindex>, memref<64x64xi1>) -> memref<64x64xf16, #gpu.address_space<private>>
      rock.blockwise_load_ptr(%25, %mask_tensor_b) -> %21 : (memref<32x64xindex>, memref<32x64xi1>) -> memref<32x64xf16, #gpu.address_space<private>>|

      rock.blockwise_gemm_accel %23 += %21 * %22 : memref<32x64xf32, #gpu.address_space<private>> += memref<32x64xf16, #gpu.address_space<private>> * memref<64x64xf16, #gpu.address_space<private>>
    }

    // Step 2. Translate blockwise_store_tile into transforms_to_ptr + blockwise_store_ptr
    // [OLD IR] rock.blockwise_store_tile {forceUnroll, useIndexDiffs} %23 -> [#transform_map6](%2) [%11, %18, %20] by  set : memref<32x64xf32, #gpu.address_space<private>> -> memref<1x384x768xf32>

    %mask_tensor_c, %c_out_ptr = rock.transforms_to_ptr %2[%11, %18, %20] : memref<1x384x768xf32> to memref<32x64xindex>, memref<32x64xi1>
    rock.blockwise_store_ptr %23 -> (%c_out_ptr, %mask_tensor_c) : memref<32x64xf32, #gpu.address_space<private>> -> (memref<32x64xindex>, memref<32x64xi1>)

    return
  }
}
