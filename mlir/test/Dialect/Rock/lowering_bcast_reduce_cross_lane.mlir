// RUN: rocmlir-opt -rock-blockwise-gemm-to-threadwise -split-input-file %s | FileCheck %s --check-prefixes=CHECK

#map4 = affine_map<(d0, d1) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 * 8 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [256, 32] -> [1, 8, 2, 16, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 8, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16, 4, 1, 8] -> [64, 128]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 128] -> [128, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [256] -> [1, 8, 2, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{8, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16] -> [2, 128]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 128] -> [128, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx1201 (RDNA4 wave32): NR-Small PermlaneX16Var LDS-skip path
// m_tid=2, n_tid=16, blockSize=256 > nrDimProd=64 => NR-Small
// partialR=2, K=1 => canUsePermlaneX16Var_NRSmall_LdsSkip

// CHECK-LABEL: func @test_permlane_nrsmall_ldsskip_gfx1201

// Threadwise partial reduction loop
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via permlanex16_var
// CHECK: amdgpu.permlane_var
// CHECK-NEXT: arith.maxnumf

// No LDS barriers (full LDS-skip path)
// CHECK-NOT: rock.lds_barrier

// Direct register readback to output
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlane_nrsmall_ldsskip_gfx1201(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1201",
               block_size = 256 : i32, grid_size = 36 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 256 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4f16 = affine_map<(d0, d1) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16, d1 floordiv 8, 0, d1 mod 8)>
#map5f16 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 * 8 + d1) * 16 + d3)>
#map6f16 = affine_map<(d0, d1) -> (d0, d1)>
#map7f16 = affine_map<(d0, d1) -> (d1, d0)>
#map8f16 = affine_map<(d0) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16)>
#map9f16 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 16 + d3)>
#map10f16 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11f16 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#tm8f16 = #rock.transform_map<#map4f16 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [256, 32] -> [1, 8, 2, 16, 4, 1, 8]>
#tm9f16 = #rock.transform_map<#map5f16 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 8, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16, 4, 1, 8] -> [64, 128]>
#tm10f16 = #rock.transform_map<#map6f16 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#tm11f16 = #rock.transform_map<#map6f16 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#tm12f16 = #rock.transform_map<#map7f16 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 128] -> [128, 64]>
#tm13f16 = #rock.transform_map<#map8f16 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [256] -> [1, 8, 2, 16]>
#tm14f16 = #rock.transform_map<#map9f16 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{8, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16] -> [2, 128]>
#tm15f16 = #rock.transform_map<#map6f16 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#tm16f16 = #rock.transform_map<#map6f16 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#tm17f16 = #rock.transform_map<#map7f16 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 128] -> [128, 2]>
#tm18f16 = #rock.transform_map<#map10f16 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#tm19f16 = #rock.transform_map<#map11f16 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#tm20f16 = #rock.transform_map<#map6f16 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#tm21f16 = #rock.transform_map<#map6f16 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#tm22f16 = #rock.transform_map<#map7f16 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx1201 (RDNA4 wave32): NR-Small PermlaneX16Var LDS-skip path (f16)
// Same layout as f32 test but with f16 element type to verify
// amdgpu.permlane_var handles sub-32-bit type decomposition correctly.

// CHECK-LABEL: func @test_permlane_nrsmall_ldsskip_f16_gfx1201

// Threadwise partial reduction loop
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via permlanex16_var (f16 type decomposition)
// CHECK: amdgpu.permlane_var
// CHECK-NEXT: arith.maxnumf

// No LDS barriers (full LDS-skip path)
// CHECK-NOT: rock.lds_barrier

// Direct register readback to output
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf16, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlane_nrsmall_ldsskip_f16_gfx1201(
    %input_reg : memref<32xf16, #gpu.address_space<private>>,
    %output_reg : memref<32xf16, #gpu.address_space<private>>,
    %ws_lds : memref<256xf16, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1201",
               block_size = 256 : i32, grid_size = 36 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#tm8f16, #tm9f16, #tm10f16, #tm10f16, #tm11f16, #tm12f16] [#tm13f16, #tm14f16, #tm15f16, #tm15f16, #tm16f16, #tm17f16] [#tm18f16, #tm19f16, #tm20f16, #tm20f16, #tm21f16, #tm22f16] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 256 : i32} : memref<32xf16, #gpu.address_space<private>> using memref<256xf16, #gpu.address_space<workgroup>> into memref<32xf16, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 * 4 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 4, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [128, 64] -> [1, 4, 2, 16, 4, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 4, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 4, 2, 16, 4, 2, 8] -> [64, 128]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 128] -> [128, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 4, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [128] -> [1, 4, 2, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{4, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 4, 2, 16] -> [2, 64]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 64] -> [64, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [4, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 2, 8] -> [32, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 2] -> [2, 32]>

// gfx1201 (RDNA4 wave32): NR-Large PermlaneX16Var path
// m_tid=2, n_tid=16, blockSize=128 <= nrDimProd=128 => NR-Large
// partialR=2 == mTidPerWave=2 => canUsePermlaneX16Var_NRLarge

// CHECK-LABEL: func @test_permlane_nrlarge_gfx1201

// Threadwise partial reduction loop
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via permlanex16_var (2 elements in nrDim)
// CHECK: amdgpu.permlane_var
// CHECK-NEXT: arith.maxnumf
// CHECK: amdgpu.permlane_var
// CHECK-NEXT: arith.maxnumf

// No LDS barriers (register-only path)
// CHECK-NOT: rock.lds_barrier

// Direct register readback to output
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlane_nrlarge_gfx1201(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1201",
               block_size = 128 : i32, grid_size = 36 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 128 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 8 + d6) * 2 + d2, (d5 * 8 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [256, 32] -> [1, 8, 2, 16, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 8, 2} ["rep_i", "wave_m", "item_i", "m_tid"] at [4, 0, 6, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 8, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16, 4, 1, 8] -> [64, 128]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 128] -> [128, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 8, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [256] -> [1, 8, 2, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{8, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 8, 2, 16] -> [2, 128]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 128] -> [2, 128]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 128] -> [128, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx1100 (RDNA3 wave32): NR-Small DsSwizzle XOR=16 LDS-skip path
// m_tid=2, n_tid=16, blockSize=256 > nrDimProd => NR-Small
// gfx11 lacks permlanex16_var => ds_swizzle XOR=16
// partialR=2, K=1 => canUseDsSwizzleW32_NRSmall_LdsSkip

// CHECK-LABEL: func @test_dsswizzle_nrsmall_ldsskip_gfx1100

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via amdgpu.swizzle_bitmode (XOR=16)
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsswizzle_nrsmall_ldsskip_gfx1100(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1100",
               block_size = 256 : i32, grid_size = 36 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 256 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 8 + d6) * 2 + d2, (d5 * 4 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 32, (d0 mod 32) floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 4, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [128, 64] -> [1, 4, 2, 16, 4, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 8, 2} ["rep_i", "wave_m", "item_i", "m_tid"] at [4, 0, 6, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 4, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 4, 2, 16, 4, 2, 8] -> [64, 128]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{128} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 128] -> [64, 128]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 128] -> [128, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 4, 2, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [128] -> [1, 4, 2, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{4, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 4, 2, 16] -> [2, 64]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 64] -> [64, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [4, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 2, 8] -> [32, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 2] -> [2, 32]>

// gfx1100 (RDNA3 wave32): NR-Large DsSwizzle XOR=16 path
// m_tid=2, n_tid=16, blockSize=128 <= nrDimProd=128 => NR-Large
// gfx11 lacks permlanex16_var => ds_swizzle XOR=16
// partialR=2 == mTidPerWave=2 => canUseDsSwizzleW32_NRLarge

// CHECK-LABEL: func @test_dsswizzle_nrlarge_gfx1100

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via amdgpu.swizzle_bitmode (2 elements)
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf

// No LDS barriers
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsswizzle_nrlarge_gfx1100(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1100",
               block_size = 128 : i32, grid_size = 36 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 128 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx942 (CDNA3 wave64): NR-Small DsSwizzleBpermute LDS-skip, SUM
// m_tid=2, n_tid=32, blockSize=64 == waveSize (single wave)
// partialR=2, K=1 => canUseDsSwizzleBpermute_NRSmall_LdsSkip
// groupSize=2 => only ds_bpermute (XOR 32), no ds_swizzle step

// CHECK-LABEL: func @test_dsbpermute_nrsmall_ldsskip_sum_gfx942

// Threadwise partial reduction loop (sum)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Cross-half-wave reduction via ds_bpermute (XOR 32)
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrsmall_ldsskip_sum_gfx942(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx950 (CDNA4 wave64): NR-Small PermlaneSwap LDS-skip, SUM
// m_tid=2, n_tid=32, blockSize=64 == waveSize (single wave)
// partialR=2, K=1 => canUsePermlaneSwap_NRSmall_LdsSkip
// groupSize=2 => only permlane32_swap (cross-half)

// CHECK-LABEL: func @test_permlaneswap_nrsmall_ldsskip_sum_gfx950

// Threadwise partial reduction loop (sum)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Cross-half-wave reduction via permlane32_swap
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.permlane32.swap
// CHECK: llvm.extractvalue
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlaneswap_nrsmall_ldsskip_sum_gfx950(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx950 (CDNA4 wave64): NR-Small PermlaneSwap LDS-skip, f16
// Same config as test_permlaneswap_nrsmall_ldsskip_sum_gfx950 but f16
// to verify the bitcast f16→i16, zext i16→i32, swap, trunc i32→i16,
// bitcast i16→f16 chain in permlaneSwapReduceStep (no v_cvt instructions).

// CHECK-LABEL: func @test_permlaneswap_nrsmall_ldsskip_f16_gfx950

// Threadwise partial reduction (sum, f16)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Cross-half-wave reduction via permlane32_swap with f16 bit-packing
// CHECK: arith.bitcast %{{.*}} : f16 to i16
// CHECK: arith.extui %{{.*}} : i16 to i32
// CHECK: rocdl.permlane32.swap
// CHECK: arith.trunci %{{.*}} : i32 to i16
// CHECK: arith.bitcast %{{.*}} : i16 to f16
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf16, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlaneswap_nrsmall_ldsskip_f16_gfx950(
    %input_reg : memref<32xf16, #gpu.address_space<private>>,
    %output_reg : memref<32xf16, #gpu.address_space<private>>,
    %ws_lds : memref<64xf16, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf16, #gpu.address_space<private>> using memref<64xf16, #gpu.address_space<workgroup>> into memref<32xf16, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 16] -> [1, 1, 4, 16, 2, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 1, 8] -> [64, 16]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 16] -> [64, 16]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 16] -> [64, 16]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 16] -> [16, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [16] -> [2, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 1, 8] -> [16, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 1] -> [16, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 1] -> [16, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 1] -> [1, 16]>

// gfx942 (CDNA3 wave64): NR-Small DsSwizzleBpermute LDS-skip, partialR=4
// m_tid=4, n_tid=16, blockSize=64 == waveSize (single wave)
// partialR=4, K=1 => canUseDsSwizzleBpermute_NRSmall_LdsSkip
// groupSize=4 => ds_swizzle XOR=16 (within-half) + ds_bpermute XOR=32 (cross-half)

// CHECK-LABEL: func @test_dsbpermute_nrsmall_ldsskip_r4_sum_gfx942

// Threadwise partial reduction (sum)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Step 1: within-half reduction via amdgpu.swizzle_bitmode (XOR 16)
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.addf

// Step 2: cross-half reduction via ds_bpermute (XOR 32)
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrsmall_ldsskip_r4_sum_gfx942(
    %input_reg : memref<16xf32, #gpu.address_space<private>>,
    %output_reg : memref<16xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<16xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<16xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 16] -> [1, 1, 4, 16, 2, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 1, 8] -> [64, 16]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 16] -> [64, 16]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 16] -> [64, 16]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 16] -> [16, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [16] -> [2, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 1, 8] -> [16, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 1] -> [16, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 1] -> [16, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 1] -> [1, 16]>

// gfx950 (CDNA4 wave64): NR-Small PermlaneSwap LDS-skip, partialR=4
// m_tid=4, n_tid=16, blockSize=64 == waveSize (single wave)
// partialR=4, K=1 => canUsePermlaneSwap_NRSmall_LdsSkip
// groupSize=4 => permlane16_swap (within-half) + permlane32_swap (cross-half)

// CHECK-LABEL: func @test_permlaneswap_nrsmall_ldsskip_r4_sum_gfx950

// Threadwise partial reduction (sum)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Step 1: within-half reduction via permlane16_swap
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.permlane16.swap
// CHECK: llvm.extractvalue
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf

// Step 2: cross-half reduction via permlane32_swap
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.permlane32.swap
// CHECK: llvm.extractvalue
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlaneswap_nrsmall_ldsskip_r4_sum_gfx950(
    %input_reg : memref<16xf32, #gpu.address_space<private>>,
    %output_reg : memref<16xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<16xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<16xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 64] -> [1, 1, 2, 32, 4, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 2, 8] -> [64, 64]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 64] -> [64, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [4, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 2, 8] -> [32, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 2] -> [2, 32]>

// gfx942 (CDNA3 wave64): NR-Large DsSwizzleBpermute, partialR=2
// m_tid=2, n_tid=32, blockSize=64 <= nrDimProd=64 => NR-Large
// partialR=2 => canUseDsSwizzleBpermute_NRLarge
// groupSize=2 => only ds_bpermute (XOR 32), applied to 2 elements

// CHECK-LABEL: func @test_dsbpermute_nrlarge_r2_gfx942

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-lane reduction: ds_bpermute on 2 nrDim elements
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf

// No LDS barriers
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrlarge_r2_gfx942(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<128xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 64] -> [1, 1, 2, 32, 4, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 2, 8] -> [64, 64]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 64] -> [64, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [4, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 2, 8] -> [32, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 2] -> [32, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 2] -> [2, 32]>

// gfx950 (CDNA4 wave64): NR-Large PermlaneSwap, partialR=2
// m_tid=2, n_tid=32, blockSize=64 <= nrDimProd=64 => NR-Large
// partialR=2 => canUsePermlaneSwap_NRLarge
// groupSize=2 => only permlane32_swap, applied to 2 elements

// CHECK-LABEL: func @test_permlaneswap_nrlarge_r2_gfx950

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-lane reduction: permlane32_swap on 2 nrDim elements
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.permlane32.swap
// CHECK: llvm.extractvalue
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.permlane32.swap
// CHECK: llvm.extractvalue
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf

// No LDS barriers
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlaneswap_nrlarge_r2_gfx950(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<128xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 32, (d1 mod 32) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 32, (d0 mod 32) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 4, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 64] -> [1, 1, 4, 16, 2, 4, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{4, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 4, 8] -> [64, 64]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 64] -> [64, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 4, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [2, 4, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{4} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 4, 8] -> [16, 4]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 4] -> [16, 4]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{4} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 4] -> [16, 4]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 4] -> [4, 16]>

// gfx942 (CDNA3 wave64): NR-Large DsSwizzleBpermute, partialR=4
// m_tid=4, n_tid=16, blockSize=64 <= nrDimProd=64 => NR-Large
// partialR=4 => canUseDsSwizzleBpermute_NRLarge
// groupSize=4 => ds_swizzle (XOR 16) + ds_bpermute (XOR 32), each on 4 elements

// CHECK-LABEL: func @test_dsbpermute_nrlarge_r4_gfx942

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Step 1: amdgpu.swizzle_bitmode XOR=16 on 4 nrDim elements
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf
// CHECK: amdgpu.swizzle_bitmode
// CHECK: arith.maxnumf

// Step 2: ds_bpermute XOR=32 on 4 nrDim elements
// CHECK: rocdl.ds_bpermute
// CHECK: arith.maxnumf
// CHECK: rocdl.ds_bpermute
// CHECK: arith.maxnumf
// CHECK: rocdl.ds_bpermute
// CHECK: arith.maxnumf
// CHECK: rocdl.ds_bpermute
// CHECK: arith.maxnumf

// No LDS barriers
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrlarge_r4_gfx942(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 32, (d1 mod 32) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 32, (d0 mod 32) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 4, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 64] -> [1, 1, 4, 16, 2, 4, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{4, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 4, 8] -> [64, 64]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 64] -> [64, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 4, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [64] -> [2, 4, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{4} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 4, 8] -> [16, 4]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 4] -> [16, 4]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{4} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 4] -> [16, 4]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 4] -> [4, 16]>

// gfx950 (CDNA4 wave64): NR-Large PermlaneSwap, partialR=4
// m_tid=4, n_tid=16, blockSize=64 <= nrDimProd=64 => NR-Large
// partialR=4 => canUsePermlaneSwap_NRLarge
// groupSize=4 => permlane16_swap + permlane32_swap, each on 4 elements

// CHECK-LABEL: func @test_permlaneswap_nrlarge_r4_gfx950

// Threadwise partial reduction
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Step 1: permlane16_swap on 4 nrDim elements
// CHECK: rocdl.permlane16.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane16.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane16.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane16.swap
// CHECK: arith.maxnumf

// Step 2: permlane32_swap on 4 nrDim elements
// CHECK: rocdl.permlane32.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane32.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane32.swap
// CHECK: arith.maxnumf
// CHECK: rocdl.permlane32.swap
// CHECK: arith.maxnumf

// No LDS barriers
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_permlaneswap_nrlarge_r4_gfx950(
    %input_reg : memref<64xf32, #gpu.address_space<private>>,
    %output_reg : memref<64xf32, #gpu.address_space<private>>,
    %ws_lds : memref<256xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<64xf32, #gpu.address_space<private>> using memref<256xf32, #gpu.address_space<workgroup>> into memref<64xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 4, 16, 2, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 2, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [2, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 2, 8] -> [16, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 2] -> [16, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 2] -> [16, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 2] -> [2, 16]>

// gfx942 (CDNA3 wave64): NR-Small DsSwizzleBpermute WITHOUT LDS-skip
// m_tid=4, n_tid=16, blockSize=64 > nrDimProd=32 => NR-Small
// K=2 => LDS-skip disabled, ds_bpermute emitted with LDS barriers

// CHECK-LABEL: func @test_dsbpermute_nrsmall_noldsskip_gfx942

// Upfront LDS barrier
// CHECK: rock.lds_barrier

// Cross-lane reduction via ds_bpermute (groupSize=2, 1 element)
// CHECK: rocdl.ds_bpermute
// CHECK: arith.maxnumf

// LDS barrier after leader write
// CHECK: rock.lds_barrier

// Final readback from LDS to output via threadwise_read_into
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_dsbpermute_nrsmall_noldsskip_gfx942(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<128xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 16, d0 mod 16, d1 floordiv 16, (d1 mod 16) floordiv 8, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 4 + d2) * 8 + d6, (d5 + d1) * 16 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 16, d0 mod 16)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2, d1 * 16 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 16, (d0 mod 16) floordiv 8, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{2, 2, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 4, 16, 2, 2, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{2, 1, 4, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 1, 16} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16, 2, 2, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 4, 16} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 4, 16]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 4} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 16} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 4, 16] -> [4, 16]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{4} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{16} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [4, 16] -> [4, 16]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [4, 16] -> [16, 4]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{2, 2, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [2, 2, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{2, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 2, 8] -> [16, 2]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [16, 2] -> [16, 2]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{16} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{2} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [16, 2] -> [16, 2]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [16, 2] -> [2, 16]>

// gfx950 (CDNA4 wave64): NR-Small PermlaneSwap WITHOUT LDS-skip
// m_tid=4, n_tid=16, blockSize=64 > nrDimProd=32 => NR-Small
// K=2 => LDS-skip disabled, permlane32_swap emitted with LDS barriers

// CHECK-LABEL: func @test_permlaneswap_nrsmall_noldsskip_gfx950

// Upfront LDS barrier
// CHECK: rock.lds_barrier

// Cross-lane reduction via permlane32_swap (groupSize=2, 1 element)
// CHECK: rocdl.permlane32.swap
// CHECK: arith.maxnumf

// LDS barrier after leader write
// CHECK: rock.lds_barrier

// Final readback from LDS to output via threadwise_read_into
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_permlaneswap_nrsmall_noldsskip_gfx950(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<128xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx950",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 * 2 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, d0 floordiv 64, (d0 mod 64) floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 2, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [128, 32] -> [1, 2, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 2, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 2, 2, 32, 4, 1, 8] -> [64, 64]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 64] -> [64, 64]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 64] -> [64, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 2, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [128] -> [1, 2, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{2, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 2, 2, 32] -> [2, 64]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{64} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 64] -> [2, 64]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 64] -> [64, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx942 (CDNA3 wave64): LDS-tree fallback — multi-wave, no fast-path
// m_tid=2, n_tid=32, blockSize=128 (2 waves) > nrDimProd=64 => NR-Small
// Multi-wave => no cross-lane fast path, falls to LDS-tree reduction

// CHECK-LABEL: func @test_ldstree_fallback_multiwave_gfx942

// No cross-lane intrinsics before first barrier
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// LDS barrier (upfront store)
// CHECK: rock.lds_barrier

// Tree reduction
// CHECK: arith.maxnumf

// LDS barrier (after tree step)
// CHECK: rock.lds_barrier

// No cross-lane intrinsics after barriers
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// Final readback from LDS to output via threadwise_read_into
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_ldstree_fallback_multiwave_gfx942(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<128xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 128 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 128 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<128xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// Near-miss: gfx1030 (RDNA2 wave32) — same layout as gfx942 NR-Small
// (m_tid=2, n_tid=32, blockSize=64) but gfx1030 does not match any
// cross-lane arch gate (not gfx908/gfx90a/gfx94x/gfx950/gfx11/gfx12).
// Also m_tid*n_tid=64 != 32 (waveSize), so layoutTilesWave is false.
// Must fall back to LDS-tree reduction.

// CHECK-LABEL: func @test_ldstree_fallback_unsupported_arch_gfx1030

// No cross-lane intrinsics
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// LDS barrier (upfront store)
// CHECK: rock.lds_barrier

// Tree reduction
// CHECK: arith.maxnumf

// LDS barrier (after tree step)
// CHECK: rock.lds_barrier

// No cross-lane intrinsics after barriers
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// Final readback
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_ldstree_fallback_unsupported_arch_gfx1030(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1030",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// Near-miss: wave64 layout on gfx1100 (wave32) — layout doesn't tile wave.
// m_tid=2, n_tid=32 => product=64 != 32 (waveSize), so layoutTilesWave
// is false. hasDsSwizzleWave32 requires layoutTilesWave, so all wave32
// fast paths are disabled. Must fall back to LDS-tree reduction.

// CHECK-LABEL: func @test_ldstree_fallback_layout_mismatch_gfx1100

// No cross-lane intrinsics
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: amdgpu.permlane_var
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// LDS barrier (upfront store)
// CHECK: rock.lds_barrier

// Tree reduction
// CHECK: arith.maxnumf

// LDS barrier (after tree step)
// CHECK: rock.lds_barrier

// No cross-lane intrinsics after barriers
// CHECK-NOT: amdgpu.swizzle_bitmode
// CHECK-NOT: amdgpu.permlane_var
// CHECK-NOT: rocdl.ds_bpermute
// CHECK-NOT: rocdl.permlane

// Final readback
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_ldstree_fallback_layout_mismatch_gfx1100(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx1100",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx942 (CDNA3 wave64): f16 sub-32-bit type coverage for ds_bpermute path
// Same config as test_dsbpermute_nrsmall_ldsskip_sum_gfx942 but with f16
// to verify bit-packing (bitcast+zext/trunc) handles sub-32-bit types in
// the ds_bpermute path correctly (no v_cvt instructions).

// CHECK-LABEL: func @test_dsbpermute_nrsmall_ldsskip_f16_gfx942

// Threadwise partial reduction (sum, f16)
// CHECK: rock.transforming_for
// CHECK: arith.addf

// Cross-half-wave reduction via ds_bpermute with f16 bit-packing
// CHECK: arith.bitcast %{{.*}} : f16 to i16
// CHECK: arith.extui %{{.*}} : i16 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.trunci %{{.*}} : i32 to i16
// CHECK: arith.bitcast %{{.*}} : i16 to f16
// CHECK: arith.addf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf16, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrsmall_ldsskip_f16_gfx942(
    %input_reg : memref<32xf16, #gpu.address_space<private>>,
    %output_reg : memref<32xf16, #gpu.address_space<private>>,
    %ws_lds : memref<64xf16, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce sum [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf16, #gpu.address_space<private>> using memref<64xf16, #gpu.address_space<workgroup>> into memref<32xf16, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx908 (MI100 wave64): NR-Small DsSwizzleBpermute LDS-skip
// Same layout as gfx942 test — verifies ds_bpermute fast path
// is correctly selected on the oldest supported CDNA arch.

// CHECK-LABEL: func @test_dsbpermute_nrsmall_ldsskip_gfx908

// Threadwise partial reduction loop
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via ds_bpermute
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrsmall_ldsskip_gfx908(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx908",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx90a (MI250 wave64): NR-Small DsSwizzleBpermute LDS-skip
// Same layout as gfx942 test — verifies ds_bpermute fast path
// is correctly selected on MI250 (gfx90a) arch.

// CHECK-LABEL: func @test_dsbpermute_nrsmall_ldsskip_gfx90a

// Threadwise partial reduction loop
// CHECK: rock.transforming_for
// CHECK: arith.maxnumf

// Cross-half-wave reduction via ds_bpermute
// CHECK: arith.bitcast %{{.*}} : f32 to i32
// CHECK: rocdl.ds_bpermute
// CHECK: arith.bitcast %{{.*}} : i32 to f32
// CHECK: arith.maxnumf

// No LDS barriers (full LDS-skip)
// CHECK-NOT: rock.lds_barrier

// Direct register readback
// CHECK: rock.transforming_for
// CHECK: rock.in_bounds_load %{{.*}} : memref<1xf32, #gpu.address_space<private>>
// CHECK: rock.in_bounds_store %{{.*}} -> %arg1
// CHECK: return

func.func @test_dsbpermute_nrsmall_ldsskip_gfx90a(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx90a",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>
  return
}

// -----

#map4 = affine_map<(d0, d1) -> (0, 0, d0 floordiv 32, d0 mod 32, d1 floordiv 8, 0, d1 mod 8)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (((d0 + d4) * 2 + d2) * 8 + d6, (d5 + d1) * 32 + d3)>
#map6 = affine_map<(d0, d1) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d1, d0)>
#map8 = affine_map<(d0) -> (0, 0, d0 floordiv 32, d0 mod 32)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d2, d1 * 32 + d3)>
#map10 = affine_map<(d0) -> (d0 floordiv 8, 0, d0 mod 8)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 8 + d2, d1)>

#transform_map8 = #rock.transform_map<#map4 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>, <Merge{4, 1, 8} ["item"] at [1] -> ["rep_i", "rep_j", "item_i"] at [4, 5, 6]>] bounds = [64, 32] -> [1, 1, 2, 32, 4, 1, 8]>
#transform_map9 = #rock.transform_map<#map5 by [<Unmerge{4, 1, 2, 8} ["rep_i", "wave_m", "m_tid", "item_i"] at [4, 0, 2, 6] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 1, 32} ["rep_j", "wave_n", "n_tid"] at [5, 1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32, 4, 1, 8] -> [64, 32]>
#transform_map10 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map11 = #rock.transform_map<#map6 by [<Unmerge{64} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [64, 32] -> [64, 32]>
#transform_map12 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [64, 32] -> [32, 64]>
#transform_map13 = #rock.transform_map<#map8 by [<Merge{1, 1, 2, 32} ["tid"] at [0] -> ["wave_m", "wave_n", "m_tid", "n_tid"] at [0, 1, 2, 3]>] bounds = [64] -> [1, 1, 2, 32]>
#transform_map14 = #rock.transform_map<#map9 by [<Unmerge{1, 2} ["wave_m", "m_tid"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1, 32} ["wave_n", "n_tid"] at [1, 3] -> ["gemmBlockN"] at [1]>] bounds = [1, 1, 2, 32] -> [2, 32]>
#transform_map15 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map16 = #rock.transform_map<#map6 by [<Unmerge{2} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{32} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [2, 32] -> [2, 32]>
#transform_map17 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [2, 32] -> [32, 2]>
#transform_map18 = #rock.transform_map<#map10 by [<Merge{4, 1, 8} ["item"] at [0] -> ["rep_i", "rep_j", "item_i"] at [0, 1, 2]>] bounds = [32] -> [4, 1, 8]>
#transform_map19 = #rock.transform_map<#map11 by [<Unmerge{4, 8} ["rep_i", "item_i"] at [0, 2] -> ["gemmBlockM"] at [0]>, <Unmerge{1} ["rep_j"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [4, 1, 8] -> [32, 1]>
#transform_map20 = #rock.transform_map<#map6 by [<PassThrough ["gemmBlockM"] at [0] -> ["gemmBlockM"] at [0]>, <PassThrough ["gemmBlockN"] at [1] -> ["gemmBlockN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map21 = #rock.transform_map<#map6 by [<Unmerge{32} ["gemmBlockM"] at [0] -> ["gemmM"] at [0]>, <Unmerge{1} ["gemmBlockN"] at [1] -> ["gemmN"] at [1]>] bounds = [32, 1] -> [32, 1]>
#transform_map22 = #rock.transform_map<#map7 by [<PassThrough ["dim1", "dim0"] at [1, 0] -> ["dim1", "dim0"] at [0, 1]>] bounds = [32, 1] -> [1, 32]>

// gfx942 (CDNA3 wave64): NR-Small extraOut fallback
// Same single-wave LDS-skip-eligible config as test_dsbpermute_nrsmall_ldsskip_sum_gfx942,
// but with extraOut present. The extraOut path reads from LDS, so the LDS-skip
// optimisation is disabled and the op falls back to the LDS round-trip with barriers.

// CHECK-LABEL: func @test_dsbpermute_extraout_fallback_gfx942

// Cross-lane reduction is still used (ds_bpermute)
// CHECK: rocdl.ds_bpermute

// LDS barriers ARE present (extraOut disables LDS-skip)
// CHECK: rock.lds_barrier

// Two threadwise_read_into: one for output, one for extraOut
// CHECK: rock.threadwise_read_into
// CHECK: rock.threadwise_read_into
// CHECK: return

func.func @test_dsbpermute_extraout_fallback_gfx942(
    %input_reg : memref<32xf32, #gpu.address_space<private>>,
    %output_reg : memref<32xf32, #gpu.address_space<private>>,
    %extra_out : memref<32xf32, #gpu.address_space<private>>,
    %ws_lds : memref<64xf32, #gpu.address_space<workgroup>>)
    attributes{rock.arch = "amdgcn-amd-amdhsa:gfx942",
               block_size = 64 : i32, grid_size = 72 : i32, rock.kernel} {
  rock.blockwise_broadcast_reduce max [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] [#transform_map13, #transform_map14, #transform_map15, #transform_map15, #transform_map16, #transform_map17] [#transform_map18, #transform_map19, #transform_map20, #transform_map20, #transform_map21, #transform_map22] %input_reg into %output_reg, [#transform_map8, #transform_map9, #transform_map10, #transform_map10, #transform_map11, #transform_map12] %extra_out using %ws_lds {axis = 1 : index, blockSize = 64 : i32} : memref<32xf32, #gpu.address_space<private>> using memref<64xf32, #gpu.address_space<workgroup>> into memref<32xf32, #gpu.address_space<private>>, memref<32xf32, #gpu.address_space<private>>
  return
}
