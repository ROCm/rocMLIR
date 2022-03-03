#gemm_padding = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">
#transform_map0 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 16 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["gemmM"] at [2]>, #miopen.transform<Unmerge{64, 16} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 64, 1024, 16] -> [1, 1024, 1024]>
#transform_map1 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, #miopen.transform<Unfold{1024, 1, 1} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 1024, 1024] -> [1, 1024, 1024, 1, 1]>
#transform_map2 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{16384, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 16, 64, 16] -> [16384]>
#transform_map3 = #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 16384} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [16384] -> [32768]>
#transform_map4 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 16 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>, #miopen.transform<Unmerge{64, 16} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 64, 25088, 16] -> [1, 1024, 25088]>
#transform_map5 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, 0, (d2 mod 196) floordiv 14, 0, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, #miopen.transform<Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ci", "y", "x"] at [2, 3, 5]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "ho", "wo"] at [0, 4, 6]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 1, 14, 1, 14]>
#transform_map6 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by[#miopen.transform<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, #miopen.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, #miopen.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [128, 1, 1024, 1, 14, 1, 14] -> [128, 1, 1024, 14, 14]>
#transform_map7 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by[#miopen.transform<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, #miopen.transform<PassThrough ["gi"] at [1] -> ["gi"] at [1]>, #miopen.transform<PassThrough ["ci"] at [2] -> ["ci"] at [2]>, #miopen.transform<Pad{0, 0, 0, 0} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [128, 1, 1024, 14, 14] -> [128, 1, 1024, 14, 14]>
#transform_map8 = #miopen.transform_map<affine_map<(d0) -> (d0 + 16384)> by[#miopen.transform<Slice{16384, 32768} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [16384] -> [32768]>
#transform_map9 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 16 + d1 * 4 + d2 * 4 + d3 * 4 + d4 * 4 + d5)> by[#miopen.transform<Embed{16, 4, 4, 4, 4, 1} ["G", "M0", "M1", "M2", "N0", "N1"] at [0, 1, 2, 3, 4, 5] -> ["raw"] at [0]>] bounds = [1, 128, 2, 4, 6272, 4] -> [16]>
#transform_map10 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 8 + d2 * 4 + d3, d4 * 4 + d5)> by[#miopen.transform<PassThrough ["G"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<Embed{8, 4, 1} ["M0", "M1", "M2"] at [1, 2, 3] -> ["gemmM"] at [1]>, #miopen.transform<Embed{4, 1} ["N0", "N1"] at [4, 5] -> ["gemmN"] at [2]>] bounds = [1, 128, 2, 4, 6272, 4] -> [1, 1024, 25088]>
#transform_map11 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 14, 14]>
module {
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x1024x1024x1x1xi8>, %arg1: memref<128x1x1024x14x14xi8>, %arg2: memref<128x1x1024x14x14xi32>) attributes {block_size = 256 : i32, grid_size = 6272 : i32, kernel = 0 : i32} {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c6272 = arith.constant 6272 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0> : vector<16xi32>
    %c8 = arith.constant 8 : index
    %0 = miopen.workgroup_id : index
    %1 = miopen.workitem_id : index
    %2 = arith.divui %0, %c6272 : index
    %3 = arith.remui %0, %c6272 : index
    %4 = arith.remui %3, %c16 : index
    %5 = arith.divui %3, %c16 : index
    %6 = arith.muli %4, %c64 : index
    %7 = arith.muli %5, %c64 : index
    %8 = arith.divui %1, %c16 : index
    %9 = arith.remui %1, %c16 : index
    %10 = arith.muli %8, %c4 : index
    %11 = arith.addi %6, %10 : index
    %12 = arith.divui %1, %c64 : index
    %13 = arith.remui %1, %c64 : index
    %14 = arith.muli %12, %c4 : index
    %15 = arith.addi %7, %13 : index
    %16 = miopen.alloc() : memref<32768xi8, 3>
    %17:64 = miopen.threadwise_load %arg0[%2, %c0, %11, %9] with [[#transform_map0, #transform_map1]] {bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x1024x1024x1x1xi8>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
    miopen.threadwise_store %17#0, %17#1, %17#2, %17#3, %17#4, %17#5, %17#6, %17#7, %17#8, %17#9, %17#10, %17#11, %17#12, %17#13, %17#14, %17#15, %17#16, %17#17, %17#18, %17#19, %17#20, %17#21, %17#22, %17#23, %17#24, %17#25, %17#26, %17#27, %17#28, %17#29, %17#30, %17#31, %17#32, %17#33, %17#34, %17#35, %17#36, %17#37, %17#38, %17#39, %17#40, %17#41, %17#42, %17#43, %17#44, %17#45, %17#46, %17#47, %17#48, %17#49, %17#50, %17#51, %17#52, %17#53, %17#54, %17#55, %17#56, %17#57, %17#58, %17#59, %17#60, %17#61, %17#62, %17#63 -> %16[%c0, %c0, %10, %9] with [[#transform_map2, #transform_map3]] {bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<32768xi8, 3>, index, index, index, index
    %18:64 = miopen.threadwise_load %arg1[%2, %14, %15, %c0] with [[#transform_map4, #transform_map5, #transform_map6, #transform_map7]] {bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x1024x14x14xi8>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
    miopen.threadwise_store %18#0, %18#1, %18#2, %18#3, %18#4, %18#5, %18#6, %18#7, %18#8, %18#9, %18#10, %18#11, %18#12, %18#13, %18#14, %18#15, %18#16, %18#17, %18#18, %18#19, %18#20, %18#21, %18#22, %18#23, %18#24, %18#25, %18#26, %18#27, %18#28, %18#29, %18#30, %18#31, %18#32, %18#33, %18#34, %18#35, %18#36, %18#37, %18#38, %18#39, %18#40, %18#41, %18#42, %18#43, %18#44, %18#45, %18#46, %18#47, %18#48, %18#49, %18#50, %18#51, %18#52, %18#53, %18#54, %18#55, %18#56, %18#57, %18#58, %18#59, %18#60, %18#61, %18#62, %18#63 -> %16[%c0, %14, %13, %c0] with [[#transform_map2, #transform_map8]] {bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<32768xi8, 3>, index, index, index, index
    %19 = arith.divui %1, %c64 : index
    %20 = arith.divui %19, %c2 : index
    %21 = arith.remui %19, %c2 : index
    %22 = arith.muli %20, %c32 : index
    %23 = arith.muli %21, %c32 : index
    %24 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %25 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %26:3 = affine.for %arg3 = 0 to 3 iter_args(%arg4 = %c0, %arg5 = %14, %arg6 = %cst) -> (index, index, vector<16xi32>) {
      %51 = arith.addi %arg4, %c16 : index
      %52:64 = miopen.threadwise_load %arg0[%2, %51, %11, %9] with [[#transform_map0, #transform_map1]] {bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x1024x1024x1x1xi8>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
      %53 = arith.addi %arg5, %c16 : index
      %54:64 = miopen.threadwise_load %arg1[%2, %53, %15, %c0] with [[#transform_map4, #transform_map5, #transform_map6, #transform_map7]] {bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x1024x14x14xi8>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
      miopen.lds_barrier
      %55 = miopen.xdlops_gemm_v2(%16, %16, %22, %23, %24, %25, %arg6) {k = 16 : i32, kpack = 16 : i32, m = 64 : i32, m_per_wave = 32 : i32, n = 64 : i32, n_per_wave = 32 : i32, transforms = [[#transform_map3], [#transform_map8]]} : memref<32768xi8, 3>, memref<32768xi8, 3>, index, index, memref<8xvector<16xi8>, 5>, memref<8xvector<16xi8>, 5>, vector<16xi32> -> vector<16xi32>
      miopen.lds_barrier
      miopen.threadwise_store %52#0, %52#1, %52#2, %52#3, %52#4, %52#5, %52#6, %52#7, %52#8, %52#9, %52#10, %52#11, %52#12, %52#13, %52#14, %52#15, %52#16, %52#17, %52#18, %52#19, %52#20, %52#21, %52#22, %52#23, %52#24, %52#25, %52#26, %52#27, %52#28, %52#29, %52#30, %52#31, %52#32, %52#33, %52#34, %52#35, %52#36, %52#37, %52#38, %52#39, %52#40, %52#41, %52#42, %52#43, %52#44, %52#45, %52#46, %52#47, %52#48, %52#49, %52#50, %52#51, %52#52, %52#53, %52#54, %52#55, %52#56, %52#57, %52#58, %52#59, %52#60, %52#61, %52#62, %52#63 -> %16[%c0, %c0, %10, %9] with [[#transform_map2, #transform_map3]] {bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<32768xi8, 3>, index, index, index, index
      miopen.threadwise_store %54#0, %54#1, %54#2, %54#3, %54#4, %54#5, %54#6, %54#7, %54#8, %54#9, %54#10, %54#11, %54#12, %54#13, %54#14, %54#15, %54#16, %54#17, %54#18, %54#19, %54#20, %54#21, %54#22, %54#23, %54#24, %54#25, %54#26, %54#27, %54#28, %54#29, %54#30, %54#31, %54#32, %54#33, %54#34, %54#35, %54#36, %54#37, %54#38, %54#39, %54#40, %54#41, %54#42, %54#43, %54#44, %54#45, %54#46, %54#47, %54#48, %54#49, %54#50, %54#51, %54#52, %54#53, %54#54, %54#55, %54#56, %54#57, %54#58, %54#59, %54#60, %54#61, %54#62, %54#63 -> %16[%c0, %14, %13, %c0] with [[#transform_map2, #transform_map8]] {bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<32768xi8, 3>, index, index, index, index
      affine.yield %51, %53, %55 : index, index, vector<16xi32>
    }
    miopen.lds_barrier
    %27 = miopen.xdlops_gemm_v2(%16, %16, %22, %23, %24, %25, %26#2) {k = 16 : i32, kpack = 16 : i32, m = 64 : i32, m_per_wave = 32 : i32, n = 64 : i32, n_per_wave = 32 : i32, transforms = [[#transform_map3], [#transform_map8]]} : memref<32768xi8, 3>, memref<32768xi8, 3>, index, index, memref<8xvector<16xi8>, 5>, memref<8xvector<16xi8>, 5>, vector<16xi32> -> vector<16xi32>
    %28 = arith.remui %1, %c64 : index
    %29 = arith.divui %28, %c32 : index
    %30 = arith.remui %28, %c32 : index
    %31 = miopen.in_warp_transpose {inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32], size = 4 : i32} %27, %28 : vector<16xi32>, index
    %32 = arith.divui %30, %c4 : index
    %33 = arith.muli %32, %c4 : index
    %34 = arith.muli %29, %c4 : index
    %35 = arith.remui %30, %c4 : index
    %36 = arith.addi %34, %35 : index
    %37 = arith.remui %19, %c2 : index
    %38 = arith.muli %37, %c32 : index
    %39 = arith.addi %38, %33 : index
    %40 = arith.divui %19, %c2 : index
    %41 = arith.muli %40, %c32 : index
    %42 = arith.addi %41, %36 : index
    %43 = arith.addi %6, %42 : index
    %44 = arith.addi %7, %39 : index
    %45 = arith.divui %43, %c8 : index
    %46 = arith.remui %43, %c8 : index
    %47 = arith.divui %46, %c4 : index
    %48 = arith.remui %43, %c4 : index
    %49 = arith.divui %44, %c4 : index
    %50 = arith.remui %44, %c4 : index
    miopen.threadwise_copy_v2 %31[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %45, %47, %48, %49, %50] with [[#transform_map9], [#transform_map10, #transform_map11]] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 0 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<16xi32>, index, index, index, index, index, index -> memref<128x1x1024x14x14xi32>, index, index, index, index, index, index
    return
  }
}

