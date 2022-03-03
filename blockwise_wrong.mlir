#gemm_padding = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">
#map0 = affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1 * 16 + d3, 0, 0)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)>
#map3 = affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, d2 mod 14)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d2 floordiv 196, d0, d1 * 16 + d3, (d2 mod 196) floordiv 14, d2 mod 14)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 1024 + d2 * 16 + d3)>
#map6 = affine_map<(d0) -> (d0 + 16384)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 1024 + d2 * 16 + d3 + 16384)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> ((d4 * 4 + d5) floordiv 196, d0, d1 * 8 + d2 * 4 + d3, ((d4 * 4 + d5) mod 196) floordiv 14, (d4 * 4 + d5) mod 14)>
#transform_map0 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, #miopen.transform<Unfold{1024, 1, 1} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 1024, 1024] -> [1, 1024, 1024, 1, 1]>
#transform_map1 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 16 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["gemmM"] at [2]>, #miopen.transform<Unmerge{64, 16} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 64, 1024, 16] -> [1, 1024, 1024]>
#transform_map2 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by[#miopen.transform<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, #miopen.transform<PassThrough ["gi"] at [1] -> ["gi"] at [1]>, #miopen.transform<PassThrough ["ci"] at [2] -> ["ci"] at [2]>, #miopen.transform<Pad{0, 0, 0, 0} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [128, 1, 1024, 14, 14] -> [128, 1, 1024, 14, 14]>
#transform_map3 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by[#miopen.transform<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, #miopen.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, #miopen.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [128, 1, 1024, 1, 14, 1, 14] -> [128, 1, 1024, 14, 14]>
#transform_map4 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, 0, (d2 mod 196) floordiv 14, 0, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, #miopen.transform<Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ci", "y", "x"] at [2, 3, 5]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "ho", "wo"] at [0, 4, 6]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 1, 14, 1, 14]>
#transform_map5 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 16 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>, #miopen.transform<Unmerge{64, 16} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 64, 25088, 16] -> [1, 1024, 25088]>
#transform_map6 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 14, 14]>
#transform_map7 = #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 16384} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [16384] -> [32768]>
#transform_map8 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{16384, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 16, 64, 16] -> [16384]>
#transform_map9 = #miopen.transform_map<affine_map<(d0) -> (d0 + 16384)> by[#miopen.transform<Slice{16384, 32768} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [16384] -> [32768]>
#transform_map10 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 8 + d2 * 4 + d3, d4 * 4 + d5)> by[#miopen.transform<PassThrough ["G"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<Embed{8, 4, 1} ["M0", "M1", "M2"] at [1, 2, 3] -> ["gemmM"] at [1]>, #miopen.transform<Embed{4, 1} ["N0", "N1"] at [4, 5] -> ["gemmN"] at [2]>] bounds = [1, 128, 2, 4, 6272, 4] -> [1, 1024, 25088]>
#transform_map11 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 16 + d1 * 4 + d2 * 4 + d3 * 4 + d4 * 4 + d5)> by[#miopen.transform<Embed{16, 4, 4, 4, 4, 1} ["G", "M0", "M1", "M2", "N0", "N1"] at [0, 1, 2, 3, 4, 5] -> ["raw"] at [0]>] bounds = [1, 128, 2, 4, 6272, 4] -> [16]>
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
    %0 = miopen.transform %arg0 by [#transform_map0] : memref<1x1024x1024x1x1xi8> to memref<1x1024x1024xi8, #map0>
    %1 = miopen.transform %0 by [#transform_map1] : memref<1x1024x1024xi8, #map0> to memref<1x64x1024x16xi8, #map1>
    %2 = miopen.transform %arg1 by [#transform_map2] : memref<128x1x1024x14x14xi8> to memref<128x1x1024x14x14xi8>
    %3 = miopen.transform %2 by [#transform_map3] : memref<128x1x1024x14x14xi8> to memref<128x1x1024x1x14x1x14xi8, #map2>
    %4 = miopen.transform %3 by [#transform_map4] : memref<128x1x1024x1x14x1x14xi8, #map2> to memref<1x1024x25088xi8, #map3>
    %5 = miopen.transform %4 by [#transform_map5] : memref<1x1024x25088xi8, #map3> to memref<1x64x25088x16xi8, #map4>
    %6 = miopen.transform %arg2 by [#transform_map6] : memref<128x1x1024x14x14xi32> to memref<1x1024x25088xi32, #map3>
    %7 = miopen.workgroup_id : index
    %8 = miopen.workitem_id : index
    %9 = arith.divui %7, %c6272 : index
    %10 = arith.remui %7, %c6272 : index
    %11 = arith.remui %10, %c16 : index
    %12 = arith.divui %10, %c16 : index
    %13 = arith.muli %11, %c64 : index
    %14 = arith.muli %12, %c64 : index
    %15 = arith.divui %8, %c16 : index
    %16 = arith.remui %8, %c16 : index
    %17 = arith.muli %15, %c4 : index
    %18 = arith.addi %13, %17 : index
    %19 = arith.divui %8, %c64 : index
    %20 = arith.remui %8, %c64 : index
    %21 = arith.muli %19, %c4 : index
    %22 = arith.addi %14, %20 : index
    %23 = miopen.alloc() : memref<32768xi8, 3>
    %24 = miopen.transform %23 by [#transform_map7] : memref<32768xi8, 3> to memref<16384xi8, 3>
    %25 = miopen.transform %24 by [#transform_map8] : memref<16384xi8, 3> to memref<1x16x64x16xi8, #map5, 3>
    %26 = miopen.transform %23 by [#transform_map9] : memref<32768xi8, 3> to memref<16384xi8, #map6, 3>
    %27 = miopen.transform %26 by [#transform_map8] : memref<16384xi8, #map6, 3> to memref<1x16x64x16xi8, #map7, 3>
    %28:64 = miopen.blockwise_load %1[%9, %c0, %18, %16] with [[]] {block_size = 256 : i32, bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, source_vector_read_dim = 1 : i32} : memref<1x64x1024x16xi8, #map1>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
    miopen.blockwise_store %28#0, %28#1, %28#2, %28#3, %28#4, %28#5, %28#6, %28#7, %28#8, %28#9, %28#10, %28#11, %28#12, %28#13, %28#14, %28#15, %28#16, %28#17, %28#18, %28#19, %28#20, %28#21, %28#22, %28#23, %28#24, %28#25, %28#26, %28#27, %28#28, %28#29, %28#30, %28#31, %28#32, %28#33, %28#34, %28#35, %28#36, %28#37, %28#38, %28#39, %28#40, %28#41, %28#42, %28#43, %28#44, %28#45, %28#46, %28#47, %28#48, %28#49, %28#50, %28#51, %28#52, %28#53, %28#54, %28#55, %28#56, %28#57, %28#58, %28#59, %28#60, %28#61, %28#62, %28#63 -> %25[%c0, %c0, %17, %16] with [[]] {block_size = 256 : i32, bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 1 : i32, source_data_per_read = 1 : i32, source_vector_read_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<1x16x64x16xi8, #map5, 3>, index, index, index, index
    %29:64 = miopen.blockwise_load %5[%9, %21, %22, %c0] with [[]] {block_size = 256 : i32, bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 2 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, source_vector_read_dim = 2 : i32} : memref<1x64x25088x16xi8, #map4>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
    miopen.blockwise_store %29#0, %29#1, %29#2, %29#3, %29#4, %29#5, %29#6, %29#7, %29#8, %29#9, %29#10, %29#11, %29#12, %29#13, %29#14, %29#15, %29#16, %29#17, %29#18, %29#19, %29#20, %29#21, %29#22, %29#23, %29#24, %29#25, %29#26, %29#27, %29#28, %29#29, %29#30, %29#31, %29#32, %29#33, %29#34, %29#35, %29#36, %29#37, %29#38, %29#39, %29#40, %29#41, %29#42, %29#43, %29#44, %29#45, %29#46, %29#47, %29#48, %29#49, %29#50, %29#51, %29#52, %29#53, %29#54, %29#55, %29#56, %29#57, %29#58, %29#59, %29#60, %29#61, %29#62, %29#63 -> %27[%c0, %21, %20, %c0] with [[]] {block_size = 256 : i32, bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 2 : i32, source_data_per_read = 1 : i32, source_vector_read_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<1x16x64x16xi8, #map7, 3>, index, index, index, index
    %30 = arith.divui %8, %c64 : index
    %31 = arith.divui %30, %c2 : index
    %32 = arith.remui %30, %c2 : index
    %33 = arith.muli %31, %c32 : index
    %34 = arith.muli %32, %c32 : index
    %35 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %36 = miopen.alloc() : memref<8xvector<16xi8>, 5>
    %37:3 = affine.for %arg3 = 0 to 3 iter_args(%arg4 = %c0, %arg5 = %21, %arg6 = %cst) -> (index, index, vector<16xi32>) {
      %63 = arith.addi %arg4, %c16 : index
      %64:64 = miopen.blockwise_load %1[%9, %63, %18, %16] with [[]] {block_size = 256 : i32, bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, source_vector_read_dim = 1 : i32} : memref<1x64x1024x16xi8, #map1>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
      %65 = arith.addi %arg5, %c16 : index
      %66:64 = miopen.blockwise_load %5[%9, %65, %22, %c0] with [[]] {block_size = 256 : i32, bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 2 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, source_vector_read_dim = 2 : i32} : memref<1x64x25088x16xi8, #map4>, index, index, index, index -> i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8
      miopen.lds_barrier
      %67 = miopen.blockwise_gemm_v2(%24, %26, %33, %34, %35, %36, %arg6) {block_size = 256 : i32, k = 16 : i32, kpack = 16 : i32, m = 64 : i32, m_per_wave = 32 : i32, m_waves = 2 : i32, n = 64 : i32, n_per_wave = 32 : i32, n_waves = 2 : i32, transforms = [[], []]} : memref<16384xi8, 3>, memref<16384xi8, #map6, 3>, index, index, memref<8xvector<16xi8>, 5>, memref<8xvector<16xi8>, 5>, vector<16xi32> -> vector<16xi32>
      miopen.lds_barrier
      miopen.blockwise_store %64#0, %64#1, %64#2, %64#3, %64#4, %64#5, %64#6, %64#7, %64#8, %64#9, %64#10, %64#11, %64#12, %64#13, %64#14, %64#15, %64#16, %64#17, %64#18, %64#19, %64#20, %64#21, %64#22, %64#23, %64#24, %64#25, %64#26, %64#27, %64#28, %64#29, %64#30, %64#31, %64#32, %64#33, %64#34, %64#35, %64#36, %64#37, %64#38, %64#39, %64#40, %64#41, %64#42, %64#43, %64#44, %64#45, %64#46, %64#47, %64#48, %64#49, %64#50, %64#51, %64#52, %64#53, %64#54, %64#55, %64#56, %64#57, %64#58, %64#59, %64#60, %64#61, %64#62, %64#63 -> %25[%c0, %c0, %17, %16] with [[]] {block_size = 256 : i32, bounds = [1 : index, 16 : index, 4 : index, 1 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 1 : i32, source_data_per_read = 1 : i32, source_vector_read_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<1x16x64x16xi8, #map5, 3>, index, index, index, index
      miopen.blockwise_store %66#0, %66#1, %66#2, %66#3, %66#4, %66#5, %66#6, %66#7, %66#8, %66#9, %66#10, %66#11, %66#12, %66#13, %66#14, %66#15, %66#16, %66#17, %66#18, %66#19, %66#20, %66#21, %66#22, %66#23, %66#24, %66#25, %66#26, %66#27, %66#28, %66#29, %66#30, %66#31, %66#32, %66#33, %66#34, %66#35, %66#36, %66#37, %66#38, %66#39, %66#40, %66#41, %66#42, %66#43, %66#44, %66#45, %66#46, %66#47, %66#48, %66#49, %66#50, %66#51, %66#52, %66#53, %66#54, %66#55, %66#56, %66#57, %66#58, %66#59, %66#60, %66#61, %66#62, %66#63 -> %27[%c0, %21, %20, %c0] with [[]] {block_size = 256 : i32, bounds = [1 : index, 4 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, dest_vector_write_dim = 2 : i32, source_data_per_read = 1 : i32, source_vector_read_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<1x16x64x16xi8, #map7, 3>, index, index, index, index
      affine.yield %63, %65, %67 : index, index, vector<16xi32>
    }
    miopen.lds_barrier
    %38 = miopen.blockwise_gemm_v2(%24, %26, %33, %34, %35, %36, %37#2) {block_size = 256 : i32, k = 16 : i32, kpack = 16 : i32, m = 64 : i32, m_per_wave = 32 : i32, m_waves = 2 : i32, n = 64 : i32, n_per_wave = 32 : i32, n_waves = 2 : i32, transforms = [[], []]} : memref<16384xi8, 3>, memref<16384xi8, #map6, 3>, index, index, memref<8xvector<16xi8>, 5>, memref<8xvector<16xi8>, 5>, vector<16xi32> -> vector<16xi32>
    %39 = arith.remui %8, %c64 : index
    %40 = arith.divui %39, %c32 : index
    %41 = arith.remui %39, %c32 : index
    %42 = miopen.in_warp_transpose {inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32], size = 4 : i32} %38, %39 : vector<16xi32>, index
    %43 = miopen.transform %6 by [#transform_map10] : memref<1x1024x25088xi32, #map3> to memref<1x128x2x4x6272x4xi32, #map8>
    %44 = arith.divui %41, %c4 : index
    %45 = arith.muli %44, %c4 : index
    %46 = arith.muli %40, %c4 : index
    %47 = arith.remui %41, %c4 : index
    %48 = arith.addi %46, %47 : index
    %49 = arith.remui %30, %c2 : index
    %50 = arith.muli %49, %c32 : index
    %51 = arith.addi %50, %45 : index
    %52 = arith.divui %30, %c2 : index
    %53 = arith.muli %52, %c32 : index
    %54 = arith.addi %53, %48 : index
    %55 = arith.addi %13, %54 : index
    %56 = arith.addi %14, %51 : index
    %57 = arith.divui %55, %c8 : index
    %58 = arith.remui %55, %c8 : index
    %59 = arith.divui %58, %c4 : index
    %60 = arith.remui %55, %c4 : index
    %61 = arith.divui %56, %c4 : index
    %62 = arith.remui %56, %c4 : index
    miopen.threadwise_copy_v2 %42[%c0, %c0, %c0, %c0, %c0, %c0] -> %43[%9, %57, %59, %60, %61, %62] with [[#transform_map11], []] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 0 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<16xi32>, index, index, index, index, index, index -> memref<1x128x2x4x6272x4xi32, #map8>, index, index, index, index, index, index
    return
  }
}

