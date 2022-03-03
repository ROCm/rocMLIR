#gemm_padding = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">
#transform_map0 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1 floordiv 9, (d1 mod 9) floordiv 3, d1 mod 3)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, #miopen.transform<Unfold{8, 3, 3} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 72, 128] -> [1, 128, 8, 3, 3]>
#transform_map1 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0 * 64 + d1 * 16 + d2)> by[#miopen.transform<Embed{64, 16, 1} ["dim0", "dim1", "dim2"] at [0, 1, 2] -> ["slice"] at [0]>] bounds = [1, 4, 16] -> [64]>
#transform_map2 = #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 64} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [64] -> [128]>
#transform_map3 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 900, d0, d1 floordiv 9, (d1 mod 9) floordiv 3, (d2 mod 900) floordiv 30, d1 mod 3, d2 mod 30)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, #miopen.transform<Merge{8, 3, 3} ["gemmK"] at [1] -> ["ci", "y", "x"] at [2, 3, 5]>, #miopen.transform<Merge{128, 30, 30} ["gemmN"] at [2] -> ["ni", "ho", "wo"] at [0, 4, 6]>] bounds = [1, 72, 115200] -> [128, 1, 8, 3, 30, 3, 30]>
#transform_map4 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by[#miopen.transform<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, #miopen.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, #miopen.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [128, 1, 8, 3, 30, 3, 30] -> [128, 1, 8, 32, 32]>
#transform_map5 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by[#miopen.transform<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, #miopen.transform<PassThrough ["gi"] at [1] -> ["gi"] at [1]>, #miopen.transform<PassThrough ["ci"] at [2] -> ["ci"] at [2]>, #miopen.transform<Pad{0, 0, 0, 0} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [128, 1, 8, 32, 32] -> [128, 1, 8, 32, 32]>
#transform_map6 = #miopen.transform_map<affine_map<(d0) -> (d0 + 64)> by[#miopen.transform<Slice{64, 128} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [64] -> [128]>
#transform_map7 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 4 + d1 * 4 + d2 * 4 + d3 * 4 + d4 * 4 + d5)> by[#miopen.transform<Embed{4, 4, 4, 4, 4, 1} ["G", "M0", "M1", "M2", "N0", "N1"] at [0, 1, 2, 3, 4, 5] -> ["raw"] at [0]>] bounds = [1, 8, 4, 4, 28800, 4] -> [4]>
#transform_map8 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 16 + d2 * 4 + d3, d4 * 4 + d5)> by[#miopen.transform<PassThrough ["G"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<Embed{16, 4, 1} ["M0", "M1", "M2"] at [1, 2, 3] -> ["gemmM"] at [1]>, #miopen.transform<Embed{4, 1} ["N0", "N1"] at [4, 5] -> ["gemmN"] at [2]>] bounds = [1, 8, 4, 4, 28800, 4] -> [1, 128, 115200]>
#transform_map9 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 900, d0, d1, (d2 mod 900) floordiv 30, d2 mod 30)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, #miopen.transform<Merge{128, 30, 30} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>] bounds = [1, 128, 115200] -> [128, 1, 128, 30, 30]>
module  {
  func @miopen_conv2d_i8(%arg0: memref<1x128x8x3x3xi8>, %arg1: memref<128x1x8x32x32xi8>, %arg2: memref<128x1x128x30x30xi32>) attributes {block_size = 64 : i32, grid_size = 57600 : i32} {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c57600 = arith.constant 57600 : index
    %cst = arith.constant dense<0> : vector<4xi32>
    %0 = miopen.workgroup_id : index
    %1 = miopen.workitem_id : index
    %2 = arith.divui %0, %c57600 : index
    %3 = arith.remui %0, %c57600 : index
    %4 = arith.remui %3, %c8 : index
    %5 = arith.divui %3, %c8 : index
    %6 = arith.muli %4, %c16 : index
    %7 = arith.muli %5, %c16 : index
    %8 = arith.remui %1, %c4 : index
    %9 = arith.divui %1, %c4 : index
    %10 = arith.addi %6, %9 : index
    %11 = arith.divui %1, %c16 : index
    %12 = arith.remui %1, %c16 : index
    %13 = arith.addi %7, %12 : index
    %14 = miopen.alloc() : memref<128xi8, 3>
    %15 = miopen.threadwise_load %arg0[%2, %8, %10] with [[#transform_map0]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x128x8x3x3xi8>, index, index, index -> i8
    miopen.threadwise_store %15 -> %14[%c0, %8, %9] with [[#transform_map1, #transform_map2]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8 -> memref<128xi8, 3>, index, index, index
    %16 = miopen.threadwise_load %arg1[%2, %11, %13] with [[#transform_map3, #transform_map4, #transform_map5]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x8x32x32xi8>, index, index, index -> i8
    miopen.threadwise_store %16 -> %14[%c0, %11, %12] with [[#transform_map1, #transform_map6]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8 -> memref<128xi8, 3>, index, index, index
    %17 = arith.divui %1, %c64 : index
    %18 = arith.muli %17, %c16 : index
    %19 = miopen.alloc() : memref<1xi8, 5>
    %20 = miopen.alloc() : memref<1xi8, 5>
    %21:3 = affine.for %arg3 = 0 to 17 iter_args(%arg4 = %8, %arg5 = %11, %arg6 = %cst) -> (index, index, vector<4xi32>) {
      %42 = arith.addi %arg4, %c4 : index
      %43 = miopen.threadwise_load %arg0[%2, %42, %10] with [[#transform_map0]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x128x8x3x3xi8>, index, index, index -> i8
      %44 = arith.addi %arg5, %c4 : index
      %45 = miopen.threadwise_load %arg1[%2, %44, %13] with [[#transform_map3, #transform_map4, #transform_map5]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x8x32x32xi8>, index, index, index -> i8
      miopen.lds_barrier
      %46 = miopen.xdlops_gemm_v2(%14, %14, %18, %c0, %19, %20, %arg6) {k = 4 : i32, kpack = 1 : i32, m = 16 : i32, m_per_wave = 16 : i32, n = 16 : i32, n_per_wave = 16 : i32, transforms = [[#transform_map2], [#transform_map6]]} : memref<128xi8, 3>, memref<128xi8, 3>, index, index, memref<1xi8, 5>, memref<1xi8, 5>, vector<4xi32> -> vector<4xi32>
      miopen.lds_barrier
      miopen.threadwise_store %43 -> %14[%c0, %8, %9] with [[#transform_map1, #transform_map2]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8 -> memref<128xi8, 3>, index, index, index
      miopen.threadwise_store %45 -> %14[%c0, %11, %12] with [[#transform_map1, #transform_map6]] {bounds = [1 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8 -> memref<128xi8, 3>, index, index, index
      affine.yield %42, %44, %46 : index, index, vector<4xi32>
    }
    miopen.lds_barrier
    %22 = miopen.xdlops_gemm_v2(%14, %14, %18, %c0, %19, %20, %21#2) {k = 4 : i32, kpack = 1 : i32, m = 16 : i32, m_per_wave = 16 : i32, n = 16 : i32, n_per_wave = 16 : i32, transforms = [[#transform_map2], [#transform_map6]]} : memref<128xi8, 3>, memref<128xi8, 3>, index, index, memref<1xi8, 5>, memref<1xi8, 5>, vector<4xi32> -> vector<4xi32>
    %23 = arith.remui %1, %c64 : index
    %24 = arith.divui %23, %c16 : index
    %25 = arith.remui %23, %c16 : index
    %26 = miopen.in_warp_transpose {inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32], size = 4 : i32} %22, %23 : vector<4xi32>, index
    %27 = arith.divui %25, %c4 : index
    %28 = arith.muli %27, %c4 : index
    %29 = arith.muli %24, %c4 : index
    %30 = arith.remui %25, %c4 : index
    %31 = arith.addi %29, %30 : index
    %32 = arith.muli %17, %c16 : index
    %33 = arith.addi %32, %31 : index
    %34 = arith.addi %6, %33 : index
    %35 = arith.addi %7, %28 : index
    %36 = arith.divui %34, %c16 : index
    %37 = arith.remui %34, %c16 : index
    %38 = arith.divui %37, %c4 : index
    %39 = arith.remui %34, %c4 : index
    %40 = arith.divui %35, %c4 : index
    %41 = arith.remui %35, %c4 : index
    miopen.threadwise_copy_v2 %26[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %36, %38, %39, %40, %41] with [[#transform_map7], [#transform_map8, #transform_map9]] {bounds = [1 : index, 1 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 0 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<4xi32>, index, index, index, index, index, index -> memref<128x1x128x30x30xi32>, index, index, index, index, index, index
    return
  }
}

