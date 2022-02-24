#gemm_padding = #miopen.padding_info<extraM = 0, extraK = 0, extraN = 0, bwdPaddingInfo = "NA">
#transform_map0 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["gemmM"] at [2]>, #miopen.transform<Unmerge{256, 4} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 256, 1024, 4] -> [1, 1024, 1024]>
#transform_map1 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1, 0, 0)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["g"] at [0]>, #miopen.transform<Unfold{1024, 1, 1} ["gemmK"] at [1] -> ["c", "y", "x"] at [2, 3, 4]>, #miopen.transform<PassThrough ["gemmM"] at [2] -> ["k"] at [1]>] bounds = [1, 1024, 1024] -> [1, 1024, 1024, 1, 1]>
#transform_map2 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 512 + d2 * 4 + d3)> by[#miopen.transform<Embed{2048, 512, 4, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 4, 128, 4] -> [2048]>
#transform_map3 = #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 2048} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [2048] -> [4096]>
#transform_map4 = #miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0, d1 * 4 + d3, d2)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<PassThrough ["gemmN"] at [2] -> ["gemmN"] at [2]>, #miopen.transform<Unmerge{256, 4} ["gemmK", "gemmKPack"] at [1, 3] -> ["gemmK"] at [1]>] bounds = [1, 256, 25088, 4] -> [1, 1024, 25088]>
#transform_map5 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, 0, (d2 mod 196) floordiv 14, 0, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["gi"] at [1]>, #miopen.transform<Merge{1024, 1, 1} ["gemmK"] at [1] -> ["ci", "y", "x"] at [2, 3, 5]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["ni", "ho", "wo"] at [0, 4, 6]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 1, 14, 1, 14]>
#transform_map6 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3 + d4, d5 + d6)> by[#miopen.transform<PassThrough ["ni", "gi", "ci"] at [0, 1, 2] -> ["ni", "gi", "ci"] at [0, 1, 2]>, #miopen.transform<Embed{1, 1} ["y", "ho"] at [3, 4] -> ["hipad"] at [3]>, #miopen.transform<Embed{1, 1} ["x", "wo"] at [5, 6] -> ["wipad"] at [4]>] bounds = [128, 1, 1024, 1, 14, 1, 14] -> [128, 1, 1024, 14, 14]>
#transform_map7 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)> by[#miopen.transform<PassThrough ["ni"] at [0] -> ["ni"] at [0]>, #miopen.transform<PassThrough ["gi"] at [1] -> ["gi"] at [1]>, #miopen.transform<PassThrough ["ci"] at [2] -> ["ci"] at [2]>, #miopen.transform<Pad{0, 0, 0, 0} ["hipad", "wipad"] at [3, 4] -> ["hi", "wi"] at [3, 4]>] bounds = [128, 1, 1024, 14, 14] -> [128, 1, 1024, 14, 14]>
#transform_map8 = #miopen.transform_map<affine_map<(d0) -> (d0 + 2048)> by[#miopen.transform<Slice{2048, 4096} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [2048] -> [4096]>
#transform_map9 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0 * 16 + d1 * 4 + d2 * 4 + d3 * 4 + d4 * 4 + d5)> by[#miopen.transform<Embed{16, 4, 4, 4, 4, 1} ["G", "M0", "M1", "M2", "N0", "N1"] at [0, 1, 2, 3, 4, 5] -> ["raw"] at [0]>] bounds = [1, 128, 2, 4, 6272, 4] -> [16]>
#transform_map10 = #miopen.transform_map<affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 8 + d2 * 4 + d3, d4 * 4 + d5)> by[#miopen.transform<PassThrough ["G"] at [0] -> ["gemmG"] at [0]>, #miopen.transform<Embed{8, 4, 1} ["M0", "M1", "M2"] at [1, 2, 3] -> ["gemmM"] at [1]>, #miopen.transform<Embed{4, 1} ["N0", "N1"] at [4, 5] -> ["gemmN"] at [2]>] bounds = [1, 128, 2, 4, 6272, 4] -> [1, 1024, 25088]>
#transform_map11 = #miopen.transform_map<affine_map<(d0, d1, d2) -> (d2 floordiv 196, d0, d1, (d2 mod 196) floordiv 14, d2 mod 14)> by[#miopen.transform<PassThrough ["gemmG"] at [0] -> ["go"] at [1]>, #miopen.transform<PassThrough ["gemmM"] at [1] -> ["ko"] at [2]>, #miopen.transform<Merge{128, 14, 14} ["gemmN"] at [2] -> ["no", "ho", "wo"] at [0, 3, 4]>] bounds = [1, 1024, 25088] -> [128, 1, 1024, 14, 14]>
module  {
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x1024x1024x1x1xf32>, %arg1: memref<128x1x1024x14x14xf32>, %arg2: memref<128x1x1024x14x14xf32>) attributes {block_size = 256 : i32, grid_size = 1568 : i32, kernel = 0 : i32} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %c128 = arith.constant 128 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c1568 = arith.constant 1568 : index
    %cst = arith.constant dense<0.000000e+00> : vector<32xf32>
    %c32 = arith.constant 32 : index
    %0 = miopen.workgroup_id : index
    %1 = miopen.workitem_id : index
    %2 = arith.divui %0, %c1568 : index
    %3 = arith.remui %0, %c1568 : index
    %4 = arith.remui %3, %c8 : index
    %5 = arith.divui %3, %c8 : index
    %6 = arith.muli %4, %c128 : index
    %7 = arith.muli %5, %c128 : index
    %8 = arith.remui %1, %c4 : index
    %9 = arith.remui %8, %c2 : index
    %10 = arith.divui %1, %c4 : index
    %11 = arith.remui %1, %c4 : index
    %12 = arith.divui %11, %c2 : index
    %13 = arith.muli %9, %c2 : index
    %14 = arith.muli %10, %c2 : index
    %15 = arith.muli %12, %c2 : index
    %16 = arith.addi %6, %14 : index
    %17 = arith.divui %1, %c64 : index
    %18 = arith.remui %1, %c64 : index
    %19 = arith.muli %18, %c2 : index
    %20 = arith.addi %7, %19 : index
    %21 = miopen.alloc() : memref<4096xf32, 3>
    %22:8 = miopen.threadwise_load %arg0[%2, %13, %16, %15] with [[#transform_map0, #transform_map1]] {bounds = [1 : index, 2 : index, 2 : index, 2 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 2 : i32, vector_read_write_dim = 3 : i32} : memref<1x1024x1024x1x1xf32>, index, index, index, index -> f32, f32, f32, f32, f32, f32, f32, f32
    miopen.threadwise_store %22#0, %22#1, %22#2, %22#3, %22#4, %22#5, %22#6, %22#7 -> %21[%c0, %13, %14, %15] with [[#transform_map2, #transform_map3]] {bounds = [1 : index, 2 : index, 2 : index, 2 : index], dest_data_per_write = 1 : i32, source_data_per_read = 2 : i32, vector_read_write_dim = 3 : i32} : f32, f32, f32, f32, f32, f32, f32, f32 -> memref<4096xf32, 3>, index, index, index, index
    %23:8 = miopen.threadwise_load %arg1[%2, %17, %20, %c0] with [[#transform_map4, #transform_map5, #transform_map6, #transform_map7]] {bounds = [1 : index, 1 : index, 2 : index, 4 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 2 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x1024x14x14xf32>, index, index, index, index -> f32, f32, f32, f32, f32, f32, f32, f32
    miopen.threadwise_store %23#0, %23#1, %23#2, %23#3, %23#4, %23#5, %23#6, %23#7 -> %21[%c0, %17, %19, %c0] with [[#transform_map2, #transform_map8]] {bounds = [1 : index, 1 : index, 2 : index, 4 : index], dest_data_per_write = 1 : i32, source_data_per_read = 2 : i32, vector_read_write_dim = 2 : i32} : f32, f32, f32, f32, f32, f32, f32, f32 -> memref<4096xf32, 3>, index, index, index, index
    %24 = arith.divui %1, %c64 : index
    %25 = arith.divui %24, %c2 : index
    %26 = arith.remui %24, %c2 : index
    %27 = arith.muli %25, %c64 : index
    %28 = arith.muli %26, %c64 : index
    %29 = miopen.alloc() : memref<4xvector<4xf32>, 5>
    %30 = miopen.alloc() : memref<4xvector<4xf32>, 5>
    %31:4 = affine.for %arg3 = 0 to 63 iter_args(%arg4 = %13, %arg5 = %17, %arg6 = %cst, %arg7 = %cst) -> (index, index, vector<32xf32>, vector<32xf32>) {
      %118 = arith.addi %arg4, %c4 : index
      %119:8 = miopen.threadwise_load %arg0[%2, %118, %16, %15] with [[#transform_map0, #transform_map1]] {bounds = [1 : index, 2 : index, 2 : index, 2 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 2 : i32, vector_read_write_dim = 3 : i32} : memref<1x1024x1024x1x1xf32>, index, index, index, index -> f32, f32, f32, f32, f32, f32, f32, f32
      %120 = arith.addi %arg5, %c4 : index
      %121:8 = miopen.threadwise_load %arg1[%2, %120, %20, %c0] with [[#transform_map4, #transform_map5, #transform_map6, #transform_map7]] {bounds = [1 : index, 1 : index, 2 : index, 4 : index], dest_data_per_write = 1 : i32, oobDims = [false, false, false, false, false], paddingInfo = #gemm_padding, source_data_per_read = 2 : i32, vector_read_write_dim = 2 : i32} : memref<128x1x1024x14x14xf32>, index, index, index, index -> f32, f32, f32, f32, f32, f32, f32, f32
      miopen.lds_barrier
      %122:2 = miopen.xdlops_gemm_v2(%21, %21, %27, %28, %29, %30, %arg6, %arg7) {k = 4 : i32, kpack = 4 : i32, m = 128 : i32, m_per_wave = 64 : i32, n = 128 : i32, n_per_wave = 64 : i32, transforms = [[#transform_map3], [#transform_map8]]} : memref<4096xf32, 3>, memref<4096xf32, 3>, index, index, memref<4xvector<4xf32>, 5>, memref<4xvector<4xf32>, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
      miopen.lds_barrier
      miopen.threadwise_store %119#0, %119#1, %119#2, %119#3, %119#4, %119#5, %119#6, %119#7 -> %21[%c0, %13, %14, %15] with [[#transform_map2, #transform_map3]] {bounds = [1 : index, 2 : index, 2 : index, 2 : index], dest_data_per_write = 1 : i32, source_data_per_read = 2 : i32, vector_read_write_dim = 3 : i32} : f32, f32, f32, f32, f32, f32, f32, f32 -> memref<4096xf32, 3>, index, index, index, index
      miopen.threadwise_store %121#0, %121#1, %121#2, %121#3, %121#4, %121#5, %121#6, %121#7 -> %21[%c0, %17, %19, %c0] with [[#transform_map2, #transform_map8]] {bounds = [1 : index, 1 : index, 2 : index, 4 : index], dest_data_per_write = 1 : i32, source_data_per_read = 2 : i32, vector_read_write_dim = 2 : i32} : f32, f32, f32, f32, f32, f32, f32, f32 -> memref<4096xf32, 3>, index, index, index, index
      affine.yield %118, %120, %122#0, %122#1 : index, index, vector<32xf32>, vector<32xf32>
    }
    miopen.lds_barrier
    %32:2 = miopen.xdlops_gemm_v2(%21, %21, %27, %28, %29, %30, %31#2, %31#3) {k = 4 : i32, kpack = 4 : i32, m = 128 : i32, m_per_wave = 64 : i32, n = 128 : i32, n_per_wave = 64 : i32, transforms = [[#transform_map3], [#transform_map8]]} : memref<4096xf32, 3>, memref<4096xf32, 3>, index, index, memref<4xvector<4xf32>, 5>, memref<4xvector<4xf32>, 5>, vector<32xf32>, vector<32xf32> -> vector<32xf32>, vector<32xf32>
    %33 = arith.remui %1, %c64 : index
    %34 = arith.divui %33, %c32 : index
    %35 = arith.remui %33, %c32 : index
    %36 = miopen.in_warp_transpose {inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32], size = 4 : i32} %32#0, %33 : vector<32xf32>, index
    %37 = miopen.in_warp_transpose {inGroupPerm = [0 : i32, 1 : i32, 2 : i32, 3 : i32], size = 4 : i32} %32#1, %33 : vector<32xf32>, index
    %38 = arith.divui %35, %c4 : index
    %39 = arith.muli %38, %c4 : index
    %40 = arith.muli %34, %c4 : index
    %41 = arith.remui %35, %c4 : index
    %42 = arith.addi %40, %41 : index
    %43 = arith.remui %24, %c2 : index
    %44 = arith.muli %43, %c64 : index
    %45 = arith.addi %44, %39 : index
    %46 = arith.divui %24, %c2 : index
    %47 = arith.muli %46, %c64 : index
    %48 = arith.addi %47, %42 : index
    %49 = arith.addi %6, %48 : index
    %50 = arith.addi %7, %45 : index
    %51 = arith.divui %49, %c8 : index
    %52 = arith.remui %49, %c8 : index
    %53 = arith.divui %52, %c4 : index
    %54 = arith.remui %49, %c4 : index
    %55 = arith.divui %50, %c4 : index
    %56 = arith.remui %50, %c4 : index
    miopen.threadwise_copy_v2 %36[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %51, %53, %54, %55, %56] with [[#transform_map9], [#transform_map10, #transform_map11]] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 0 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<32xf32>, index, index, index, index, index, index -> memref<128x1x1024x14x14xf32>, index, index, index, index, index, index
    %57 = arith.divui %35, %c4 : index
    %58 = arith.muli %57, %c4 : index
    %59 = arith.addi %58, %c32 : index
    %60 = arith.muli %34, %c4 : index
    %61 = arith.remui %35, %c4 : index
    %62 = arith.addi %60, %61 : index
    %63 = arith.remui %24, %c2 : index
    %64 = arith.muli %63, %c64 : index
    %65 = arith.addi %64, %59 : index
    %66 = arith.divui %24, %c2 : index
    %67 = arith.muli %66, %c64 : index
    %68 = arith.addi %67, %62 : index
    %69 = arith.addi %6, %68 : index
    %70 = arith.addi %7, %65 : index
    %71 = arith.divui %69, %c8 : index
    %72 = arith.remui %69, %c8 : index
    %73 = arith.divui %72, %c4 : index
    %74 = arith.remui %69, %c4 : index
    %75 = arith.divui %70, %c4 : index
    %76 = arith.remui %70, %c4 : index
    miopen.threadwise_copy_v2 %36[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %71, %73, %74, %75, %76] with [[#transform_map9], [#transform_map10, #transform_map11]] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 16 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<32xf32>, index, index, index, index, index, index -> memref<128x1x1024x14x14xf32>, index, index, index, index, index, index
    %77 = arith.divui %35, %c4 : index
    %78 = arith.muli %77, %c4 : index
    %79 = arith.muli %34, %c4 : index
    %80 = arith.remui %35, %c4 : index
    %81 = arith.addi %79, %80 : index
    %82 = arith.addi %81, %c32 : index
    %83 = arith.remui %24, %c2 : index
    %84 = arith.muli %83, %c64 : index
    %85 = arith.addi %84, %78 : index
    %86 = arith.divui %24, %c2 : index
    %87 = arith.muli %86, %c64 : index
    %88 = arith.addi %87, %82 : index
    %89 = arith.addi %6, %88 : index
    %90 = arith.addi %7, %85 : index
    %91 = arith.divui %89, %c8 : index
    %92 = arith.remui %89, %c8 : index
    %93 = arith.divui %92, %c4 : index
    %94 = arith.remui %89, %c4 : index
    %95 = arith.divui %90, %c4 : index
    %96 = arith.remui %90, %c4 : index
    miopen.threadwise_copy_v2 %37[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %91, %93, %94, %95, %96] with [[#transform_map9], [#transform_map10, #transform_map11]] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 0 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<32xf32>, index, index, index, index, index, index -> memref<128x1x1024x14x14xf32>, index, index, index, index, index, index
    %97 = arith.divui %35, %c4 : index
    %98 = arith.muli %97, %c4 : index
    %99 = arith.addi %98, %c32 : index
    %100 = arith.muli %34, %c4 : index
    %101 = arith.remui %35, %c4 : index
    %102 = arith.addi %100, %101 : index
    %103 = arith.addi %102, %c32 : index
    %104 = arith.remui %24, %c2 : index
    %105 = arith.muli %104, %c64 : index
    %106 = arith.addi %105, %99 : index
    %107 = arith.divui %24, %c2 : index
    %108 = arith.muli %107, %c64 : index
    %109 = arith.addi %108, %103 : index
    %110 = arith.addi %6, %109 : index
    %111 = arith.addi %7, %106 : index
    %112 = arith.divui %110, %c8 : index
    %113 = arith.remui %110, %c8 : index
    %114 = arith.divui %113, %c4 : index
    %115 = arith.remui %110, %c4 : index
    %116 = arith.divui %111, %c4 : index
    %117 = arith.remui %111, %c4 : index
    miopen.threadwise_copy_v2 %37[%c0, %c0, %c0, %c0, %c0, %c0] -> %arg2[%2, %112, %114, %115, %116, %117] with [[#transform_map9], [#transform_map10, #transform_map11]] {bounds = [1 : index, 4 : index, 1 : index, 1 : index, 1 : index, 4 : index], dataOperation = 0 : i32, destOobDims = [false, false, false, false, false], dest_data_per_write = 4 : i32, paddingInfo = #gemm_padding, sourceOffset = 16 : index, source_data_per_read = 4 : i32, upper_vector_read_dim = 5 : i32, vector_read_write_dim = 4 : i32} : vector<32xf32>, index, index, index, index, index, index -> memref<128x1x1024x14x14xf32>, index, index, index, index, index, index
    return
  }
}

