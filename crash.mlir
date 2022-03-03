vectorizationSize: 16 dataPerThreadCopy: 1
vectorizationSize: 16 dataPerThreadCopy: 1

threadwise_store op:
miopen.threadwise_store %32#0, %32#1, %32#2, %32#3, %32#4, %32#5, %32#6, %32#7, %32#8, %32#9, %32#10, %32#11, %32#12, %32#13, %32#14, %32#15 -> %18[%c0, %15, %16, %c0] with [[#miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{4096, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 4, 64, 16] -> [4096]>, #miopen.transform_map<affine_map<(d0) -> (d0 + 4096)> by[#miopen.transform<Slice{4096, 8192} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [4096] -> [8192]>]] {bounds = [1 : index, 1 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<8192xi8, 3>, index, index, index, index

slice lengths: 1 1 1 16 
vector dim: 2
dest data per write: 1
modified lengths: 1 1 1 16 
IVs: 0 0 0 0 
computing lower index
bottom idxes: %40 = arith.addi %39, %c4096_2 : index
inputsIndex: 0
destination index:%40 = arith.addi %39, %c4096_2 : indexIVs: 0 0 0 1 
computing lower index
bottom idxes: %42 = arith.addi %40, %c1_21 : index
inputsIndex: 1
destination index:%42 = arith.addi %40, %c1_21 : indexIVs: 0 0 0 2 
computing lower index
bottom idxes: %44 = arith.addi %40, %c2_33 : index
inputsIndex: 2
destination index:%44 = arith.addi %40, %c2_33 : indexIVs: 0 0 0 3 
computing lower index
bottom idxes: %46 = arith.addi %40, %c3_44 : index
inputsIndex: 3
destination index:%46 = arith.addi %40, %c3_44 : indexIVs: 0 0 0 4 
computing lower index
bottom idxes: %48 = arith.addi %40, %c4_56 : index
inputsIndex: 4
destination index:%48 = arith.addi %40, %c4_56 : indexIVs: 0 0 0 5 
computing lower index
bottom idxes: %50 = arith.addi %40, %c5_67 : index
inputsIndex: 5
destination index:%50 = arith.addi %40, %c5_67 : indexIVs: 0 0 0 6 
computing lower index
bottom idxes: %52 = arith.addi %40, %c6_78 : index
inputsIndex: 6
destination index:%52 = arith.addi %40, %c6_78 : indexIVs: 0 0 0 7 
computing lower index
bottom idxes: %54 = arith.addi %40, %c7_89 : index
inputsIndex: 7
destination index:%54 = arith.addi %40, %c7_89 : indexIVs: 0 0 0 8 
computing lower index
bottom idxes: %56 = arith.addi %40, %c8_101 : index
inputsIndex: 8
destination index:%56 = arith.addi %40, %c8_101 : indexIVs: 0 0 0 9 
computing lower index
bottom idxes: %58 = arith.addi %40, %c9_112 : index
inputsIndex: 9
destination index:%58 = arith.addi %40, %c9_112 : indexIVs: 0 0 0 10 
computing lower index
bottom idxes: %60 = arith.addi %40, %c10_123 : index
inputsIndex: 10
destination index:%60 = arith.addi %40, %c10_123 : indexIVs: 0 0 0 11 
computing lower index
bottom idxes: %62 = arith.addi %40, %c11_134 : index
inputsIndex: 11
destination index:%62 = arith.addi %40, %c11_134 : indexIVs: 0 0 0 12 
computing lower index
bottom idxes: %64 = arith.addi %40, %c12_145 : index
inputsIndex: 12
destination index:%64 = arith.addi %40, %c12_145 : indexIVs: 0 0 0 13 
computing lower index
bottom idxes: %66 = arith.addi %40, %c13_156 : index
inputsIndex: 13
destination index:%66 = arith.addi %40, %c13_156 : indexIVs: 0 0 0 14 
computing lower index
bottom idxes: %68 = arith.addi %40, %c14_167 : index
inputsIndex: 14
destination index:%68 = arith.addi %40, %c14_167 : indexIVs: 0 0 0 15 
computing lower index
bottom idxes: %70 = arith.addi %40, %c15_178 : index
inputsIndex: 15
destination index:%70 = arith.addi %40, %c15_178 : index
threadwise_store op:
miopen.threadwise_store %30#0, %30#1, %30#2, %30#3, %30#4, %30#5, %30#6, %30#7, %30#8, %30#9, %30#10, %30#11, %30#12, %30#13, %30#14, %30#15 -> %18[%c0, %13, %10, %12] with [[#miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{4096, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 4, 64, 16] -> [4096]>, #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 4096} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [4096] -> [8192]>]] {bounds = [1 : index, 16 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<8192xi8, 3>, index, index, index, index

slice lengths: 1 16 1 1 
vector dim: 1
dest data per write: 1
modified lengths: 1 16 1 1 
IVs: 0 0 0 0 
computing lower index
bottom idxes: %39 = arith.addi %38, %12 : index
inputsIndex: 0
destination index:%39 = arith.addi %38, %12 : indexIVs: 0 1 0 0 
computing lower index
bottom idxes: %41 = arith.addi %39, %c1024_27 : index
inputsIndex: 1
destination index:%41 = arith.addi %39, %c1024_27 : indexIVs: 0 2 0 0 
computing lower index
bottom idxes: %43 = arith.addi %39, %c2048_38 : index
inputsIndex: 2
destination index:%43 = arith.addi %39, %c2048_38 : indexIVs: 0 3 0 0 
computing lower index
bottom idxes: %45 = arith.addi %39, %c3072_49 : index
inputsIndex: 3
destination index:%45 = arith.addi %39, %c3072_49 : indexIVs: 0 4 0 0 
computing lower index
bottom idxes: %47 = arith.addi %39, %c4096_61 : index
inputsIndex: 4
destination index:%47 = arith.addi %39, %c4096_61 : indexIVs: 0 5 0 0 
computing lower index
bottom idxes: %49 = arith.addi %39, %c5120_72 : index
inputsIndex: 5
destination index:%49 = arith.addi %39, %c5120_72 : indexIVs: 0 6 0 0 
computing lower index
bottom idxes: %51 = arith.addi %39, %c6144_83 : index
inputsIndex: 6
destination index:%51 = arith.addi %39, %c6144_83 : indexIVs: 0 7 0 0 
computing lower index
bottom idxes: %53 = arith.addi %39, %c7168_94 : index
inputsIndex: 7
destination index:%53 = arith.addi %39, %c7168_94 : indexIVs: 0 8 0 0 
computing lower index
bottom idxes: %55 = arith.addi %39, %c8192_105 : index
inputsIndex: 8
destination index:%55 = arith.addi %39, %c8192_105 : indexIVs: 0 9 0 0 
computing lower index
bottom idxes: %57 = arith.addi %39, %c9216_116 : index
inputsIndex: 9
destination index:%57 = arith.addi %39, %c9216_116 : indexIVs: 0 10 0 0 
computing lower index
bottom idxes: %59 = arith.addi %39, %c10240_127 : index
inputsIndex: 10
destination index:%59 = arith.addi %39, %c10240_127 : indexIVs: 0 11 0 0 
computing lower index
bottom idxes: %61 = arith.addi %39, %c11264_138 : index
inputsIndex: 11
destination index:%61 = arith.addi %39, %c11264_138 : indexIVs: 0 12 0 0 
computing lower index
bottom idxes: %63 = arith.addi %39, %c12288_149 : index
inputsIndex: 12
destination index:%63 = arith.addi %39, %c12288_149 : indexIVs: 0 13 0 0 
computing lower index
bottom idxes: %65 = arith.addi %39, %c13312_160 : index
inputsIndex: 13
destination index:%65 = arith.addi %39, %c13312_160 : indexIVs: 0 14 0 0 
computing lower index
bottom idxes: %67 = arith.addi %39, %c14336_171 : index
inputsIndex: 14
destination index:%67 = arith.addi %39, %c14336_171 : indexIVs: 0 15 0 0 
computing lower index
bottom idxes: %69 = arith.addi %39, %c15360_182 : index
inputsIndex: 15
destination index:%69 = arith.addi %39, %c15360_182 : index
threadwise_store op:
miopen.threadwise_store %20#0, %20#1, %20#2, %20#3, %20#4, %20#5, %20#6, %20#7, %20#8, %20#9, %20#10, %20#11, %20#12, %20#13, %20#14, %20#15 -> %18[%c0, %15, %16, %c0] with [[#miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{4096, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 4, 64, 16] -> [4096]>, #miopen.transform_map<affine_map<(d0) -> (d0 + 4096)> by[#miopen.transform<Slice{4096, 8192} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [4096] -> [8192]>]] {bounds = [1 : index, 1 : index, 1 : index, 16 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 2 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<8192xi8, 3>, index, index, index, index

slice lengths: 1 1 1 16 
vector dim: 2
dest data per write: 1
modified lengths: 1 1 1 16 
IVs: 0 0 0 0 
computing lower index
bottom idxes: %27 = arith.addi %26, %c4096_9 : index
inputsIndex: 0
destination index:%27 = arith.addi %26, %c4096_9 : indexIVs: 0 0 0 1 
computing lower index
bottom idxes: %29 = arith.addi %27, %c1_29 : index
inputsIndex: 1
destination index:%29 = arith.addi %27, %c1_29 : indexIVs: 0 0 0 2 
computing lower index
bottom idxes: %31 = arith.addi %27, %c2_41 : index
inputsIndex: 2
destination index:%31 = arith.addi %27, %c2_41 : indexIVs: 0 0 0 3 
computing lower index
bottom idxes: %33 = arith.addi %27, %c3_53 : index
inputsIndex: 3
destination index:%33 = arith.addi %27, %c3_53 : indexIVs: 0 0 0 4 
computing lower index
bottom idxes: %35 = arith.addi %27, %c4_65 : index
inputsIndex: 4
destination index:%35 = arith.addi %27, %c4_65 : indexIVs: 0 0 0 5 
computing lower index
bottom idxes: %37 = arith.addi %27, %c5_77 : index
inputsIndex: 5
destination index:%37 = arith.addi %27, %c5_77 : indexIVs: 0 0 0 6 
computing lower index
bottom idxes: %39 = arith.addi %27, %c6_89 : index
inputsIndex: 6
destination index:%39 = arith.addi %27, %c6_89 : indexIVs: 0 0 0 7 
computing lower index
bottom idxes: %41 = arith.addi %27, %c7_101 : index
inputsIndex: 7
destination index:%41 = arith.addi %27, %c7_101 : indexIVs: 0 0 0 8 
computing lower index
bottom idxes: %43 = arith.addi %27, %c8_113 : index
inputsIndex: 8
destination index:%43 = arith.addi %27, %c8_113 : indexIVs: 0 0 0 9 
computing lower index
bottom idxes: %45 = arith.addi %27, %c9_125 : index
inputsIndex: 9
destination index:%45 = arith.addi %27, %c9_125 : indexIVs: 0 0 0 10 
computing lower index
bottom idxes: %47 = arith.addi %27, %c10_137 : index
inputsIndex: 10
destination index:%47 = arith.addi %27, %c10_137 : indexIVs: 0 0 0 11 
computing lower index
bottom idxes: %49 = arith.addi %27, %c11_149 : index
inputsIndex: 11
destination index:%49 = arith.addi %27, %c11_149 : indexIVs: 0 0 0 12 
computing lower index
bottom idxes: %51 = arith.addi %27, %c12_161 : index
inputsIndex: 12
destination index:%51 = arith.addi %27, %c12_161 : indexIVs: 0 0 0 13 
computing lower index
bottom idxes: %53 = arith.addi %27, %c13_173 : index
inputsIndex: 13
destination index:%53 = arith.addi %27, %c13_173 : indexIVs: 0 0 0 14 
computing lower index
bottom idxes: %55 = arith.addi %27, %c14_185 : index
inputsIndex: 14
destination index:%55 = arith.addi %27, %c14_185 : indexIVs: 0 0 0 15 
computing lower index
bottom idxes: %57 = arith.addi %27, %c15_197 : index
inputsIndex: 15
destination index:%57 = arith.addi %27, %c15_197 : index
threadwise_store op:
miopen.threadwise_store %19#0, %19#1, %19#2, %19#3, %19#4, %19#5, %19#6, %19#7, %19#8, %19#9, %19#10, %19#11, %19#12, %19#13, %19#14, %19#15 -> %18[%c0, %13, %10, %12] with [[#miopen.transform_map<affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 1024 + d2 * 16 + d3)> by[#miopen.transform<Embed{4096, 1024, 16, 1} ["dim0", "dim1", "dim2", "dim3"] at [0, 1, 2, 3] -> ["slice"] at [0]>] bounds = [1, 4, 64, 16] -> [4096]>, #miopen.transform_map<affine_map<(d0) -> (d0)> by[#miopen.transform<Slice{0, 4096} ["slice"] at [0] -> ["buffer"] at [0]>] bounds = [4096] -> [8192]>]] {bounds = [1 : index, 16 : index, 1 : index, 1 : index], dest_data_per_write = 1 : i32, source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 -> memref<8192xi8, 3>, index, index, index, index

slice lengths: 1 16 1 1 
vector dim: 1
dest data per write: 1
modified lengths: 1 16 1 1 
IVs: 0 0 0 0 
computing lower index
bottom idxes: %25 = arith.addi %24, %12 : index
inputsIndex: 0
destination index:%25 = arith.addi %24, %12 : indexIVs: 0 1 0 0 
computing lower index
bottom idxes: %27 = arith.addi %25, %c1024_28 : index
inputsIndex: 1
destination index:%27 = arith.addi %25, %c1024_28 : indexIVs: 0 2 0 0 
computing lower index
bottom idxes: %29 = arith.addi %25, %c2048_40 : index
inputsIndex: 2
destination index:%29 = arith.addi %25, %c2048_40 : indexIVs: 0 3 0 0 
computing lower index
bottom idxes: %31 = arith.addi %25, %c3072_52 : index
inputsIndex: 3
destination index:%31 = arith.addi %25, %c3072_52 : indexIVs: 0 4 0 0 
computing lower index
bottom idxes: %33 = arith.addi %25, %c4096_64 : index
inputsIndex: 4
destination index:%33 = arith.addi %25, %c4096_64 : indexIVs: 0 5 0 0 
computing lower index
bottom idxes: %35 = arith.addi %25, %c5120_76 : index
inputsIndex: 5
destination index:%35 = arith.addi %25, %c5120_76 : indexIVs: 0 6 0 0 
computing lower index
bottom idxes: %37 = arith.addi %25, %c6144_88 : index
inputsIndex: 6
destination index:%37 = arith.addi %25, %c6144_88 : indexIVs: 0 7 0 0 
computing lower index
bottom idxes: %39 = arith.addi %25, %c7168_100 : index
inputsIndex: 7
destination index:%39 = arith.addi %25, %c7168_100 : indexIVs: 0 8 0 0 
computing lower index
bottom idxes: %41 = arith.addi %25, %c8192_112 : index
inputsIndex: 8
destination index:%41 = arith.addi %25, %c8192_112 : indexIVs: 0 9 0 0 
computing lower index
bottom idxes: %43 = arith.addi %25, %c9216_124 : index
inputsIndex: 9
destination index:%43 = arith.addi %25, %c9216_124 : indexIVs: 0 10 0 0 
computing lower index
bottom idxes: %45 = arith.addi %25, %c10240_136 : index
inputsIndex: 10
destination index:%45 = arith.addi %25, %c10240_136 : indexIVs: 0 11 0 0 
computing lower index
bottom idxes: %47 = arith.addi %25, %c11264_148 : index
inputsIndex: 11
destination index:%47 = arith.addi %25, %c11264_148 : indexIVs: 0 12 0 0 
computing lower index
bottom idxes: %49 = arith.addi %25, %c12288_160 : index
inputsIndex: 12
destination index:%49 = arith.addi %25, %c12288_160 : indexIVs: 0 13 0 0 
computing lower index
bottom idxes: %51 = arith.addi %25, %c13312_172 : index
inputsIndex: 13
destination index:%51 = arith.addi %25, %c13312_172 : indexIVs: 0 14 0 0 
computing lower index
bottom idxes: %53 = arith.addi %25, %c14336_184 : index
inputsIndex: 14
destination index:%53 = arith.addi %25, %c14336_184 : indexIVs: 0 15 0 0 
computing lower index
bottom idxes: %55 = arith.addi %25, %c15360_196 : index
inputsIndex: 15
destination index:%55 = arith.addi %25, %c15360_196 : indexmodule {
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x128x128x1x1xi8>, %arg1: memref<128x1x128x16x16xi8>, %arg2: memref<128x1x128x16x16xi32>) attributes {block_size = 256 : i32, grid_size = 1024 : i32, kernel = 0 : i32} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant dense<0> : vector<16xi32>
    %c0_i8 = arith.constant 0 : i8
    %c1024 = arith.constant 1024 : index
    %c2048 = arith.constant 2048 : index
    %c3072 = arith.constant 3072 : index
    %c4096 = arith.constant 4096 : index
    %c5120 = arith.constant 5120 : index
    %c6144 = arith.constant 6144 : index
    %c7168 = arith.constant 7168 : index
    %c8192 = arith.constant 8192 : index
    %c9216 = arith.constant 9216 : index
    %c10240 = arith.constant 10240 : index
    %c11264 = arith.constant 11264 : index
    %c12288 = arith.constant 12288 : index
    %c13312 = arith.constant 13312 : index
    %c14336 = arith.constant 14336 : index
    %c15360 = arith.constant 15360 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c10 = arith.constant 10 : index
    %c11 = arith.constant 11 : index
    %c12 = arith.constant 12 : index
    %c13 = arith.constant 13 : index
    %c14 = arith.constant 14 : index
    %c15 = arith.constant 15 : index
    %0 = miopen.workgroup_id : index
    %1 = miopen.workitem_id : index
    %2 = arith.remui %1, %c0 : index
    %3 = arith.remui %2, %c0 : index
    %4 = arith.divui %1, %c0 : index
    %5 = arith.remui %1, %c0 : index
    %6 = arith.divui %5, %c0 : index
    %7 = arith.muli %3, %c16 : index
    %8 = arith.divui %1, %c64 : index
    %9 = arith.remui %1, %c64 : index
    %10 = miopen.alloc() : memref<8192xi8, 3>
    %11 = arith.muli %7, %c1024 : index
    %12 = arith.muli %4, %c16 : index
    %13 = arith.addi %11, %12 : index
    %14 = arith.addi %13, %6 : index
    memref.store %c0_i8, %10[%14] : memref<8192xi8, 3>
    %15 = arith.addi %14, %c1024 : index
    memref.store %c0_i8, %10[%15] : memref<8192xi8, 3>
    %16 = arith.addi %14, %c2048 : index
    memref.store %c0_i8, %10[%16] : memref<8192xi8, 3>
    %17 = arith.addi %14, %c3072 : index
    memref.store %c0_i8, %10[%17] : memref<8192xi8, 3>
    %18 = arith.addi %14, %c4096 : index
    memref.store %c0_i8, %10[%18] : memref<8192xi8, 3>
    %19 = arith.addi %14, %c5120 : index
    memref.store %c0_i8, %10[%19] : memref<8192xi8, 3>
    %20 = arith.addi %14, %c6144 : index
    memref.store %c0_i8, %10[%20] : memref<8192xi8, 3>
    %21 = arith.addi %14, %c7168 : index
    memref.store %c0_i8, %10[%21] : memref<8192xi8, 3>
    %22 = arith.addi %14, %c8192 : index
    memref.store %c0_i8, %10[%22] : memref<8192xi8, 3>
    %23 = arith.addi %14, %c9216 : index
    memref.store %c0_i8, %10[%23] : memref<8192xi8, 3>
    %24 = arith.addi %14, %c10240 : index
    memref.store %c0_i8, %10[%24] : memref<8192xi8, 3>
    %25 = arith.addi %14, %c11264 : index
    memref.store %c0_i8, %10[%25] : memref<8192xi8, 3>
    %26 = arith.addi %14, %c12288 : index
    memref.store %c0_i8, %10[%26] : memref<8192xi8, 3>
    %27 = arith.addi %14, %c13312 : index
    memref.store %c0_i8, %10[%27] : memref<8192xi8, 3>
    %28 = arith.addi %14, %c14336 : index
    memref.store %c0_i8, %10[%28] : memref<8192xi8, 3>
    %29 = arith.addi %14, %c15360 : index
    memref.store %c0_i8, %10[%29] : memref<8192xi8, 3>
    %30 = arith.muli %8, %c1024 : index
    %31 = arith.muli %9, %c16 : index
    %32 = arith.addi %30, %31 : index
    %33 = arith.addi %32, %c4096 : index
    memref.store %c0_i8, %10[%33] : memref<8192xi8, 3>
    %34 = arith.addi %33, %c1 : index
    memref.store %c0_i8, %10[%34] : memref<8192xi8, 3>
    %35 = arith.addi %33, %c2 : index
    memref.store %c0_i8, %10[%35] : memref<8192xi8, 3>
    %36 = arith.addi %33, %c3 : index
    memref.store %c0_i8, %10[%36] : memref<8192xi8, 3>
    %37 = arith.addi %33, %c4 : index
    memref.store %c0_i8, %10[%37] : memref<8192xi8, 3>
    %38 = arith.addi %33, %c5 : index
    memref.store %c0_i8, %10[%38] : memref<8192xi8, 3>
    %39 = arith.addi %33, %c6 : index
    memref.store %c0_i8, %10[%39] : memref<8192xi8, 3>
    %40 = arith.addi %33, %c7 : index
    memref.store %c0_i8, %10[%40] : memref<8192xi8, 3>
    %41 = arith.addi %33, %c8 : index
    memref.store %c0_i8, %10[%41] : memref<8192xi8, 3>
    %42 = arith.addi %33, %c9 : index
    memref.store %c0_i8, %10[%42] : memref<8192xi8, 3>
    %43 = arith.addi %33, %c10 : index
    memref.store %c0_i8, %10[%43] : memref<8192xi8, 3>
    %44 = arith.addi %33, %c11 : index
    memref.store %c0_i8, %10[%44] : memref<8192xi8, 3>
    %45 = arith.addi %33, %c12 : index
    memref.store %c0_i8, %10[%45] : memref<8192xi8, 3>
    %46 = arith.addi %33, %c13 : index
    memref.store %c0_i8, %10[%46] : memref<8192xi8, 3>
    %47 = arith.addi %33, %c14 : index
    memref.store %c0_i8, %10[%47] : memref<8192xi8, 3>
    %48 = arith.addi %33, %c15 : index
    memref.store %c0_i8, %10[%48] : memref<8192xi8, 3>
    %49:3 = affine.for %arg3 = 0 to 1 iter_args(%arg4 = %7, %arg5 = %8, %arg6 = %cst) -> (index, index, vector<16xi32>) {
      %50 = arith.addi %arg4, %c4 : index
      %51 = arith.addi %arg5, %c4 : index
      miopen.lds_barrier
      miopen.lds_barrier
      %52 = arith.muli %7, %c1024 : index
      %53 = arith.muli %4, %c16 : index
      %54 = arith.addi %52, %53 : index
      %55 = arith.addi %54, %6 : index
      memref.store %c0_i8, %10[%55] : memref<8192xi8, 3>
      %56 = arith.addi %55, %c1024 : index
      memref.store %c0_i8, %10[%56] : memref<8192xi8, 3>
      %57 = arith.addi %55, %c2048 : index
      memref.store %c0_i8, %10[%57] : memref<8192xi8, 3>
      %58 = arith.addi %55, %c3072 : index
      memref.store %c0_i8, %10[%58] : memref<8192xi8, 3>
      %59 = arith.addi %55, %c4096 : index
      memref.store %c0_i8, %10[%59] : memref<8192xi8, 3>
      %60 = arith.addi %55, %c5120 : index
      memref.store %c0_i8, %10[%60] : memref<8192xi8, 3>
      %61 = arith.addi %55, %c6144 : index
      memref.store %c0_i8, %10[%61] : memref<8192xi8, 3>
      %62 = arith.addi %55, %c7168 : index
      memref.store %c0_i8, %10[%62] : memref<8192xi8, 3>
      %63 = arith.addi %55, %c8192 : index
      memref.store %c0_i8, %10[%63] : memref<8192xi8, 3>
      %64 = arith.addi %55, %c9216 : index
      memref.store %c0_i8, %10[%64] : memref<8192xi8, 3>
      %65 = arith.addi %55, %c10240 : index
      memref.store %c0_i8, %10[%65] : memref<8192xi8, 3>
      %66 = arith.addi %55, %c11264 : index
      memref.store %c0_i8, %10[%66] : memref<8192xi8, 3>
      %67 = arith.addi %55, %c12288 : index
      memref.store %c0_i8, %10[%67] : memref<8192xi8, 3>
      %68 = arith.addi %55, %c13312 : index
      memref.store %c0_i8, %10[%68] : memref<8192xi8, 3>
      %69 = arith.addi %55, %c14336 : index
      memref.store %c0_i8, %10[%69] : memref<8192xi8, 3>
      %70 = arith.addi %55, %c15360 : index
      memref.store %c0_i8, %10[%70] : memref<8192xi8, 3>
      %71 = arith.muli %8, %c1024 : index
      %72 = arith.muli %9, %c16 : index
      %73 = arith.addi %71, %72 : index
      %74 = arith.addi %73, %c4096 : index
      memref.store %c0_i8, %10[%74] : memref<8192xi8, 3>
      %75 = arith.addi %74, %c1 : index
      memref.store %c0_i8, %10[%75] : memref<8192xi8, 3>
      %76 = arith.addi %74, %c2 : index
      memref.store %c0_i8, %10[%76] : memref<8192xi8, 3>
      %77 = arith.addi %74, %c3 : index
      memref.store %c0_i8, %10[%77] : memref<8192xi8, 3>
      %78 = arith.addi %74, %c4 : index
      memref.store %c0_i8, %10[%78] : memref<8192xi8, 3>
      %79 = arith.addi %74, %c5 : index
      memref.store %c0_i8, %10[%79] : memref<8192xi8, 3>
      %80 = arith.addi %74, %c6 : index
      memref.store %c0_i8, %10[%80] : memref<8192xi8, 3>
      %81 = arith.addi %74, %c7 : index
      memref.store %c0_i8, %10[%81] : memref<8192xi8, 3>
      %82 = arith.addi %74, %c8 : index
      memref.store %c0_i8, %10[%82] : memref<8192xi8, 3>
      %83 = arith.addi %74, %c9 : index
      memref.store %c0_i8, %10[%83] : memref<8192xi8, 3>
      %84 = arith.addi %74, %c10 : index
      memref.store %c0_i8, %10[%84] : memref<8192xi8, 3>
      %85 = arith.addi %74, %c11 : index
      memref.store %c0_i8, %10[%85] : memref<8192xi8, 3>
      %86 = arith.addi %74, %c12 : index
      memref.store %c0_i8, %10[%86] : memref<8192xi8, 3>
      %87 = arith.addi %74, %c13 : index
      memref.store %c0_i8, %10[%87] : memref<8192xi8, 3>
      %88 = arith.addi %74, %c14 : index
      memref.store %c0_i8, %10[%88] : memref<8192xi8, 3>
      %89 = arith.addi %74, %c15 : index
      memref.store %c0_i8, %10[%89] : memref<8192xi8, 3>
      affine.yield %50, %51, %cst : index, index, vector<16xi32>
    }
    miopen.lds_barrier
    return
  }
}

