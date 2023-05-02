#map0 = affine_map<(d0, d1) -> (d1, d0, 0, 0)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 + d3, d4 + d5)>
#map2 = affine_map<(d0, d1) -> (d1 floordiv 196, d0, (d1 mod 196) floordiv 14, (d1 mod 196) mod 14)>
#map3 = affine_map<(d0, d1) -> (d1 + d0 * 256)>
#map4 = affine_map<(d0) -> (d0 + 4096)>
#map5 = affine_map<(d0, d1) -> (d1 + d0 * 128 + 4096)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 floordiv 196, d0 * 8 + d1 * 4 + d2, (d3 mod 196) floordiv 14, (d3 mod 196) mod 14)>


module {
  func @rock_conv2d_kcyx_nchw_nkhw(%arg0: memref<1024x1024x1x1xf32>, %arg1: memref<128x1024x14x14xf32>, %arg2: memref<128x1024x14x14xf32>) {
    %c1 = constant 1 : index
    %c256 = constant 256 : index
    %c128 = constant 128 : index
    %c2 = constant 2 : index
    %c0 = constant 0 : index
    %c4096 = constant 4096 : index
    %cst = constant dense<0.000000e+00> : vector<32xf32>
    %c63 = constant 63 : index
    %c4 = constant 4 : index
    %c4_i32 = constant 4 : i32
    %c8_i32 = constant 8 : i32
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c64 = constant 64 : index
    %c32 = constant 32 : index
    %c16_i32 = constant 16 : i32
    %0 = rock.transform(%arg0) {gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmK"], source_dimensions = [1 : i32, 2 : i32, 3 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [1 : i32], names = ["gemmM"], source_dimensions = [0 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmK", "gemmM"], source_layout = ["k", "c", "y", "x"]} : memref<1024x1024x1x1xf32> to memref<1024x1024xf32, #map0>
    %1 = rock.transform(%arg1) {layout = [{dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ci"], source_dimensions = [1 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [2 : i32, 3 : i32], names = ["hipad", "wipad"], parameters = [0 : i32, 0 : i32], source_dimensions = [2 : i32, 3 : i32], source_names = ["hi", "wi"], transformation = "Pad"}], output_layout = ["ni", "ci", "hipad", "wipad"], source_layout = ["ni", "ci", "hi", "wi"]} : memref<128x1024x14x14xf32> to memref<128x1024x14x14xf32>
    %2 = rock.transform(%1) {intermediate_layout = ["ni", "ci", "hipad", "wipad"], layout = [{dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ci"], source_dimensions = [1 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [2 : i32, 3 : i32], names = ["y", "ho"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [2 : i32], source_names = ["hipad"], transformation = "Embed"}, {dimensions = [4 : i32, 5 : i32], names = ["x", "wo"], parameters = [1 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [3 : i32], source_names = ["wipad"], transformation = "Embed"}], output_layout = ["ni", "ci", "y", "ho", "x", "wo"]} : memref<128x1024x14x14xf32> to memref<128x1024x1x14x1x14xf32, #map1>
    %3 = rock.transform(%2) {gridwise_gemm_argument_position = 1 : i32, intermediate_layout = ["ni", "ci", "y", "ho", "x", "wo"], layout = [{dimensions = [0 : i32], names = ["gemmK"], source_dimensions = [1 : i32, 2 : i32, 4 : i32], source_names = ["ci", "y", "x"], transformation = "Merge"}, {dimensions = [1 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 3 : i32, 5 : i32], source_names = ["ni", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmK", "gemmN"]} : memref<128x1024x1x14x1x14xf32, #map1> to memref<1024x25088xf32, #map2>
    %4 = rock.transform(%arg2) {gridwise_gemm_argument_position = 2 : i32, layout = [{dimensions = [0 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["ko"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 2 : i32, 3 : i32], source_names = ["no", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmM", "gemmN"], source_layout = ["no", "ko", "ho", "wo"]} : memref<128x1024x14x14xf32> to memref<1024x25088xf32, #map2>
    %5 = rock.workgroup_id : index
    %6 = rock.workitem_id : index
    %7 = remi_signed %5, %c4 : index
    %8 = divi_signed %5, %c4 : index
    %9 = muli %7, %c256 : index
    %10 = muli %8, %c128 : index
    %11 = index_cast %9 : index to i32
    %12 = index_cast %10 : index to i32
    %13 = remi_signed %6, %c4 : index
    %14 = divi_signed %6, %c4 : index
    %15 = muli %13, %c4 : index
    %16 = muli %14, %c4 : index
    %17 = index_cast %15 : index to i32
    %18 = index_cast %16 : index to i32
    %19 = addi %11, %18 : i32
    %20 = divi_signed %6, %c32 : index
    %21 = remi_signed %6, %c32 : index
    %22 = muli %20, %c2 : index
    %23 = muli %21, %c4 : index
    %24 = index_cast %22 : index to i32
    %25 = index_cast %23 : index to i32
    %26 = addi %12, %25 : i32
    %27 = rock.alloc() : memref<6144xf32, 3>
    %28 = rock.subview(%27, %c0) : memref<6144xf32, 3> to memref<4096xf32, 3>
    %29 = rock.subview(%28, %c0) : memref<4096xf32, 3> to memref<16x256xf32, #map3, 3>
    %30 = rock.subview(%27, %c4096) : memref<6144xf32, 3> to memref<2048xf32, #map4, 3>
    %31 = rock.subview(%30, %c0) : memref<2048xf32, #map4, 3> to memref<16x128xf32, #map5, 3>
    %32 = rock.alloc() : memref<4x4xf32, 5>
    %33 = rock.alloc() : memref<2x4xf32, 5>

    %orig34 = splat %c0_i32 : vector<2xi32>
    %y34 = vector.insertelement %17, %orig34[%c0_i32 : i32] : vector<2xi32>
    %34   = vector.insertelement %19, %y34[%c1_i32 : i32] : vector<2xi32>

    %orig35 = splat %c0_i32 : vector<2xi32>
    %y35 = vector.insertelement %17, %orig35[%c0_i32 : i32] : vector<2xi32>
    %35   = vector.insertelement %18, %y35[%c1_i32 : i32] : vector<2xi32>

    %orig36 = splat %c0_i32 : vector<2xi32>
    %y36 = vector.insertelement %24, %orig36[%c0_i32 : i32] : vector<2xi32>
    %36   = vector.insertelement %26, %y36[%c1_i32 : i32] : vector<2xi32>

    %orig37 = splat %c0_i32 : vector<2xi32>
    %y37 = vector.insertelement %24, %orig37[%c0_i32 : i32] : vector<2xi32>
    %37   = vector.insertelement %25, %y37[%c1_i32 : i32] : vector<2xi32>

    rock.blockwise_copy_v2(%0, %29, %34, %35, %32) {block_size = 256 : i32, dest_data_per_write = 1 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<1024x1024xf32, #map0>, memref<16x256xf32, #map3, 3>, vector<2xi32>, vector<2xi32>, memref<4x4xf32, 5>
    rock.blockwise_copy_v2(%3, %31, %36, %37, %33) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<1024x25088xf32, #map2>, memref<16x128xf32, #map5, 3>, vector<2xi32>, vector<2xi32>, memref<2x4xf32, 5>

    %38 = divi_signed %6, %c64 : index
    %39 = divi_signed %38, %c2 : index
    %40 = remi_signed %38, %c2 : index
    %41 = muli %39, %c128 : index
    %42 = muli %40, %c64 : index
    %43 = rock.alloc() : memref<32xf32, 5>
    %44 = rock.alloc() : memref<16xf32, 5>

    %45:4, %blah, %blah2 = scf.for %arg3 = %c0 to %c63 step %c1 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %cst, %arg7 = %cst, %arg8 = %34, %arg9 = %36) -> (vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<2xi32>, vector<2xi32>) {
      %a_y = vector.extractelement %arg8[%c0_i32 : i32] : vector<2xi32>
      %a_y_new = addi %a_y, %c16_i32 : i32
      %a_src = vector.insertelement %a_y_new, %arg8[%c0_i32 : i32] : vector<2xi32>

      rock.blockwise_copy_v2(%0, %32, %a_src, %35) {block_size = 256 : i32, dest_data_per_write = 1 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<1024x1024xf32, #map0>, memref<4x4xf32, 5>, vector<2xi32>, vector<2xi32>

      %b_y = vector.extractelement %arg9[%c0_i32 : i32] : vector<2xi32>
      %b_y_new = addi %b_y, %c16_i32 : i32
      %b_src = vector.insertelement %b_y_new, %arg9[%c0_i32 : i32] : vector<2xi32>

      rock.blockwise_copy_v2(%3, %33, %b_src, %37) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<1024x25088xf32, #map2>, memref<2x4xf32, 5>, vector<2xi32>, vector<2xi32>

      rock.workgroup_barrier
      %204:4 = rock.blockwise_gemm_accel(%29, %31, %41, %42, %43, %44, %arg4, %arg5, %arg6, %arg7) {block_size = 256 : i32, coord_transforms = [], k = 16 : i32, m = 256 : i32, m_per_wave = 128 : i32, m_waves = 2 : i32, n = 128 : i32, n_per_wave = 64 : i32, n_waves = 2 : i32} : memref<16x256xf32, #map3, 3>, memref<16x128xf32, #map5, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
      rock.workgroup_barrier

      rock.blockwise_copy_v2(%32, %29, %a_src, %35) {block_size = 256 : i32, dest_data_per_write = 1 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<4x4xf32, 5>, memref<16x256xf32, #map3, 3>, vector<2xi32>, vector<2xi32>
      rock.blockwise_copy_v2(%33, %31, %b_src, %37) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<2x4xf32, 5>, memref<16x128xf32, #map5, 3>, vector<2xi32>, vector<2xi32>
      scf.yield %204#0, %204#1, %204#2, %204#3, %a_src, %b_src : vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<2xi32>, vector<2xi32>
    }
    rock.workgroup_barrier
    %46:4 = rock.blockwise_gemm_accel(%29, %31, %41, %42, %43, %44, %45#0, %45#1, %45#2, %45#3) {block_size = 256 : i32, coord_transforms = [], k = 16 : i32, m = 256 : i32, m_per_wave = 128 : i32, m_waves = 2 : i32, n = 128 : i32, n_per_wave = 64 : i32, n_waves = 2 : i32} : memref<16x256xf32, #map3, 3>, memref<16x128xf32, #map5, 3>, index, index, memref<32xf32, 5>, memref<16xf32, 5>, vector<32xf32>, vector<32xf32>, vector<32xf32>, vector<32xf32>
    %47 = rock.transform(%4) : memref<1024x25088xf32, #map2> to memref<128x2x4x25088xf32, #map6>
    %48 = remi_signed %6, %c64 : index
    %49 = divi_signed %48, %c32 : index
    %50 = remi_signed %48, %c32 : index
    %51 = muli %49, %c4 : index
    %52 = remi_signed %38, %c2 : index
    %53 = muli %52, %c64 : index
    %54 = addi %53, %50 : index
    %55 = index_cast %54 : index to i32
    %56 = divi_signed %38, %c2 : index
    %57 = muli %56, %c128 : index
    %58 = addi %57, %51 : index
    %59 = index_cast %58 : index to i32
    %60 = addi %11, %59 : i32
    %61 = addi %12, %55 : i32
    %62 = divi_signed %60, %c8_i32 : i32
    %63 = remi_signed %60, %c8_i32 : i32
    %64 = divi_signed %63, %c4_i32 : i32
    %65 = remi_signed %60, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#0, %47, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %62, %64, %65, %61) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %66 = remi_signed %6, %c64 : index
    %67 = divi_signed %66, %c32 : index
    %68 = remi_signed %66, %c32 : index
    %69 = addi %68, %c32 : index
    %70 = muli %67, %c4 : index
    %71 = remi_signed %38, %c2 : index
    %72 = muli %71, %c64 : index
    %73 = addi %72, %69 : index
    %74 = index_cast %73 : index to i32
    %75 = divi_signed %38, %c2 : index
    %76 = muli %75, %c128 : index
    %77 = addi %76, %70 : index
    %78 = index_cast %77 : index to i32
    %79 = addi %11, %78 : i32
    %80 = addi %12, %74 : i32
    %81 = divi_signed %79, %c8_i32 : i32
    %82 = remi_signed %79, %c8_i32 : i32
    %83 = divi_signed %82, %c4_i32 : i32
    %84 = remi_signed %79, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#0, %47, %c16_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %81, %83, %84, %80) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %85 = remi_signed %6, %c64 : index
    %86 = divi_signed %85, %c32 : index
    %87 = remi_signed %85, %c32 : index
    %88 = muli %86, %c4 : index
    %89 = addi %88, %c32 : index
    %90 = remi_signed %38, %c2 : index
    %91 = muli %90, %c64 : index
    %92 = addi %91, %87 : index
    %93 = index_cast %92 : index to i32
    %94 = divi_signed %38, %c2 : index
    %95 = muli %94, %c128 : index
    %96 = addi %95, %89 : index
    %97 = index_cast %96 : index to i32
    %98 = addi %11, %97 : i32
    %99 = addi %12, %93 : i32
    %100 = divi_signed %98, %c8_i32 : i32
    %101 = remi_signed %98, %c8_i32 : i32
    %102 = divi_signed %101, %c4_i32 : i32
    %103 = remi_signed %98, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#1, %47, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %100, %102, %103, %99) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %104 = remi_signed %6, %c64 : index
    %105 = divi_signed %104, %c32 : index
    %106 = remi_signed %104, %c32 : index
    %107 = addi %106, %c32 : index
    %108 = muli %105, %c4 : index
    %109 = addi %108, %c32 : index
    %110 = remi_signed %38, %c2 : index
    %111 = muli %110, %c64 : index
    %112 = addi %111, %107 : index
    %113 = index_cast %112 : index to i32
    %114 = divi_signed %38, %c2 : index
    %115 = muli %114, %c128 : index
    %116 = addi %115, %109 : index
    %117 = index_cast %116 : index to i32
    %118 = addi %11, %117 : i32
    %119 = addi %12, %113 : i32
    %120 = divi_signed %118, %c8_i32 : i32
    %121 = remi_signed %118, %c8_i32 : i32
    %122 = divi_signed %121, %c4_i32 : i32
    %123 = remi_signed %118, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#1, %47, %c16_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %120, %122, %123, %119) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %124 = remi_signed %6, %c64 : index
    %125 = divi_signed %124, %c32 : index
    %126 = remi_signed %124, %c32 : index
    %127 = muli %125, %c4 : index
    %128 = addi %127, %c64 : index
    %129 = remi_signed %38, %c2 : index
    %130 = muli %129, %c64 : index
    %131 = addi %130, %126 : index
    %132 = index_cast %131 : index to i32
    %133 = divi_signed %38, %c2 : index
    %134 = muli %133, %c128 : index
    %135 = addi %134, %128 : index
    %136 = index_cast %135 : index to i32
    %137 = addi %11, %136 : i32
    %138 = addi %12, %132 : i32
    %139 = divi_signed %137, %c8_i32 : i32
    %140 = remi_signed %137, %c8_i32 : i32
    %141 = divi_signed %140, %c4_i32 : i32
    %142 = remi_signed %137, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#2, %47, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %139, %141, %142, %138) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %143 = remi_signed %6, %c64 : index
    %144 = divi_signed %143, %c32 : index
    %145 = remi_signed %143, %c32 : index
    %146 = addi %145, %c32 : index
    %147 = muli %144, %c4 : index
    %148 = addi %147, %c64 : index
    %149 = remi_signed %38, %c2 : index
    %150 = muli %149, %c64 : index
    %151 = addi %150, %146 : index
    %152 = index_cast %151 : index to i32
    %153 = divi_signed %38, %c2 : index
    %154 = muli %153, %c128 : index
    %155 = addi %154, %148 : index
    %156 = index_cast %155 : index to i32
    %157 = addi %11, %156 : i32
    %158 = addi %12, %152 : i32
    %159 = divi_signed %157, %c8_i32 : i32
    %160 = remi_signed %157, %c8_i32 : i32
    %161 = divi_signed %160, %c4_i32 : i32
    %162 = remi_signed %157, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#2, %47, %c16_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %159, %161, %162, %158) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %163 = remi_signed %6, %c64 : index
    %164 = divi_signed %163, %c32 : index
    %165 = remi_signed %163, %c32 : index
    %166 = muli %164, %c4 : index
    %167 = addi %166, %c32 : index
    %168 = addi %167, %c64 : index
    %169 = remi_signed %38, %c2 : index
    %170 = muli %169, %c64 : index
    %171 = addi %170, %165 : index
    %172 = index_cast %171 : index to i32
    %173 = divi_signed %38, %c2 : index
    %174 = muli %173, %c128 : index
    %175 = addi %174, %168 : index
    %176 = index_cast %175 : index to i32
    %177 = addi %11, %176 : i32
    %178 = addi %12, %172 : i32
    %179 = divi_signed %177, %c8_i32 : i32
    %180 = remi_signed %177, %c8_i32 : i32
    %181 = divi_signed %180, %c4_i32 : i32
    %182 = remi_signed %177, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#3, %47, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %179, %181, %182, %178) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    %183 = remi_signed %6, %c64 : index
    %184 = divi_signed %183, %c32 : index
    %185 = remi_signed %183, %c32 : index
    %186 = addi %185, %c32 : index
    %187 = muli %184, %c4 : index
    %188 = addi %187, %c32 : index
    %189 = addi %188, %c64 : index
    %190 = remi_signed %38, %c2 : index
    %191 = muli %190, %c64 : index
    %192 = addi %191, %186 : index
    %193 = index_cast %192 : index to i32
    %194 = divi_signed %38, %c2 : index
    %195 = muli %194, %c128 : index
    %196 = addi %195, %189 : index
    %197 = index_cast %196 : index to i32
    %198 = addi %11, %197 : i32
    %199 = addi %12, %193 : i32
    %200 = divi_signed %198, %c8_i32 : i32
    %201 = remi_signed %198, %c8_i32 : i32
    %202 = divi_signed %201, %c4_i32 : i32
    %203 = remi_signed %198, %c4_i32 : i32
    rock.threadwise_copy_v2(%46#3, %47, %c16_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %200, %202, %203, %199) {bound = [4 : i32, 1 : i32, 4 : i32, 1 : i32], coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1, d2, d3) -> (d0 * 4 + d2)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 3 : i32} : vector<32xf32>, memref<128x2x4x25088xf32, #map6>
    return
  }
}
