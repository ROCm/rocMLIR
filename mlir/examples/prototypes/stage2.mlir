#map0 = affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>
#map2 = affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>
#map3 = affine_map<(d0, d1) -> (d1 floordiv 900, d0, (d1 mod 900) floordiv 30, (d1 mod 900) mod 30)>
#map4 = affine_map<(d0) -> (d0 + 1024)>
#map5 = affine_map<(d0, d1) -> (d1 + d0 * 128)>
#map6 = affine_map<(d0, d1) -> (d1 + d0 * 128 + 1024)>
#map7 = affine_map<(d0) -> (d0 + 2048)>
#map8 = affine_map<(d0) -> (d0 + 3072)>
#map9 = affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>
#map10 = affine_map<(d0, d1) -> (d1 + d0 * 128 + 3072)>
#map11 = affine_map<(d0, d1, d2, d3) -> ((d3 + d2 * 64) floordiv 900, d1 + d0 * 64, ((d3 + d2 * 64) mod 900) floordiv 30, ((d3 + d2 * 64) mod 900) mod 30)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d1 + d0 * 4, d3 + d2 * 4)>


module attributes {gpu.container_module} {
  func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>) {
    %c1 = constant 1 : index
    %c256 = constant 256 : index
    %c900 = constant 900 : index
    "gpu.launch_func"(%c1, %c1, %c1, %c256, %c1, %c1, %arg0, %arg1, %arg2) {kernel = @rock_kernel_module::@rock_conv2d_kcyx_nchw_nkhw} : (index, index, index, index, index, index, memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
    return
  }
  func @main() {
    %cst = constant 1.000000e+00 : f32
    %cst_0 = constant 0.000000e+00 : f32
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %0 = alloc() : memref<128x8x3x3xf32>
    %1 = alloc() : memref<128x8x32x32xf32>
    %2 = alloc() : memref<128x128x30x30xf32>
    %3 = memref_cast %0 : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
    %4 = memref_cast %1 : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
    %5 = memref_cast %2 : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>
    call @mcpuMemset4DFloat(%3, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
    call @mcpuMemset4DFloat(%5, %cst_0) : (memref<?x?x?x?xf32>, f32) -> ()
    %6 = call @mgpuMemAlloc4DFloat(%3) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %7 = call @mgpuMemAlloc4DFloat(%4) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    %8 = call @mgpuMemAlloc4DFloat(%5) : (memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
    call @mgpuMemCopy4DFloat(%3, %6, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy4DFloat(%4, %7, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy4DFloat(%5, %8, %c1_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    %9 = memref_cast %6 : memref<?x?x?x?xf32> to memref<128x8x3x3xf32>
    %10 = memref_cast %7 : memref<?x?x?x?xf32> to memref<128x8x32x32xf32>
    %11 = memref_cast %8 : memref<?x?x?x?xf32> to memref<128x128x30x30xf32>
    call @conv2d(%9, %10, %11) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
    call @mgpuMemCopy4DFloat(%8, %5, %c2_i32) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
    %12 = memref_cast %5 : memref<?x?x?x?xf32> to memref<*xf32>
    call @print_memref_f32(%12) : (memref<*xf32>) -> ()
    call @mgpuMemDealloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc4DFloat(%7) : (memref<?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc4DFloat(%8) : (memref<?x?x?x?xf32>) -> ()
    dealloc %0 : memref<128x8x3x3xf32>
    dealloc %1 : memref<128x8x32x32xf32>
    dealloc %2 : memref<128x128x30x30xf32>
    return
  }
  func @mcpuMemset4DFloat(memref<?x?x?x?xf32>, f32)
  func @mgpuMemAlloc4DFloat(memref<?x?x?x?xf32>) -> memref<?x?x?x?xf32>
  func @mgpuMemDealloc4DFloat(memref<?x?x?x?xf32>)
  func @mgpuMemCopy4DFloat(memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32)
  func @print_memref_f32(memref<*xf32>)
  gpu.module @rock_kernel_module {
    gpu.func @rock_conv2d_kcyx_nchw_nkhw(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>) kernel {
    %cst = constant 0.000000e+00 : f32
    %c0_i32 = constant 0 : i32
    %c1 = constant 1 : index
    %c900 = constant 900 : index
    %c128 = constant 128 : index
    %c8 = constant 8 : index
    %c32 = constant 32 : index
    %c2048 = constant 2048 : index
    %c0 = constant 0 : index
    %c1024 = constant 1024 : index
    %c16 = constant 16 : index
    %c4 = constant 4 : index
    %c8_i32 = constant 8 : i32
    %c64_i32 = constant 64 : i32
    %0 = rock.transform(%arg0) {gridwise_gemm_argument_position = 0 : i32, layout = [{dimensions = [0 : i32], names = ["gemmK"], source_dimensions = [1 : i32, 2 : i32, 3 : i32], source_names = ["c", "y", "x"], transformation = "Unfold"}, {dimensions = [1 : i32], names = ["gemmM"], source_dimensions = [0 : i32], source_names = ["k"], transformation = "PassThrough"}], output_layout = ["gemmK", "gemmM"], source_layout = ["k", "c", "y", "x"]} : memref<128x8x3x3xf32> to memref<72x128xf32, #map0>
    %1 = rock.transform(%arg1) {layout = [{dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ci"], source_dimensions = [1 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [2 : i32, 3 : i32], names = ["hipad", "wipad"], parameters = [0 : i32, 0 : i32], source_dimensions = [2 : i32, 3 : i32], source_names = ["hi", "wi"], transformation = "Pad"}], output_layout = ["ni", "ci", "hipad", "wipad"], source_layout = ["ni", "ci", "hi", "wi"]} : memref<128x8x32x32xf32> to memref<128x8x32x32xf32>
    %2 = rock.transform(%1) {intermediate_layout = ["ni", "ci", "hipad", "wipad"], layout = [{dimensions = [0 : i32], names = ["ni"], source_dimensions = [0 : i32], source_names = ["ni"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["ci"], source_dimensions = [1 : i32], source_names = ["ci"], transformation = "PassThrough"}, {dimensions = [2 : i32, 3 : i32], names = ["y", "ho"], parameters = [2 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [2 : i32], source_names = ["hipad"], transformation = "Embed"}, {dimensions = [4 : i32, 5 : i32], names = ["x", "wo"], parameters = [2 : i32, 1 : i32, 1 : i32, 0 : i32], source_dimensions = [3 : i32], source_names = ["wipad"], transformation = "Embed"}], output_layout = ["ni", "ci", "y", "ho", "x", "wo"]} : memref<128x8x32x32xf32> to memref<128x8x3x30x3x30xf32, #map1>
    %3 = rock.transform(%2) {gridwise_gemm_argument_position = 1 : i32, intermediate_layout = ["ni", "ci", "y", "ho", "x", "wo"], layout = [{dimensions = [0 : i32], names = ["gemmK"], source_dimensions = [1 : i32, 2 : i32, 4 : i32], source_names = ["ci", "y", "x"], transformation = "Merge"}, {dimensions = [1 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 3 : i32, 5 : i32], source_names = ["ni", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmK", "gemmN"]} : memref<128x8x3x30x3x30xf32, #map1> to memref<72x115200xf32, #map2>
    %4 = rock.transform(%arg2) {gridwise_gemm_argument_position = 2 : i32, layout = [{dimensions = [0 : i32], names = ["gemmM"], source_dimensions = [1 : i32], source_names = ["ko"], transformation = "PassThrough"}, {dimensions = [1 : i32], names = ["gemmN"], source_dimensions = [0 : i32, 2 : i32, 3 : i32], source_names = ["no", "ho", "wo"], transformation = "Merge"}], output_layout = ["gemmM", "gemmN"], source_layout = ["no", "ko", "ho", "wo"]} : memref<128x128x30x30xf32> to memref<128x115200xf32, #map3>
    //%5 = rock.workgroup_id : index
    %5 = constant 899 : index
    //%5 = constant 0 : index
    %6 = divi_signed %5, %c900 : index
    %7 = remi_signed %5, %c900 : index
    %8 = muli %6, %c128 : index
    %9 = muli %7, %c128 : index
    %10 = index_cast %8 : index to i32
    %11 = index_cast %9 : index to i32
    %12 = rock.workitem_id : index
    %13 = remi_signed %12, %c8 : index
    %14 = divi_signed %12, %c8 : index
    %15 = muli %14, %c4 : index
    %16 = index_cast %13 : index to i32
    %17 = index_cast %15 : index to i32
    %18 = addi %10, %17 : i32
    %19 = divi_signed %12, %c32 : index
    %20 = remi_signed %12, %c32 : index
    %21 = muli %20, %c4 : index
    %22 = index_cast %19 : index to i32
    %23 = index_cast %21 : index to i32
    %24 = addi %11, %23 : i32
    %25 = rock.alloc() : memref<4096xf32, 3>
    %26 = rock.subview(%25, %c0) : memref<4096xf32, 3> to memref<2048xf32, 3>
    %27 = rock.subview(%26, %c0) : memref<2048xf32, 3> to memref<1024xf32, 3>
    %28 = rock.subview(%26, %c1024) : memref<2048xf32, 3> to memref<1024xf32, #map4, 3>
    %29 = rock.subview(%27, %c0) : memref<1024xf32, 3> to memref<8x128xf32, #map5, 3>
    %30 = rock.subview(%28, %c0) : memref<1024xf32, #map4, 3> to memref<8x128xf32, #map6, 3>
    %31 = rock.subview(%25, %c2048) : memref<4096xf32, 3> to memref<2048xf32, #map7, 3>
    %32 = rock.subview(%31, %c0) : memref<2048xf32, #map7, 3> to memref<1024xf32, #map7, 3>
    %33 = rock.subview(%31, %c1024) : memref<2048xf32, #map7, 3> to memref<1024xf32, #map8, 3>
    %34 = rock.subview(%32, %c0) : memref<1024xf32, #map7, 3> to memref<8x128xf32, #map9, 3>
    %35 = rock.subview(%33, %c0) : memref<1024xf32, #map8, 3> to memref<8x128xf32, #map10, 3>
    %36 = rock.alloc() : memref<8x8xf32, 5>
    %37 = rock.alloc() : memref<1x4xf32, 5>
    %38 = rock.alloc() : memref<1x4xf32, 5>
    %39 = rock.alloc() : memref<1x4xf32, 5>
    %40 = rock.alloc() : memref<1x4xf32, 5>
    rock.fill(%36, %cst) : memref<8x8xf32, 5>
    %41 = rock.alloc() : memref<2xi32, 5>
    store %16, %41[%c0] : memref<2xi32, 5>
    store %18, %41[%c1] : memref<2xi32, 5>
    %42 = rock.alloc() : memref<2xi32, 5>
    store %16, %42[%c0] : memref<2xi32, 5>
    store %17, %42[%c1] : memref<2xi32, 5>
    %43 = rock.alloc() : memref<2xi32, 5>
    store %22, %43[%c0] : memref<2xi32, 5>
    store %24, %43[%c1] : memref<2xi32, 5>
    %44 = rock.alloc() : memref<2xi32, 5>
    store %22, %44[%c0] : memref<2xi32, 5>
    store %23, %44[%c1] : memref<2xi32, 5>
    %45 = divi_signed %12, %c16 : index
    %46 = divi_signed %45, %c4 : index
    %47 = remi_signed %45, %c4 : index
    %48 = remi_signed %12, %c16 : index
    %49 = divi_signed %48, %c4 : index
    %50 = remi_signed %48, %c4 : index
    %51 = muli %49, %c4 : index
    %52 = muli %46, %c16 : index
    %53 = addi %52, %51 : index
    %54 = index_cast %53 : index to i32
    %55 = muli %50, %c4 : index
    %56 = muli %47, %c16 : index
    %57 = addi %56, %55 : index
    %58 = index_cast %57 : index to i32
    %59 = addi %10, %54 : i32
    %60 = addi %11, %58 : i32
    rock.blockwise_copy(%0, %29, %41, %42, %38) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<72x128xf32, #map0>, memref<8x128xf32, #map5, 3>, memref<2xi32, 5>, memref<2xi32, 5>, memref<1x4xf32, 5>
    rock.blockwise_copy(%3, %34, %43, %44, %40) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<72x115200xf32, #map2>, memref<8x128xf32, #map9, 3>, memref<2xi32, 5>, memref<2xi32, 5>, memref<1x4xf32, 5>

    // // XXX. write out coordinate for Matrix A.
    // %y_a_0_i32 = load %41[%c0] : memref<2xi32, 5>
    // %y_a_0_f32 = sitofp %y_a_0_i32 : i32 to f32
    // %x_a_0_i32 = load %41[%c1] : memref<2xi32, 5>
    // %x_a_0_f32 = sitofp %x_a_0_i32 : i32 to f32
    // store %y_a_0_f32, %arg2[%c0, %12, %c0, %c0] : memref<128x128x30x30xf32>
    // store %x_a_0_f32, %arg2[%c0, %12, %c0, %c1] : memref<128x128x30x30xf32>

    // // XXX. write out coordinate for Matrix B.
    // %y_b_0_i32 = load %43[%c0] : memref<2xi32, 5>
    // %y_b_0_f32 = sitofp %y_b_0_i32 : i32 to f32
    // %x_b_0_i32 = load %43[%c1] : memref<2xi32, 5>
    // %x_b_0_f32 = sitofp %x_b_0_i32 : i32 to f32
    // store %y_b_0_f32, %arg2[%c0, %12, %c1, %c0] : memref<128x128x30x30xf32>
    // store %x_b_0_f32, %arg2[%c0, %12, %c1, %c1] : memref<128x128x30x30xf32>



    %c2 = constant 2 : index
    %c3 = constant 3 : index

    %bid_y = divi_signed %5, %c128 : index
    %bid_x = remi_signed %5, %c128 : index

    scf.for %arg3 = %c0 to %c4 step %c1 {
      rock.workgroup_barrier

      rock.move_pos(%41, %c8_i32, %c0_i32) : memref<2xi32, 5>

      // XXX. write out coordinate for Matrix A.
      %y_a_0_i32 = load %41[%c0] : memref<2xi32, 5>
      %y_a_0_f32 = sitofp %y_a_0_i32 : i32 to f32
      %x_a_0_i32 = load %41[%c1] : memref<2xi32, 5>
      %x_a_0_f32 = sitofp %x_a_0_i32 : i32 to f32
      store %y_a_0_f32, %arg2[%c0, %12, %arg3, %c0] : memref<128x128x30x30xf32>
      store %x_a_0_f32, %arg2[%c0, %12, %arg3, %c1] : memref<128x128x30x30xf32>

      rock.blockwise_copy(%0, %37, %41, %42) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<72x128xf32, #map0>, memref<1x4xf32, 5>, memref<2xi32, 5>, memref<2xi32, 5>

      // matrix B
      rock.move_pos(%43, %c8_i32, %c0_i32) : memref<2xi32, 5>

      // XXX. write out coordinate for Matrix B.
      %y_b_0_i32 = load %43[%c0] : memref<2xi32, 5>
      %y_b_0_f32 = sitofp %y_b_0_i32 : i32 to f32
      %x_b_0_i32 = load %43[%c1] : memref<2xi32, 5>
      %x_b_0_f32 = sitofp %x_b_0_i32 : i32 to f32
      store %y_b_0_f32, %arg2[%c1, %12, %arg3, %c0] : memref<128x128x30x30xf32>
      store %x_b_0_f32, %arg2[%c1, %12, %arg3, %c1] : memref<128x128x30x30xf32>
 
      rock.blockwise_copy(%3, %39, %43, %44) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<72x115200xf32, #map2>, memref<1x4xf32, 5>, memref<2xi32, 5>, memref<2xi32, 5>

      //rock.blockwise_gemm(%29, %34, %36, %53, %57) {block_size = 256 : i32, k_per_thread = 1 : i32, m_level0_cluster = 4 : i32, m_level1_cluster = 4 : i32, m_per_thread = 4 : i32, n_level0_cluster = 4 : i32, n_level1_cluster = 4 : i32, n_per_thread = 4 : i32} : memref<8x128xf32, #map5, 3>, memref<8x128xf32, #map9, 3>, memref<8x8xf32, 5>, index, index

      rock.blockwise_copy(%37, %30, %41, %42) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<1x4xf32, 5>, memref<8x128xf32, #map6, 3>, memref<2xi32, 5>, memref<2xi32, 5>

      // matrix B
      rock.blockwise_copy(%39, %35, %43, %44) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<1x4xf32, 5>, memref<8x128xf32, #map10, 3>, memref<2xi32, 5>, memref<2xi32, 5>

      rock.workgroup_barrier

      rock.move_pos(%41, %c8_i32, %c0_i32) : memref<2xi32, 5>

      // XXX. write out coordinate for Matrix A.
      %y_a_1_i32 = load %41[%c0] : memref<2xi32, 5>
      %y_a_1_f32 = sitofp %y_a_1_i32 : i32 to f32
      %x_a_1_i32 = load %41[%c1] : memref<2xi32, 5>
      %x_a_1_f32 = sitofp %x_a_1_i32 : i32 to f32
      store %y_a_1_f32, %arg2[%c0, %12, %arg3, %c2] : memref<128x128x30x30xf32>
      store %x_a_1_f32, %arg2[%c0, %12, %arg3, %c3] : memref<128x128x30x30xf32>

      rock.blockwise_copy(%0, %38, %41, %42) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<72x128xf32, #map0>, memref<1x4xf32, 5>, memref<2xi32, 5>, memref<2xi32, 5>

      // matrix B. crashes at workgroup 899.
      rock.move_pos(%43, %c8_i32, %c0_i32) : memref<2xi32, 5>

      // XXX. write out coordinate for Matrix B.
      %y_b_1_i32 = load %43[%c0] : memref<2xi32, 5>
      %y_b_1_f32 = sitofp %y_b_1_i32 : i32 to f32
      %x_b_1_i32 = load %43[%c1] : memref<2xi32, 5>
      %x_b_1_f32 = sitofp %x_b_1_i32 : i32 to f32
      store %y_b_1_f32, %arg2[%c1, %12, %arg3, %c2] : memref<128x128x30x30xf32>
      store %x_b_1_f32, %arg2[%c1, %12, %arg3, %c3] : memref<128x128x30x30xf32>

      rock.blockwise_copy(%3, %40, %43, %44) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<72x115200xf32, #map2>, memref<1x4xf32, 5>, memref<2xi32, 5>, memref<2xi32, 5>

      //rock.blockwise_gemm(%30, %35, %36, %53, %57) {block_size = 256 : i32, k_per_thread = 1 : i32, m_level0_cluster = 4 : i32, m_level1_cluster = 4 : i32, m_per_thread = 4 : i32, n_level0_cluster = 4 : i32, n_level1_cluster = 4 : i32, n_per_thread = 4 : i32} : memref<8x128xf32, #map6, 3>, memref<8x128xf32, #map10, 3>, memref<8x8xf32, 5>, index, index

      rock.blockwise_copy(%38, %29, %41, %42) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [1 : i32, 0 : i32], source_vector_read_dim = 0 : i32} : memref<1x4xf32, 5>, memref<8x128xf32, #map5, 3>, memref<2xi32, 5>, memref<2xi32, 5>
      // matrix B
      rock.blockwise_copy(%40, %34, %43, %44) {block_size = 256 : i32, dest_data_per_write = 4 : i32, dest_dim_access_order = [0 : i32, 1 : i32], dest_vector_write_dim = 1 : i32, source_data_per_read = 4 : i32, source_dim_access_order = [0 : i32, 1 : i32], source_vector_read_dim = 1 : i32} : memref<1x4xf32, 5>, memref<8x128xf32, #map9, 3>, memref<2xi32, 5>, memref<2xi32, 5>
    }

    // rock.workgroup_barrier
    // rock.blockwise_gemm(%30, %35, %36, %53, %57) {block_size = 256 : i32, k_per_thread = 1 : i32, m_level0_cluster = 4 : i32, m_level1_cluster = 4 : i32, m_per_thread = 4 : i32, n_level0_cluster = 4 : i32, n_level1_cluster = 4 : i32, n_per_thread = 4 : i32} : memref<8x128xf32, #map6, 3>, memref<8x128xf32, #map10, 3>, memref<8x8xf32, 5>, index, index
    // %61 = rock.transform(%4) : memref<128x115200xf32, #map3> to memref<2x64x1800x64xf32, #map11>
    // %62 = rock.transform(%36) : memref<8x8xf32, 5> to memref<2x4x2x4xf32, #map12, 5>
    // %63 = divi_signed %59, %c64_i32 : i32
    // %64 = remi_signed %59, %c64_i32 : i32
    // %65 = divi_signed %60, %c64_i32 : i32
    // %66 = remi_signed %60, %c64_i32 : i32
    // rock.threadwise_copy(%62, %61, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %63, %64, %65, %66) {dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32, 2 : i32, 3 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<2x4x2x4xf32, #map12, 5>, memref<2x64x1800x64xf32, #map11>
    gpu.return
  }
  } // gpu.module
}
