

module attributes {gpu.container_module} {
  func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>) {
    %cst = constant 1 : index
    %cst256 = constant 256 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst256, %cst, %cst, %arg0, %arg1, %arg2) { kernel = @rock_kernel_module::@rock_conv2d_kcyx_nchw_nkhw} : (index, index, index, index, index, index, memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()
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
    %c900 = constant 900 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c16 = constant 16 : index
    %c8_i32 = constant 8 : i32
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c5 = constant 5 : index
    %c6 = constant 6 : index
    %c7 = constant 7 : index
    %c0_i32 = constant 0 : i32
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c4_i32 = constant 4 : i32
    %c64_i32 = constant 64 : i32
    %c8 = constant 8 : index
    %0 = rock.workgroup_id : index
    %1 = divi_signed %0, %c900 : index
    %2 = remi_signed %0, %c900 : index
    %3 = muli %1, %c128 : index
    %4 = muli %2, %c128 : index
    %5 = index_cast %3 : index to i32
    %6 = index_cast %4 : index to i32
    %7 = rock.workitem_id : index
    %8 = divi_signed %7, %c8 : index
    %9 = remi_signed %7, %c8 : index
    %10 = muli %9, %c4 : index
    %11 = index_cast %8 : index to i32
    %12 = index_cast %10 : index to i32
    %13 = addi %5, %12 : i32
    %14 = divi_signed %7, %c32 : index
    %15 = remi_signed %7, %c32 : index
    %16 = muli %15, %c4 : index
    %17 = index_cast %14 : index to i32
    %18 = index_cast %16 : index to i32
    %19 = addi %6, %18 : i32

    // LDS double buffer for matrix A and matrix B
    %20 = rock.alloc() : memref<4096xf32, 3>

    // VGPR buffer for matrix C
    %21 = rock.alloc() : memref<8x8xf32, 5>

    // %22 = rock.alloc() : memref<1x4xf32, 5>
    %23 = rock.alloc() : memref<1x4xf32, 5>
    // %24 = rock.alloc() : memref<1x4xf32, 5>
    // %25 = rock.alloc() : memref<1x4xf32, 5>
    scf.for %arg3 = %c0 to %c8 step %c1 {
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c0, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c0, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c1, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c1, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c2, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c2, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c3, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c3, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c4, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c4, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c5, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c5, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c6, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c6, %c7] : memref<8x8xf32, 5>
      }
      scf.for %arg4 = %c0 to %c8 step %c1 {
        store %cst, %21[%c7, %c0] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c1] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c2] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c3] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c4] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c5] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c6] : memref<8x8xf32, 5>
        store %cst, %21[%c7, %c7] : memref<8x8xf32, 5>
      }
    }

    // coordinates for matrix A
    %26 = rock.alloc() : memref<2xi32, 5>
    store %11, %26[%c0] : memref<2xi32, 5>
    store %13, %26[%c1] : memref<2xi32, 5>
    %27 = rock.alloc() : memref<2xi32, 5>
    store %11, %27[%c0] : memref<2xi32, 5>
    store %12, %27[%c1] : memref<2xi32, 5>

    // coordinates for matrix B
    // %28 = rock.alloc() : memref<2xi32, 5>
    // store %17, %28[%c0] : memref<2xi32, 5>
    // store %19, %28[%c1] : memref<2xi32, 5>
    // %29 = rock.alloc() : memref<2xi32, 5>
    // store %17, %29[%c0] : memref<2xi32, 5>
    // store %18, %29[%c1] : memref<2xi32, 5>

    %30 = divi_signed %7, %c16 : index
    %31 = divi_signed %30, %c4 : index
    %32 = remi_signed %30, %c4 : index
    %33 = remi_signed %7, %c16 : index
    %34 = divi_signed %33, %c4 : index
    %35 = remi_signed %33, %c4 : index
    %36 = muli %34, %c4 : index
    %37 = muli %31, %c16 : index
    %38 = addi %37, %36 : index
    %39 = index_cast %38 : index to i32
    %40 = muli %35, %c4 : index
    %41 = muli %32, %c16 : index
    %42 = addi %41, %40 : index
    %43 = index_cast %42 : index to i32
    %44 = addi %5, %39 : i32
    %45 = addi %6, %43 : i32

    // read matrix A into LDS
    %46 = load %26[%c0] : memref<2xi32, 5>
    %47 = load %26[%c1] : memref<2xi32, 5>
    rock.threadwise_copy(%arg0, %23, %46, %47, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
    %48 = load %27[%c0] : memref<2xi32, 5>
    %49 = load %27[%c1] : memref<2xi32, 5>
    rock.threadwise_copy(%23, %20, %c0_i32, %c0_i32, %48, %49) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>

    // read matrix B into LDS
    // %50 = load %28[%c0] : memref<2xi32, 5>
    // %51 = load %28[%c1] : memref<2xi32, 5>
    // rock.threadwise_copy(%arg1, %25, %50, %51, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>
    // %52 = load %29[%c0] : memref<2xi32, 5>
    // %53 = load %29[%c1] : memref<2xi32, 5>
    // rock.threadwise_copy(%25, %20, %c0_i32, %c0_i32, %52, %53) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>

    // main loop
    // scf.for %arg3 = %c0 to %c4 step %c1 {
    //   rock.workgroup_barrier
    //   %56 = load %26[%c0] : memref<2xi32, 5>
    //   %57 = addi %56, %c8_i32 : i32
    //   store %57, %26[%c0] : memref<2xi32, 5>
    //   %58 = load %26[%c1] : memref<2xi32, 5>
    //   store %58, %26[%c1] : memref<2xi32, 5>
    //   %59 = load %26[%c0] : memref<2xi32, 5>
    //   %60 = load %26[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%arg0, %22, %59, %60, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
    //   %61 = load %28[%c0] : memref<2xi32, 5>
    //   %62 = addi %61, %c8_i32 : i32
    //   store %62, %28[%c0] : memref<2xi32, 5>
    //   %63 = load %28[%c1] : memref<2xi32, 5>
    //   store %63, %28[%c1] : memref<2xi32, 5>
    //   %64 = load %28[%c0] : memref<2xi32, 5>
    //   %65 = load %28[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%arg1, %24, %64, %65, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>
    //   %66 = rock.alloc() : memref<1x8xf32, 5>
    //   %67 = rock.alloc() : memref<1x8xf32, 5>
    //   scf.for %arg4 = %c0 to %c8 step %c1 {
    //     %88 = index_cast %arg4 : index to i32
    //     scf.for %arg5 = %c0 to %c2 step %c1 {
    //       %89 = index_cast %arg5 : index to i32
    //       %90 = index_cast %38 : index to i32
    //       %91 = muli %89, %c64_i32 : i32
    //       %92 = addi %91, %90 : i32
    //       %93 = muli %89, %c4_i32 : i32
    //       rock.threadwise_copy(%20, %66, %88, %92, %c0_i32, %93) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //     }
    //     scf.for %arg5 = %c0 to %c2 step %c1 {
    //       %89 = index_cast %arg5 : index to i32
    //       %90 = index_cast %42 : index to i32
    //       %91 = muli %89, %c64_i32 : i32
    //       %92 = addi %91, %90 : i32
    //       %93 = muli %89, %c4_i32 : i32
    //       rock.threadwise_copy(%20, %67, %88, %92, %c0_i32, %93) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //     }
    //     rock.threadwise_gemm(%66, %67, %21) : memref<1x8xf32, 5>, memref<1x8xf32, 5>, memref<8x8xf32, 5>
    //   }
    //   %68 = load %27[%c0] : memref<2xi32, 5>
    //   %69 = load %27[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%22, %20, %c0_i32, %c0_i32, %68, %69) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 1024)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
    //   %70 = load %29[%c0] : memref<2xi32, 5>
    //   %71 = load %29[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%24, %20, %c0_i32, %c0_i32, %70, %71) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 3072)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
    //   rock.workgroup_barrier
    //   %72 = load %26[%c0] : memref<2xi32, 5>
    //   %73 = addi %72, %c8_i32 : i32
    //   store %73, %26[%c0] : memref<2xi32, 5>
    //   %74 = load %26[%c1] : memref<2xi32, 5>
    //   store %74, %26[%c1] : memref<2xi32, 5>
    //   %75 = load %26[%c0] : memref<2xi32, 5>
    //   %76 = load %26[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%arg0, %23, %75, %76, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
    //   %77 = load %28[%c0] : memref<2xi32, 5>
    //   %78 = addi %77, %c8_i32 : i32
    //   store %78, %28[%c0] : memref<2xi32, 5>
    //   %79 = load %28[%c1] : memref<2xi32, 5>
    //   store %79, %28[%c1] : memref<2xi32, 5>
    //   %80 = load %28[%c0] : memref<2xi32, 5>
    //   %81 = load %28[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%arg1, %25, %80, %81, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>
    //   %82 = rock.alloc() : memref<1x8xf32, 5>
    //   %83 = rock.alloc() : memref<1x8xf32, 5>
    //   scf.for %arg4 = %c0 to %c8 step %c1 {
    //     %88 = index_cast %arg4 : index to i32
    //     scf.for %arg5 = %c0 to %c2 step %c1 {
    //       %89 = index_cast %arg5 : index to i32
    //       %90 = index_cast %38 : index to i32
    //       %91 = muli %89, %c64_i32 : i32
    //       %92 = addi %91, %90 : i32
    //       %93 = muli %89, %c4_i32 : i32
    //       rock.threadwise_copy(%20, %82, %88, %92, %c0_i32, %93) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 1024)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //     }
    //     scf.for %arg5 = %c0 to %c2 step %c1 {
    //       %89 = index_cast %arg5 : index to i32
    //       %90 = index_cast %42 : index to i32
    //       %91 = muli %89, %c64_i32 : i32
    //       %92 = addi %91, %90 : i32
    //       %93 = muli %89, %c4_i32 : i32
    //       rock.threadwise_copy(%20, %83, %88, %92, %c0_i32, %93) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 3072)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //     }
    //     rock.threadwise_gemm(%82, %83, %21) : memref<1x8xf32, 5>, memref<1x8xf32, 5>, memref<8x8xf32, 5>
    //   }
    //   %84 = load %27[%c0] : memref<2xi32, 5>
    //   %85 = load %27[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%23, %20, %c0_i32, %c0_i32, %84, %85) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
    //   %86 = load %29[%c0] : memref<2xi32, 5>
    //   %87 = load %29[%c1] : memref<2xi32, 5>
    //   rock.threadwise_copy(%25, %20, %c0_i32, %c0_i32, %86, %87) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
    // }

    rock.workgroup_barrier

    // loop tail
    // %54 = rock.alloc() : memref<1x8xf32, 5>
    // %55 = rock.alloc() : memref<1x8xf32, 5>
    // scf.for %arg3 = %c0 to %c8 step %c1 {
    //   %56 = index_cast %arg3 : index to i32
    //   scf.for %arg4 = %c0 to %c2 step %c1 {
    //     %57 = index_cast %arg4 : index to i32
    //     %58 = index_cast %38 : index to i32
    //     %59 = muli %57, %c64_i32 : i32
    //     %60 = addi %59, %58 : i32
    //     %61 = muli %57, %c4_i32 : i32
    //     rock.threadwise_copy(%20, %54, %56, %60, %c0_i32, %61) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 1024)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //   }
    //   scf.for %arg4 = %c0 to %c2 step %c1 {
    //     %57 = index_cast %arg4 : index to i32
    //     %58 = index_cast %42 : index to i32
    //     %59 = muli %57, %c64_i32 : i32
    //     %60 = addi %59, %58 : i32
    //     %61 = muli %57, %c4_i32 : i32
    //     rock.threadwise_copy(%20, %55, %56, %60, %c0_i32, %61) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 3072)>]}], data_per_access = 1 : i32, n_slice_col = 4 : i32, n_slice_row = 1 : i32} : memref<4096xf32, 3>, memref<1x8xf32, 5>
    //   }
    //   rock.threadwise_gemm(%54, %55, %21) : memref<1x8xf32, 5>, memref<1x8xf32, 5>, memref<8x8xf32, 5>
    // }

    // write out
    //rock.threadwise_copy(%21, %arg2, %c0_i32, %c0_i32, %44, %45) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0, (d1 mod 900) floordiv 30, (d1 mod 900) mod 30)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<8x8xf32, 5>, memref<128x128x30x30xf32>

    // check LDS content
    %c1024 = constant 1024 : index
    %c30 = constant 30 : index
    %cmp = cmpi "eq", %7, %c0 : index
    scf.if %cmp {
      scf.for %i0 = %c0 to %c1024 step %c1 {
        %value = load %20[%i0] : memref<4096xf32, 3>
        %w = remi_signed %i0, %c30 : index
        %nch = divi_signed %i0, %c30 : index
        %h = remi_signed %nch, %c30 : index
        %nc = divi_signed %i0, %c900 : index
        store %value, %arg2[%c0, %nc, %h, %w] : memref<128x128x30x30xf32>
      }
    }
 
    gpu.return
  }
  } // gpu.module
}
