

module attributes {gpu.container_module} {
  func @conv2d(%arg0: memref<128x8x3x3xf32>, %arg1: memref<128x8x32x32xf32>, %arg2: memref<128x128x30x30xf32>) {
    %c1 = constant 1 : index
    %c256 = constant 256 : index
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
      %c32 = constant 32 : index
      %c8_i32 = constant 8 : i32
      %c115072_i32 = constant 115072 : i32
      %c8 = constant 8 : index
      %c4 = constant 4 : index
      %c5 = constant 5 : index
      %c6 = constant 6 : index
      %c7 = constant 7 : index
      %c2 = constant 2 : index
      %c3 = constant 3 : index
      %c0_i32 = constant 0 : i32
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      %0 = rock.workitem_id : index
      %1 = remi_signed %0, %c8 : index
      %2 = divi_signed %0, %c8 : index
      %3 = muli %2, %c4 : index
      %4 = index_cast %1 : index to i32
      %5 = index_cast %3 : index to i32
      %6 = divi_signed %0, %c32 : index
      %7 = remi_signed %0, %c32 : index
      %8 = muli %7, %c4 : index
      %9 = index_cast %6 : index to i32
      %10 = index_cast %8 : index to i32
      %11 = addi %10, %c115072_i32 : i32
      %12 = rock.alloc() : memref<4096xf32, 3>
      %13 = rock.alloc() : memref<8x8xf32, 5>
      %14 = rock.alloc() : memref<1x4xf32, 5>
      %15 = rock.alloc() : memref<1x4xf32, 5>
      %16 = rock.alloc() : memref<1x4xf32, 5>
      %17 = rock.alloc() : memref<1x4xf32, 5>
      scf.for %arg3 = %c0 to %c8 step %c1 {
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c0, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c0, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c1, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c1, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c2, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c2, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c3, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c3, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c4, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c4, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c5, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c5, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c6, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c6, %c7] : memref<8x8xf32, 5>
        }
        scf.for %arg4 = %c0 to %c8 step %c1 {
          store %cst, %13[%c7, %c0] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c1] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c2] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c3] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c4] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c5] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c6] : memref<8x8xf32, 5>
          store %cst, %13[%c7, %c7] : memref<8x8xf32, 5>
        }
      }
      %18 = rock.alloc() : memref<2xi32, 5>
      store %4, %18[%c0] : memref<2xi32, 5>
      store %5, %18[%c1] : memref<2xi32, 5>
      %19 = rock.alloc() : memref<2xi32, 5>
      store %4, %19[%c0] : memref<2xi32, 5>
      store %5, %19[%c1] : memref<2xi32, 5>
      %20 = rock.alloc() : memref<2xi32, 5>
      store %9, %20[%c0] : memref<2xi32, 5>
      store %11, %20[%c1] : memref<2xi32, 5>
      %21 = rock.alloc() : memref<2xi32, 5>
      store %9, %21[%c0] : memref<2xi32, 5>
      store %10, %21[%c1] : memref<2xi32, 5>
      %22 = load %18[%c0] : memref<2xi32, 5>
      %23 = load %18[%c1] : memref<2xi32, 5>
      rock.threadwise_copy(%arg0, %15, %22, %23, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 128 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
      %24 = load %19[%c0] : memref<2xi32, 5>
      %25 = load %19[%c1] : memref<2xi32, 5>
      rock.threadwise_copy(%15, %12, %c0_i32, %c0_i32, %24, %25) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
      %26 = load %20[%c0] : memref<2xi32, 5>
      %27 = load %20[%c1] : memref<2xi32, 5>
      rock.threadwise_copy(%arg1, %17, %26, %27, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 115200 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>
      %28 = load %21[%c0] : memref<2xi32, 5>
      %29 = load %21[%c1] : memref<2xi32, 5>
      rock.threadwise_copy(%17, %12, %c0_i32, %c0_i32, %28, %29) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
      scf.for %arg3 = %c0 to %c4 step %c1 {
        rock.workgroup_barrier
        %30 = load %18[%c0] : memref<2xi32, 5>
        %31 = addi %30, %c8_i32 : i32
        store %31, %18[%c0] : memref<2xi32, 5>
        %32 = load %18[%c1] : memref<2xi32, 5>
        store %32, %18[%c1] : memref<2xi32, 5>
        %33 = load %18[%c0] : memref<2xi32, 5>
        %34 = sitofp %33 : i32 to f32
        %35 = load %18[%c1] : memref<2xi32, 5>
        %36 = sitofp %35 : i32 to f32
        store %34, %arg2[%c0, %0, %arg3, %c0] : memref<128x128x30x30xf32>
        store %36, %arg2[%c0, %0, %arg3, %c1] : memref<128x128x30x30xf32>
        %37 = load %18[%c0] : memref<2xi32, 5>
        %38 = load %18[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%arg0, %14, %37, %38, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 128 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
        %39 = load %20[%c0] : memref<2xi32, 5>
        %40 = addi %39, %c8_i32 : i32
        store %40, %20[%c0] : memref<2xi32, 5>
        %41 = load %20[%c1] : memref<2xi32, 5>
        store %41, %20[%c1] : memref<2xi32, 5>
        %42 = load %20[%c0] : memref<2xi32, 5>
        %43 = sitofp %42 : i32 to f32
        %44 = load %20[%c1] : memref<2xi32, 5>
        %45 = sitofp %44 : i32 to f32
        store %43, %arg2[%c1, %0, %arg3, %c0] : memref<128x128x30x30xf32>
        store %45, %arg2[%c1, %0, %arg3, %c1] : memref<128x128x30x30xf32>
        %46 = load %20[%c0] : memref<2xi32, 5>
        %47 = load %20[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%arg1, %16, %46, %47, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 115200 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>
        %48 = load %19[%c0] : memref<2xi32, 5>
        %49 = load %19[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%14, %12, %c0_i32, %c0_i32, %48, %49) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 1024)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
        %50 = load %21[%c0] : memref<2xi32, 5>
        %51 = load %21[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%16, %12, %c0_i32, %c0_i32, %50, %51) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 3072)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
        rock.workgroup_barrier
        %52 = load %18[%c0] : memref<2xi32, 5>
        %53 = addi %52, %c8_i32 : i32
        store %53, %18[%c0] : memref<2xi32, 5>
        %54 = load %18[%c1] : memref<2xi32, 5>
        store %54, %18[%c1] : memref<2xi32, 5>
        %55 = load %18[%c0] : memref<2xi32, 5>
        %56 = sitofp %55 : i32 to f32
        %57 = load %18[%c1] : memref<2xi32, 5>
        %58 = sitofp %57 : i32 to f32
        store %56, %arg2[%c0, %0, %arg3, %c2] : memref<128x128x30x30xf32>
        store %58, %arg2[%c0, %0, %arg3, %c3] : memref<128x128x30x30xf32>
        %59 = load %18[%c0] : memref<2xi32, 5>
        %60 = load %18[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%arg0, %15, %59, %60, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 128 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>
        %61 = load %20[%c0] : memref<2xi32, 5>
        %62 = addi %61, %c8_i32 : i32
        store %62, %20[%c0] : memref<2xi32, 5>
        %63 = load %20[%c1] : memref<2xi32, 5>
        store %63, %20[%c1] : memref<2xi32, 5>
        %64 = load %20[%c0] : memref<2xi32, 5>
        %65 = sitofp %64 : i32 to f32
        %66 = load %20[%c1] : memref<2xi32, 5>
        %67 = sitofp %66 : i32 to f32
        store %65, %arg2[%c1, %0, %arg3, %c2] : memref<128x128x30x30xf32>
        store %67, %arg2[%c1, %0, %arg3, %c3] : memref<128x128x30x30xf32>
        %68 = load %20[%c0] : memref<2xi32, 5>
        %69 = load %20[%c1] : memref<2xi32, 5>
        rock.threadwise_copy(%arg1, %17, %68, %69, %c0_i32, %c0_i32) {coord_transforms = [{domain = [72 : i32, 115200 : i32], operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1 floordiv 900, d0 floordiv 9, ((d0 mod 9) floordiv 3) * 2 + (d1 mod 900) floordiv 30, ((d0 mod 9) mod 3) * 2 + (d1 mod 900) mod 30)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2 * 2 + d3, d4 * 2 + d5)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<128x8x32x32xf32>, memref<1x4xf32, 5>

        // %70 = load %19[%c0] : memref<2xi32, 5>
        // %71 = load %19[%c1] : memref<2xi32, 5>
        // rock.threadwise_copy(%15, %12, %c0_i32, %c0_i32, %70, %71) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
        // %72 = load %21[%c0] : memref<2xi32, 5>
        // %73 = load %21[%c1] : memref<2xi32, 5>
        // rock.threadwise_copy(%17, %12, %c0_i32, %c0_i32, %72, %73) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128 + 2048)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 1 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
      }
      gpu.return
    }
  }
}
