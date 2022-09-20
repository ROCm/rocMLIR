

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
      %20 = rock.alloc() : memref<4096xf32, 3>
      %21 = rock.alloc() : memref<8x8xf32, 5>
      %22 = rock.alloc() : memref<1x4xf32, 5>
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
      %23 = rock.alloc() : memref<2xi32, 5>
      store %11, %23[%c0] : memref<2xi32, 5>
      store %13, %23[%c1] : memref<2xi32, 5>
      %24 = rock.alloc() : memref<2xi32, 5>
      store %11, %24[%c0] : memref<2xi32, 5>
      store %12, %24[%c1] : memref<2xi32, 5>
      %25 = divi_signed %7, %c16 : index
      %26 = divi_signed %25, %c4 : index
      %27 = remi_signed %25, %c4 : index
      %28 = remi_signed %7, %c16 : index
      %29 = divi_signed %28, %c4 : index
      %30 = remi_signed %28, %c4 : index
      %31 = muli %29, %c4 : index
      %32 = muli %26, %c16 : index
      %33 = addi %32, %31 : index
      %34 = index_cast %33 : index to i32
      %35 = muli %30, %c4 : index
      %36 = muli %27, %c16 : index
      %37 = addi %36, %35 : index
      %38 = index_cast %37 : index to i32
      %39 = addi %5, %34 : i32
      %40 = addi %6, %38 : i32
      %41 = load %23[%c0] : memref<2xi32, 5>
      %42 = load %23[%c1] : memref<2xi32, 5>
      rock.threadwise_copy(%arg0, %22, %41, %42, %c0_i32, %c0_i32) {coord_transforms = [{operand = 0 : i32, transforms = [affine_map<(d0, d1) -> (d1, d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>]}], dest_data_per_write = 1 : i32, dim_access_order = [1 : i32, 0 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<128x8x3x3xf32>, memref<1x4xf32, 5>

      scf.for %iter = %c0 to %c4 step %c1 {
        %v = load %22[%c0, %iter] : memref<1x4xf32, 5>
        store %v, %arg2[%c0, %c0, %7, %iter] : memref<128x128x30x30xf32>
      }

      //%43 = load %24[%c0] : memref<2xi32, 5>
      //%44 = load %24[%c1] : memref<2xi32, 5>
      //rock.threadwise_copy(%22, %20, %c0_i32, %c0_i32, %43, %44) {coord_transforms = [{operand = 1 : i32, transforms = [affine_map<(d0, d1) -> (d1 + d0 * 128)>]}], dest_data_per_write = 1 : i32, dim_access_order = [0 : i32, 1 : i32], source_data_per_read = 1 : i32, vector_read_write_dim = 0 : i32} : memref<1x4xf32, 5>, memref<4096xf32, 3>
      //rock.workgroup_barrier
      //%c30 = constant 30 : index
      //%45 = cmpi "eq", %7, %c0 : index
      //scf.if %45 {
      //  scf.for %arg3 = %c0 to %c30 step %c1 {
      //    %46 = load %20[%arg3] : memref<4096xf32, 3>
      //    store %46, %arg2[%c0, %c0, %c0, %arg3] : memref<128x128x30x30xf32>
      //  }
      //}
      gpu.return
    }
  }
}
