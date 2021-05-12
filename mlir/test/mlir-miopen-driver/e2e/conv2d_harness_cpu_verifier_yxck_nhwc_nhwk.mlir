// RUN: mlir-miopen-driver -p -fil_layout=gyxck -in_layout=nhwgc -out_layout=nhwgk --host %s -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// filter: GYXCK
// input : NHWGC
// output: NHWGK
module {
  func @main() {
    // allocate CPU memory for gpu_conv
    %0 = alloc() : memref<1x3x3x8x128xf32>
    %1 = alloc() : memref<128x32x32x1x8xf32>
    %2 = alloc() : memref<128x30x30x1x128xf32>

    %3 = memref_cast %0 : memref<1x3x3x8x128xf32> to memref<?x?x?x?x?xf32>
    %4 = memref_cast %1 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    %5 = memref_cast %2 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>

    // populate initial values
    %c0_i16 = constant 0 : i16
    %c1_i16 = constant 1 : i16
    %c1_i16_0 = constant 1 : i16
    %c1_i32 = constant 1 : i32
    call @mcpuMemset5DFloatRandInt(%3, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
    call @mcpuMemset5DFloatRandInt(%4, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
    call @mcpuMemset5DFloatRandInt(%5, %c0_i16, %c0_i16, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

    // launch gpu convolution
    call @gpu_conv(%0, %1, %2) : (memref<1x3x3x8x128xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()

    // allocate CPU memory and initialized
    %6 = alloc() : memref<1x3x3x8x128xf32>
    %7 = memref_cast %6 : memref<1x3x3x8x128xf32> to memref<?x?x?x?x?xf32>
    call @mcpuMemset5DFloatRandInt(%7, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
    %9 = alloc() : memref<128x32x32x1x8xf32>
    %10 = memref_cast %9 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    call @mcpuMemset5DFloatRandInt(%10, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()
    %12 = alloc() : memref<128x30x30x1x128xf32>
    %13 = memref_cast %12 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>
    call @mcpuMemset5DFloatRandInt(%13, %c0_i16, %c0_i16, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

    // launch cpu convolution
    call @conv2d_host(%6, %9, %12) : (memref<1x3x3x8x128xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()

    // verity results
    call @verify_results(%12, %2) : (memref<128x30x30x1x128xf32>, memref<128x30x30x1x128xf32>) -> ()

    // deallocate CPU memory.
    dealloc %0 : memref<1x3x3x8x128xf32>
    dealloc %1 : memref<128x32x32x1x8xf32>
    dealloc %2 : memref<128x30x30x1x128xf32>
    dealloc %6 : memref<1x3x3x8x128xf32>
    dealloc %9 : memref<128x32x32x1x8xf32>
    dealloc %12 : memref<128x30x30x1x128xf32>
    return
  }

  func private @mcpuMemset5DFloatRandInt(memref<?x?x?x?x?xf32>, i16, i16, i32)
  func private @mcpuMemCopy5DFloat(memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)

  func @gpu_conv(%arg0: memref<1x3x3x8x128xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    %0 = memref_cast %arg0 : memref<1x3x3x8x128xf32> to memref<?x?x?x?x?xf32>
    %1 = memref_cast %arg1 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    %2 = memref_cast %arg2 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>

    // allocate GPU memory.
    %3 = call @mgpuMemAlloc5DFloat(%0) : (memref<?x?x?x?x?xf32>) -> memref<?x?x?x?x?xf32>
    %4 = call @mgpuMemAlloc5DFloat(%1) : (memref<?x?x?x?x?xf32>) -> memref<?x?x?x?x?xf32>
    %5 = call @mgpuMemAlloc5DFloat(%2) : (memref<?x?x?x?x?xf32>) -> memref<?x?x?x?x?xf32>

    // copy direction constants.
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
  
    // transfer data CPU -> GPU.
    call @mgpuMemCopy5DFloat(%0, %3, %c1_i32) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy5DFloat(%1, %4, %c1_i32) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy5DFloat(%2, %5, %c1_i32) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

    // launch kernel.
    %6 = memref_cast %3 : memref<?x?x?x?x?xf32> to memref<1x3x3x8x128xf32>
    %7 = memref_cast %4 : memref<?x?x?x?x?xf32> to memref<128x32x32x1x8xf32>
    %8 = memref_cast %5 : memref<?x?x?x?x?xf32> to memref<128x30x30x1x128xf32>
    call @conv2d(%6, %7, %8) : (memref<1x3x3x8x128xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()
    call @mgpuMemCopy5DFloat(%5, %2, %c2_i32) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

    // deallocate GPU memory.
    call @mgpuMemDealloc5DFloat(%0) : (memref<?x?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc5DFloat(%1) : (memref<?x?x?x?x?xf32>) -> ()
    call @mgpuMemDealloc5DFloat(%2) : (memref<?x?x?x?x?xf32>) -> ()
    return
  }

  func private @mgpuMemAlloc5DFloat(memref<?x?x?x?x?xf32>) -> memref<?x?x?x?x?xf32>
  func private @mgpuMemCopy5DFloat(memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32)

  func @conv2d(%arg0: memref<1x3x3x8x128xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    return
  }

  func private @mgpuMemDealloc5DFloat(memref<?x?x?x?x?xf32>)

  func @conv2d_host(%arg0: memref<1x3x3x8x128xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    %0 = memref_cast %arg0 : memref<1x3x3x8x128xf32> to memref<?x?x?x?x?xf32>
    %1 = memref_cast %arg1 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    %2 = memref_cast %arg2 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>

    %3 = memref_cast %0 : memref<?x?x?x?x?xf32> to memref<*xf32>
    %4 = memref_cast %1 : memref<?x?x?x?x?xf32> to memref<*xf32>
    %5 = memref_cast %2 : memref<?x?x?x?x?xf32> to memref<*xf32>

    // set up strides, paddings and dilations
    %c1_i32 = constant 1 : i32
    %c1_i32_0 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %c0_i32_1 = constant 0 : i32
    %c1_i32_2 = constant 1 : i32
    %c1_i32_3 = constant 1 : i32

    // set up constant indices
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    
    // set up constants (ascii code) for layout letters
    %g = constant 103 : i8
    %k = constant 107 : i8
    %c = constant 99 : i8
    %y = constant 121 : i8
    %x = constant 120 : i8
    %n = constant 110 : i8
    %h = constant 104 : i8
    %w = constant 119 : i8
 
    // allocate memory for layouts 
    %6 = alloca() : memref<5xi8>
    %7 = alloca() : memref<5xi8>
    %8 = alloca() : memref<5xi8>

    // store layouts
    store %g, %6[%c0] : memref<5xi8> 
    store %y, %6[%c1] : memref<5xi8>
    store %x, %6[%c2] : memref<5xi8>
    store %c, %6[%c3] : memref<5xi8>
    store %k, %6[%c4] : memref<5xi8>
    store %n, %7[%c0] : memref<5xi8>
    store %h, %7[%c1] : memref<5xi8>
    store %w, %7[%c2] : memref<5xi8>
    store %g, %7[%c3] : memref<5xi8>
    store %c, %7[%c4] : memref<5xi8>
    store %n, %8[%c0] : memref<5xi8>
    store %h, %8[%c1] : memref<5xi8>
    store %w, %8[%c2] : memref<5xi8>
    store %g, %8[%c3] : memref<5xi8>
    store %k, %8[%c4] : memref<5xi8>

    %9 = memref_cast %6 : memref<5xi8> to memref<*xi8>
    %10 = memref_cast %7 : memref<5xi8> to memref<*xi8>
    %11 = memref_cast %8 : memref<5xi8> to memref<*xi8>
    call @mcpuConv2d(%3, %4, %5, %9, %10, %11, %c1_i32, %c1_i32_0, %c0_i32, %c0_i32_1, %c0_i32, %c0_i32_1, %c1_i32_2, %c1_i32_3) : 
                    ( memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    
    return
  }
  func private @mcpuConv2d(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32, i32, i32)

  func @verify_results(%arg0: memref<128x30x30x1x128xf32>, %arg1: memref<128x30x30x1x128xf32>) {
    %c0 = constant 0 : index
    %0 = alloca() : memref<1xi32>
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    store %c1_i32, %0[%c0] : memref<1xi32>
    %c1 = constant 1 : index
    %c128 = constant 128 : index
    %c30 = constant 30 : index
    %c30_0 = constant 30 : index
    %c128_1 = constant 128 : index
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c30 step %c1 {
        scf.for %arg4 = %c0 to %c30_0 step %c1 {
          scf.for %arg5 = %c0 to %c1 step %c1 {
            scf.for %arg6 = %c0 to %c128_1 step %c1 {
              %2 = load %arg0[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf32>
              %3 = load %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf32>
              %4 = cmpf "une", %2, %3 : f32
              scf.if %4 {
                store %c0_i32, %0[%c0] : memref<1xi32>
              }
            }
          }
        }
      }
    }
    %1 = memref_cast %0 : memref<1xi32> to memref<*xi32>
    call @print_memref_i32(%1) : (memref<*xi32>) -> ()
    return
  }
  func private @print_memref_i32(memref<*xi32>)
}
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// E2E: [1]
