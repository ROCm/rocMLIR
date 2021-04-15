// RUN: mlir-miopen-driver -p -t f16 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk --host %s -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

module  {
  func @main() {
    // allocate CPU memory for gpu_conv.    
    %0 = alloc() : memref<1x128x3x3x8xf16>
    %1 = alloc() : memref<128x32x32x1x8xf16>
    %2 = alloc() : memref<128x30x30x1x128xf16>

    %3 = memref_cast %0 : memref<1x128x3x3x8xf16> to memref<?x?x?x?x?xf16>
    %4 = memref_cast %1 : memref<128x32x32x1x8xf16> to memref<?x?x?x?x?xf16>
    %5 = memref_cast %2 : memref<128x30x30x1x128xf16> to memref<?x?x?x?x?xf16>

    // populate initial values.
    %c0_i16 = constant 0 : i16
    %c1_i16 = constant 1 : i16
    %c1_i16_0 = constant 1 : i16
    %c1_i32 = constant 1 : i32
    call @mcpuMemset5DHalfRand(%3, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf16>, i16, i16, i32) -> ()
    call @mcpuMemset5DHalfRand(%4, %c1_i16, %c1_i16_0, %c1_i32) : (memref<?x?x?x?x?xf16>, i16, i16, i32) -> ()
    call @mcpuMemset5DHalfRand(%5, %c0_i16, %c0_i16, %c1_i32) : (memref<?x?x?x?x?xf16>, i16, i16, i32) -> ()

    // launch gpu convolution
    call @gpu_conv(%0, %1, %2) : (memref<1x128x3x3x8xf16>, memref<128x32x32x1x8xf16>, memref<128x30x30x1x128xf16>) -> ()
    %6 = alloc() : memref<128x30x30x1x128xf32>

    // convert gpu results to f32
    call @convert_tensor128x30x30x1x128(%2, %6) : (memref<128x30x30x1x128xf16>, memref<128x30x30x1x128xf32>) -> ()

    // allocate CPU memory for cpu filter tensor
    %7 = alloc() : memref<1x128x3x3x8xf32>
    %8 = memref_cast %7 : memref<1x128x3x3x8xf32> to memref<?x?x?x?x?xf32>

    // convert gpu filter tensor to f32
    %9 = alloc() : memref<1x128x3x3x8xf32>
    %10 = memref_cast %9 : memref<1x128x3x3x8xf32> to memref<?x?x?x?x?xf32>
    call @convert_tensor1x128x3x3x8(%0, %9) : (memref<1x128x3x3x8xf16>, memref<1x128x3x3x8xf32>) -> ()

    // copy values of gpu filter tensor to cpu filter tensor 
    call @mcpuMemCopy5DFloat(%10, %8) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()

    // allocate CPU memory for cpu input tensor
    %11 = alloc() : memref<128x32x32x1x8xf32>
    %12 = memref_cast %11 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>

    // convert gpu input tensor to f32
    %13 = alloc() : memref<128x32x32x1x8xf32>
    %14 = memref_cast %13 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    call @convert_tensor128x32x32x1x8(%1, %13) : (memref<128x32x32x1x8xf16>, memref<128x32x32x1x8xf32>) -> ()

    // copy values of gpu input tensor to cpu input tensor 
    call @mcpuMemCopy5DFloat(%14, %12) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()

    // allocate CPU memory for cpu output tensor and initialize
    %15 = alloc() : memref<128x30x30x1x128xf32>
    %16 = memref_cast %15 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>
    call @mcpuMemset5DFloatRand(%16, %c0_i16, %c0_i16, %c1_i32) : (memref<?x?x?x?x?xf32>, i16, i16, i32) -> ()

    // launch cpu convolution
    call @conv2d_host(%7, %11, %15) : (memref<1x128x3x3x8xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()

    // verify results
    call @verify_results(%15, %6) : (memref<128x30x30x1x128xf32>, memref<128x30x30x1x128xf32>) -> ()

    // deallocate CPU memory
    dealloc %6 : memref<128x30x30x1x128xf32>
    dealloc %0 : memref<1x128x3x3x8xf16>
    dealloc %1 : memref<128x32x32x1x8xf16>
    dealloc %2 : memref<128x30x30x1x128xf16>
    dealloc %7 : memref<1x128x3x3x8xf32>
    dealloc %11 : memref<128x32x32x1x8xf32>
    dealloc %15 : memref<128x30x30x1x128xf32>
    return
  }

  func private @mcpuMemset5DHalfRand(memref<?x?x?x?x?xf16>, i16, i16, i32)

  func @gpu_conv(%arg0: memref<1x128x3x3x8xf16>, %arg1: memref<128x32x32x1x8xf16>, %arg2: memref<128x30x30x1x128xf16>) {
    %0 = memref_cast %arg0 : memref<1x128x3x3x8xf16> to memref<?x?x?x?x?xf16>
    %1 = memref_cast %arg1 : memref<128x32x32x1x8xf16> to memref<?x?x?x?x?xf16>
    %2 = memref_cast %arg2 : memref<128x30x30x1x128xf16> to memref<?x?x?x?x?xf16>
    
    // allocate GPU memory
    %3 = call @mgpuMemAlloc5DHalf(%0) : (memref<?x?x?x?x?xf16>) -> memref<?x?x?x?x?xf16>
    %4 = call @mgpuMemAlloc5DHalf(%1) : (memref<?x?x?x?x?xf16>) -> memref<?x?x?x?x?xf16>
    %5 = call @mgpuMemAlloc5DHalf(%2) : (memref<?x?x?x?x?xf16>) -> memref<?x?x?x?x?xf16>

    // copy direction constants
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32

    // transfer data CPU -> GPU
    call @mgpuMemCopy5DHalf(%0, %3, %c1_i32) : (memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, i32) -> ()
    call @mgpuMemCopy5DHalf(%1, %4, %c1_i32) : (memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, i32) -> ()
    call @mgpuMemCopy5DHalf(%2, %5, %c1_i32) : (memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, i32) -> ()

    // launch kernel 
    %6 = memref_cast %3 : memref<?x?x?x?x?xf16> to memref<1x128x3x3x8xf16>
    %7 = memref_cast %4 : memref<?x?x?x?x?xf16> to memref<128x32x32x1x8xf16>
    %8 = memref_cast %5 : memref<?x?x?x?x?xf16> to memref<128x30x30x1x128xf16>
    call @conv2d(%6, %7, %8) : (memref<1x128x3x3x8xf16>, memref<128x32x32x1x8xf16>, memref<128x30x30x1x128xf16>) -> ()
    call @mgpuMemCopy5DHalf(%5, %2, %c2_i32) : (memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, i32) -> ()

    // deallocate GPU memory
    call @mgpuMemDealloc5DHalf(%0) : (memref<?x?x?x?x?xf16>) -> ()
    call @mgpuMemDealloc5DHalf(%1) : (memref<?x?x?x?x?xf16>) -> ()
    call @mgpuMemDealloc5DHalf(%2) : (memref<?x?x?x?x?xf16>) -> ()
    return
  }

  func private @mgpuMemAlloc5DHalf(memref<?x?x?x?x?xf16>) -> memref<?x?x?x?x?xf16>

  func private @mgpuMemCopy5DHalf(memref<?x?x?x?x?xf16>, memref<?x?x?x?x?xf16>, i32)

  func @conv2d(%arg0: memref<1x128x3x3x8xf16>, %arg1: memref<128x32x32x1x8xf16>, %arg2: memref<128x30x30x1x128xf16>) {
    return
  }

  func private @mgpuMemDealloc5DHalf(memref<?x?x?x?x?xf16>)

  func @convert_tensor128x30x30x1x128(%arg0: memref<128x30x30x1x128xf16>, %arg1: memref<128x30x30x1x128xf32>) {
    %c0 = constant 0 : index
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
              %0 = load %arg0[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf16>
              %1 = fpext %0 : f16 to f32
              store %1, %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf32>
            }
          }
        }
      }
    }
    return
  }

  func private @mcpuMemCopy5DFloat(memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>)

  func private @mcpuMemset5DFloatRand(memref<?x?x?x?x?xf32>, i16, i16, i32)

  func @convert_tensor1x128x3x3x8(%arg0: memref<1x128x3x3x8xf16>, %arg1: memref<1x128x3x3x8xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c128 = constant 128 : index
    %c3 = constant 3 : index
    %c3_0 = constant 3 : index
    %c8 = constant 8 : index
    scf.for %arg2 = %c0 to %c1 step %c1 {
      scf.for %arg3 = %c0 to %c128 step %c1 {
        scf.for %arg4 = %c0 to %c3 step %c1 {
          scf.for %arg5 = %c0 to %c3_0 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              %0 = load %arg0[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<1x128x3x3x8xf16>
              %1 = fpext %0 : f16 to f32
              store %1, %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<1x128x3x3x8xf32>
            }
          }
        }
      }
    }
    return
  }

  func @convert_tensor128x32x32x1x8(%arg0: memref<128x32x32x1x8xf16>, %arg1: memref<128x32x32x1x8xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c128 = constant 128 : index
    %c32 = constant 32 : index
    %c32_0 = constant 32 : index
    %c8 = constant 8 : index
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c32 step %c1 {
        scf.for %arg4 = %c0 to %c32_0 step %c1 {
          scf.for %arg5 = %c0 to %c1 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              %0 = load %arg0[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x32x32x1x8xf16>
              %1 = fpext %0 : f16 to f32
              store %1, %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x32x32x1x8xf32>
            }
          }
        }
      }
    }
    return
  }

  func @conv2d_host(%arg0: memref<1x128x3x3x8xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    %0 = memref_cast %arg0 : memref<1x128x3x3x8xf32> to memref<*xf32>
    %1 = memref_cast %arg1 : memref<128x32x32x1x8xf32> to memref<*xf32>
    %2 = memref_cast %arg2 : memref<128x30x30x1x128xf32> to memref<*xf32>
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
    %c103_i8 = constant 103 : i8
    %c107_i8 = constant 107 : i8
    %c99_i8 = constant 99 : i8
    %c121_i8 = constant 121 : i8
    %c120_i8 = constant 120 : i8
    %c110_i8 = constant 110 : i8
    %c104_i8 = constant 104 : i8
    %c119_i8 = constant 119 : i8

    // allocate memory for layouts
    %3 = alloca() : memref<5xi8>
    %4 = alloca() : memref<5xi8>
    %5 = alloca() : memref<5xi8>

    // store layouts
    store %c103_i8, %3[%c0] : memref<5xi8>
    store %c107_i8, %3[%c1] : memref<5xi8>
    store %c121_i8, %3[%c2] : memref<5xi8>
    store %c120_i8, %3[%c3] : memref<5xi8>
    store %c99_i8, %3[%c4] : memref<5xi8>
    store %c110_i8, %4[%c0] : memref<5xi8>
    store %c104_i8, %4[%c1] : memref<5xi8>
    store %c119_i8, %4[%c2] : memref<5xi8>
    store %c103_i8, %4[%c3] : memref<5xi8>
    store %c99_i8, %4[%c4] : memref<5xi8>
    store %c110_i8, %5[%c0] : memref<5xi8>
    store %c104_i8, %5[%c1] : memref<5xi8>
    store %c119_i8, %5[%c2] : memref<5xi8>
    store %c103_i8, %5[%c3] : memref<5xi8>
    store %c107_i8, %5[%c4] : memref<5xi8>
    %6 = memref_cast %3 : memref<5xi8> to memref<*xi8>
    %7 = memref_cast %4 : memref<5xi8> to memref<*xi8>
    %8 = memref_cast %5 : memref<5xi8> to memref<*xi8>
    call @mcpuConv2d(%0, %1, %2, %6, %7, %8, %c1_i32, %c1_i32_0, %c0_i32, %c0_i32_1, %c1_i32_2, %c1_i32_3) : (memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32) -> ()
    return
  }

  func private @mcpuConv2d(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32)

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
              %4 = cmpf une, %2, %3 : f32
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
