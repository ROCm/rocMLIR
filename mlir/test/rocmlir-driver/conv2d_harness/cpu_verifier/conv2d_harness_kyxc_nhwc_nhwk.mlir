// RUN: rocmlir-gen --arch %arch -p -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk %s | rocmlir-driver -c | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// filter: GKYXC
// input : NHWGC
// output: NHWGK
module {
  func.func private @rock_conv2d_gkyxc_nhwgc_nhwgk_0(%arg0: memref<1x128x3x3x8xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>)
  func.func @main() {
    // allocate CPU memory for gpu_conv.
    %0 = memref.alloc() : memref<1x128x3x3x8xf32>
    %1 = memref.alloc() : memref<128x32x32x1x8xf32>
    %2 = memref.alloc() : memref<128x30x30x1x128xf32>

    %cst = arith.constant 1.0 : f32
    linalg.fill ins(%cst : f32) outs(%0 : memref<1x128x3x3x8xf32>)
    linalg.fill ins(%cst : f32) outs(%1 : memref<128x32x32x1x8xf32>)
    linalg.fill ins(%cst : f32) outs(%2 : memref<128x30x30x1x128xf32>)

    // launch gpu convolution
    call @gpu_conv(%0, %1, %2) : (memref<1x128x3x3x8xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()

    // allocate CPU memory and initialize
    %6 = memref.alloc() : memref<1x128x3x3x8xf32>
    linalg.fill ins(%cst : f32) outs(%6 : memref<1x128x3x3x8xf32>)

    %9 = memref.alloc() : memref<128x32x32x1x8xf32>
    linalg.fill ins(%cst : f32) outs(%9 : memref<128x32x32x1x8xf32>)

    %12 = memref.alloc() : memref<128x30x30x1x128xf32>
    linalg.fill ins(%cst : f32) outs(%12 : memref<128x30x30x1x128xf32>)

    // launch cpu convolution
    call @conv2d_host(%6, %9, %12) : (memref<1x128x3x3x8xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()

    // verity results
    call @verify_results(%12, %2) : (memref<128x30x30x1x128xf32>, memref<128x30x30x1x128xf32>) -> ()

    // deallocate CPU memory.
    memref.dealloc %0 : memref<1x128x3x3x8xf32>
    memref.dealloc %1 : memref<128x32x32x1x8xf32>
    memref.dealloc %2 : memref<128x30x30x1x128xf32>
    memref.dealloc %6 : memref<1x128x3x3x8xf32>
    memref.dealloc %9 : memref<128x32x32x1x8xf32>
    memref.dealloc %12 : memref<128x30x30x1x128xf32>
    return
  }

  func.func @gpu_conv(%arg0: memref<1x128x3x3x8xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    // allocate GPU memory.
    %0 = gpu.alloc  () : memref<1x128x3x3x8xf32>
    %1 = gpu.alloc  () : memref<128x32x32x1x8xf32>
    %2 = gpu.alloc  () : memref<128x30x30x1x128xf32>

    // transfer data CPU -> GPU.
    gpu.memcpy  %0, %arg0 : memref<1x128x3x3x8xf32>, memref<1x128x3x3x8xf32>
    gpu.memcpy  %1, %arg1 : memref<128x32x32x1x8xf32>, memref<128x32x32x1x8xf32>
    gpu.memcpy  %2, %arg2 : memref<128x30x30x1x128xf32>, memref<128x30x30x1x128xf32>

    // launch kernel.
    call @rock_conv2d_gkyxc_nhwgc_nhwgk_0(%0, %1, %2) : (memref<1x128x3x3x8xf32>, memref<128x32x32x1x8xf32>, memref<128x30x30x1x128xf32>) -> ()
    gpu.memcpy  %arg2, %2 : memref<128x30x30x1x128xf32>, memref<128x30x30x1x128xf32>

    // deallocate GPU memory.
    gpu.dealloc  %0 : memref<1x128x3x3x8xf32>
    gpu.dealloc  %1 : memref<128x32x32x1x8xf32>
    gpu.dealloc  %2 : memref<128x30x30x1x128xf32>
    return
  }

  func.func @conv2d_host(%arg0: memref<1x128x3x3x8xf32>, %arg1: memref<128x32x32x1x8xf32>, %arg2: memref<128x30x30x1x128xf32>) {
    %0 = memref.cast %arg0 : memref<1x128x3x3x8xf32> to memref<?x?x?x?x?xf32>
    %1 = memref.cast %arg1 : memref<128x32x32x1x8xf32> to memref<?x?x?x?x?xf32>
    %2 = memref.cast %arg2 : memref<128x30x30x1x128xf32> to memref<?x?x?x?x?xf32>

    %3 = memref.cast %0 : memref<?x?x?x?x?xf32> to memref<*xf32>
    %4 = memref.cast %1 : memref<?x?x?x?x?xf32> to memref<*xf32>
    %5 = memref.cast %2 : memref<?x?x?x?x?xf32> to memref<*xf32>

    // set up strides, paddings and dilations
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i32_1 = arith.constant 0 : i32
    %c1_i32_2 = arith.constant 1 : i32
    %c1_i32_3 = arith.constant 1 : i32

    // set up constant indices
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // set up constants (ascii code) for layout letters
    %g = arith.constant 103 : i8
    %k = arith.constant 107 : i8
    %c = arith.constant 99 : i8
    %y = arith.constant 121 : i8
    %x = arith.constant 120 : i8
    %n = arith.constant 110 : i8
    %h = arith.constant 104 : i8
    %w = arith.constant 119 : i8

    // allocate memory for layouts
    %6 = memref.alloca() : memref<5xi8>
    %7 = memref.alloca() : memref<5xi8>
    %8 = memref.alloca() : memref<5xi8>

    // store layouts
    memref.store %g, %6[%c0] : memref<5xi8>
    memref.store %k, %6[%c1] : memref<5xi8>
    memref.store %y, %6[%c2] : memref<5xi8>
    memref.store %x, %6[%c3] : memref<5xi8>
    memref.store %c, %6[%c4] : memref<5xi8>
    memref.store %n, %7[%c0] : memref<5xi8>
    memref.store %h, %7[%c1] : memref<5xi8>
    memref.store %w, %7[%c2] : memref<5xi8>
    memref.store %g, %7[%c3] : memref<5xi8>
    memref.store %c, %7[%c4] : memref<5xi8>
    memref.store %n, %8[%c0] : memref<5xi8>
    memref.store %h, %8[%c1] : memref<5xi8>
    memref.store %w, %8[%c2] : memref<5xi8>
    memref.store %g, %8[%c3] : memref<5xi8>
    memref.store %k, %8[%c4] : memref<5xi8>

    %9 = memref.cast %6 : memref<5xi8> to memref<*xi8>
    %10 = memref.cast %7 : memref<5xi8> to memref<*xi8>
    %11 = memref.cast %8 : memref<5xi8> to memref<*xi8>
    call @mcpuConv2dFloat(%3, %4, %5, %9, %10, %11, %c1_i32, %c1_i32_0, %c0_i32, %c0_i32_1, %c0_i32, %c0_i32_1,%c1_i32_2, %c1_i32_3) :
                    ( memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()

    return
  }
  func.func private @mcpuConv2dFloat(memref<*xf32>, memref<*xf32>, memref<*xf32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32,i32, i32)

  func.func @verify_results(%arg0: memref<128x30x30x1x128xf32>, %arg1: memref<128x30x30x1x128xf32>) {
    %c0 = arith.constant 0 : index
    %0 = memref.alloca() : memref<1xi32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    memref.store %c1_i32, %0[%c0] : memref<1xi32>
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c30 = arith.constant 30 : index
    %c30_0 = arith.constant 30 : index
    %c128_1 = arith.constant 128 : index
    scf.for %arg2 = %c0 to %c128 step %c1 {
      scf.for %arg3 = %c0 to %c30 step %c1 {
        scf.for %arg4 = %c0 to %c30_0 step %c1 {
          scf.for %arg5 = %c0 to %c1 step %c1 {
            scf.for %arg6 = %c0 to %c128_1 step %c1 {
              %2 = memref.load %arg0[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf32>
              %3 = memref.load %arg1[%arg2, %arg3, %arg4, %arg5, %arg6] : memref<128x30x30x1x128xf32>
              %cst = arith.constant 1.000000e-07 : f32
              %4 = arith.subf %2, %3 : f32
              %5 = math.absf %4 : f32
              %6 = arith.cmpf ugt, %5, %cst : f32
              scf.if %6 {
                memref.store %c0_i32, %0[%c0] : memref<1xi32>
              }
            }
          }
        }
      }
    }
    %1 = memref.cast %0 : memref<1xi32> to memref<*xi32>
    call @printMemrefI32(%1) : (memref<*xi32>) -> ()
    return
  }
  func.func private @printMemrefI32(memref<*xi32>)
}
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// E2E: [1]
