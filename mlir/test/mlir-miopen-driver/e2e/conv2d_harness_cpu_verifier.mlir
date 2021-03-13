// RUN: mlir-miopen-driver -p -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk --host %s -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

// filter: YXCK
// input : NHWC
// output: NHWK
func @conv2d_host(%filter: memref<3x3x8x128xf32>,
                  %input : memref<128x32x32x8xf32>,
                  %output: memref<128x30x30x128xf32>) {
  linalg.conv(%filter, %input, %output) {strides=[1,1], dilations=[1,1], padding=dense<0>:tensor<2x2xi64>} : memref<3x3x8x128xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>
  return
}

func @conv2d(%filter : memref<3x3x8x128xf32>, %input : memref<128x32x32x8xf32>, %output : memref<128x30x30x128xf32>) {
  // Convolution host-side logic would be populated here.
  return
}

func @main() {
  // allocate CPU memory.
  %0 = alloc() : memref<3x3x8x128xf32>
  %1 = alloc() : memref<128x32x32x8xf32>
  %2 = alloc() : memref<128x30x30x128xf32>
  %ref2 = alloc() : memref<128x30x30x128xf32>

  %3 = memref_cast %0 : memref<3x3x8x128xf32> to memref<?x?x?x?xf32>
  %4 = memref_cast %1 : memref<128x32x32x8xf32> to memref<?x?x?x?xf32>
  %5 = memref_cast %2 : memref<128x30x30x128xf32> to memref<?x?x?x?xf32>
  %ref5 = memref_cast %ref2 : memref<128x30x30x128xf32> to memref<?x?x?x?xf32>

  // populate initial values.
  %cst = constant 1.0 : f32
  %cst0 = constant 0.0 : f32
  call @mcpuMemset4DFloat(%3, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset4DFloat(%5, %cst0) : (memref<?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset4DFloat(%ref5, %cst0) : (memref<?x?x?x?xf32>, f32) -> ()

  // allocate GPU memory.
  %6 = call @mgpuMemAlloc4DFloat(%3) : (memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)
  %7 = call @mgpuMemAlloc4DFloat(%4) : (memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)
  %8 = call @mgpuMemAlloc4DFloat(%5) : (memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)

  // copy direction constants.
  %cst_h2d = constant 1 : i32
  %cst_d2h = constant 2 : i32

  // transfer data CPU -> GPU.
  call @mgpuMemCopy4DFloat(%3, %6, %cst_h2d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy4DFloat(%4, %7, %cst_h2d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy4DFloat(%5, %8, %cst_h2d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()

  // launch kernel.
  %filter = memref_cast %6 : memref<?x?x?x?xf32> to memref<3x3x8x128xf32>
  %input = memref_cast %7 : memref<?x?x?x?xf32> to memref<128x32x32x8xf32>
  %output = memref_cast %8 : memref<?x?x?x?xf32> to memref<128x30x30x128xf32>
  call @conv2d(%filter, %input, %output) : (memref<3x3x8x128xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy4DFloat(%8, %5, %cst_d2h) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()

  // launch CPU convolution.
  call @conv2d_host(%0, %1, %ref2) : (memref<3x3x8x128xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>) -> ()

  // verify result.
  %c0 = constant 0 : index
  %result = alloca() : memref<1xi32>
  %c0_i32 = constant 0 : i32
  %c1_i32 = constant 1 : i32
  store %c1_i32, %result[%c0] : memref<1xi32>

  %c1 = constant 1 : index
  %c30 = constant 30 : index
  %c128 = constant 128 : index
  scf.for %arg0 = %c0 to %c128 step %c1 {
    scf.for %arg1 = %c0 to %c30 step %c1 {
      scf.for %arg2 = %c0 to %c30 step %c1 {
        scf.for %arg3 = %c0 to %c128 step %c1 {
          %cpu_result = load %ref2[%arg0, %arg1, %arg2, %arg3] : memref<128x30x30x128xf32>
          %gpu_result = load %2[%arg0, %arg1, %arg2, %arg3] : memref<128x30x30x128xf32>
          %cmp_result = cmpf "oeq", %cpu_result, %gpu_result : f32

          scf.if %cmp_result {
          } else {
            store %c0_i32, %result[%c0] : memref<1xi32>
          }
        }
      }
    }
  }
  %9 = memref_cast %result : memref<1xi32> to memref<*xi32>
  call @print_memref_i32(%9) : (memref<*xi32>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%7) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%8) : (memref<?x?x?x?xf32>) -> ()

  // deallocate CPU memory.
  dealloc %0 : memref<3x3x8x128xf32>
  dealloc %1 : memref<128x32x32x8xf32>
  dealloc %2 : memref<128x30x30x128xf32>
  dealloc %ref2 : memref<128x30x30x128xf32>

  return
}

func @mcpuMemset4DFloat(%ptr : memref<?x?x?x?xf32>, %value: f32) -> ()
func @mgpuMemAlloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)
func @mgpuMemDealloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> ()
func @mgpuMemCopy4DFloat(%src : memref<?x?x?x?xf32>, %dst : memref<?x?x?x?xf32>, %dir : i32) -> ()
func @print_memref_i32(%ptr : memref<*xi32>)
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// E2E: [1]
