// RUN: mlir-miopen-driver -p --host %s | FileCheck %s --check-prefix=HARNESS
// RUN: mlir-miopen-driver -pc --host %s | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-miopen-driver -pc --host %s | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func @conv2d(%filter : memref<1x128x8x3x3xf32>, %input : memref<128x1x8x32x32xf32>, %output : memref<128x1x128x30x30xf32>) {
  // Convolution host-side logic would be populated here.
  return
}
// HARNESS: module
// HARNESS: func @conv2d([[FILTER_MEMREF:%.*]]: memref<1x128x8x3x3xf32>, [[INPUT_MEMREF:%.*]]: memref<128x1x8x32x32xf32>, [[OUTPUT_MEMREF:%.*]]: memref<128x1x128x30x30xf32>)
// LOWERING: module
// LOWERING: func @conv2d([[FILTER_MEMREF:%.*]]: memref<1x128x8x3x3xf32>, [[INPUT_MEMREF:%.*]]: memref<128x1x8x32x32xf32>, [[OUTPUT_MEMREF:%.*]]: memref<128x1x128x30x30xf32>)
// LOWERING: gpu.launch_func  @miopen_conv2d_gkcyx_ngchw_ngkhw_0_module::@miopen_conv2d_gkcyx_ngchw_ngkhw_0  blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) args([[FILTER_MEMREF]] : memref<1x128x8x3x3xf32>, [[INPUT_MEMREF]] : memref<128x1x8x32x32xf32>, [[OUTPUT_MEMREF]] : memref<128x1x128x30x30xf32>)

func @main() {
  // memref.allocate CPU memory.
  %0 = memref.alloc() : memref<1x128x8x3x3xf32>
  %1 = memref.alloc() : memref<128x1x8x32x32xf32>
  %2 = memref.alloc() : memref<128x1x128x30x30xf32>

  %3 = memref.cast %0 : memref<1x128x8x3x3xf32> to memref<?x?x?x?x?xf32>
  %4 = memref.cast %1 : memref<128x1x8x32x32xf32> to memref<?x?x?x?x?xf32>
  %5 = memref.cast %2 : memref<128x1x128x30x30xf32> to memref<?x?x?x?x?xf32>

  // populate initial values.
  %cst = constant 1.0 : f32
  %cst0 = constant 0.0 : f32
  call @mcpuMemset5DFloat(%3, %cst) : (memref<?x?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset5DFloat(%4, %cst) : (memref<?x?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset5DFloat(%5, %cst0) : (memref<?x?x?x?x?xf32>, f32) -> ()

  // memref.allocate GPU memory.
  %6 = call @mgpuMemAlloc5DFloat(%3) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
  %7 = call @mgpuMemAlloc5DFloat(%4) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
  %8 = call @mgpuMemAlloc5DFloat(%5) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)

  // copy direction constants.
  %cst_h2d = constant 1 : i32
  %cst_d2h = constant 2 : i32

  // transfer data CPU -> GPU.
  call @mgpuMemCopy5DFloat(%3, %6, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy5DFloat(%4, %7, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy5DFloat(%5, %8, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

  // launch kernel.
  %filter = memref.cast %6 : memref<?x?x?x?x?xf32> to memref<1x128x8x3x3xf32>
  %input = memref.cast %7 : memref<?x?x?x?x?xf32> to memref<128x1x8x32x32xf32>
  %output = memref.cast %8 : memref<?x?x?x?x?xf32> to memref<128x1x128x30x30xf32>
  call @conv2d(%filter, %input, %output) : (memref<1x128x8x3x3xf32>, memref<128x1x8x32x32xf32>, memref<128x1x128x30x30xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy5DFloat(%8, %5, %cst_d2h) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

  // verify result.
  // TBD. Add more verifying logic.
  %9 = memref.cast %5 : memref<?x?x?x?x?xf32> to memref<*xf32>
  call @print_memref_f32(%9) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc5DFloat(%6) : (memref<?x?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc5DFloat(%7) : (memref<?x?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc5DFloat(%8) : (memref<?x?x?x?x?xf32>) -> ()

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<1x128x8x3x3xf32>
  memref.dealloc %1 : memref<128x1x8x32x32xf32>
  memref.dealloc %2 : memref<128x1x128x30x30xf32>

  return
}

func private @mcpuMemset5DFloat(%ptr : memref<?x?x?x?x?xf32>, %value: f32) -> ()
func private @mgpuMemAlloc5DFloat(%ptr : memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
func private @mgpuMemDealloc5DFloat(%ptr : memref<?x?x?x?x?xf32>) -> ()
func private @mgpuMemCopy5DFloat(%src : memref<?x?x?x?x?xf32>, %dst : memref<?x?x?x?x?xf32>, %dir : i32) -> ()
func private @print_memref_f32(%ptr : memref<*xf32>)
// LOWERING: gpu.module @miopen_conv2d_gkcyx_ngchw_ngkhw_0_module
// LOWERING: gpu.func @miopen_conv2d_gkcyx_ngchw_ngkhw_0
// TBD. Add more verifying logic.
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [128, 1, 128, 30, 30] strides = [115200, 115200, 900, 30, 1] data =
