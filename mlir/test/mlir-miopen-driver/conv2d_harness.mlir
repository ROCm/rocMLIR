// RUN: mlir-miopen-driver -p --host %s | FileCheck %s --check-prefix=HARNESS

// TBD. lowering test.
// TBD: mlir-miopen-driver -pc --host %s | FileCheck %s --check-prefix=LOWERING

// TBD. e2e exuction test.
// TBD: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @conv2d(%filter : memref<128x8x3x3xf32>, %input : memref<128x8x32x32xf32>, %output : memref<128x128x30x30xf32>) {
  // Convolution host-side logic would be populated here.
  return
}
// HARNESS: module
// HARNESS: func @conv2d([[FILTER_MEMREF:%.*]]: memref<128x8x3x3xf32>, [[INPUT_MEMREF:%.*]]: memref<128x8x32x32xf32>, [[OUTPUT_MEMREF:%.*]]: memref<128x128x30x30xf32>)

func @main() {
  // allocate CPU memory.
  %0 = alloc() : memref<128x8x3x3xf32>
  %1 = alloc() : memref<128x8x32x32xf32>
  %2 = alloc() : memref<128x128x30x30xf32>

  %3 = memref_cast %0 : memref<128x8x3x3xf32> to memref<?x?x?x?xf32>
  %4 = memref_cast %1 : memref<128x8x32x32xf32> to memref<?x?x?x?xf32>
  %5 = memref_cast %2 : memref<128x128x30x30xf32> to memref<?x?x?x?xf32>

  // populate initial values.
  %cst = constant 1.0 : f32
  %cst0 = constant 0.0 : f32
  call @mcpuMemset4DFloat(%3, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset4DFloat(%4, %cst) : (memref<?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset4DFloat(%5, %cst0) : (memref<?x?x?x?xf32>, f32) -> ()

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

  // launch kernel.
  %filter = memref_cast %6 : memref<?x?x?x?xf32> to memref<128x8x3x3xf32>
  %input = memref_cast %7 : memref<?x?x?x?xf32> to memref<128x8x32x32xf32>
  %output = memref_cast %8 : memref<?x?x?x?xf32> to memref<128x128x30x30xf32>
  call @conv2d(%filter, %input, %output) : (memref<128x8x3x3xf32>, memref<128x8x32x32xf32>, memref<128x128x30x30xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy4DFloat(%8, %5, %cst_d2h) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()

  // verify result.
  // TBD.

  // dellocate GPU memory.
  call @mgpuMemDealloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%7) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%8) : (memref<?x?x?x?xf32>) -> ()

  // deallocate CPU memory.
  dealloc %0 : memref<128x8x3x3xf32>
  dealloc %1 : memref<128x8x32x32xf32>
  dealloc %2 : memref<128x128x30x30xf32>

  return
}

func @mcpuMemset4DFloat(%ptr : memref<?x?x?x?xf32>, %value: f32) -> ()
func @mgpuMemAlloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)
func @mgpuMemDealloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> ()
func @mgpuMemCopy4DFloat(%src : memref<?x?x?x?xf32>, %dst : memref<?x?x?x?xf32>, %dir : i32) -> ()
