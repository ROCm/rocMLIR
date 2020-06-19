// RUN: mlir-miopen-driver -p --host %s | FileCheck %s --check-prefix=HARNESS
// RUN: mlir-miopen-driver -pc --host %s | FileCheck %s --check-prefix=LOWERING
// RUN: mlir-miopen-driver -p -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -c --host %s | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func @conv2d(%filter : memref<128x3x3x8xf32>, %input : memref<128x32x32x8xf32>, %output : memref<128x30x30x128xf32>) {
  // Convolution host-side logic would be populated here.
  return
}
// HARNESS: module
// HARNESS: func @conv2d([[FILTER_MEMREF:%.*]]: memref<128x3x3x8xf32>, [[INPUT_MEMREF:%.*]]: memref<128x32x32x8xf32>, [[OUTPUT_MEMREF:%.*]]: memref<128x30x30x128xf32>)
// LOWERING: module
// LOWERING: func @conv2d([[FILTER_MEMREF:%.*]]: memref<128x3x3x8xf32>, [[INPUT_MEMREF:%.*]]: memref<128x32x32x8xf32>, [[OUTPUT_MEMREF:%.*]]: memref<128x30x30x128xf32>)
// LOWERING: "gpu.launch_func"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, [[FILTER_MEMREF]], [[INPUT_MEMREF]], [[OUTPUT_MEMREF]]) {kernel = @miopen_kernel_module::@miopen_conv2d_kcyx_nchw_nkhw} : (index, index, index, index, index, index, memref<128x3x3x8xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>) -> ()

func @main() {
  // allocate CPU memory.
  %0 = alloc() : memref<128x3x3x8xf32>
  %1 = alloc() : memref<128x32x32x8xf32>
  %2 = alloc() : memref<128x30x30x128xf32>

  %3 = memref_cast %0 : memref<128x3x3x8xf32> to memref<?x?x?x?xf32>
  %4 = memref_cast %1 : memref<128x32x32x8xf32> to memref<?x?x?x?xf32>
  %5 = memref_cast %2 : memref<128x30x30x128xf32> to memref<?x?x?x?xf32>

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
  call @mgpuMemCopy4DFloat(%5, %8, %cst_h2d) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()

  // launch kernel.
  %filter = memref_cast %6 : memref<?x?x?x?xf32> to memref<128x3x3x8xf32>
  %input = memref_cast %7 : memref<?x?x?x?xf32> to memref<128x32x32x8xf32>
  %output = memref_cast %8 : memref<?x?x?x?xf32> to memref<128x30x30x128xf32>
  call @conv2d(%filter, %input, %output) : (memref<128x3x3x8xf32>, memref<128x32x32x8xf32>, memref<128x30x30x128xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy4DFloat(%8, %5, %cst_d2h) : (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>, i32) -> ()

  // verify result.
  // TBD. Add more verifying logic.
  %9 = memref_cast %5 : memref<?x?x?x?xf32> to memref<*xf32>
  call @print_memref_f32(%9) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc4DFloat(%6) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%7) : (memref<?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc4DFloat(%8) : (memref<?x?x?x?xf32>) -> ()

  // deallocate CPU memory.
  dealloc %0 : memref<128x3x3x8xf32>
  dealloc %1 : memref<128x32x32x8xf32>
  dealloc %2 : memref<128x30x30x128xf32>

  return
}

func @mcpuMemset4DFloat(%ptr : memref<?x?x?x?xf32>, %value: f32) -> ()
func @mgpuMemAlloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> (memref<?x?x?x?xf32>)
func @mgpuMemDealloc4DFloat(%ptr : memref<?x?x?x?xf32>) -> ()
func @mgpuMemCopy4DFloat(%src : memref<?x?x?x?xf32>, %dst : memref<?x?x?x?xf32>, %dir : i32) -> ()
func @print_memref_f32(%ptr : memref<*xf32>)
// LOWERING: gpu.module @miopen_kernel_module
// LOWERING: gpu.func @miopen_conv2d_kcyx_nchw_nkhw
// TBD. Add more verifying logic.
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [128, 30, 30, 128] strides = [115200, 3840, 128, 1] data =
