// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw --host %s | FileCheck %s --check-prefix=HARNESS
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw --host %s | rocmlir-driver -c | FileCheck %s --check-prefix=LOWERING
// RUN: rocmlir-gen --arch %arch -p -fil_layout=gckyx -in_layout=gcnhw -out_layout=gknhw --host %s | rocmlir-driver -c | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func.func private @rock_conv2d_gckyx_gcnhw_gknhw_0(%filter : memref<1x8x128x3x3xf32>, %input : memref<1x8x128x32x32xf32>, %output : memref<1x128x128x30x30xf32>) -> ()

// HARNESS: module
// HARNESS: func.func @rock_conv2d_gckyx_gcnhw_gknhw_0([[FILTER_MEMREF:%.*]]: memref<1x8x128x3x3xf32>, [[INPUT_MEMREF:%.*]]: memref<1x8x128x32x32xf32>, [[OUTPUT_MEMREF:%.*]]: memref<1x128x128x30x30xf32>)
// LOWERING: module
// LOWERING: llvm.mlir.global internal constant @rock_conv2d_gckyx_gcnhw_gknhw_0_module_gpubin_cst

func.func @main() {
  // memref.allocate CPU memory.
  %0 = memref.alloc() : memref<1x8x128x3x3xf32>
  %1 = memref.alloc() : memref<1x8x128x32x32xf32>
  %2 = memref.alloc() : memref<1x128x128x30x30xf32>

  // populate initial values.
  %cst = arith.constant 1.0 : f32
  linalg.fill ins(%cst : f32) outs(%0 : memref<1x8x128x3x3xf32>)
  linalg.fill ins(%cst : f32) outs(%1 : memref<1x8x128x32x32xf32>)
  linalg.fill ins(%cst : f32) outs(%2 : memref<1x128x128x30x30xf32>)

  // memref.allocate GPU memory.
  %filter = gpu.alloc  () : memref<1x8x128x3x3xf32>
  %input = gpu.alloc  () : memref<1x8x128x32x32xf32>
  %output = gpu.alloc  () : memref<1x128x128x30x30xf32>

  // transfer data CPU -> GPU.
  gpu.memcpy  %filter, %0 : memref<1x8x128x3x3xf32>, memref<1x8x128x3x3xf32>
  gpu.memcpy  %input, %1 : memref<1x8x128x32x32xf32>, memref<1x8x128x32x32xf32>
  gpu.memcpy  %output, %2 : memref<1x128x128x30x30xf32>, memref<1x128x128x30x30xf32>

  // launch kernel.
  call @rock_conv2d_gckyx_gcnhw_gknhw_0(%filter, %input, %output) : (memref<1x8x128x3x3xf32>, memref<1x8x128x32x32xf32>, memref<1x128x128x30x30xf32>) -> ()

  // transfer data GPU -> CPU.
  gpu.memcpy  %2, %output : memref<1x128x128x30x30xf32>, memref<1x128x128x30x30xf32>

  // verify result.
  // TBD. Add more verifying logic.
  %6 = memref.cast %2 : memref<1x128x128x30x30xf32> to memref<*xf32>
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  gpu.dealloc  %filter : memref<1x8x128x3x3xf32>
  gpu.dealloc  %input : memref<1x8x128x32x32xf32>
  gpu.dealloc  %output : memref<1x128x128x30x30xf32>

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<1x8x128x3x3xf32>
  memref.dealloc %1 : memref<1x8x128x32x32xf32>
  memref.dealloc %2 : memref<1x128x128x30x30xf32>

  return
}

func.func private @printMemrefF32(%ptr : memref<*xf32>)
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 128, 128, 30, 30] strides = [14745600, 115200, 900, 30, 1] data =
