// RUN: rocmlir-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | FileCheck %s --check-prefix=HARNESS
// RUN: rocmlir-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | rocmlir-driver -c | FileCheck %s --check-prefix=LOWERING
// RUN: rocmlir-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | rocmlir-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func.func private @rock_conv2d_gcyxk_gchwn_gkhwn_0(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x32x32x128xf32>, %output : memref<1x128x30x30x128xf32>) -> ()

// HARNESS: module
// HARNESS: func.func @rock_conv2d_gcyxk_gchwn_gkhwn_0([[FILTER_MEMREF:%.*]]: memref<1x8x3x3x128xf32>, [[INPUT_MEMREF:%.*]]: memref<1x8x32x32x128xf32>, [[OUTPUT_MEMREF:%.*]]: memref<1x128x30x30x128xf32>)
// LOWERING: module
// LOWERING: gpu.launch_func  @rock_conv2d_gcyxk_gchwn_gkhwn_0_module::@rock_conv2d_gcyxk_gchwn_gkhwn_0 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) dynamic_shared_memory_size %{{.*}} args(%{{.*}} : memref<1x8x3x3x128xf32>, %{{.*}} : memref<1x8x32x32x128xf32>, %{{.*}} : memref<1x128x30x30x128xf32>)

func.func @main() {
  // memref.allocate CPU memory.
  %0 = memref.alloc() : memref<1x8x3x3x128xf32>
  %1 = memref.alloc() : memref<1x8x32x32x128xf32>
  %2 = memref.alloc() : memref<1x128x30x30x128xf32>

  %3 = memref.cast %0 : memref<1x8x3x3x128xf32> to memref<?x?x?x?x?xf32>
  %4 = memref.cast %1 : memref<1x8x32x32x128xf32> to memref<?x?x?x?x?xf32>
  %5 = memref.cast %2 : memref<1x128x30x30x128xf32> to memref<?x?x?x?x?xf32>

  // populate initial values.
  %cst = arith.constant 1.0 : f32
  %cst0 = arith.constant 0.0 : f32
  call @mcpuMemset5DFloat(%3, %cst) : (memref<?x?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset5DFloat(%4, %cst) : (memref<?x?x?x?x?xf32>, f32) -> ()
  call @mcpuMemset5DFloat(%5, %cst0) : (memref<?x?x?x?x?xf32>, f32) -> ()

  // memref.allocate GPU memory.
  %filter = gpu.alloc  () : memref<1x8x3x3x128xf32>
  %input = gpu.alloc  () : memref<1x8x32x32x128xf32>
  %output = gpu.alloc  () : memref<1x128x30x30x128xf32>

  // transfer data CPU -> GPU.
  gpu.memcpy  %filter, %0 : memref<1x8x3x3x128xf32>, memref<1x8x3x3x128xf32>
  gpu.memcpy  %input, %1 : memref<1x8x32x32x128xf32>, memref<1x8x32x32x128xf32>
  gpu.memcpy  %output, %2 : memref<1x128x30x30x128xf32>, memref<1x128x30x30x128xf32>

  // launch kernel.
  call @rock_conv2d_gcyxk_gchwn_gkhwn_0(%filter, %input, %output) : (memref<1x8x3x3x128xf32>, memref<1x8x32x32x128xf32>, memref<1x128x30x30x128xf32>) -> ()

  // transfer data GPU -> CPU.
  gpu.memcpy  %2, %output : memref<1x128x30x30x128xf32>, memref<1x128x30x30x128xf32>

  // verify result.
  // TBD. Add more verifying logic.
  %6 = memref.cast %2 : memref<1x128x30x30x128xf32> to memref<*xf32>
  call @printMemrefF32(%6) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  gpu.dealloc  %filter : memref<1x8x3x3x128xf32>
  gpu.dealloc  %input : memref<1x8x32x32x128xf32>
  gpu.dealloc  %output : memref<1x128x30x30x128xf32>

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<1x8x3x3x128xf32>
  memref.dealloc %1 : memref<1x8x32x32x128xf32>
  memref.dealloc %2 : memref<1x128x30x30x128xf32>

  return
}

func.func private @mcpuMemset5DFloat(%ptr : memref<?x?x?x?x?xf32>, %value: f32) -> ()
func.func private @printMemrefF32(%ptr : memref<*xf32>)
// LOWERING: gpu.module @rock_conv2d_gcyxk_gchwn_gkhwn_0_module
// LOWERING: gpu.func @rock_conv2d_gcyxk_gchwn_gkhwn_0
// TBD. Add more verifying logic.
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 128, 30, 30, 128] strides = [14745600, 115200, 3840, 128, 1] data =
