// RUN: miopen-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | FileCheck %s --check-prefix=HARNESS
// RUN: miopen-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | mlir-miopen-driver -c | FileCheck %s --check-prefix=LOWERING
// RUN: miopen-gen -p -fil_layout=gcyxk -in_layout=gchwn -out_layout=gkhwn --host %s | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=E2E

func.func private @miopen_conv2d_gcyxk_gchwn_gkhwn_0(%filter : memref<1x8x3x3x128xf32>, %input : memref<1x8x32x32x128xf32>, %output : memref<1x128x30x30x128xf32>) -> ()

// HARNESS: module
// HARNESS: func.func @miopen_conv2d_gcyxk_gchwn_gkhwn_0([[FILTER_MEMREF:%.*]]: memref<1x8x3x3x128xf32>, [[INPUT_MEMREF:%.*]]: memref<1x8x32x32x128xf32>, [[OUTPUT_MEMREF:%.*]]: memref<1x128x30x30x128xf32>)
// LOWERING: module
// LOWERING: gpu.launch_func  @miopen_conv2d_gcyxk_gchwn_gkhwn_0_module::@miopen_conv2d_gcyxk_gchwn_gkhwn_0 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) dynamic_shared_memory_size %{{.*}} args(%{{.*}} : memref<1x8x3x3x128xf32>, %{{.*}} : memref<1x8x32x32x128xf32>, %{{.*}} : memref<1x128x30x30x128xf32>)

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
  %6 = call @mgpuMemAlloc5DFloat(%3) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
  %7 = call @mgpuMemAlloc5DFloat(%4) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
  %8 = call @mgpuMemAlloc5DFloat(%5) : (memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)

  // copy direction constants.
  %cst_h2d = arith.constant 1 : i32
  %cst_d2h = arith.constant 2 : i32

  // transfer data CPU -> GPU.
  call @mgpuMemCopy5DFloat(%3, %6, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy5DFloat(%4, %7, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()
  call @mgpuMemCopy5DFloat(%5, %8, %cst_h2d) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

  // launch kernel.
  %filter = memref.cast %6 : memref<?x?x?x?x?xf32> to memref<1x8x3x3x128xf32>
  %input = memref.cast %7 : memref<?x?x?x?x?xf32> to memref<1x8x32x32x128xf32>
  %output = memref.cast %8 : memref<?x?x?x?x?xf32> to memref<1x128x30x30x128xf32>
  call @miopen_conv2d_gcyxk_gchwn_gkhwn_0(%filter, %input, %output) : (memref<1x8x3x3x128xf32>, memref<1x8x32x32x128xf32>, memref<1x128x30x30x128xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy5DFloat(%8, %5, %cst_d2h) : (memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, i32) -> ()

  // verify result.
  // TBD. Add more verifying logic.
  %9 = memref.cast %5 : memref<?x?x?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%9) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc5DFloat(%6) : (memref<?x?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc5DFloat(%7) : (memref<?x?x?x?x?xf32>) -> ()
  call @mgpuMemDealloc5DFloat(%8) : (memref<?x?x?x?x?xf32>) -> ()

  // memref.deallocate CPU memory.
  memref.dealloc %0 : memref<1x8x3x3x128xf32>
  memref.dealloc %1 : memref<1x8x32x32x128xf32>
  memref.dealloc %2 : memref<1x128x30x30x128xf32>

  return
}

func.func private @mcpuMemset5DFloat(%ptr : memref<?x?x?x?x?xf32>, %value: f32) -> ()
func.func private @mgpuMemAlloc5DFloat(%ptr : memref<?x?x?x?x?xf32>) -> (memref<?x?x?x?x?xf32>)
func.func private @mgpuMemDealloc5DFloat(%ptr : memref<?x?x?x?x?xf32>) -> ()
func.func private @mgpuMemCopy5DFloat(%src : memref<?x?x?x?x?xf32>, %dst : memref<?x?x?x?x?xf32>, %dir : i32) -> ()
func.func private @printMemrefF32(%ptr : memref<*xf32>)
// LOWERING: gpu.module @miopen_conv2d_gcyxk_gchwn_gkhwn_0_module
// LOWERING: gpu.func @miopen_conv2d_gcyxk_gchwn_gkhwn_0
// TBD. Add more verifying logic.
// E2E: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 128, 30, 30, 128] strides = [14745600, 115200, 3840, 128, 1] data =
