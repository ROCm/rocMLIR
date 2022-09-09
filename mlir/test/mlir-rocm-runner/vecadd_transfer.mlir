// RUN: mlir-rocm-runner %s \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

func.func @vecadd(%arg0 : memref<16xf32>, %arg1 : memref<16xf32>, %arg2 : memref<16xf32>) {
  %cst = arith.constant 1 : index
  %cst0 = arith.constant 0 : index
  %cst2 = memref.dim %arg0, %cst0 : memref<16xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    %a = memref.load %arg0[%tx] : memref<16xf32>
    %b = memref.load %arg1[%tx] : memref<16xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %arg2[%tx] : memref<16xf32>
    gpu.terminator
  }
  return
}

// CHECK: [2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46]
func.func @main() {
  // allocate CPU memory.
  %0 = memref.alloc() : memref<16xf32>
  %1 = memref.alloc() : memref<16xf32>
  %2 = memref.alloc() : memref<16xf32>
  %cst64 = arith.constant 64 : i64

  %3 = memref.cast %0 : memref<16xf32> to memref<?xf32>
  %4 = memref.cast %1 : memref<16xf32> to memref<?xf32>
  %5 = memref.cast %2 : memref<16xf32> to memref<?xf32>

  // populate initial values.
  %cst = arith.constant 1.23 : f32
  %cst0 = arith.constant 0.0 : f32
  call @mcpuMemset(%3, %cst) : (memref<?xf32>, f32) -> ()
  call @mcpuMemset(%4, %cst) : (memref<?xf32>, f32) -> ()
  call @mcpuMemset(%5, %cst0) : (memref<?xf32>, f32) -> ()

  // allocate GPU memory.
  %6 = gpu.alloc () : memref<16xf32>
  %7 = gpu.alloc () : memref<16xf32>
  %8 = gpu.alloc () : memref<16xf32>

  // copy direction constants.
  %cst_h2d = arith.constant 1 : i32
  %cst_d2h = arith.constant 2 : i32

  // transfer data CPU -> GPU.
  gpu.memcpy %6, %0 : memref<16xf32>, memref<16xf32>
  gpu.memcpy %7, %1 : memref<16xf32>, memref<16xf32>

  // launch kernel.
  call @vecadd(%6, %7, %8) : (memref<16xf32>, memref<16xf32>, memref<16xf32>) -> ()

  // transfer data GPU -> CPU.
  gpu.memcpy %2, %8 : memref<16xf32>, memref<16xf32>

  // print result.
  %9 = memref.cast %5 : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%9) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  gpu.dealloc %6 : memref<16xf32>
  gpu.dealloc %7 : memref<16xf32>
  gpu.dealloc %8 : memref<16xf32>

  // deallocate CPU memory.
  memref.dealloc %0 : memref<16xf32>
  memref.dealloc %1 : memref<16xf32>
  memref.dealloc %2 : memref<16xf32>

  return
}

func.func private @mcpuMemset(%ptr : memref<?xf32>, %value: f32) -> ()
func.func private @printMemrefF32(%ptr : memref<*xf32>) -> ()
