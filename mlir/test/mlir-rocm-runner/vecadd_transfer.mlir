// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func.func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  %cst = arith.constant 1 : index
  %cst0 = arith.constant 0 : index
  %cst2 = memref.dim %arg0, %cst0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    %a = memref.load %arg0[%tx] : memref<?xf32>
    %b = memref.load %arg1[%tx] : memref<?xf32>
    %c = arith.addf %a, %b : f32
    memref.store %c, %arg2[%tx] : memref<?xf32>
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
  %6 = call @mgpuMemAlloc(%3) : (memref<?xf32>) -> (memref<?xf32>)
  %7 = call @mgpuMemAlloc(%4) : (memref<?xf32>) -> (memref<?xf32>)
  %8 = call @mgpuMemAlloc(%5) : (memref<?xf32>) -> (memref<?xf32>)

  // copy direction constants.
  %cst_h2d = arith.constant 1 : i32
  %cst_d2h = arith.constant 2 : i32

  // transfer data CPU -> GPU.
  call @mgpuMemCopy(%3, %6, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()
  call @mgpuMemCopy(%4, %7, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()

  // launch kernel.
  call @vecadd(%6, %7, %8) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()

  // transfer data GPU -> CPU.
  call @mgpuMemCopy(%8, %5, %cst_d2h) : (memref<?xf32>, memref<?xf32>, i32) -> ()

  // print result.
  %9 = memref.cast %5 : memref<?xf32> to memref<*xf32>
  call @printMemrefF32(%9) : (memref<*xf32>) -> ()

  // dellocate GPU memory.
  call @mgpuMemDealloc(%6) : (memref<?xf32>) -> ()
  call @mgpuMemDealloc(%7) : (memref<?xf32>) -> ()
  call @mgpuMemDealloc(%8) : (memref<?xf32>) -> ()

  // deallocate CPU memory.
  memref.dealloc %0 : memref<16xf32>
  memref.dealloc %1 : memref<16xf32>
  memref.dealloc %2 : memref<16xf32>

  return
}

func.func private @mcpuMemset(%ptr : memref<?xf32>, %value: f32) -> ()
func.func private @mgpuMemAlloc(%ptr : memref<?xf32>) -> (memref<?xf32>)
func.func private @mgpuMemDealloc(%ptr : memref<?xf32>) -> ()
func.func private @mgpuMemCopy(%src : memref<?xf32>, %dst : memref<?xf32>, %dir : i32) -> ()
func.func private @printMemrefF32(%ptr : memref<*xf32>) -> ()
