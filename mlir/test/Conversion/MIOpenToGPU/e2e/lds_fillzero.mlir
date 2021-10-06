// RUN: miopen-opt -test-miopen-lowering-gpu-module -convert-miopen-to-gpu %s | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_kernels {
    gpu.func @lds_fillzero_kernel(%arg2 : memref<?xf32>) kernel {
      %d = miopen.alloc() : memref<16xf32, 3>
      %cst = constant 0.0 : f32
      miopen.fill(%d, %cst) : memref<16xf32, 3>, f32
      miopen.workgroup_barrier

      %tx = miopen.workitem_id : index
      %e = memref.load %d[%tx] : memref<16xf32, 3>
      memref.store %e, %arg2[%tx] : memref<?xf32>
      gpu.return
    }
  }

  func @lds_fillzero(%arg2 : memref<?xf32>) {
    %cst = constant 1 : index
    %c0 = constant 0 : index
    %cst2 = memref.dim %arg2, %c0 : memref<?xf32>
    "gpu.launch_func"(%cst, %cst, %cst, %cst2, %cst, %cst, %arg2) { kernel = @gpu_kernels::@lds_fillzero_kernel, operand_segment_sizes = dense<[0,1,1,1,1,1,1,1]> : vector<8xi32> } : (index, index, index, index, index, index, memref<?xf32>) -> ()
    return
  }

  // CHECK: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  func @main() {
    // allocate CPU memory.
    %2 = memref.alloc() : memref<16xf32>

    %5 = memref.cast %2 : memref<16xf32> to memref<?xf32>

    // populate initial values.
    %cst = constant 1.23 : f32
    call @mcpuMemset(%5, %cst) : (memref<?xf32>, f32) -> ()

    // allocate GPU memory.
    %8 = call @mgpuMemAlloc(%5) : (memref<?xf32>) -> (memref<?xf32>)

    // copy direction constants.
    %cst_h2d = constant 1 : i32
    %cst_d2h = constant 2 : i32

    // launch kernel.
    call @lds_fillzero(%8) : (memref<?xf32>) -> ()

    // transfer data GPU -> CPU.
    call @mgpuMemCopy(%8, %5, %cst_d2h) : (memref<?xf32>, memref<?xf32>, i32) -> ()

    // print result.
    %9 = memref.cast %5 : memref<?xf32> to memref<*xf32>
    call @print_memref_f32(%9) : (memref<*xf32>) -> ()

    // dellocate GPU memory.
    call @mgpuMemDealloc(%8) : (memref<?xf32>) -> ()

    // deallocate CPU memory.
    memref.dealloc %2 : memref<16xf32>

    return
  }

  func private @mcpuMemset(%ptr : memref<?xf32>, %value: f32) -> ()
  func private @mgpuMemAlloc(%ptr : memref<?xf32>) -> (memref<?xf32>)
  func private @mgpuMemDealloc(%ptr : memref<?xf32>) -> ()
  func private @mgpuMemCopy(%src : memref<?xf32>, %dst : memref<?xf32>, %dir : i32) -> ()
  func private @print_memref_f32(%ptr : memref<*xf32>) -> ()
}
