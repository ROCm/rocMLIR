// RUN: mlir-rocm-runner %s \
// RUN:   --bare-ptr-memref-kernels=false \
// RUN:   --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   --entry-point-result=void \
// RUN: | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_kernels {
    gpu.func @vecadd_kernel(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) workgroup(%arg3 : memref<16xf32, 3>) private(%arg4 : memref<16xf32, 5>) kernel {
      %tx = gpu.thread_id x
      %a = memref.load %arg0[%tx] : memref<?xf32>
      %b = memref.load %arg1[%tx] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      memref.store %c, %arg2[%tx] : memref<?xf32>
      gpu.return
    }
  }

  func.func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
    %cst = arith.constant 1 : index
    %cst0 = arith.constant 0 : index
    %cst2 = memref.dim %arg0, %cst0 : memref<?xf32>
    "gpu.launch_func"(%cst, %cst, %cst, %cst2, %cst, %cst, %arg0, %arg1, %arg2) { kernel = @gpu_kernels::@vecadd_kernel, operand_segment_sizes = dense<[0,1,1,1,1,1,1,0,3]> : vector<9xi32> } : (index, index, index, index, index, index, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
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
    %6 = gpu.alloc () : memref<16xf32>
    %7 = gpu.alloc () : memref<16xf32>
    %8 = gpu.alloc () : memref<16xf32>

    // transfer data CPU -> GPU.
    gpu.memcpy %6, %0 : memref<16xf32>, memref<16xf32>
    gpu.memcpy %7, %1 : memref<16xf32>, memref<16xf32>

    %26 = memref.cast %6 : memref<16xf32> to memref<?xf32>
    %27 = memref.cast %7 : memref<16xf32> to memref<?xf32>
    %28 = memref.cast %8 : memref<16xf32> to memref<?xf32>

    // launch kernel.
    call @vecadd(%26, %27, %28) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()

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
}
