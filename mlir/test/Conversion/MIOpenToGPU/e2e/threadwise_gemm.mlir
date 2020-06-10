// RUN: mlir-opt -test-miopen-lowering-gpu-module -convert-miopen-to-gpu="kernel-name=threadwise_gemm_kernel gpu-module-name=gpu_kernels" %s | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_kernels {
    gpu.func @threadwise_gemm_kernel(%arg0 : memref<4x8xf32>, %arg1 : memref<4x8xf32>, %arg2 : memref<8x8xf32>) kernel {
      miopen.threadwise_gemm(%arg0, %arg1, %arg2) : memref<4x8xf32>, memref<4x8xf32>, memref<8x8xf32>
      gpu.return
    }
  }
  
  func @threadwise_gemm(%arg0 : memref<4x8xf32>, %arg1 : memref<4x8xf32>, %arg2 : memref<8x8xf32>) {
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %arg0, %arg1, %arg2) { kernel = @gpu_kernels::@threadwise_gemm_kernel} : (index, index, index, index, index, index, memref<4x8xf32>, memref<4x8xf32>, memref<8x8xf32>) -> ()
    return
  }
  
  // CHECK:       4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4
  // CHECK-NEXT:  4,   4,   4,   4,   4,   4,   4,   4

  func @main() {
    // allocate CPU memory.
    %0 = alloc() : memref<4x8xf32>
    %1 = alloc() : memref<4x8xf32>
    %2 = alloc() : memref<8x8xf32>
  
    %3 = memref_cast %0 : memref<4x8xf32> to memref<?x?xf32>
    %4 = memref_cast %1 : memref<4x8xf32> to memref<?x?xf32>
    %5 = memref_cast %2 : memref<8x8xf32> to memref<?x?xf32>
  
    // populate initial values.
    %cst1 = constant 1.0 : f32
    %cst0 = constant 0.0 : f32
    call @mcpuMemset2DFloat(%3, %cst1) : (memref<?x?xf32>, f32) -> ()
    call @mcpuMemset2DFloat(%4, %cst1) : (memref<?x?xf32>, f32) -> ()
    call @mcpuMemset2DFloat(%5, %cst0) : (memref<?x?xf32>, f32) -> ()
  
    %lhs_cpu = memref_cast %3 : memref<?x?xf32> to memref<*xf32>
    %rhs_cpu = memref_cast %4 : memref<?x?xf32> to memref<*xf32>
    %output_cpu = memref_cast %5 : memref<?x?xf32> to memref<*xf32>

    // Print data on CPU prior to kernel launch.
    // call @print_memref_f32(%lhs_cpu) : (memref<*xf32>) -> ()
    // call @print_memref_f32(%rhs_cpu) : (memref<*xf32>) -> ()
    // call @print_memref_f32(%output_cpu) : (memref<*xf32>) -> ()

    // allocate GPU memory.
    %6 = call @mgpuMemAlloc2DFloat(%3) : (memref<?x?xf32>) -> (memref<?x?xf32>)
    %7 = call @mgpuMemAlloc2DFloat(%4) : (memref<?x?xf32>) -> (memref<?x?xf32>)
    %8 = call @mgpuMemAlloc2DFloat(%5) : (memref<?x?xf32>) -> (memref<?x?xf32>)
  
    // copy direction constants.
    %cst_h2d = constant 1 : i32
    %cst_d2h = constant 2 : i32
  
    // transfer data CPU -> GPU.
    call @mgpuMemCopy2DFloat(%3, %6, %cst_h2d) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
    call @mgpuMemCopy2DFloat(%4, %7, %cst_h2d) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
    call @mgpuMemCopy2DFloat(%5, %8, %cst_h2d) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
  
    %lhs = memref_cast %6 : memref<?x?xf32> to memref<4x8xf32>
    %rhs = memref_cast %7 : memref<?x?xf32> to memref<4x8xf32>
    %output = memref_cast %8 : memref<?x?xf32> to memref<8x8xf32>

    // launch kernel.
    call @threadwise_gemm(%lhs, %rhs, %output) : (memref<4x8xf32>, memref<4x8xf32>, memref<8x8xf32>) -> ()
  
    // transfer data GPU -> CPU.
    call @mgpuMemCopy2DFloat(%8, %5, %cst_d2h) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
    call @mgpuMemCopy2DFloat(%7, %4, %cst_d2h) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
    call @mgpuMemCopy2DFloat(%6, %3, %cst_d2h) : (memref<?x?xf32>, memref<?x?xf32>, i32) -> ()
  
    // Print result after GPU kernel execution.
    //call @print_memref_f32(%lhs_cpu) : (memref<*xf32>) -> ()
    //call @print_memref_f32(%rhs_cpu) : (memref<*xf32>) -> ()
    call @print_memref_f32(%output_cpu) : (memref<*xf32>) -> ()

    // dellocate GPU memory.
    call @mgpuMemDealloc2DFloat(%6) : (memref<?x?xf32>) -> ()
    call @mgpuMemDealloc2DFloat(%7) : (memref<?x?xf32>) -> ()
    call @mgpuMemDealloc2DFloat(%8) : (memref<?x?xf32>) -> ()
  
    // deallocate CPU memory.
    dealloc %0 : memref<4x8xf32>
    dealloc %1 : memref<4x8xf32>
    dealloc %2 : memref<8x8xf32>
  
    return
  }
  
  func @mcpuMemset2DFloat(%ptr : memref<?x?xf32>, %value: f32) -> ()
  func @mgpuMemAlloc2DFloat(%ptr : memref<?x?xf32>) -> (memref<?x?xf32>)
  func @mgpuMemDealloc2DFloat(%ptr : memref<?x?xf32>) -> ()
  func @mgpuMemCopy2DFloat(%src : memref<?x?xf32>, %dst : memref<?x?xf32>, %dir : i32) -> ()
  func @print_memref_f32(%ptr : memref<*xf32>) -> ()
}
