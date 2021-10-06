// RUN: mlir-opt -test-miopen-lowering-gpu-module -convert-miopen-to-gpu %s | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_kernels {
    gpu.func @threadwise_gemm_kernel(%arg0 : memref<1x4x8xf32>, %arg1 : memref<1x4x8xf32>, %arg2 : memref<1x8x8xf32>) kernel {
      miopen.threadwise_gemm(%arg0, %arg1, %arg2) : memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>
      gpu.return
    }
  }
  
  func @threadwise_gemm(%arg0 : memref<1x4x8xf32>, %arg1 : memref<1x4x8xf32>, %arg2 : memref<1x8x8xf32>) {
    %cst = constant 1 : index
    "gpu.launch_func"(%cst, %cst, %cst, %cst, %cst, %cst, %arg0, %arg1, %arg2) { kernel = @gpu_kernels::@threadwise_gemm_kernel, operand_segment_sizes = dense<[0,1,1,1,1,1,1,3]> : vector<8xi32> } : (index, index, index, index, index, index, memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>) -> ()
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
    %0 = alloc() : memref<1x4x8xf32>
    %1 = alloc() : memref<1x4x8xf32>
    %2 = alloc() : memref<1x8x8xf32>
  
    %3 = memref_cast %0 : memref<1x4x8xf32> to memref<?x?x?xf32>
    %4 = memref_cast %1 : memref<1x4x8xf32> to memref<?x?x?xf32>
    %5 = memref_cast %2 : memref<1x8x8xf32> to memref<?x?x?xf32>
  
    // populate initial values.
    %cst1 = constant 1.0 : f32
    %cst0 = constant 0.0 : f32
    call @mcpuMemset3DFloat(%3, %cst1) : (memref<?x?x?xf32>, f32) -> ()
    call @mcpuMemset3DFloat(%4, %cst1) : (memref<?x?x?xf32>, f32) -> ()
    call @mcpuMemset3DFloat(%5, %cst0) : (memref<?x?x?xf32>, f32) -> ()
  
    %lhs_cpu = memref_cast %3 : memref<?x?x?xf32> to memref<*xf32>
    %rhs_cpu = memref_cast %4 : memref<?x?x?xf32> to memref<*xf32>
    %output_cpu = memref_cast %5 : memref<?x?x?xf32> to memref<*xf32>

    // Print data on CPU prior to kernel launch.
    // call @print_memref_f32(%lhs_cpu) : (memref<*xf32>) -> ()
    // call @print_memref_f32(%rhs_cpu) : (memref<*xf32>) -> ()
    // call @print_memref_f32(%output_cpu) : (memref<*xf32>) -> ()

    // allocate GPU memory.
    %6 = call @mgpuMemAlloc3DFloat(%3) : (memref<?x?x?xf32>) -> (memref<?x?x?xf32>)
    %7 = call @mgpuMemAlloc3DFloat(%4) : (memref<?x?x?xf32>) -> (memref<?x?x?xf32>)
    %8 = call @mgpuMemAlloc3DFloat(%5) : (memref<?x?x?xf32>) -> (memref<?x?x?xf32>)
  
    // copy direction constants.
    %cst_h2d = constant 1 : i32
    %cst_d2h = constant 2 : i32
  
    // transfer data CPU -> GPU.
    call @mgpuMemCopy3DFloat(%3, %6, %cst_h2d) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy3DFloat(%4, %7, %cst_h2d) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy3DFloat(%5, %8, %cst_h2d) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
  
    %lhs = memref_cast %6 : memref<?x?x?xf32> to memref<1x4x8xf32>
    %rhs = memref_cast %7 : memref<?x?x?xf32> to memref<1x4x8xf32>
    %output = memref_cast %8 : memref<?x?x?xf32> to memref<1x8x8xf32>

    // launch kernel.
    call @threadwise_gemm(%lhs, %rhs, %output) : (memref<1x4x8xf32>, memref<1x4x8xf32>, memref<1x8x8xf32>) -> ()
  
    // transfer data GPU -> CPU.
    call @mgpuMemCopy3DFloat(%8, %5, %cst_d2h) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy3DFloat(%7, %4, %cst_d2h) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
    call @mgpuMemCopy3DFloat(%6, %3, %cst_d2h) : (memref<?x?x?xf32>, memref<?x?x?xf32>, i32) -> ()
  
    // Print result after GPU kernel execution.
    //call @print_memref_f32(%lhs_cpu) : (memref<*xf32>) -> ()
    //call @print_memref_f32(%rhs_cpu) : (memref<*xf32>) -> ()
    call @print_memref_f32(%output_cpu) : (memref<*xf32>) -> ()

    // dellocate GPU memory.
    call @mgpuMemDealloc3DFloat(%6) : (memref<?x?x?xf32>) -> ()
    call @mgpuMemDealloc3DFloat(%7) : (memref<?x?x?xf32>) -> ()
    call @mgpuMemDealloc3DFloat(%8) : (memref<?x?x?xf32>) -> ()
  
    // deallocate CPU memory.
    dealloc %0 : memref<1x4x8xf32>
    dealloc %1 : memref<1x4x8xf32>
    dealloc %2 : memref<1x8x8xf32>
  
    return
  }
  
  func private @mcpuMemset3DFloat(%ptr : memref<?x?x?xf32>, %value: f32) -> ()
  func private @mgpuMemAlloc3DFloat(%ptr : memref<?x?x?xf32>) -> (memref<?x?x?xf32>)
  func private @mgpuMemDealloc3DFloat(%ptr : memref<?x?x?xf32>) -> ()
  func private @mgpuMemCopy3DFloat(%src : memref<?x?x?xf32>, %dst : memref<?x?x?xf32>, %dir : i32) -> ()
  func private @print_memref_f32(%ptr : memref<*xf32>) -> ()
}
