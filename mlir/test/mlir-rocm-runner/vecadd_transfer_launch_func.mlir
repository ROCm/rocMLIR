// RUN: mlir-rocm-runner --target=gfx908 --feature=+code-object-v3 %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

module attributes {gpu.container_module} {
  gpu.module @gpu_kernels {
    gpu.func @vecadd_kernel(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) workgroup(%arg3 : memref<16xf32, 3>) private(%arg4 : memref<16xf32, 5>) kernel {
      %tx = "gpu.thread_id"() {dimension = "x"} : () -> (index)
      %a = load %arg0[%tx] : memref<?xf32>
      %b = load %arg1[%tx] : memref<?xf32>
      %c = addf %a, %b : f32
      store %c, %arg2[%tx] : memref<?xf32>
      gpu.return
    }
  }
  
  func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
    %cst = constant 1 : index
    %cst2 = dim %arg0, 0 : memref<?xf32>
    "gpu.launch_func"(%cst, %cst, %cst, %cst2, %cst, %cst, %arg0, %arg1, %arg2) { kernel = @gpu_kernels::@vecadd_kernel} : (index, index, index, index, index, index, memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
    return
  }
  
  // CHECK: [2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46, 2.46]
  func @main() {
    // allocate CPU memory.
    %0 = alloc() : memref<16xf32>
    %1 = alloc() : memref<16xf32>
    %2 = alloc() : memref<16xf32>
  
    %3 = memref_cast %0 : memref<16xf32> to memref<?xf32>
    %4 = memref_cast %1 : memref<16xf32> to memref<?xf32>
    %5 = memref_cast %2 : memref<16xf32> to memref<?xf32>
  
    // populate initial values.
    %cst = constant 1.23 : f32
    %cst0 = constant 0.0 : f32
    call @mcpuMemset(%3, %cst) : (memref<?xf32>, f32) -> ()
    call @mcpuMemset(%4, %cst) : (memref<?xf32>, f32) -> ()
    call @mcpuMemset(%5, %cst0) : (memref<?xf32>, f32) -> ()
  
    // allocate GPU memory.
    %6 = call @mgpuMemAlloc(%3) : (memref<?xf32>) -> (memref<?xf32>)
    %7 = call @mgpuMemAlloc(%4) : (memref<?xf32>) -> (memref<?xf32>)
    %8 = call @mgpuMemAlloc(%5) : (memref<?xf32>) -> (memref<?xf32>)
  
    // copy direction constants.
    %cst_h2d = constant 1 : i32
    %cst_d2h = constant 2 : i32
  
    // transfer data CPU -> GPU.
    call @mgpuMemCopy(%3, %6, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()
    call @mgpuMemCopy(%4, %7, %cst_h2d) : (memref<?xf32>, memref<?xf32>, i32) -> ()
  
    // launch kernel.
    call @vecadd(%6, %7, %8) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  
    // transfer data GPU -> CPU.
    call @mgpuMemCopy(%8, %5, %cst_d2h) : (memref<?xf32>, memref<?xf32>, i32) -> ()
  
    // print result.
    %9 = memref_cast %5 : memref<?xf32> to memref<*xf32>
    call @print_memref_f32(%9) : (memref<*xf32>) -> ()
  
    // dellocate GPU memory.
    call @mgpuMemDealloc(%6) : (memref<?xf32>) -> ()
    call @mgpuMemDealloc(%7) : (memref<?xf32>) -> ()
    call @mgpuMemDealloc(%8) : (memref<?xf32>) -> ()
  
    // deallocate CPU memory.
    dealloc %0 : memref<16xf32>
    dealloc %1 : memref<16xf32>
    dealloc %2 : memref<16xf32>
  
    return
  }
  
  func @mcpuMemset(%ptr : memref<?xf32>, %value: f32) -> ()
  func @mgpuMemAlloc(%ptr : memref<?xf32>) -> (memref<?xf32>)
  func @mgpuMemDealloc(%ptr : memref<?xf32>) -> ()
  func @mgpuMemCopy(%src : memref<?xf32>, %dst : memref<?xf32>, %dir : i32) -> ()
  func @print_memref_f32(%ptr : memref<*xf32>) -> ()
}
