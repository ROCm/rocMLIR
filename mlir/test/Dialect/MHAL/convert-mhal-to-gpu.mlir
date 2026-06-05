// RUN: rocmlir-opt --convert-mhal-to-gpu --split-input-file %s | FileCheck %s

// COM: Exercises ConvertMHALToGPUPass (external/mlir-hal/lib/Conversion/
// COM: MHALToGPU/MHALToGPU.cpp). The pattern KernelFuncCallRewritePattern
// COM: rewrites a bufferized func.call (zero results) whose callee carries
// COM: mhal.targets = [#mhal.kernel_pkg<GPU = ..., kernel_name, [grid, block]
// COM: -> #mhal.target_obj<...>>] into gpu.binary + gpu.alloc + gpu.memcpy +
// COM: gpu.launch_func + gpu.wait, sets the gpu.container_module attribute on
// COM: the module, and strips the mhal.targets attr from each touched func.
// COM: The target_obj's binary payload must already be a #gpu.object so it can
// COM: be inserted into gpu.binary's objects list directly.

// COM: ---- 1: a single memref operand with no write_access should turn into
// COM: alloc + memcpy host->device + launch + wait. There is no back-copy.

// CHECK: module attributes {gpu.container_module}

// CHECK-LABEL: func.func @host_call_one_arg
// CHECK-NOT: mhal.targets
// CHECK-DAG: %[[ONE:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[BLOCK:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[GRID:.+]] = arith.constant 16 : index
// CHECK: gpu.wait async
// CHECK: gpu.alloc async
// CHECK: gpu.memcpy async
// CHECK: %[[LAUNCH:.+]] = gpu.launch_func async
// CHECK-SAME: @kernel_one_module::@kernel_one
// CHECK-SAME: blocks in (%[[GRID]], %[[ONE]], %[[ONE]])
// CHECK-SAME: threads in (%[[BLOCK]], %[[ONE]], %[[ONE]])
// CHECK: gpu.wait
// CHECK: gpu.binary @kernel_one_module
// CHECK-SAME: #gpu.object<#rocdl.target<chip = "gfx90a">, "BINARY">
// CHECK-NOT: func.call @kernel_one
module {
  func.func private @kernel_one(%arg0: memref<256xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_one [16, 64]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "BINARY">>>]} {
    return
  }
  func.func @host_call_one_arg(%arg0: memref<256xf32>) {
    func.call @kernel_one(%arg0) : (memref<256xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 2: a write_access argument triggers the copy-back path: after the
// COM: launch, an additional gpu.memcpy is emitted to bring device memory
// COM: back to the host memref.

// CHECK-LABEL: func.func @host_call_writes
// CHECK: gpu.alloc async
// CHECK: gpu.alloc async
// CHECK: %[[LAUNCH:.+]] = gpu.launch_func async
// CHECK-SAME: @kernel_inout_module::@kernel_inout
// CHECK: gpu.memcpy async
// CHECK: gpu.wait
module {
  func.func private @kernel_inout(%arg0: memref<128xf32> {mhal.read_access},
                                  %arg1: memref<128xf32> {mhal.write_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_inout [4, 256]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "BINARY">>>]} {
    return
  }
  func.func @host_call_writes(%in: memref<128xf32>, %out: memref<128xf32>) {
    func.call @kernel_inout(%in, %out) : (memref<128xf32>, memref<128xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 3: when mhal.targets contains both CPU and GPU packages, the
// COM: pattern uses getGPUTarget() (MHALToGPU.cpp) to pick the GPU one.
// COM: Here CPU comes first so the GPU package (named kernel_picked) must
// COM: be the one referenced in the launch.

// CHECK-LABEL: func.func @host_call_picks_gpu
// CHECK: gpu.launch_func {{.*}} @kernel_picked_module::@kernel_picked
// CHECK-NOT: kernel_cpu
module {
  func.func private @kernel_picked(%arg0: memref<16xf32> {mhal.read_access})
      attributes {mhal.targets = [
        #mhal.kernel_pkg<CPU = "x86_64" : kernel_cpu [1, 1]
          -> #mhal.target_obj<ELF = "x86_64" ->
               #gpu.object<#rocdl.target<chip = "gfx90a">, "CBIN">>>,
        #mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_picked [2, 64]
          -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
               #gpu.object<#rocdl.target<chip = "gfx90a">, "GBIN">>>]} {
    return
  }
  func.func @host_call_picks_gpu(%arg0: memref<16xf32>) {
    func.call @kernel_picked(%arg0) : (memref<16xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 4: a bufferized func.call to a function without any mhal.targets
// COM: attr MUST NOT be rewritten - the pattern returns notifyMatchFailure
// COM: ("callee has no mhal.targets[gpu]" from KernelFuncCallRewritePattern
// COM: ::matchAndRewrite). This guards plain host calls from being
// COM: accidentally moved to the device.

// CHECK-LABEL: func.func @host_call_no_target
// CHECK: call @plain_callee
// CHECK-NOT: gpu.launch_func
module {
  func.func private @plain_callee(%arg0: memref<8xf32>) {
    return
  }
  func.func @host_call_no_target(%arg0: memref<8xf32>) {
    func.call @plain_callee(%arg0) : (memref<8xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 5: a function with only a CPU target (no GPU) is also skipped
// COM: because getGPUTarget returns std::nullopt when no kernel package's
// COM: type is TargetType::GPU.

// CHECK-LABEL: func.func @host_call_only_cpu
// CHECK: call @cpu_only
// CHECK-NOT: gpu.launch_func
module {
  func.func private @cpu_only(%arg0: memref<8xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<CPU = "x86_64" : cpu_only [1, 1]
        -> #mhal.target_obj<ELF = "x86_64" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @host_call_only_cpu(%arg0: memref<8xf32>) {
    func.call @cpu_only(%arg0) : (memref<8xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 6: non-memref operands (scalars) skip moveMemory entirely and
// COM: are forwarded directly into the gpu.launch_func args list. Only the
// COM: memref operand goes through gpu.alloc/gpu.memcpy.

// CHECK-LABEL: func.func @host_call_scalar_arg
// CHECK-DAG: %[[C:.+]] = arith.constant 4.200000e+01 : f32
// CHECK: gpu.alloc async
// CHECK: gpu.launch_func {{.*}} args(%{{.*}} : memref<4xf32>, %[[C]] : f32)
module {
  func.func private @kernel_scalar(%arg0: memref<4xf32> {mhal.write_access}, %arg1: f32)
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_scalar [1, 4]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @host_call_scalar_arg(%arg0: memref<4xf32>) {
    %cst = arith.constant 4.200000e+01 : f32
    func.call @kernel_scalar(%arg0, %cst) : (memref<4xf32>, f32) -> ()
    return
  }
}
