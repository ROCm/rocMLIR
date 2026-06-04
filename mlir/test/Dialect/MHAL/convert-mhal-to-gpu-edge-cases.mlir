// RUN: rocmlir-opt --convert-mhal-to-gpu --split-input-file %s | FileCheck %s

// COM: Edge-case coverage for ConvertMHALToGPUPass (external/mlir-hal/lib/
// COM: Conversion/MHALToGPU/MHALToGPU.cpp). The companion file
// COM: convert-mhal-to-gpu.mlir exercises the basic happy-path; this file
// COM: targets the secondary branches of lowerKernelCallToGpu and
// COM: KernelFuncCallRewritePattern that the basic file leaves cold:
// COM: pre-existing gpu.binary reuse, wrong-arity launch dims, kernel
// COM: operand defined by memref.alloc / gpu.alloc, multiple copy-back
// COM: tokens, all-scalar (no-memref) operand lists, and the early-exit
// COM: when the call carries results.

// COM: ---- 1: two calls to the same kernel must reuse a single gpu.binary.
// COM: First rewrite hits the `if (!binaryOp)` create-path (MHALToGPU.cpp
// COM: L79-86); second rewrite finds the binary already in the symbol
// COM: table and falls through. We CHECK that only ONE gpu.binary appears
// COM: in the output and that there are TWO gpu.launch_func ops referring
// COM: to it.

// CHECK-LABEL: func.func @two_calls_share_binary
// CHECK-COUNT-2: gpu.launch_func {{.*}} @shared_kernel_module::@shared_kernel
// CHECK-NOT: gpu.launch_func
// CHECK: gpu.binary @shared_kernel_module
// CHECK-NOT: gpu.binary
module {
  func.func private @shared_kernel(%arg0: memref<8xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : shared_kernel [1, 8]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "BIN">>>]} {
    return
  }
  func.func @two_calls_share_binary(%a: memref<8xf32>, %b: memref<8xf32>) {
    func.call @shared_kernel(%a) : (memref<8xf32>) -> ()
    func.call @shared_kernel(%b) : (memref<8xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 2: launch dims with arity 1 trigger the
// COM: `launchDims.size() != 2` notifyMatchFailure at MHALToGPU.cpp L69-70.
// COM: The pattern returns failure so the original func.call survives and
// COM: NO gpu.launch_func is emitted.

// CHECK-LABEL: func.func @bad_launch_dims_one
// CHECK: call @kernel_one_dim
// CHECK-NOT: gpu.launch_func
module {
  func.func private @kernel_one_dim(%arg0: memref<4xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_one_dim [16]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @bad_launch_dims_one(%arg0: memref<4xf32>) {
    func.call @kernel_one_dim(%arg0) : (memref<4xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 3: same `launchDims.size() != 2` guard, this time exercised
// COM: from the other side with a 3-element launch dim list.

// CHECK-LABEL: func.func @bad_launch_dims_three
// CHECK: call @kernel_three_dims
// CHECK-NOT: gpu.launch_func
module {
  func.func private @kernel_three_dims(%arg0: memref<4xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_three_dims [16, 4, 8]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @bad_launch_dims_three(%arg0: memref<4xf32>) {
    func.call @kernel_three_dims(%arg0) : (memref<4xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 4: a memref operand defined by memref.alloc (rather than a
// COM: BlockArgument) takes the moveMemory(oprAllocOp != null) path in
// COM: MHALToGPU.cpp L117-118 and L132-138. Because the alloc'd memref is
// COM: also returned (via memref.dealloc on the host), it has a non-on-
// COM: device user, so the `else` branch (L136-137) runs:
// COM: anchor->replaceUsesOfWith(opr, dstMem) + runCopy().

// CHECK-LABEL: func.func @operand_from_memref_alloc
// CHECK: %[[ALLOC:.+]] = memref.alloc() : memref<8xf32>
// CHECK: gpu.alloc async
// CHECK: gpu.memcpy async
// CHECK: gpu.launch_func {{.*}} @kernel_alloc_module::@kernel_alloc
// CHECK: memref.dealloc %[[ALLOC]]
module {
  func.func private @kernel_alloc(%arg0: memref<8xf32> {mhal.read_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_alloc [1, 8]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @operand_from_memref_alloc() {
    %buf = memref.alloc() : memref<8xf32>
    func.call @kernel_alloc(%buf) : (memref<8xf32>) -> ()
    memref.dealloc %buf : memref<8xf32>
    return
  }
}

// -----

// COM: ---- 5: two write_access memref args produce two copy-back
// COM: gpu.memcpy ops after the launch, so `tokens.size() > 1` at
// COM: MHALToGPU.cpp L194-195 is true and an extra gpu.wait is emitted to
// COM: merge them before the final wait. Both args should come back to host.

// CHECK-LABEL: func.func @two_write_access_args
// CHECK: gpu.launch_func {{.*}} @kernel_two_writes_module::@kernel_two_writes
// CHECK-COUNT-2: gpu.memcpy async
// CHECK: gpu.wait async
// CHECK: gpu.wait
module {
  func.func private @kernel_two_writes(%a: memref<16xf32> {mhal.write_access},
                                       %b: memref<16xf32> {mhal.write_access})
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_two_writes [2, 16]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @two_write_access_args(%a: memref<16xf32>, %b: memref<16xf32>) {
    func.call @kernel_two_writes(%a, %b) : (memref<16xf32>, memref<16xf32>) -> ()
    return
  }
}

// -----

// COM: ---- 6: a bufferized call MUST have zero results. If we feed the
// COM: pattern a call whose callee returns a value, the early-exit at
// COM: KernelFuncCallRewritePattern::matchAndRewrite (MHALToGPU.cpp
// COM: L215-217) fires with "expected bufferized call (zero results)" and
// COM: the call survives untouched.

// CHECK-LABEL: func.func @call_with_result_skipped
// CHECK: %{{.+}} = call @kernel_with_result
// CHECK-NOT: gpu.launch_func
module {
  func.func private @kernel_with_result(%arg0: memref<4xf32> {mhal.read_access}) -> i32
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_with_result [1, 4]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    %0 = arith.constant 0 : i32
    return %0 : i32
  }
  func.func @call_with_result_skipped(%arg0: memref<4xf32>) -> i32 {
    %r = func.call @kernel_with_result(%arg0) : (memref<4xf32>) -> i32
    return %r : i32
  }
}

// -----

// COM: ---- 7: a kernel whose operand list is entirely scalar exercises the
// COM: `asyncDeps.empty()` branch at MHALToGPU.cpp L167-168: with no memref
// COM: operands, moveMemory is never called, asyncDeps stays empty, and a
// COM: dependency-free gpu.wait must be synthesized so the launch has an
// COM: async-token operand.

// CHECK-LABEL: func.func @all_scalar_args
// CHECK-DAG: %[[C0:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[C1:.+]] = arith.constant 2.500000e+00 : f32
// CHECK: %[[W:.+]] = gpu.wait async
// CHECK: gpu.launch_func {{.*}} @kernel_scalars_module::@kernel_scalars {{.*}} args(%[[C0]] : i32, %[[C1]] : f32)
module {
  func.func private @kernel_scalars(%a: i32, %b: f32)
      attributes {mhal.targets = [#mhal.kernel_pkg<GPU = "amdgcn-amd-amdhsa:gfx90a" : kernel_scalars [1, 1]
        -> #mhal.target_obj<ELF = "amdgcn-amd-amdhsa:gfx90a" ->
             #gpu.object<#rocdl.target<chip = "gfx90a">, "B">>>]} {
    return
  }
  func.func @all_scalar_args() {
    %i = arith.constant 1 : i32
    %f = arith.constant 2.5 : f32
    func.call @kernel_scalars(%i, %f) : (i32, f32) -> ()
    return
  }
}
