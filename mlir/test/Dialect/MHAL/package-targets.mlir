// RUN: rocmlir-opt --mhal-package-targets --split-input-file %s | FileCheck %s

// COM: Exercises MHALPackageTargetsPass (external/mlir-hal/lib/Dialect/MHAL/
// COM: Transforms/PackageTargets.cpp). The pass walks inner ModuleOps with
// COM: the mhal.module attribute, for each gpu.binary in such a module it
// COM: builds a #mhal.target_obj wrapping the gpu.object, builds a
// COM: #mhal.kernel_pkg keyed on the host func referenced by kernel metadata
// COM: 'original_func', and attaches the resulting list to that host func via
// COM: mhal.targets = [...]. Finally the kernel module is erased.

// CHECK-LABEL: func.func private @host_part_0
// CHECK-SAME: attributes {mhal.targets =
// CHECK-SAME: #mhal.kernel_pkg<GPU = gfx90a : host_part_0 [16, 64]
// CHECK-SAME: -> #mhal.target_obj<ELF = gfx90a -> #gpu.object<#rocdl.target<chip = "gfx90a">,
// CHECK-SAME: kernels = <[#gpu.kernel_metadata<"host_part_0",
// CHECK-NOT: module @__xmodule
// CHECK-NOT: gpu.binary
module {
  func.func private @host_part_0(%arg0: memref<256xf32> {mhal.read_access},
                                 %arg1: memref<256xf32> {mhal.write_access}) {
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a", mhal.module} {
    gpu.binary @host_part_0_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_part_0", () -> (), metadata = {
                      original_func = @host_part_0,
                      grid_size = 16 : i32,
                      block_size = 64 : i32}>]>,
                  "BIN0">]
  }
}

// -----

// COM: Two kernel modules for different architectures each contribute a
// COM: package for the same host function, producing mhal.targets with two
// COM: entries (one per arch).

// CHECK-LABEL: func.func private @host_multi
// CHECK-SAME: mhal.targets = [
// CHECK-DAG: #mhal.kernel_pkg<GPU = gfx90a : host_multi [8, 128] -> #mhal.target_obj<ELF = gfx90a -> #gpu.object<#rocdl.target<chip = "gfx90a">,
// CHECK-DAG: #mhal.kernel_pkg<GPU = gfx942 : host_multi [16, 64] -> #mhal.target_obj<ELF = gfx942 -> #gpu.object<#rocdl.target<chip = "gfx942">,
// CHECK-NOT: module @__xmodule
module {
  func.func private @host_multi(%arg0: memref<128xf32> {mhal.read_access}) {
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a", mhal.module} {
    gpu.binary @host_multi_bin90a [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_multi", () -> (), metadata = {
                      original_func = @host_multi,
                      grid_size = 8 : i32,
                      block_size = 128 : i32}>]>,
                  "BIN_90A">]
  }
  module @__xmodule_gfx942 attributes {mhal.arch = "gfx942", mhal.module} {
    gpu.binary @host_multi_bin942 [
      #gpu.object<#rocdl.target<chip = "gfx942">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_multi", () -> (), metadata = {
                      original_func = @host_multi,
                      grid_size = 16 : i32,
                      block_size = 64 : i32}>]>,
                  "BIN_942">]
  }
}

// -----

// COM: A single kernel module can hold multiple gpu.binary ops and multiple
// COM: kernel metadata entries that target distinct host functions. Each
// COM: host function ends up with exactly one mhal.targets entry pointing at
// COM: the binary that contains its kernel.

// CHECK-LABEL: func.func private @host_a
// CHECK-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = gfx90a : host_a [1, 256]
// CHECK-LABEL: func.func private @host_b
// CHECK-SAME: mhal.targets = [#mhal.kernel_pkg<GPU = gfx90a : host_b [2, 128]
// CHECK-NOT: module @__xmodule
module {
  func.func private @host_a(%arg0: memref<64xf32> {mhal.read_access}) {
    return
  }
  func.func private @host_b(%arg0: memref<32xf32> {mhal.read_access}) {
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a", mhal.module} {
    gpu.binary @host_a_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_a", () -> (), metadata = {
                      original_func = @host_a,
                      grid_size = 1 : i32,
                      block_size = 256 : i32}>]>,
                  "BIN_A">]
    gpu.binary @host_b_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_b", () -> (), metadata = {
                      original_func = @host_b,
                      grid_size = 2 : i32,
                      block_size = 128 : i32}>]>,
                  "BIN_B">]
  }
}

// -----

// COM: A nested module without the mhal.module marker is ignored: its
// COM: gpu.binary stays in place and no mhal.targets attribute is attached
// COM: to the host function.

// CHECK-LABEL: func.func private @host_unmarked
// CHECK-NOT: mhal.targets
// CHECK: module @__xmodule_unmarked
// CHECK: gpu.binary @host_unmarked_bin
module {
  func.func private @host_unmarked(%arg0: memref<8xf32>) {
    return
  }
  module @__xmodule_unmarked attributes {mhal.arch = "gfx90a"} {
    gpu.binary @host_unmarked_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"host_unmarked", () -> (), metadata = {
                      original_func = @host_unmarked,
                      grid_size = 1 : i32,
                      block_size = 32 : i32}>]>,
                  "BIN">]
  }
}

// -----

// COM: Kernel metadata whose original_func does not resolve to a func.func
// COM: in the outer module is silently skipped: no error, no crash, just no
// COM: mhal.targets attached anywhere. The kernel module is still erased.

// CHECK-LABEL: func.func private @existing_host
// CHECK-NOT: mhal.targets
// CHECK-NOT: module @__xmodule
module {
  func.func private @existing_host(%arg0: memref<8xf32>) {
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a", mhal.module} {
    gpu.binary @dangling_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"nope", () -> (), metadata = {
                      original_func = @does_not_exist,
                      grid_size = 1 : i32,
                      block_size = 32 : i32}>]>,
                  "BIN">]
  }
}

// -----

// COM: Kernel metadata with no 'original_func' attribute at all takes the
// COM: `if (auto attr = kernel.getAttr<SymbolRefAttr>("original_func"))`
// COM: false branch: the kernel is skipped silently, no mhal.targets is
// COM: attached, and the kernel module is still erased like in the previous
// COM: case.

// CHECK-LABEL: func.func private @host_no_orig_attr
// CHECK-NOT: mhal.targets
// CHECK-NOT: module @__xmodule
module {
  func.func private @host_no_orig_attr(%arg0: memref<8xf32>) {
    return
  }
  module @__xmodule_gfx90a attributes {mhal.arch = "gfx90a", mhal.module} {
    gpu.binary @no_orig_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  kernels = #gpu.kernel_table<[
                    #gpu.kernel_metadata<"orphan", () -> (), metadata = {
                      grid_size = 1 : i32,
                      block_size = 32 : i32}>]>,
                  "BIN">]
  }
}
