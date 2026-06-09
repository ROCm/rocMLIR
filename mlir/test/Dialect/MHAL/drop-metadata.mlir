// RUN: rocmlir-opt --mhal-drop-binary-metadata --split-input-file %s | FileCheck %s

// COM: Exercises MHALDropBinaryMetadataPass (external/mlir-hal/lib/Dialect/
// COM: MHAL/Transforms/DropMetadata.cpp). For each top-level gpu.binary in
// COM: the module the pass clears discardable attributes on the binary op
// COM: and rebuilds every gpu.object inside it preserving only target,
// COM: format, and the raw binary string -- dropping the 'properties' dict
// COM: and the 'kernels' table.

// COM: ---- 1: properties and kernels are stripped, target/format/binary
// COM: stay.

// CHECK-LABEL: gpu.binary @bin_basic
// CHECK-SAME: [#gpu.object<#rocdl.target<chip = "gfx90a">, "RAW">]
// CHECK-NOT: kernels
// CHECK-NOT: properties
gpu.binary @bin_basic [
  #gpu.object<#rocdl.target<chip = "gfx90a">,
              properties = {foo = "bar", count = 7 : i64,
                            k = [#mhal.mhal.prefill<0, 0.0 : f32>]},
              kernels = #gpu.kernel_table<[
                #gpu.kernel_metadata<"k", () -> (), metadata = {x = 1 : i32}>]>,
              "RAW">]

// -----

// COM: ---- 2: discardable attributes on the gpu.binary op itself (e.g.
// COM: rock.blocks_per_cu) are dropped via setDiscardableAttrs({}).

// CHECK-LABEL: gpu.binary @bin_with_discardables
// CHECK-NOT: rock.blocks_per_cu
// CHECK-NOT: custom_attr
gpu.binary @bin_with_discardables {rock.blocks_per_cu = 4 : i32, custom_attr = "drop_me"} [
  #gpu.object<#rocdl.target<chip = "gfx90a">, "RAW">]

// -----

// COM: ---- 3: multiple objects (e.g. fat binary for several arches) all
// COM: have their metadata stripped in one pass run.

// CHECK-LABEL: gpu.binary @bin_multi_obj
// CHECK-SAME: [#gpu.object<#rocdl.target<chip = "gfx90a">, "RAW_A">,
// CHECK-SAME: #gpu.object<#rocdl.target<chip = "gfx942">, "RAW_B">]
// CHECK-NOT: properties
// CHECK-NOT: kernels
gpu.binary @bin_multi_obj [
  #gpu.object<#rocdl.target<chip = "gfx90a">,
              properties = {O = 3 : i32},
              kernels = #gpu.kernel_table<[#gpu.kernel_metadata<"a", () -> ()>]>,
              "RAW_A">,
  #gpu.object<#rocdl.target<chip = "gfx942">,
              properties = {O = 2 : i32},
              kernels = #gpu.kernel_table<[#gpu.kernel_metadata<"b", () -> ()>]>,
              "RAW_B">]

// -----

// COM: ---- 4: a binary that already has no metadata is idempotent under
// COM: the pass.

// CHECK-LABEL: gpu.binary @bin_already_clean
// CHECK-SAME: [#gpu.object<#rocdl.target<chip = "gfx90a">, "RAW">]
gpu.binary @bin_already_clean [
  #gpu.object<#rocdl.target<chip = "gfx90a">, "RAW">]

// -----

// COM: ---- 5: only TOP-LEVEL binaries are processed by getOps<BinaryOp>().
// COM: A binary nested inside an inner module is left untouched.

// CHECK-LABEL: gpu.binary @top_bin
// CHECK-NOT: properties
// CHECK: module @nested
// CHECK: gpu.binary @nested_bin
// CHECK-SAME: properties = {keep_me = "yes"}
module {
  gpu.binary @top_bin [
    #gpu.object<#rocdl.target<chip = "gfx90a">,
                properties = {drop_me = "yes"},
                "RAW">]
  module @nested {
    gpu.binary @nested_bin [
      #gpu.object<#rocdl.target<chip = "gfx90a">,
                  properties = {keep_me = "yes"},
                  "RAW">]
  }
}

// -----

// COM: ---- 6: a module with no gpu.binary at all is a no-op (covers the
// COM: empty-loop path through runOnOperation).

// CHECK-LABEL: func.func @no_binary
// CHECK-NEXT: return
func.func @no_binary() {
  return
}
