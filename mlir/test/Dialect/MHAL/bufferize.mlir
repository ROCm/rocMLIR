// RUN: rocmlir-opt --mhal-bufferize --split-input-file %s | FileCheck %s

// COM: Exercises MHALBufferizePass (external/mlir-hal/lib/Dialect/MHAL/
// COM: Transforms/Bufferize.cpp). The pass is a thin wrapper around
// COM: bufferization::bufferizeOp with the partial-bufferization options
// COM: and an opFilter restricted to mhal::MHALDialect. Post-#2333 MHAL is
// COM: attribute-only and has no ops, so the pass is effectively a no-op
// COM: that must still register its dependent dialects, walk each func.func,
// COM: and leave the IR unchanged.

// COM: ---- 1: a fully bufferized function passes through unchanged.

// CHECK-LABEL: func.func @already_bufferized
// CHECK: arith.constant
// CHECK-NEXT: %[[A:.+]] = memref.alloc
// CHECK-NEXT: linalg.fill
// CHECK-NEXT: memref.dealloc
// CHECK-NEXT: return
func.func @already_bufferized() {
  %cst = arith.constant 0.0 : f32
  %0 = memref.alloc() : memref<8x8xf32>
  linalg.fill ins(%cst : f32) outs(%0 : memref<8x8xf32>)
  memref.dealloc %0 : memref<8x8xf32>
  return
}

// -----

// COM: ---- 2: an empty function body must not crash the pass.

// CHECK-LABEL: func.func @empty_body
// CHECK-NEXT: return
func.func @empty_body() {
  return
}

// -----

// COM: ---- 3: MHAL-specific argument attributes (mhal.read_access,
// COM: mhal.write_access) are preserved on the bufferized func boundary.

// CHECK-LABEL: func.func @preserves_mhal_arg_attrs
// CHECK-SAME: %{{.*}}: memref<32xf32> {mhal.read_access}
// CHECK-SAME: %{{.*}}: memref<32xf32> {mhal.write_access}
func.func @preserves_mhal_arg_attrs(%arg0: memref<32xf32> {mhal.read_access},
                                    %arg1: memref<32xf32> {mhal.write_access}) {
  memref.copy %arg0, %arg1 : memref<32xf32> to memref<32xf32>
  return
}

// -----

// COM: ---- 4: a func carrying mhal.targets keeps that attribute through the
// COM: bufferize pass intact (the opFilter does not strip discardable
// COM: attributes).

// CHECK-LABEL: func.func @preserves_mhal_targets
// CHECK-SAME: attributes {mhal.targets = [#mhal.kernel_pkg<GPU = gfx90a : preserves_mhal_targets [16, 64]
func.func @preserves_mhal_targets(%arg0: memref<16xf32>) attributes {mhal.targets = [
  #mhal.kernel_pkg<GPU = "gfx90a" : preserves_mhal_targets [16, 64]
    -> #mhal.target_obj<ELF = "gfx90a" -> "B">>]} {
  return
}

// -----

// COM: ---- 5: multiple functions in one module are each visited.

// CHECK-LABEL: func.func @first
// CHECK-NEXT: return
// CHECK-LABEL: func.func @second
// CHECK-NEXT: return
module {
  func.func @first() {
    return
  }
  func.func @second() {
    return
  }
}
