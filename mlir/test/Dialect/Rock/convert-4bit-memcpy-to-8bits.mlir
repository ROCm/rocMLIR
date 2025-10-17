// filepath: \home\umayadav\repo\rocMLIR\mlir\test\Dialect\Rock\convert-4bit-memcpy-to-8bits.mlir
// RUN: rocmlir-opt --rock-convert-4bit-memcpy-to-8bit %s | FileCheck %s

// The pass rewrites gpu.alloc/dealloc of any 4-bit element (i4 or f4E2M1FN) to an
// i8 buffer whose last dimension is halved (ceil(N/2)), inserts an
// unrealized_conversion_cast back to the original 4-bit memref, and rewrites
// gpu.memcpy only when BOTH operands are such casts, replacing them with a
// memcpy on the raw i8 memrefs.

// ---------------------------------------------------------------------------
// Callee kernels (no allocs inside -> untouched).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @kernel_i4
func.func @kernel_i4(%in : memref<32xi4>, %out : memref<32xi4>) {
  func.return
}
// CHECK-LABEL: func.func @kernel_f4
func.func @kernel_f4(%in : memref<32xf4E2M1FN>, %out : memref<32xf4E2M1FN>) {
  func.return
}

// ---------------------------------------------------------------------------
// Two i4 allocs + memcpy between them (tests alloc + memcpy + dealloc rewrite).
// 4x10 -> 4x5
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @alloc_pair_i4
// CHECK: %[[A_RAW:.*]] = gpu.alloc() : memref<4x5xi8>
// CHECK: %[[A_CAST:.*]] = unrealized_conversion_cast %[[A_RAW]] : memref<4x5xi8> to memref<4x10xi4>
// CHECK: %[[B_RAW:.*]] = gpu.alloc() : memref<4x5xi8>
// CHECK: %[[B_CAST:.*]] = unrealized_conversion_cast %[[B_RAW]] : memref<4x5xi8> to memref<4x10xi4>
// CHECK: gpu.memcpy %[[B_RAW]], %[[A_RAW]] : memref<4x5xi8>, memref<4x5xi8>
// CHECK: gpu.dealloc %[[A_RAW]] : memref<4x5xi8>
// CHECK: gpu.dealloc %[[B_RAW]] : memref<4x5xi8>
func.func @alloc_pair_i4() {
  %a = gpu.alloc() : memref<4x10xi4>
  %b = gpu.alloc() : memref<4x10xi4>
  gpu.memcpy %b, %a : memref<4x10xi4>, memref<4x10xi4>
  gpu.dealloc %a : memref<4x10xi4>
  gpu.dealloc %b : memref<4x10xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Two f4E2M1FN allocs + memcpy between them (float 4-bit also rewritten).
// 4x10 -> 4x5
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @alloc_pair_f4
// CHECK: %[[A_RAW:.*]] = gpu.alloc() : memref<4x5xi8>
// CHECK: %[[A_CAST:.*]] = unrealized_conversion_cast %[[A_RAW]] : memref<4x5xi8> to memref<4x10xf4E2M1FN>
// CHECK: %[[B_RAW:.*]] = gpu.alloc() : memref<4x5xi8>
// CHECK: %[[B_CAST:.*]] = unrealized_conversion_cast %[[B_RAW]] : memref<4x5xi8> to memref<4x10xf4E2M1FN>
// CHECK: gpu.memcpy %[[B_RAW]], %[[A_RAW]] : memref<4x5xi8>, memref<4x5xi8>
// CHECK: gpu.dealloc %[[A_RAW]] : memref<4x5xi8>
// CHECK: gpu.dealloc %[[B_RAW]] : memref<4x5xi8>
func.func @alloc_pair_f4() {
  %a = gpu.alloc() : memref<4x10xf4E2M1FN>
  %b = gpu.alloc() : memref<4x10xf4E2M1FN>
  gpu.memcpy %b, %a : memref<4x10xf4E2M1FN>, memref<4x10xf4E2M1FN>
  gpu.dealloc %a : memref<4x10xf4E2M1FN>
  gpu.dealloc %b : memref<4x10xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// i4 process flow: arg_in -> in_tmp, kernel(in_tmp,out_tmp), out_tmp -> arg_out.
// Memcpys with one cast operand (no rewrite). Deallocs use raw.
// 32 -> 16
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @process_i4_32
// CHECK: %[[IN_RAW:.*]] = gpu.alloc() : memref<16xi8>
// CHECK: %[[IN_CAST:.*]] = unrealized_conversion_cast %[[IN_RAW]] : memref<16xi8> to memref<32xi4>
// CHECK: %[[OUT_RAW:.*]] = gpu.alloc() : memref<16xi8>
// CHECK: %[[OUT_CAST:.*]] = unrealized_conversion_cast %[[OUT_RAW]] : memref<16xi8> to memref<32xi4>
// CHECK: gpu.memcpy %[[IN_CAST]], %arg_in : memref<32xi4>, memref<32xi4>
// CHECK: func.call @kernel_i4(%[[IN_CAST]], %[[OUT_CAST]]) : (memref<32xi4>, memref<32xi4>) -> ()
// CHECK: gpu.memcpy %arg_out, %[[OUT_CAST]] : memref<32xi4>, memref<32xi4>
// CHECK: gpu.dealloc %[[IN_RAW]] : memref<16xi8>
// CHECK: gpu.dealloc %[[OUT_RAW]] : memref<16xi8>
func.func @process_i4_32(%arg_in : memref<32xi4>, %arg_out : memref<32xi4>) {
  %in_tmp = gpu.alloc() : memref<32xi4>
  %out_tmp = gpu.alloc() : memref<32xi4>
  gpu.memcpy %in_tmp, %arg_in : memref<32xi4>, memref<32xi4>
  func.call @kernel_i4(%in_tmp, %out_tmp) : (memref<32xi4>, memref<32xi4>) -> ()
  gpu.memcpy %arg_out, %out_tmp : memref<32xi4>, memref<32xi4>
  gpu.dealloc %in_tmp : memref<32xi4>
  gpu.dealloc %out_tmp : memref<32xi4>
  func.return
}

// ---------------------------------------------------------------------------
// f4 process flow: same as above but float 4-bit also rewritten.
// 32 -> 16; memcpys not rewritten (single cast operand).
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @process_f4_32
// CHECK: %[[IN_RAW:.*]] = gpu.alloc() : memref<16xi8>
// CHECK: %[[IN_CAST:.*]] = unrealized_conversion_cast %[[IN_RAW]] : memref<16xi8> to memref<32xf4E2M1FN>
// CHECK: %[[OUT_RAW:.*]] = gpu.alloc() : memref<16xi8>
// CHECK: %[[OUT_CAST:.*]] = unrealized_conversion_cast %[[OUT_RAW]] : memref<16xi8> to memref<32xf4E2M1FN>
// CHECK: gpu.memcpy %[[IN_CAST]], %arg_in : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
// CHECK: func.call @kernel_f4(%[[IN_CAST]], %[[OUT_CAST]]) : (memref<32xf4E2M1FN>, memref<32xf4E2M1FN>) -> ()
// CHECK: gpu.memcpy %arg_out, %[[OUT_CAST]] : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
// CHECK: gpu.dealloc %[[IN_RAW]] : memref<16xi8>
// CHECK: gpu.dealloc %[[OUT_RAW]] : memref<16xi8>
func.func @process_f4_32(%arg_in : memref<32xf4E2M1FN>, %arg_out : memref<32xf4E2M1FN>) {
  %in_tmp = gpu.alloc() : memref<32xf4E2M1FN>
  %out_tmp = gpu.alloc() : memref<32xf4E2M1FN>
  gpu.memcpy %in_tmp, %arg_in : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
  func.call @kernel_f4(%in_tmp, %out_tmp) : (memref<32xf4E2M1FN>, memref<32xf4E2M1FN>) -> ()
  gpu.memcpy %arg_out, %out_tmp : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
  gpu.dealloc %in_tmp : memref<32xf4E2M1FN>
  gpu.dealloc %out_tmp : memref<32xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// i4 multi-dim (odd last dim): 3x5x9 -> 3x5x5 raw.
// Memcpys single cast -> unchanged.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @process_i4_3x5x9
// CHECK: %[[IN_RAW:.*]] = gpu.alloc() : memref<3x5x5xi8>
// CHECK: %[[IN_CAST:.*]] = unrealized_conversion_cast %[[IN_RAW]] : memref<3x5x5xi8> to memref<3x5x9xi4>
// CHECK: %[[OUT_RAW:.*]] = gpu.alloc() : memref<3x5x5xi8>
// CHECK: %[[OUT_CAST:.*]] = unrealized_conversion_cast %[[OUT_RAW]] : memref<3x5x5xi8> to memref<3x5x9xi4>
// CHECK: gpu.memcpy %[[IN_CAST]], %arg_in : memref<3x5x9xi4>, memref<3x5x9xi4>
// CHECK: func.call @kernel_i4(%[[IN_CAST]], %[[OUT_CAST]]) : (memref<3x5x9xi4>, memref<3x5x9xi4>) -> ()
// CHECK: gpu.memcpy %arg_out, %[[OUT_CAST]] : memref<3x5x9xi4>, memref<3x5x9xi4>
// CHECK: gpu.dealloc %[[IN_RAW]] : memref<3x5x5xi8>
// CHECK: gpu.dealloc %[[OUT_RAW]] : memref<3x5x5xi8>
func.func @process_i4_3x5x9(%arg_in : memref<3x5x9xi4>, %arg_out : memref<3x5x9xi4>) {
  %in_tmp = gpu.alloc() : memref<3x5x9xi4>
  %out_tmp = gpu.alloc() : memref<3x5x9xi4>
  gpu.memcpy %in_tmp, %arg_in : memref<3x5x9xi4>, memref<3x5x9xi4>
  func.call @kernel_i4(%in_tmp, %out_tmp) : (memref<3x5x9xi4>, memref<3x5x9xi4>) -> ()
  gpu.memcpy %arg_out, %out_tmp : memref<3x5x9xi4>, memref<3x5x9xi4>
  gpu.dealloc %in_tmp : memref<3x5x9xi4>
  gpu.dealloc %out_tmp : memref<3x5x9xi4>
  func.return
}

// ---------------------------------------------------------------------------
// f4 multi-dim (odd last dim) also rewritten.
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @process_f4_3x5x9
// CHECK: %[[IN_RAW:.*]] = gpu.alloc() : memref<3x5x5xi8>
// CHECK: %[[IN_CAST:.*]] = unrealized_conversion_cast %[[IN_RAW]] : memref<3x5x5xi8> to memref<3x5x9xf4E2M1FN>
// CHECK: %[[OUT_RAW:.*]] = gpu.alloc() : memref<3x5x5xi8>
// CHECK: %[[OUT_CAST:.*]] = unrealized_conversion_cast %[[OUT_RAW]] : memref<3x5x5xi8> to memref<3x5x9xf4E2M1FN>
// CHECK: gpu.memcpy %[[IN_CAST]], %arg_in : memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>
// CHECK: func.call @kernel_f4(%[[IN_CAST]], %[[OUT_CAST]]) : (memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>) -> ()
// CHECK: gpu.memcpy %arg_out, %[[OUT_CAST]] : memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>
// CHECK: gpu.dealloc %[[IN_RAW]] : memref<3x5x5xi8>
// CHECK: gpu.dealloc %[[OUT_RAW]] : memref<3x5x5xi8>
func.func @process_f4_3x5x9(%arg_in : memref<3x5x9xf4E2M1FN>, %arg_out : memref<3x5x9xf4E2M1FN>) {
  %in_tmp = gpu.alloc() : memref<3x5x9xf4E2M1FN>
  %out_tmp = gpu.alloc() : memref<3x5x9xf4E2M1FN>
  gpu.memcpy %in_tmp, %arg_in : memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>
  func.call @kernel_f4(%in_tmp, %out_tmp) : (memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>) -> ()
  gpu.memcpy %arg_out, %out_tmp : memref<3x5x9xf4E2M1FN>, memref<3x5x9xf4E2M1FN>
  gpu.dealloc %in_tmp : memref<3x5x9xf4E2M1FN>
  gpu.dealloc %out_tmp : memref<3x5x9xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// i4 large flat: 64 -> 32
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @process_i4_64
// CHECK: %[[IN_RAW:.*]] = gpu.alloc() : memref<32xi8>
// CHECK: %[[IN_CAST:.*]] = unrealized_conversion_cast %[[IN_RAW]] : memref<32xi8> to memref<64xi4>
// CHECK: %[[OUT_RAW:.*]] = gpu.alloc() : memref<32xi8>
// CHECK: %[[OUT_CAST:.*]] = unrealized_conversion_cast %[[OUT_RAW]] : memref<32xi8> to memref<64xi4>
// CHECK: gpu.memcpy %[[IN_CAST]], %arg_in : memref<64xi4>, memref<64xi4>
// CHECK: func.call @kernel_i4(%[[IN_CAST]], %[[OUT_CAST]]) : (memref<64xi4>, memref<64xi4>) -> ()
// CHECK: gpu.memcpy %arg_out, %[[OUT_CAST]] : memref<64xi4>, memref<64xi4>
// CHECK: gpu.dealloc %[[IN_RAW]] : memref<32xi8>
// CHECK: gpu.dealloc %[[OUT_RAW]] : memref<32xi8>
func.func @process_i4_64(%arg_in : memref<64xi4>, %arg_out : memref<64xi4>) {
  %in_tmp = gpu.alloc() : memref<64xi4>
  %out_tmp = gpu.alloc() : memref<64xi4>
  gpu.memcpy %in_tmp, %arg_in : memref<64xi4>, memref<64xi4>
  func.call @kernel_i4(%in_tmp, %out_tmp) : (memref<64xi4>, memref<64xi4>) -> ()
  gpu.memcpy %arg_out, %out_tmp : memref<64xi4>, memref<64xi4>
  gpu.dealloc %in_tmp : memref<64xi4>
  gpu.dealloc %out_tmp : memref<64xi4>
  func.return
}
