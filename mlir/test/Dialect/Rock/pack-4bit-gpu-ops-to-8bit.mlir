// RUN: rocmlir-opt --rock-pack-4bit-gpu-ops-to-8bit %s | FileCheck %s

// The pass rewrites gpu.alloc/dealloc of any 4-bit element (i4 or f4E2M1FN) to an
// i8 buffer whose last dimension is halved (ceil(N/2)), inserts an
// unrealized_conversion_cast back to the original 4-bit memref, and rewrites
// gpu.memcpy only when BOTH operands are such casts, replacing them with a
// memcpy on the raw i8 memrefs.

// ===========================================================================
// Callee function declarations (remain untouched)
// ===========================================================================

// CHECK-LABEL: func.func @kernel_i4
func.func @kernel_i4(%in : memref<*xi4>, %out : memref<*xi4>) {
  func.return
}

// CHECK-LABEL: func.func @kernel_f4
func.func @kernel_f4(%in : memref<*xf4E2M1FN>, %out : memref<*xf4E2M1FN>) {
  func.return
}

// ===========================================================================
// Basic transformations - i4 and f4 types
// ===========================================================================

// ---------------------------------------------------------------------------
// Two i4 allocs + memcpy between them (tests alloc + memcpy + dealloc rewrite).
// 4x10 -> 4x5 (halves the last dimension and converts to i8)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @alloc_pair_i4
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<4x5xi8>
// CHECK: %[[B:.*]] = gpu.alloc{{.*}}: memref<4x5xi8>
// CHECK: gpu.memcpy{{.*}}%[[B]], %[[A]]{{.*}}: memref<4x5xi8>, memref<4x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<4x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[B]]{{.*}}: memref<4x5xi8>
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
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<4x5xi8>
// CHECK: %[[B:.*]] = gpu.alloc{{.*}}: memref<4x5xi8>
// CHECK: gpu.memcpy{{.*}}%[[B]], %[[A]]{{.*}}: memref<4x5xi8>, memref<4x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<4x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[B]]{{.*}}: memref<4x5xi8>
func.func @alloc_pair_f4() {
  %a = gpu.alloc() : memref<4x10xf4E2M1FN>
  %b = gpu.alloc() : memref<4x10xf4E2M1FN>
  gpu.memcpy %b, %a : memref<4x10xf4E2M1FN>, memref<4x10xf4E2M1FN>
  gpu.dealloc %a : memref<4x10xf4E2M1FN>
  gpu.dealloc %b : memref<4x10xf4E2M1FN>
  func.return
}

// ===========================================================================
// 1D tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1D case: 64 -> 32
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @alloc_1d
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<32xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<32xi8>
func.func @alloc_1d() {
  %a = gpu.alloc() : memref<64xi4>
  gpu.dealloc %a : memref<64xi4>
  func.return
}

// ===========================================================================
// 3D tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test with odd last dimension: 3x5x9 -> 3x5x5 (ceil(9/2) = 5)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @alloc_odd_dimension
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<3x5x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<3x5x5xi8>
func.func @alloc_odd_dimension() {
  %a = gpu.alloc() : memref<3x5x9xi4>
  gpu.dealloc %a : memref<3x5x9xi4>
  func.return
}

// ===========================================================================
// Edge case dimensions - small and large odd dimensions
// ===========================================================================

// ---------------------------------------------------------------------------
// Test edge case: dimension of 1 - ceil(1/2) = 1
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_dim_one
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<4x1xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<4x1xi8>
func.func @test_dim_one() {
  %a = gpu.alloc() : memref<4x1xi4>
  gpu.dealloc %a : memref<4x1xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test edge case: dimension of 3 - ceil(3/2) = 2
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_dim_three
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<2xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<2xi8>
func.func @test_dim_three() {
  %a = gpu.alloc() : memref<3xi4>
  gpu.dealloc %a : memref<3xi4>
  func.return
}

// ===========================================================================
// Host-shared attribute tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test host_shared allocation
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_host_shared
// CHECK: %[[A:.*]] = gpu.alloc host_shared () : memref<64xi8>
// CHECK: gpu.dealloc %[[A]] : memref<64xi8>
func.func @test_host_shared() {
  %a = gpu.alloc host_shared () : memref<128xi4>
  gpu.dealloc %a : memref<128xi4>
  func.return
}

// ===========================================================================
// Async operation tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test async alloc/dealloc - async token must be preserved
// Verifies that async tokens are properly threaded through the transformation
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_async_alloc_dealloc
// CHECK: %[[MEM:.*]], %[[TOKEN:.*]] = gpu.alloc async () : memref<64xi8>
// CHECK: %[[CAST:.*]]:2 = builtin.unrealized_conversion_cast %[[MEM]], %[[TOKEN]]
// CHECK: gpu.dealloc async [%[[CAST]]#1] %[[CAST]]#0
func.func @test_async_alloc_dealloc() {
  %mem, %token = gpu.alloc async () : memref<128xi4>
  %token_dealloc = gpu.dealloc async [%token] %mem : memref<128xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test async with f4E2M1FN type
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_async_f4
// CHECK: %[[MEM:.*]], %[[TOKEN:.*]] = gpu.alloc async () : memref<16xi8>
// CHECK: %[[CAST:.*]]:2 = builtin.unrealized_conversion_cast %[[MEM]], %[[TOKEN]]
// CHECK: gpu.dealloc async [%[[CAST]]#1] %[[CAST]]#0
func.func @test_async_f4() {
  %mem, %token = gpu.alloc async () : memref<32xf4E2M1FN>
  %token_dealloc = gpu.dealloc async [%token] %mem : memref<32xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// Test async host_shared - both attributes preserved
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_async_host_shared
// CHECK: %[[MEM:.*]], %[[TOKEN:.*]] = gpu.alloc async host_shared () : memref<32xi8>
// CHECK: %[[CAST:.*]]:2 = builtin.unrealized_conversion_cast %[[MEM]], %[[TOKEN]]
// CHECK: gpu.dealloc async [%[[CAST]]#1] %[[CAST]]#0
func.func @test_async_host_shared() {
  %mem, %token = gpu.alloc async host_shared () : memref<64xi4>
  %token_dealloc = gpu.dealloc async [%token] %mem : memref<64xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test async 2D with odd last dimension
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_async_2d_odd
// CHECK: %[[MEM:.*]], %[[TOKEN:.*]] = gpu.alloc async () : memref<8x4xi8>
// CHECK: %[[CAST:.*]]:2 = builtin.unrealized_conversion_cast %[[MEM]], %[[TOKEN]]
// CHECK: gpu.dealloc async [%[[CAST]]#1] %[[CAST]]#0
func.func @test_async_2d_odd() {
  %mem, %token = gpu.alloc async () : memref<8x7xf4E2M1FN>
  %token_dealloc = gpu.dealloc async [%token] %mem : memref<8x7xf4E2M1FN>
  func.return
}

// ===========================================================================
// Function call integration tests (after rock-emulate-narrow-type pass)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test with function call - simulating input from rock-emulate-narrow-type
// The function signatures have already been converted to i8 by previous pass
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_i4
// CHECK-SAME: memref<32xi8>
func.func private @callee_i4(%arg0: memref<32xi8>, %arg1: memref<64xf16>)

// ---------------------------------------------------------------------------
// Test GPU alloc + memcpy + function call pattern
// Input: After rock-emulate-narrow-type (args are i8, with casts to i4)
// Output: All casts removed, everything uses i8, memcpy between casts rewritten
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_with_call
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU_I8:.*]] = gpu.alloc{{.*}}: memref<32xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU_I8]], %arg0{{.*}}: memref<32xi8>, memref<32xi8>
// CHECK: %[[F16_ALLOC:.*]] = gpu.alloc{{.*}}: memref<64xf16>
// CHECK: gpu.memcpy{{.*}}%[[F16_ALLOC]], %arg1{{.*}}: memref<64xf16>, memref<64xf16>
// CHECK: call @callee_i4(%[[GPU_I8]], %[[F16_ALLOC]]) : (memref<32xi8>, memref<64xf16>) -> ()
// CHECK: gpu.memcpy{{.*}}%arg0, %[[GPU_I8]]{{.*}}: memref<32xi8>, memref<32xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU_I8]]{{.*}}: memref<32xi8>
// CHECK: gpu.memcpy{{.*}}%arg1, %[[F16_ALLOC]]{{.*}}: memref<64xf16>, memref<64xf16>
// CHECK: gpu.dealloc{{.*}}%[[F16_ALLOC]]{{.*}}: memref<64xf16>
func.func @test_with_call(%arg0: memref<32xi8>, %arg1: memref<64xf16>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<32xi8> to memref<64xi4>
  %i4_gpu = gpu.alloc() : memref<64xi4>
  %i4_gpu_i8 = builtin.unrealized_conversion_cast %i4_gpu : memref<64xi4> to memref<32xi8>
  gpu.memcpy %i4_gpu, %arg0_i4 : memref<64xi4>, memref<64xi4>
  %f16_gpu = gpu.alloc() : memref<64xf16>
  gpu.memcpy %f16_gpu, %arg1 : memref<64xf16>, memref<64xf16>
  func.call @callee_i4(%i4_gpu_i8, %f16_gpu) : (memref<32xi8>, memref<64xf16>) -> ()
  gpu.memcpy %arg0_i4, %i4_gpu : memref<64xi4>, memref<64xi4>
  gpu.dealloc %i4_gpu : memref<64xi4>
  gpu.memcpy %arg1, %f16_gpu : memref<64xf16>, memref<64xf16>
  gpu.dealloc %f16_gpu : memref<64xf16>
  func.return
}

// ---------------------------------------------------------------------------
// Test with f4 in function call (after rock-emulate-narrow-type)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_f4
// CHECK-SAME: memref<16xi8>
func.func private @callee_f4(%arg0: memref<16xi8>)

// CHECK-LABEL: func.func @test_f4_with_call
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU_I8:.*]] = gpu.alloc{{.*}}: memref<16xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU_I8]], %arg0{{.*}}: memref<16xi8>, memref<16xi8>
// CHECK: call @callee_f4(%[[GPU_I8]]) : (memref<16xi8>) -> ()
// CHECK: gpu.memcpy{{.*}}%arg0, %[[GPU_I8]]{{.*}}: memref<16xi8>, memref<16xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU_I8]]{{.*}}: memref<16xi8>
func.func @test_f4_with_call(%arg0: memref<16xi8>) {
  %arg0_f4 = builtin.unrealized_conversion_cast %arg0 : memref<16xi8> to memref<32xf4E2M1FN>
  %f4_gpu = gpu.alloc() : memref<32xf4E2M1FN>
  %f4_gpu_i8 = builtin.unrealized_conversion_cast %f4_gpu : memref<32xf4E2M1FN> to memref<16xi8>
  gpu.memcpy %f4_gpu, %arg0_f4 : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
  func.call @callee_f4(%f4_gpu_i8) : (memref<16xi8>) -> ()
  gpu.memcpy %arg0_f4, %f4_gpu : memref<32xf4E2M1FN>, memref<32xf4E2M1FN>
  gpu.dealloc %f4_gpu : memref<32xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// Test with multiple i4 allocations passed to function (after rock-emulate-narrow-type)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_multi
// CHECK-SAME: memref<32xi8>
// CHECK-SAME: memref<32xi8>
func.func private @callee_multi(%arg0: memref<32xi8>, %arg1: memref<32xi8>)

// CHECK-LABEL: func.func @test_multi_i4_call
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU1:.*]] = gpu.alloc{{.*}}: memref<32xi8>
// CHECK: %[[GPU2:.*]] = gpu.alloc{{.*}}: memref<32xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU1]], %arg0{{.*}}: memref<32xi8>, memref<32xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU2]], %arg1{{.*}}: memref<32xi8>, memref<32xi8>
// CHECK: call @callee_multi(%[[GPU1]], %[[GPU2]]) : (memref<32xi8>, memref<32xi8>) -> ()
// CHECK: gpu.dealloc{{.*}}%[[GPU1]]{{.*}}: memref<32xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU2]]{{.*}}: memref<32xi8>
func.func @test_multi_i4_call(%arg0: memref<32xi8>, %arg1: memref<32xi8>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<32xi8> to memref<64xi4>
  %arg1_i4 = builtin.unrealized_conversion_cast %arg1 : memref<32xi8> to memref<64xi4>
  %gpu1 = gpu.alloc() : memref<64xi4>
  %gpu1_i8 = builtin.unrealized_conversion_cast %gpu1 : memref<64xi4> to memref<32xi8>
  %gpu2 = gpu.alloc() : memref<64xi4>
  %gpu2_i8 = builtin.unrealized_conversion_cast %gpu2 : memref<64xi4> to memref<32xi8>
  gpu.memcpy %gpu1, %arg0_i4 : memref<64xi4>, memref<64xi4>
  gpu.memcpy %gpu2, %arg1_i4 : memref<64xi4>, memref<64xi4>
  func.call @callee_multi(%gpu1_i8, %gpu2_i8) : (memref<32xi8>, memref<32xi8>) -> ()
  gpu.dealloc %gpu1 : memref<64xi4>
  gpu.dealloc %gpu2 : memref<64xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test 2D memref with even last dimension: 4x10xi4 -> 4x5xi8
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_2d
// CHECK-SAME: memref<4x5xi8>
func.func private @callee_2d(%arg0: memref<4x5xi8>)

// CHECK-LABEL: func.func @test_2d_even
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU:.*]] = gpu.alloc{{.*}}: memref<4x5xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU]], %arg0{{.*}}: memref<4x5xi8>, memref<4x5xi8>
// CHECK: call @callee_2d(%[[GPU]]) : (memref<4x5xi8>) -> ()
// CHECK: gpu.memcpy{{.*}}%arg0, %[[GPU]]{{.*}}: memref<4x5xi8>, memref<4x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU]]{{.*}}: memref<4x5xi8>
func.func @test_2d_even(%arg0: memref<4x5xi8>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<4x5xi8> to memref<4x10xi4>
  %gpu = gpu.alloc() : memref<4x10xi4>
  %gpu_i8 = builtin.unrealized_conversion_cast %gpu : memref<4x10xi4> to memref<4x5xi8>
  gpu.memcpy %gpu, %arg0_i4 : memref<4x10xi4>, memref<4x10xi4>
  func.call @callee_2d(%gpu_i8) : (memref<4x5xi8>) -> ()
  gpu.memcpy %arg0_i4, %gpu : memref<4x10xi4>, memref<4x10xi4>
  gpu.dealloc %gpu : memref<4x10xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test 3D memref with odd last dimension: 3x5x9xi4 -> 3x5x5xi8 (ceil(9/2) = 5)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_3d_odd
// CHECK-SAME: memref<3x5x5xi8>
func.func private @callee_3d_odd(%arg0: memref<3x5x5xi8>)

// CHECK-LABEL: func.func @test_3d_odd
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU:.*]] = gpu.alloc{{.*}}: memref<3x5x5xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU]], %arg0{{.*}}: memref<3x5x5xi8>, memref<3x5x5xi8>
// CHECK: call @callee_3d_odd(%[[GPU]]) : (memref<3x5x5xi8>) -> ()
// CHECK: gpu.memcpy{{.*}}%arg0, %[[GPU]]{{.*}}: memref<3x5x5xi8>, memref<3x5x5xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU]]{{.*}}: memref<3x5x5xi8>
func.func @test_3d_odd(%arg0: memref<3x5x5xi8>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<3x5x5xi8> to memref<3x5x9xi4>
  %gpu = gpu.alloc() : memref<3x5x9xi4>
  %gpu_i8 = builtin.unrealized_conversion_cast %gpu : memref<3x5x9xi4> to memref<3x5x5xi8>
  gpu.memcpy %gpu, %arg0_i4 : memref<3x5x9xi4>, memref<3x5x9xi4>
  func.call @callee_3d_odd(%gpu_i8) : (memref<3x5x5xi8>) -> ()
  gpu.memcpy %arg0_i4, %gpu : memref<3x5x9xi4>, memref<3x5x9xi4>
  gpu.dealloc %gpu : memref<3x5x9xi4>
  func.return
}

// ---------------------------------------------------------------------------
// Test 2D f4 with odd last dimension: 8x7xf4 -> 8x4xi8 (ceil(7/2) = 4)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_f4_2d_odd
// CHECK-SAME: memref<8x4xi8>
func.func private @callee_f4_2d_odd(%arg0: memref<8x4xi8>)

// CHECK-LABEL: func.func @test_f4_2d_odd
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU:.*]] = gpu.alloc{{.*}}: memref<8x4xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU]], %arg0{{.*}}: memref<8x4xi8>, memref<8x4xi8>
// CHECK: call @callee_f4_2d_odd(%[[GPU]]) : (memref<8x4xi8>) -> ()
// CHECK: gpu.memcpy{{.*}}%arg0, %[[GPU]]{{.*}}: memref<8x4xi8>, memref<8x4xi8>
// CHECK: gpu.dealloc{{.*}}%[[GPU]]{{.*}}: memref<8x4xi8>
func.func @test_f4_2d_odd(%arg0: memref<8x4xi8>) {
  %arg0_f4 = builtin.unrealized_conversion_cast %arg0 : memref<8x4xi8> to memref<8x7xf4E2M1FN>
  %gpu = gpu.alloc() : memref<8x7xf4E2M1FN>
  %gpu_i8 = builtin.unrealized_conversion_cast %gpu : memref<8x7xf4E2M1FN> to memref<8x4xi8>
  gpu.memcpy %gpu, %arg0_f4 : memref<8x7xf4E2M1FN>, memref<8x7xf4E2M1FN>
  func.call @callee_f4_2d_odd(%gpu_i8) : (memref<8x4xi8>) -> ()
  gpu.memcpy %arg0_f4, %gpu : memref<8x7xf4E2M1FN>, memref<8x7xf4E2M1FN>
  gpu.dealloc %gpu : memref<8x7xf4E2M1FN>
  func.return
}

// ---------------------------------------------------------------------------
// Test 4D with even last dimension: 2x3x4x16xi4 -> 2x3x4x8xi8
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func private @callee_4d
// CHECK-SAME: memref<2x3x4x8xi8>
func.func private @callee_4d(%arg0: memref<2x3x4x8xi8>)

// CHECK-LABEL: func.func @test_4d_even
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[GPU:.*]] = gpu.alloc{{.*}}: memref<2x3x4x8xi8>
// CHECK: gpu.memcpy{{.*}}%[[GPU]], %arg0{{.*}}: memref<2x3x4x8xi8>, memref<2x3x4x8xi8>
// CHECK: call @callee_4d(%[[GPU]]) : (memref<2x3x4x8xi8>) -> ()
// CHECK: gpu.dealloc{{.*}}%[[GPU]]{{.*}}: memref<2x3x4x8xi8>
func.func @test_4d_even(%arg0: memref<2x3x4x8xi8>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<2x3x4x8xi8> to memref<2x3x4x16xi4>
  %gpu = gpu.alloc() : memref<2x3x4x16xi4>
  %gpu_i8 = builtin.unrealized_conversion_cast %gpu : memref<2x3x4x16xi4> to memref<2x3x4x8xi8>
  gpu.memcpy %gpu, %arg0_i4 : memref<2x3x4x16xi4>, memref<2x3x4x16xi4>
  func.call @callee_4d(%gpu_i8) : (memref<2x3x4x8xi8>) -> ()
  gpu.dealloc %gpu : memref<2x3x4x16xi4>
  func.return
}

// ===========================================================================
// Complex memcpy chain tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test mixed: 2D with odd, multiple allocations with memcpy between them
// 6x11xi4 -> 6x6xi8 (ceil(11/2) = 6)
// ---------------------------------------------------------------------------
// CHECK-LABEL: func.func @test_mixed_2d_odd_multi_alloc
// CHECK-NOT: unrealized_conversion_cast
// CHECK: %[[A:.*]] = gpu.alloc{{.*}}: memref<6x6xi8>
// CHECK: %[[B:.*]] = gpu.alloc{{.*}}: memref<6x6xi8>
// CHECK: gpu.memcpy{{.*}}%[[A]], %arg0{{.*}}: memref<6x6xi8>, memref<6x6xi8>
// CHECK: gpu.memcpy{{.*}}%[[B]], %[[A]]{{.*}}: memref<6x6xi8>, memref<6x6xi8>
// CHECK: gpu.memcpy{{.*}}%arg0, %[[B]]{{.*}}: memref<6x6xi8>, memref<6x6xi8>
// CHECK: gpu.dealloc{{.*}}%[[A]]{{.*}}: memref<6x6xi8>
// CHECK: gpu.dealloc{{.*}}%[[B]]{{.*}}: memref<6x6xi8>
func.func @test_mixed_2d_odd_multi_alloc(%arg0: memref<6x6xi8>) {
  %arg0_i4 = builtin.unrealized_conversion_cast %arg0 : memref<6x6xi8> to memref<6x11xi4>
  %a = gpu.alloc() : memref<6x11xi4>
  %a_i8 = builtin.unrealized_conversion_cast %a : memref<6x11xi4> to memref<6x6xi8>
  %b = gpu.alloc() : memref<6x11xi4>
  %b_i8 = builtin.unrealized_conversion_cast %b : memref<6x11xi4> to memref<6x6xi8>
  gpu.memcpy %a, %arg0_i4 : memref<6x11xi4>, memref<6x11xi4>
  gpu.memcpy %b, %a : memref<6x11xi4>, memref<6x11xi4>
  gpu.memcpy %arg0_i4, %b : memref<6x11xi4>, memref<6x11xi4>
  gpu.dealloc %a : memref<6x11xi4>
  gpu.dealloc %b : memref<6x11xi4>
  func.return
}

