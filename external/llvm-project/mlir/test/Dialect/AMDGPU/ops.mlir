// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: mlir-opt -allow-unregistered-dialect %s | mlir-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// FIXME: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic %s | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @raw_buffer_load_f32_from_rank_1
func.func @raw_buffer_load_f32_from_rank_1(%src : memref<128xf32>, %offset : i32, %idx0 : i32) -> f32 {
  // CHECK: amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %{{.*}}[{{.*}}] sgprOffset %{{.*}} : memref<128xf32>, i32 -> f32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0] sgprOffset %offset : memref<128xf32>, i32 -> f32
  func.return %0 : f32
}

// CHECK-LABEL: func @raw_buffer_load_f32_from_rank_4
func.func @raw_buffer_load_f32_from_rank_4(%src : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) -> f32 {
  // CHECK: amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> f32
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> f32
  func.return %0 : f32
}

// CHECK-LABEL: func @raw_buffer_load_4xf32_from_rank_4
func.func @raw_buffer_load_4xf32_from_rank_4(%src : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) -> vector<4xf32> {
  // CHECK: amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> vector<4xf32>
  %0 = amdgpu.raw_buffer_load {boundsCheck = true, indexOffset = 1 : i32} %src[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : memref<128x64x32x16xf32>, i32, i32, i32, i32 -> vector<4xf32>
  func.return %0 : vector<4xf32>
}

// CHECK-LABEL: func @raw_buffer_store_f32_to_rank_1
func.func @raw_buffer_store_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset : i32, %idx0 : i32) {
  // CHECK: amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128xf32>, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0] sgprOffset %offset : f32 -> memref<128xf32>, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_store_f32_to_rank_4
func.func @raw_buffer_store_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_store_4xf32_to_rank_4
func.func @raw_buffer_store_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : vector<4xf32> -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_store {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : vector<4xf32> -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_atomic_fadd_f32_to_rank_1
func.func @raw_buffer_atomic_fadd_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset : i32, %idx0 : i32) {
  // CHECK: amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128xf32>, i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0] sgprOffset %offset : f32 -> memref<128xf32>, i32
  func.return
}

// CHECK-LABEL: func @raw_buffer_atomic_fadd_f32_to_rank_4
func.func @raw_buffer_atomic_fadd_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset : i32, %idx0 : i32, %idx1 : i32, %idx2 : i32, %idx3 : i32) {
  // CHECK: amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %{{.*}} -> %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] sgprOffset %{{.*}} : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  amdgpu.raw_buffer_atomic_fadd {boundsCheck = true, indexOffset = 1 : i32} %value -> %dst[%idx0, %idx1, %idx2, %idx3] sgprOffset %offset : f32 -> memref<128x64x32x16xf32>, i32, i32, i32, i32
  func.return
}

// CHECK-LABEL: func @lds_barrier
func.func @lds_barrier() {
  // CHECK: amdgpu.lds_barrier
  amdgpu.lds_barrier
  func.return
}

// CHECK-LABEL: func @mfma
func.func @mfma(%arg0 : f32, %arg1 : vector<32xf32>, %arg2 : vector<16xf32>,
                %arg3 : vector<4xf32>, %arg4 : vector<4xf16>,
                %arg5 : vector<4xi8>, %arg6 : vector<32xi32>,
                %arg7 : vector<16xi32>, %arg8 : vector<4xi32>,
                %arg9 : vector<2xbf16>, %arg10 : vector<4xbf16>, %arg11 : f64,
                %arg12 : vector<4xf64>, %arg13 : vector<8xi8>,
                %arg14 : vector<2xf32>) {
  // CHECK: amdgpu.mfma f32_32x32x1f32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f32, vector<32xf32>
  amdgpu.mfma f32_32x32x1f32 %arg0 * %arg0 + %arg1 cbsz = 0 abid = 0 blgp = 0 : f32, vector<32xf32>
  // CHECK: amdgpu.mfma f32_16x16x1f32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f32, vector<16xf32>
  amdgpu.mfma f32_16x16x1f32 %arg0 * %arg0 + %arg2 cbsz = 0 abid = 0 blgp = 0 : f32, vector<16xf32>
  // CHECK: amdgpu.mfma f32_4x4x1f32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f32, vector<4xf32>
  amdgpu.mfma f32_4x4x1f32 %arg0 * %arg0 + %arg3 cbsz = 0 abid = 0 blgp = 0 : f32, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x2f32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f32, vector<16xf32>
  amdgpu.mfma f32_32x32x2f32 %arg0 * %arg0 + %arg2 cbsz = 0 abid = 0 blgp = 0 : f32, vector<16xf32>
  // CHECK: amdgpu.mfma f32_16x16x4f32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f32, vector<4xf32>
  amdgpu.mfma f32_16x16x4f32 %arg0 * %arg0 + %arg3 cbsz = 0 abid = 0 blgp = 0 : f32, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x4f16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<32xf32>
  amdgpu.mfma f32_32x32x4f16 %arg4 * %arg4 + %arg1 cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<32xf32>
  // CHECK: amdgpu.mfma f32_16x16x4f16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<16xf32>
  amdgpu.mfma f32_16x16x4f16 %arg4 * %arg4 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_4x4x4f16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<4xf32>
  amdgpu.mfma f32_4x4x4f16 %arg4 * %arg4 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x8f16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<16xf32>
  amdgpu.mfma f32_32x32x8f16 %arg4 * %arg4 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_16x16x16f16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<4xf32>
  amdgpu.mfma f32_16x16x16f16 %arg4 * %arg4 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<4xf16>, vector<4xf32>
  // CHECK: amdgpu.mfma i32_32x32x4i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<32xi32>
  amdgpu.mfma i32_32x32x4i8 %arg5 * %arg5 + %arg6 cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<32xi32>
  // CHECK: amdgpu.mfma i32_16x16x4i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<16xi32>
  amdgpu.mfma i32_16x16x4i8 %arg5 * %arg5 + %arg7 cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<16xi32>
  // CHECK: amdgpu.mfma i32_4x4x4i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<4xi32>
  amdgpu.mfma i32_4x4x4i8 %arg5 * %arg5 + %arg8 cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<4xi32>
  // CHECK: amdgpu.mfma i32_32x32x8i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<16xi32>
  amdgpu.mfma i32_32x32x8i8 %arg5 * %arg5 + %arg7 cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<16xi32>
  // CHECK: amdgpu.mfma i32_16x16x16i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<4xi32>
  amdgpu.mfma i32_16x16x16i8 %arg5 * %arg5 + %arg8 cbsz = 0 abid = 0 blgp = 0 : vector<4xi8>, vector<4xi32>
  // CHECK: amdgpu.mfma f32_32x32x2bf16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<32xf32>
  amdgpu.mfma f32_32x32x2bf16 %arg9 * %arg9 + %arg1 cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<32xf32>
  // CHECK: amdgpu.mfma f32_16x16x2bf16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<16xf32>
  amdgpu.mfma f32_16x16x2bf16 %arg9 * %arg9 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_4x4x2bf16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<4xf32>
  amdgpu.mfma f32_4x4x2bf16 %arg9 * %arg9 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x4bf16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<16xf32>
  amdgpu.mfma f32_32x32x4bf16 %arg9 * %arg9 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_16x16x8bf16 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<4xf32>
  amdgpu.mfma f32_16x16x8bf16 %arg9 * %arg9 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<2xbf16>, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x4bf16_1k %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<32xf32>
  amdgpu.mfma f32_32x32x4bf16_1k %arg10 * %arg10 + %arg1 cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<32xf32>
  // CHECK: amdgpu.mfma f32_16x16x4bf16_1k %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<16xf32>
  amdgpu.mfma f32_16x16x4bf16_1k %arg10 * %arg10 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_4x4x4bf16_1k %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<4xf32>
  amdgpu.mfma f32_4x4x4bf16_1k %arg10 * %arg10 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x8bf16_1k %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<16xf32>
  amdgpu.mfma f32_32x32x8bf16_1k %arg10 * %arg10 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<16xf32>
  // CHECK: amdgpu.mfma f32_16x16x16bf16_1k %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<4xf32>
  amdgpu.mfma f32_16x16x16bf16_1k %arg10 * %arg10 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<4xbf16>, vector<4xf32>
  // CHECK: amdgpu.mfma f64_16x16x4f64 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f64, vector<4xf64>
  amdgpu.mfma f64_16x16x4f64 %arg11 * %arg11 + %arg12 cbsz = 0 abid = 0 blgp = 0 : f64, vector<4xf64>
  // CHECK: amdgpu.mfma f64_4x4x4f64 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : f64, f64
  amdgpu.mfma f64_4x4x4f64 %arg11 * %arg11 + %arg11 cbsz = 0 abid = 0 blgp = 0 : f64, f64
  // CHECK: amdgpu.mfma i32_16x16x32_i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<8xi8>, vector<4xi32>
  amdgpu.mfma i32_16x16x32_i8 %arg13 * %arg13 + %arg8 cbsz = 0 abid = 0 blgp = 0 : vector<8xi8>, vector<4xi32>
  // CHECK: amdgpu.mfma i32_32x32x16_i8 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<8xi8>, vector<16xi32>
  amdgpu.mfma i32_32x32x16_i8 %arg13 * %arg13 + %arg7 cbsz = 0 abid = 0 blgp = 0 : vector<8xi8>, vector<16xi32>
  // CHECK: amdgpu.mfma f32_16x16x8_xf32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xf32>, vector<4xf32>
  amdgpu.mfma f32_16x16x8_xf32 %arg14 * %arg14 + %arg3 cbsz = 0 abid = 0 blgp = 0 : vector<2xf32>, vector<4xf32>
  // CHECK: amdgpu.mfma f32_32x32x4_xf32 %{{.*}} * %{{.*}} + %{{.*}} cbsz = 0 abid = 0 blgp = 0 : vector<2xf32>, vector<16xf32>
  amdgpu.mfma f32_32x32x4_xf32 %arg14 * %arg14 + %arg2 cbsz = 0 abid = 0 blgp = 0 : vector<2xf32>, vector<16xf32>
  func.return
}
