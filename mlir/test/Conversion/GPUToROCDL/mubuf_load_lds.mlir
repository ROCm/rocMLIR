// RUN: mlir-opt %s -convert-gpu-to-rocdl | FileCheck %s --dump-input-on-failure

gpu.module @mubuf_load {
  // f32 tests.

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_f32
  func @buffer_load_from_rank_1_to_f32(%src : memref<128xf32, 3>, %offset0 : i32) -> f32 {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 3>, f32
    return %result : f32
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_2xf32
  func @buffer_load_from_rank_1_to_2xf32(%src : memref<128xf32, 3>, %offset0 : i32) -> vector<2xf32> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 3>, vector<2xf32>
    return %result : vector<2xf32>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_4xf32
  func @buffer_load_from_rank_1_to_4xf32(%src : memref<128xf32, 3>, %offset0 : i32) -> vector<4xf32> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 3>, vector<4xf32>
    return %result : vector<4xf32>
  }

  // f16 tests.

  func @buffer_load_from_rank_1_to_f16(%src : memref<128xf16, 3>, %offset0 : i32) -> f16 {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 3>, f16
    return %result : f16
  }

  func @buffer_load_from_rank_1_to_2xf16(%src : memref<128xf16, 3>, %offset0 : i32) -> vector<2xf16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 3>, vector<2xf16>
    return %result : vector<2xf16>
  }

  func @buffer_load_from_rank_1_to_4xf16(%src : memref<128xf16, 3>, %offset0 : i32) -> vector<4xf16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 3>, vector<4xf16>
    return %result : vector<4xf16>
  }

  func @buffer_load_from_rank_1_to_8xf16(%src : memref<128xf16, 3>, %offset0 : i32) -> vector<8xf16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 3>, vector<8xf16>
    return %result : vector<8xf16>
  }

  // i16 (bf16) tests.

  func @buffer_load_from_rank_1_to_i16(%src : memref<128xi16, 3>, %offset0 : i32) -> i16 {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xi16, 3>, i16
    return %result : i16
  }

  func @buffer_load_from_rank_1_to_2xi16(%src : memref<128xi16, 3>, %offset0 : i32) -> vector<2xi16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xi16, 3>, vector<2xi16>
    return %result : vector<2xi16>
  }

  func @buffer_load_from_rank_1_to_4xi16(%src : memref<128xi16, 3>, %offset0 : i32) -> vector<4xi16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xi16, 3>, vector<4xi16>
    return %result : vector<4xi16>
  }

  func @buffer_load_from_rank_1_to_8xi16(%src : memref<128xi16, 3>, %offset0 : i32) -> vector<8xi16> {
    %result = gpu.buffer_load(%src, %offset0) : memref<128xi16, 3>, vector<8xi16>
    return %result : vector<8xi16>
  }
}
