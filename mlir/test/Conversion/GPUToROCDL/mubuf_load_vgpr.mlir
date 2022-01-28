// RUN: miopen-opt %s -convert-gpu-to-rocdl | FileCheck %s

gpu.module @mubuf_load {
  // f32 tests.

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_f32
  gpu.func @buffer_load_from_rank_1_to_f32(%src : memref<128xf32, 5>, %offset0 : i32) -> f32 {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<f32, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 5>, f32, i32
    gpu.return %result : f32
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_2xf32
  gpu.func @buffer_load_from_rank_1_to_2xf32(%src : memref<128xf32, 5>, %offset0 : i32) -> vector<2xf32> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<2xf32>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 5>, vector<2xf32>, i32
    gpu.return %result : vector<2xf32>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_4xf32
  gpu.func @buffer_load_from_rank_1_to_4xf32(%src : memref<128xf32, 5>, %offset0 : i32) -> vector<4xf32> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf32, 5>, vector<4xf32>, i32
    gpu.return %result : vector<4xf32>
  }

  // f16 tests.

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_f16
  gpu.func @buffer_load_from_rank_1_to_f16(%src : memref<128xf16, 5>, %offset0 : i32) -> f16 {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<f16, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 5>, f16, i32
    gpu.return %result : f16
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_2xf16
  gpu.func @buffer_load_from_rank_1_to_2xf16(%src : memref<128xf16, 5>, %offset0 : i32) -> vector<2xf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<2xf16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 5>, vector<2xf16>, i32
    gpu.return %result : vector<2xf16>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_4xf16
  gpu.func @buffer_load_from_rank_1_to_4xf16(%src : memref<128xf16, 5>, %offset0 : i32) -> vector<4xf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<4xf16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 5>, vector<4xf16>, i32
    gpu.return %result : vector<4xf16>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_8xf16
  gpu.func @buffer_load_from_rank_1_to_8xf16(%src : memref<128xf16, 5>, %offset0 : i32) -> vector<8xf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<8xf16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xf16, 5>, vector<8xf16>, i32
    gpu.return %result : vector<8xf16>
  }

  // bf16 -> i16 tests.

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_bf16
  gpu.func @buffer_load_from_rank_1_to_bf16(%src : memref<128xbf16, 5>, %offset0 : i32) -> bf16 {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<i16, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16, 5>, bf16, i32
    gpu.return %result : bf16
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_2xbf16
  gpu.func @buffer_load_from_rank_1_to_2xbf16(%src : memref<128xbf16, 5>, %offset0 : i32) -> vector<2xbf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<2xi16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16, 5>, vector<2xbf16>, i32
    gpu.return %result : vector<2xbf16>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_4xbf16
  gpu.func @buffer_load_from_rank_1_to_4xbf16(%src : memref<128xbf16, 5>, %offset0 : i32) -> vector<4xbf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<4xi16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16, 5>, vector<4xbf16>, i32
    gpu.return %result : vector<4xbf16>
  }

  // CHECK-LABEL: func @buffer_load_from_rank_1_to_8xbf16
  gpu.func @buffer_load_from_rank_1_to_8xbf16(%src : memref<128xbf16, 5>, %offset0 : i32) -> vector<8xbf16> {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>, 5>
    %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16, 5>, vector<8xbf16>, i32
    gpu.return %result : vector<8xbf16>
  }
}
