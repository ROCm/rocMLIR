// RUN: mlir-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' | FileCheck %s

gpu.module @mubuf_load {
  // f32 tests.

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_f32
  func @raw_buffer_load_from_rank_1_to_f32(%src : memref<128xf32>, %shift : i32, %offset0 : i32) -> f32 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf32>, f32
    return %result : f32
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_f32
  func @raw_buffer_load_from_rank_4_to_f32(%src : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> f32 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, f32
    return %result : f32
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_2xf32
  func @raw_buffer_load_from_rank_1_to_2xf32(%src : memref<128xf32>, %shift : i32, %offset0 : i32) -> vector<2xf32> {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf32>, vector<2xf32>
    return %result : vector<2xf32>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_2xf32
  func @raw_buffer_load_from_rank_4_to_2xf32(%src : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xf32> {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, vector<2xf32>
    return %result : vector<2xf32>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_4xf32
  func @raw_buffer_load_from_rank_4_to_4xf32(%src : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xf32> {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, vector<4xf32>
    return %result : vector<4xf32>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_4xf32
  func @raw_buffer_load_from_rank_1_to_4xf32(%src : memref<128xf32>, %shift : i32, %offset0 : i32) -> vector<4xf32> {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf32>, vector<4xf32>
    return %result : vector<4xf32>
  }

  // f16 tests.

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_f16
  func @raw_buffer_load_from_rank_1_to_f16(%src : memref<128xf16>, %shift : i32, %offset0 : i32) -> f16 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f16
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf16>, f16
    return %result : f16
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_f16
  func @raw_buffer_load_from_rank_4_to_f16(%src : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> f16 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f16
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, f16
    return %result : f16
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_2xf16
  func @raw_buffer_load_from_rank_1_to_2xf16(%src : memref<128xf16>, %shift : i32, %offset0 : i32) -> vector<2xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : f32 to vector<2xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf16>, vector<2xf16>
    return %result : vector<2xf16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_2xf16
  func @raw_buffer_load_from_rank_4_to_2xf16(%src : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : f32 to vector<2xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<2xf16>
    return %result : vector<2xf16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_4xf16
  func @raw_buffer_load_from_rank_1_to_4xf16(%src : memref<128xf16>, %shift : i32, %offset0 : i32) -> vector<4xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<2xf32> to vector<4xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf16>, vector<4xf16>
    return %result : vector<4xf16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_4xf16
  func @raw_buffer_load_from_rank_4_to_4xf16(%src : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<2xf32> to vector<4xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<4xf16>
    return %result : vector<4xf16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_8xf16
  func @raw_buffer_load_from_rank_1_to_8xf16(%src : memref<128xf16>, %shift : i32, %offset0 : i32) -> vector<8xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<4xf32> to vector<8xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xf16>, vector<8xf16>
    return %result : vector<8xf16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_8xf16
  func @raw_buffer_load_from_rank_4_to_8xf16(%src : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<8xf16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<4xf32> to vector<8xf16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<8xf16>
    return %result : vector<8xf16>
  }

  // i16 (bf16) tests.

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_i16
  func @raw_buffer_load_from_rank_1_to_i16(%src : memref<128xi16>, %shift : i32, %offset0 : i32) -> i16 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xi16>, i16
    return %result : i16
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_i16
  func @raw_buffer_load_from_rank_4_to_i16(%src : memref<128x64x32x16xi16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> i16 {
    // CHECK: rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xi16>, i16
    return %result : i16
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_2xi16
  func @raw_buffer_load_from_rank_1_to_2xi16(%src : memref<128xi16>, %shift : i32, %offset0 : i32) -> vector<2xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : f32 to vector<2xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xi16>, vector<2xi16>
    return %result : vector<2xi16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_2xi16
  func @raw_buffer_load_from_rank_4_to_2xi16(%src : memref<128x64x32x16xi16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : f32 to vector<2xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xi16>, vector<2xi16>
    return %result : vector<2xi16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_4xi16
  func @raw_buffer_load_from_rank_1_to_4xi16(%src : memref<128xi16>, %shift : i32, %offset0 : i32) -> vector<4xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<2xf32> to vector<4xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xi16>, vector<4xi16>
    return %result : vector<4xi16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_4xi16
  func @raw_buffer_load_from_rank_4_to_4xi16(%src : memref<128x64x32x16xi16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<2xf32> to vector<4xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xi16>, vector<4xi16>
    return %result : vector<4xi16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_1_to_8xi16
  func @raw_buffer_load_from_rank_1_to_8xi16(%src : memref<128xi16>, %shift : i32, %offset0 : i32) -> vector<8xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<4xf32> to vector<8xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0) : memref<128xi16>, vector<8xi16>
    return %result : vector<8xi16>
  }

  // CHECK-LABEL: func @raw_buffer_load_from_rank_4_to_8xi16
  func @raw_buffer_load_from_rank_4_to_8xi16(%src : memref<128x64x32x16xi16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<8xi16> {
    // CHECK: [[LOAD:%[a-zA-Z_0-9]+]] = rocdl.raw.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    // CHECK-NEXT: llvm.bitcast [[LOAD]] : vector<4xf32> to vector<8xi16>
    %result = gpu.raw_buffer_load(%src, %shift, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xi16>, vector<8xi16>
    return %result : vector<8xi16>
  }
}
