// RUN: miopen-opt %s -convert-gpu-to-rocdl='index-bitwidth=32' | FileCheck %s

gpu.module @mubuf_store {
  // f32 tests.

  // CHECK-LABEL: func @buffer_store_f32_to_rank_1
  gpu.func @buffer_store_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0) : f32, memref<128xf32>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_f32_to_rank_4
  gpu.func @buffer_store_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xf32_to_rank_1
  gpu.func @buffer_store_2xf32_to_rank_1(%value : vector<2xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xf32>, memref<128xf32>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xf32_to_rank_4
  gpu.func @buffer_store_2xf32_to_rank_4(%value : vector<2xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xf32_to_rank_1
  gpu.func @buffer_store_4xf32_to_rank_1(%value : vector<4xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xf32>, memref<128xf32>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xf32_to_rank_4
  gpu.func @buffer_store_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: rocdl.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.return
  }

  // f16 tests.

  // CHECK-LABEL: func @buffer_store_f16_to_rank_1
  gpu.func @buffer_store_f16_to_rank_1(%value : f16, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<f16>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : f16, memref<128xf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_f16_to_rank_4
  gpu.func @buffer_store_f16_to_rank_4(%value : f16, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<f16>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f16, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xf16_to_rank_1
  gpu.func @buffer_store_2xf16_to_rank_1(%value : vector<2xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<2xf16> to f32
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xf16>, memref<128xf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xf16_to_rank_4
  gpu.func @buffer_store_2xf16_to_rank_4(%value : vector<2xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<2xf16> to f32
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xf16_to_rank_1
  gpu.func @buffer_store_4xf16_to_rank_1(%value : vector<4xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<4xf16> to vector<2xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xf16>, memref<128xf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xf16_to_rank_4
  gpu.func @buffer_store_4xf16_to_rank_4(%value : vector<4xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<4xf16> to vector<2xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_8xf16_to_rank_1
  gpu.func @buffer_store_8xf16_to_rank_1(%value : vector<8xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<8xf16> to vector<4xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<8xf16>, memref<128xf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_8xf16_to_rank_4
  gpu.func @buffer_store_8xf16_to_rank_4(%value : vector<8xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<8xf16> to vector<4xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // bf16 -> i16 tests.

  // CHECK-LABEL: func @buffer_store_bf16_to_rank_1
  gpu.func @buffer_store_bf16_to_rank_1(%value : bf16, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<i16>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : bf16, memref<128xbf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_bf16_to_rank_4
  gpu.func @buffer_store_bf16_to_rank_4(%value : bf16, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr<i16>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : bf16, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xbf16_to_rank_1
  gpu.func @buffer_store_2xbf16_to_rank_1(%value : vector<2xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<2xi16> to f32
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xbf16>, memref<128xbf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_2xbf16_to_rank_4
  gpu.func @buffer_store_2xbf16_to_rank_4(%value : vector<2xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<2xi16> to f32
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : f32
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xbf16_to_rank_1
  gpu.func @buffer_store_4xbf16_to_rank_1(%value : vector<4xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<4xi16> to vector<2xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xbf16>, memref<128xbf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_4xbf16_to_rank_4
  gpu.func @buffer_store_4xbf16_to_rank_4(%value : vector<4xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<4xi16> to vector<2xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<2xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_8xbf16_to_rank_1
  gpu.func @buffer_store_8xbf16_to_rank_1(%value : vector<8xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<8xi16> to vector<4xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<8xbf16>, memref<128xbf16>, i32, i32
    gpu.return
  }

  // CHECK-LABEL: func @buffer_store_8xbf16_to_rank_4
  gpu.func @buffer_store_8xbf16_to_rank_4(%value : vector<8xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
    // CHECK: [[BITCAST:%[a-zA-Z_0-9]+]] = llvm.bitcast %{{.*}} : vector<8xi16> to vector<4xf32>
    // CHECK-NEXT: rocdl.buffer.store [[BITCAST]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : vector<4xf32>
    gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.return
  }
}
