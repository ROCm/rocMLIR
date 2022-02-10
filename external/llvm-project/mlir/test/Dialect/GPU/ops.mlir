// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

module attributes {gpu.container_module} {

  // CHECK-LABEL:func @no_args(%{{.*}}: index)
  func @no_args(%sz : index) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %sz, %grid_y = %sz, %grid_z = %sz)
               threads(%tx, %ty, %tz) in (%block_x = %sz, %block_y = %sz, %block_z = %sz) {
      // CHECK: gpu.terminator
      gpu.terminator
    }
    return
  }

  // CHECK-LABEL:func @args(%{{.*}}: index, %{{.*}}: index, %{{.*}}: f32, %{{.*}}: memref<?xf32, 1>) {
  func @args(%blk : index, %thrd : index, %float : f32, %data : memref<?xf32,1>) {
    // CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}})
    gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %blk, %grid_y = %blk, %grid_z = %blk)
               threads(%tx, %ty, %tz) in (%block_x = %thrd, %block_y = %thrd, %block_z = %thrd) {
      "use"(%float) : (f32) -> ()
      "use"(%data) : (memref<?xf32,1>) -> ()
      // CHECK: gpu.terminator
      gpu.terminator
    }
    return
  }

  gpu.module @kernels {
    gpu.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>) kernel {
      %tIdX = gpu.thread_id x
      %tIdY = gpu.thread_id y
      %tIdZ = gpu.thread_id z

      %bDimX = gpu.block_dim x
      %bDimY = gpu.block_dim y
      %bDimZ = gpu.block_dim z

      %bIdX = gpu.block_id x
      %bIdY = gpu.block_id y
      %bIdZ = gpu.block_id z

      %gDimX = gpu.grid_dim x
      %gDimY = gpu.grid_dim y
      %gDimZ = gpu.grid_dim z

      %sgId = gpu.subgroup_id : index
      %numSg = gpu.num_subgroups : index
      %SgSi = gpu.subgroup_size : index

      %one = arith.constant 1.0 : f32
      %sum = gpu.all_reduce add %one {} : (f32) -> (f32)

      %width = arith.constant 7 : i32
      %offset = arith.constant 3 : i32
      // CHECK: gpu.shuffle xor %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl, %pred = gpu.shuffle xor %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle up %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl1, %pred1 = gpu.shuffle up %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle down %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl2, %pred2 = gpu.shuffle down %arg0, %offset, %width : f32
      // CHECK: gpu.shuffle idx %{{.*}}, %{{.*}}, %{{.*}} : f32
      %shfl3, %pred3 = gpu.shuffle idx %arg0, %offset, %width : f32

      "gpu.barrier"() : () -> ()

      "gpu.lds_barrier"() : () -> ()

      "some_op"(%bIdX, %tIdX) : (index, index) -> ()
      %42 = memref.load %arg1[%bIdX] : memref<?xf32, 1>
      gpu.return
    }

    gpu.func @kernel_2() kernel {
      gpu.return
    }
  }

  func @foo() {
    %0 = "op"() : () -> (f32)
    %1 = "op"() : () -> (memref<?xf32, 1>)
    // CHECK: %{{.*}} = arith.constant 8
    %cst = arith.constant 8 : index
    %c0 = arith.constant 0 : i32
    %t0 = gpu.wait async

    // CHECK: gpu.launch_func @kernels::@kernel_1 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}}) args(%{{.*}} : f32, %{{.*}} : memref<?xf32, 1>)
    gpu.launch_func @kernels::@kernel_1 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) args(%0 : f32, %1 : memref<?xf32, 1>)

    gpu.launch_func @kernels::@kernel_1 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst) dynamic_shared_memory_size %c0 args(%0 : f32, %1 : memref<?xf32, 1>)

    // CHECK: gpu.launch_func @kernels::@kernel_2 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
    gpu.launch_func @kernels::@kernel_2 blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)

    // CHECK: %{{.*}} = gpu.launch_func async [%{{.*}}] @kernels::@kernel_2 blocks in (%{{.*}}, %{{.*}}, %{{.*}}) threads in (%{{.*}}, %{{.*}}, %{{.*}})
    %t1 = gpu.launch_func async [%t0] @kernels::@kernel_2  blocks in (%cst, %cst, %cst) threads in (%cst, %cst, %cst)

    return
  }

  gpu.module @gpu_funcs {
    // CHECK-LABEL: gpu.func @kernel_1({{.*}}: f32)
    // CHECK:       workgroup
    // CHECK:       private
    // CHECK:       attributes
    gpu.func @kernel_1(%arg0: f32)
        workgroup(%arg1: memref<42xf32, 3>)
        private(%arg2: memref<2xf32, 5>, %arg3: memref<1xf32, 5>)
        kernel
        attributes {foo="bar"} {
      "use"(%arg1) : (memref<42xf32, 3>) -> ()
      "use"(%arg2) : (memref<2xf32, 5>) -> ()
      "use"(%arg3) : (memref<1xf32, 5>) -> ()
      gpu.return
    }

    // CHECK-LABEL gpu.func @printf_test
    // CHECK: (%[[ARG0:.*]]: i32)
    // CHECK: gpu.printf "Value: %d" %[[ARG0]] : i32
    gpu.func @printf_test(%arg0 : i32) {
      gpu.printf "Value: %d" %arg0 : i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @no_attribution
    // CHECK: {
    gpu.func @no_attribution(%arg0: f32) {
      gpu.return
    }

    // CHECK-LABEL gpu.func @printf_test
    // CHECK: (%[[ARG0:.*]]: i32)
    // CHECK: gpu.printf {format = "Value: %d"} %[[ARG0]] : i32
    gpu.func @printf_test(%arg0 : i32) {
      gpu.printf {format = "Value: %d"} %arg0 : i32
      gpu.return
    }

    // CHECK-LABEL: @no_attribution_attrs
    // CHECK:       attributes
    // CHECK:       {
    gpu.func @no_attribution_attrs(%arg0: f32) attributes {foo="bar"} {
      gpu.return
    }

    // CHECK-LABEL: @workgroup_only
    // CHECK:       workgroup({{.*}}: {{.*}})
    // CHECK:       {
    gpu.func @workgroup_only() workgroup(%arg0: memref<42xf32, 3>) {
      gpu.return
    }
    // CHECK-LABEL: @private_only
    // CHECK:       private({{.*}}: {{.*}})
    // CHECK:       {
    gpu.func @private_only() private(%arg0: memref<2xf32, 5>) {
      gpu.return
    }

    // CHECK-LABEL: @empty_attribution
    // CHECK:       {
    gpu.func @empty_attribution(%arg0: f32) workgroup() private() {
      gpu.return
    }
  }

  gpu.module @explicit_attributions {
    // CHECK-LABEL: gpu.func @kernel_1({{.*}}: f32, {{.*}}: memref<?xf32>) workgroup({{.*}}: memref<5xf32, 3>) private({{.*}}: memref<5xf32, 5>)
    "gpu.func"() ({
    ^bb0(%arg0: f32, %arg1: memref<?xf32>, %arg2: memref<5xf32, 3>, %arg3: memref<5xf32, 5>):
      "gpu.return"() : () -> ()
    } ) {gpu.kernel, sym_name = "kernel_1", type = (f32, memref<?xf32>) -> (), workgroup_attributions = 1: i64} : () -> ()
  }

  func @alloc() {
    // CHECK-LABEL: func @alloc()

    // CHECK: %[[m0:.*]] = gpu.alloc () : memref<13xf32, 1>
    %m0 = gpu.alloc () : memref<13xf32, 1>
    // CHECK: gpu.dealloc %[[m0]] : memref<13xf32, 1>
    gpu.dealloc %m0 : memref<13xf32, 1>

    %t0 = gpu.wait async
    // CHECK: %[[m1:.*]], %[[t1:.*]] = gpu.alloc async [{{.*}}] () : memref<13xf32, 1>
    %m1, %t1 = gpu.alloc async [%t0] () : memref<13xf32, 1>
    // CHECK: gpu.dealloc async [%[[t1]]] %[[m1]] : memref<13xf32, 1>
    %t2 = gpu.dealloc async [%t1] %m1 : memref<13xf32, 1>

    return
  }

  func @async_token(%arg0 : !gpu.async.token) -> !gpu.async.token {
    // CHECK-LABEL: func @async_token({{.*}}: !gpu.async.token)
    // CHECK: return {{.*}} : !gpu.async.token
    return %arg0 : !gpu.async.token
  }

  func @async_wait() {
    // CHECK-LABEL: func @async_wait
    // CHECK: %[[t0:.*]] = gpu.wait async
    %0 = gpu.wait async
    // CHECK: %[[t1:.*]] = gpu.wait async [%[[t0]]]
    %1 = gpu.wait async [%0]
    // CHECK: %{{.*}} = gpu.wait async [%[[t0]], %[[t1]]]
    %2 = gpu.wait async [%0, %1]
    // CHECK: gpu.wait [%[[t0]], %[[t1]]]
    // CHECK-NOT: async
    gpu.wait [%0, %1]
    // CHECK: gpu.wait
    // CHECK-NOT: async
    gpu.wait // Valid, but a no-op.
    return
  }

  func @memcpy(%dst : memref<3x7xf32>, %src : memref<3x7xf32, 1>) {
    // CHECK-LABEL: func @memcpy
    // CHECK: gpu.memcpy {{.*}}, {{.*}} : memref<3x7xf32>, memref<3x7xf32, 1>
    gpu.memcpy %dst, %src : memref<3x7xf32>, memref<3x7xf32, 1>
    // CHECK: %[[t0:.*]] = gpu.wait async
    %0 = gpu.wait async
    // CHECK: {{.*}} = gpu.memcpy async [%[[t0]]] {{.*}}, {{.*}} : memref<3x7xf32>, memref<3x7xf32, 1>
    %1 = gpu.memcpy async [%0] %dst, %src : memref<3x7xf32>, memref<3x7xf32, 1>
    return
  }

  func @warp_swizzle(%in : i32) -> i32 {
    // CHECK-LABEL func @warp_swizzle
    // CHECK %{{.*}} gpu.warp_swizzle { selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %{{.*}} : i32
    %0 = gpu.warp_swizzle { selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32] } %in : i32
    return %0 : i32
  }

  gpu.module @mfma {
    // CHECK-LABEL: gpu.func @mfma_f32
    //   CHECK:      gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : f32, vector<32xf32>
    //   CHECK-NEXT: gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : f32, vector<32xf32>
    gpu.func @mfma_f32(%a : f32, %b : f32, %c : vector<32xf32>) {
      gpu.mfma(%a, %b, %c) : f32, vector<32xf32>
      %d = gpu.mfma(%a, %b, %c) : f32, vector<32xf32>

      gpu.return
    }

    // CHECK-LABEL: gpu.func @mfma_f16
    //   CHECK:      gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, vector<32xf32>
    //   CHECK-NEXT: gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, vector<32xf32>
    gpu.func @mfma_f16(%a : vector<4xf16>, %b : vector<4xf16>, %c : vector<32xf32>) {
      gpu.mfma(%a, %b, %c) : vector<4xf16>, vector<32xf32>
      %d = gpu.mfma(%a, %b, %c) : vector<4xf16>, vector<32xf32>

      gpu.return
    }

    // CHECK-LABEL: gpu.func @mfma_bf16
    //   CHECK:      gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, vector<32xf32>
    //   CHECK-NEXT: gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, vector<32xf32>
    gpu.func @mfma_bf16(%a : vector<2xbf16>, %b : vector<2xbf16>, %c : vector<32xf32>) {
      gpu.mfma(%a, %b, %c) : vector<2xbf16>, vector<32xf32>
      %d = gpu.mfma(%a, %b, %c) : vector<2xbf16>, vector<32xf32>

      gpu.return
    }

    // CHECK-LABEL: gpu.func @mfma_i8_4xi32
    // CHECK:       gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<4xi32>
    // CHECK-NEXT:  gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<4xi32>
    gpu.func @mfma_i8_4xi32(%a : i32, %b : i32, %c : vector<4xi32>) {
      gpu.mfma(%a, %b, %c) : i32, vector<4xi32>
      %d = gpu.mfma(%a, %b, %c) : i32, vector<4xi32>

      gpu.return
    }

    // CHECK-LABEL: gpu.func @mfma_i8_16xi32
    // CHECK:       gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<16xi32>
    // CHECK-NEXT:  gpu.mfma(%{{.*}}, %{{.*}}, %{{.*}}) : i32, vector<16xi32>
    gpu.func @mfma_i8_16xi32(%a : i32, %b : i32, %c : vector<16xi32>) {
      gpu.mfma(%a, %b, %c) : i32, vector<16xi32>
      %d = gpu.mfma(%a, %b, %c) : i32, vector<16xi32>

      gpu.return
    }
  }

  gpu.module @mubuf_load {
    // f32 tests.

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_f32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf32>, f32, i32
    gpu.func @buffer_load_from_rank_1_to_f32(%src : memref<128xf32>, %offset0 : i32) -> f32 {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf32>, f32, i32
      gpu.return %result : f32
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_f32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf32>, f32, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_f32(%src : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> f32 {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, f32, i32, i32, i32, i32
      gpu.return %result : f32
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_2xf32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf32>, vector<2xf32>, i32
    gpu.func @buffer_load_from_rank_1_to_2xf32(%src : memref<128xf32>, %offset0 : i32) -> vector<2xf32> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf32>, vector<2xf32>, i32
      gpu.return %result : vector<2xf32>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_2xf32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf32>, vector<2xf32>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_2xf32(%src : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xf32> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, vector<2xf32>, i32, i32, i32, i32
      gpu.return %result : vector<2xf32>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_4xf32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf32>, vector<4xf32>, i32
    gpu.func @buffer_load_from_rank_1_to_4xf32(%src : memref<128xf32>, %offset0 : i32) -> vector<4xf32> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf32>, vector<4xf32>, i32
      gpu.return %result : vector<4xf32>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_4xf32
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf32>, vector<4xf32>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_4xf32(%src : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xf32> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf32>, vector<4xf32>, i32, i32, i32, i32
      gpu.return %result : vector<4xf32>
    }

    // f16 tests.

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_f16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf16>, f16, i32
    gpu.func @buffer_load_from_rank_1_to_f16(%src : memref<128xf16>, %offset0 : i32) -> f16 {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf16>, f16, i32
      gpu.return %result : f16
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_f16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf16>, f16, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_f16(%src : memref<128x64x32x16xf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> f16 {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, f16, i32, i32, i32, i32
      gpu.return %result : f16
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_2xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf16>, vector<2xf16>, i32
    gpu.func @buffer_load_from_rank_1_to_2xf16(%src : memref<128xf16>, %offset0 : i32) -> vector<2xf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf16>, vector<2xf16>, i32
      gpu.return %result : vector<2xf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_2xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf16>, vector<2xf16>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_2xf16(%src : memref<128x64x32x16xf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xf16> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<2xf16>, i32, i32, i32, i32
      gpu.return %result : vector<2xf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_4xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf16>, vector<4xf16>, i32
    gpu.func @buffer_load_from_rank_1_to_4xf16(%src : memref<128xf16>, %offset0 : i32) -> vector<4xf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf16>, vector<4xf16>, i32
      gpu.return %result : vector<4xf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_4xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf16>, vector<4xf16>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_4xf16(%src : memref<128x64x32x16xf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xf16> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<4xf16>, i32, i32, i32, i32
      gpu.return %result : vector<4xf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_8xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xf16>, vector<8xf16>, i32
    gpu.func @buffer_load_from_rank_1_to_8xf16(%src : memref<128xf16>, %offset0 : i32) -> vector<8xf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xf16>, vector<8xf16>, i32
      gpu.return %result : vector<8xf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_8xf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xf16>, vector<8xf16>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_8xf16(%src : memref<128x64x32x16xf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<8xf16> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xf16>, vector<8xf16>, i32, i32, i32, i32
      gpu.return %result : vector<8xf16>
    }

    // bf16 tests.

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_bf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xbf16>, bf16, i32
    gpu.func @buffer_load_from_rank_1_to_bf16(%src : memref<128xbf16>, %offset0 : i32) -> bf16 {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16>, bf16, i32
      gpu.return %result : bf16
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_bf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xbf16>, bf16, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_bf16(%src : memref<128x64x32x16xbf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> bf16 {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xbf16>, bf16, i32, i32, i32, i32
      gpu.return %result : bf16
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_2xbf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xbf16>, vector<2xbf16>, i32
    gpu.func @buffer_load_from_rank_1_to_2xbf16(%src : memref<128xbf16>, %offset0 : i32) -> vector<2xbf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16>, vector<2xbf16>, i32
      gpu.return %result : vector<2xbf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_2xbf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xbf16>, vector<2xbf16>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_2xbf16(%src : memref<128x64x32x16xbf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<2xbf16> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xbf16>, vector<2xbf16>, i32, i32, i32, i32
      gpu.return %result : vector<2xbf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_4xbf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xbf16>, vector<4xbf16>, i32
    gpu.func @buffer_load_from_rank_1_to_4xbf16(%src : memref<128xbf16>, %offset0 : i32) -> vector<4xbf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16>, vector<4xbf16>, i32
      gpu.return %result : vector<4xbf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_4_to_4xbf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : memref<128x64x32x16xbf16>, vector<4xbf16>, i32, i32, i32, i32
    gpu.func @buffer_load_from_rank_4_to_4xbf16(%src : memref<128x64x32x16xbf16>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) -> vector<4xbf16> {
      %result = gpu.buffer_load(%src, %offset0, %offset1, %offset2, %offset3) : memref<128x64x32x16xbf16>, vector<4xbf16>, i32, i32, i32, i32
      gpu.return %result : vector<4xbf16>
    }

    // CHECK-LABEL: gpu.func @buffer_load_from_rank_1_to_8xbf16
    //   CHECK: gpu.buffer_load(%{{.*}}, %{{.*}}) : memref<128xbf16>, vector<8xbf16>, i32
    gpu.func @buffer_load_from_rank_1_to_8xbf16(%src : memref<128xbf16>, %offset0 : i32) -> vector<8xbf16> {
      %result = gpu.buffer_load(%src, %offset0) : memref<128xbf16>, vector<8xbf16>, i32
      gpu.return %result : vector<8xbf16>
    }
  }

  gpu.module @mubuf_store {
    // f32 tests.

    // CHECK-LABEL: gpu.func @buffer_store_f32_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128xf32>, i32, i32
    gpu.func @buffer_store_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : f32, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_f32_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf32_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128xf32>, i32, i32
    gpu.func @buffer_store_2xf32_to_rank_1(%value : vector<2xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xf32>, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf32_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xf32_to_rank_4(%value : vector<2xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf32_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128xf32>, i32, i32
    gpu.func @buffer_store_4xf32_to_rank_1(%value : vector<4xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xf32>, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf32_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // f16 tests.

    // CHECK-LABEL: gpu.func @buffer_store_f16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : f16, memref<128xf16>, i32, i32
    gpu.func @buffer_store_f16_to_rank_1(%value : f16, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : f16, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_f16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : f16, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_f16_to_rank_4(%value : f16, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f16, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_2xf16_to_rank_1(%value : vector<2xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xf16_to_rank_4(%value : vector<2xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_4xf16_to_rank_1(%value : vector<4xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xf16_to_rank_4(%value : vector<4xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_8xf16_to_rank_1(%value : vector<8xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<8xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<8xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_8xf16_to_rank_4(%value : vector<8xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // bf16 tests.

    // CHECK-LABEL: gpu.func @buffer_store_bf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : bf16, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_bf16_to_rank_1(%value : bf16, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : bf16, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_bf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : bf16, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_bf16_to_rank_4(%value : bf16, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : bf16, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xbf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_2xbf16_to_rank_1(%value : vector<2xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<2xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xbf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xbf16_to_rank_4(%value : vector<2xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xbf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_4xbf16_to_rank_1(%value : vector<4xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<4xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xbf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xbf16_to_rank_4(%value : vector<4xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xbf16_to_rank_1
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_8xbf16_to_rank_1(%value : vector<8xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0) : vector<8xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xbf16_to_rank_4
    //   CHECK: gpu.buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<8xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_8xbf16_to_rank_4(%value : vector<8xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }
  }

  func @mmamatrix_valid_element_type(){
    // CHECK-LABEL: func @mmamatrix_valid_element_type
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    // CHECK: %[[wg:.*]] = memref.alloca()
    %i = arith.constant 16 : index
    // CHECK: %[[i:.*]] = arith.constant 16 : index
     %cst = arith.constant 1.000000e+00 : f32
    // CHECK: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %i] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK: gpu.subgroup_mma_load_matrix %[[wg]][%[[i]], %[[i]]] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    %1 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK: gpu.subgroup_mma_elementwise addf %{{.*}}, %{{.*}} : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    %2 = gpu.subgroup_mma_elementwise addf %1, %1 : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK: gpu.subgroup_mma_elementwise maxf %{{.*}}, %{{.*}} : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    %3 = gpu.subgroup_mma_elementwise maxf %2, %1 : (!gpu.mma_matrix<16x16xf32, "COp">, !gpu.mma_matrix<16x16xf32, "COp">) -> !gpu.mma_matrix<16x16xf32, "COp">
    return
  }

  gpu.module @rawbuf_store {
    // f32 tests.

    // CHECK-LABEL: gpu.func @buffer_store_f32_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128xf32>, i32, i32
    gpu.func @buffer_store_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : f32, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_f32_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf32_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128xf32>, i32, i32
    gpu.func @buffer_store_2xf32_to_rank_1(%value : vector<2xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<2xf32>, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf32_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xf32_to_rank_4(%value : vector<2xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf32_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128xf32>, i32, i32
    gpu.func @buffer_store_4xf32_to_rank_1(%value : vector<4xf32>, %dst : memref<128xf32>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<4xf32>, memref<128xf32>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf32_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32, i32
      gpu.return
    }

    // f16 tests.

    // CHECK-LABEL: gpu.func @buffer_store_f16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : f16, memref<128xf16>, i32, i32
    gpu.func @buffer_store_f16_to_rank_1(%value : f16, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : f16, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_f16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : f16, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_f16_to_rank_4(%value : f16, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : f16, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_2xf16_to_rank_1(%value : vector<2xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<2xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xf16_to_rank_4(%value : vector<2xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_4xf16_to_rank_1(%value : vector<4xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<4xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xf16_to_rank_4(%value : vector<4xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xf16>, memref<128xf16>, i32, i32
    gpu.func @buffer_store_8xf16_to_rank_1(%value : vector<8xf16>, %dst : memref<128xf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<8xf16>, memref<128xf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<8xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_8xf16_to_rank_4(%value : vector<8xf16>, %dst : memref<128x64x32x16xf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xf16>, memref<128x64x32x16xf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // bf16 tests.

    // CHECK-LABEL: gpu.func @buffer_store_bf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : bf16, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_bf16_to_rank_1(%value : bf16, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : bf16, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_bf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : bf16, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_bf16_to_rank_4(%value : bf16, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : bf16, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xbf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_2xbf16_to_rank_1(%value : vector<2xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<2xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_2xbf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_2xbf16_to_rank_4(%value : vector<2xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<2xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xbf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_4xbf16_to_rank_1(%value : vector<4xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<4xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_4xbf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_4xbf16_to_rank_4(%value : vector<4xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<4xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xbf16_to_rank_1
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}) : vector<8xbf16>, memref<128xbf16>, i32, i32
    gpu.func @buffer_store_8xbf16_to_rank_1(%value : vector<8xbf16>, %dst : memref<128xbf16>, %shift : i32, %offset0 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0) : vector<8xbf16>, memref<128xbf16>, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @buffer_store_8xbf16_to_rank_4
    //   CHECK: gpu.raw_buffer_store(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<8xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
    gpu.func @buffer_store_8xbf16_to_rank_4(%value : vector<8xbf16>, %dst : memref<128x64x32x16xbf16>, %shift : i32, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.raw_buffer_store(%value, %dst, %shift, %offset0, %offset1, %offset2, %offset3) : vector<8xbf16>, memref<128x64x32x16xbf16>, i32, i32, i32, i32, i32
      gpu.return
    }
  }

  gpu.module @atomic_fadd {
    // f32 tests.

    // CHECK-LABEL: gpu.func @atomic_fadd_f32_to_rank_1
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128xf32>, i32
    gpu.func @atomic_fadd_f32_to_rank_1(%value : f32, %dst : memref<128xf32>, %offset0 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0) : f32, memref<128xf32>, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @atomic_fadd_f32_to_rank_4
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.func @atomic_fadd_f32_to_rank_4(%value : f32, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : f32, memref<128x64x32x16xf32>, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @atomic_fadd_2xf32_to_rank_1
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128xf32>, i32
    gpu.func @atomic_fadd_2xf32_to_rank_1(%value : vector<2xf32>, %dst : memref<128xf32>, %offset0 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0) : vector<2xf32>, memref<128xf32>, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @atomic_fadd_2xf32_to_rank_4
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.func @atomic_fadd_2xf32_to_rank_4(%value : vector<2xf32>, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : vector<2xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @atomic_fadd_4xf32_to_rank_1
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128xf32>, i32
    gpu.func @atomic_fadd_4xf32_to_rank_1(%value : vector<4xf32>, %dst : memref<128xf32>, %offset0 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0) : vector<4xf32>, memref<128xf32>, i32
      gpu.return
    }

    // CHECK-LABEL: gpu.func @atomic_fadd_4xf32_to_rank_4
    //   CHECK: gpu.atomic_fadd(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
    gpu.func @atomic_fadd_4xf32_to_rank_4(%value : vector<4xf32>, %dst : memref<128x64x32x16xf32>, %offset0 : i32, %offset1 : i32, %offset2 : i32, %offset3 : i32) {
      gpu.atomic_fadd(%value, %dst, %offset0, %offset1, %offset2, %offset3) : vector<4xf32>, memref<128x64x32x16xf32>, i32, i32, i32, i32
      gpu.return
    }
  }

  gpu.module @subgroup_mma_load_matrix {
    // mma tests.

      gpu.func @mmamatrix_valid_element_type(){
        // CHECK-LABEL: func @mmamatrix_valid_element_type
        %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
        // CHECK: %[[wg:.*]] = memref.alloca()
        %i = arith.constant 16 : index
        // CHECK: %[[i:.*]] = arith.constant 16 : index
         %cst = arith.constant 1.000000e+00 : f32
        // CHECK: %[[cst:.*]] = arith.constant 1.000000e+00 : f32
        %0 = gpu.subgroup_mma_load_matrix %wg[%i, %i] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
        // CHECK: gpu.subgroup_mma_load_matrix %[[wg]][%[[i]], %[[i]]] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
        %1 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf32, "COp">
        // CHECK: gpu.subgroup_mma_constant_matrix %[[cst]] : !gpu.mma_matrix<16x16xf32, "COp">
        gpu.return
      }
   }

   // CHECK-LABEL: func @set_default_device
   func @set_default_device() {
     // CHECK-NEXT: gpu.set_default_device {devIndex = 0 : i32}
     gpu.set_default_device {devIndex = 0 : i32}
     return
   }
}
