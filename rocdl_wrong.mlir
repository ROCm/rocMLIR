module attributes {gpu.container_module} {
  func @main() {
    %c1_i16 = arith.constant 1 : i16
    %c1_i32 = arith.constant 1 : i32
    %c16384 = arith.constant 16384 : index
    %c2097152 = arith.constant 2097152 : index
    %c0_i16 = arith.constant 0 : i16
    %c2654208 = arith.constant 2654208 : index
    %0 = memref.alloc() : memref<1x128x128x1x1xi8>
    %1 = memref.cast %0 : memref<1x128x128x1x1xi8> to memref<?x?x?x?x?xi8>
    call @mcpuMemset5DInt8RandInt(%1, %c1_i16, %c1_i16, %c1_i32) : (memref<?x?x?x?x?xi8>, i16, i16, i32) -> ()
    %2 = memref.alloc() : memref<1x128x128x1x1xi8>
    %3 = memref.collapse_shape %0 [[0, 1, 2, 3, 4]] : memref<1x128x128x1x1xi8> into memref<16384xi8>
    %4 = memref.cast %3 : memref<16384xi8> to memref<?xi8>
    %5 = memref.collapse_shape %2 [[0, 1, 2, 3, 4]] : memref<1x128x128x1x1xi8> into memref<16384xi8>
    %6 = memref.cast %5 : memref<16384xi8> to memref<?xi8>
    call @_memcpy_i8_i8(%4, %6, %c16384) : (memref<?xi8>, memref<?xi8>, index) -> ()
    %7 = memref.alloc() : memref<64x1x128x16x16xi8>
    %8 = memref.cast %7 : memref<64x1x128x16x16xi8> to memref<?x?x?x?x?xi8>
    call @mcpuMemset5DInt8RandInt(%8, %c1_i16, %c1_i16, %c1_i32) : (memref<?x?x?x?x?xi8>, i16, i16, i32) -> ()
    %9 = memref.alloc() : memref<64x1x128x16x16xi8>
    %10 = memref.collapse_shape %7 [[0, 1, 2, 3, 4]] : memref<64x1x128x16x16xi8> into memref<2097152xi8>
    %11 = memref.cast %10 : memref<2097152xi8> to memref<?xi8>
    %12 = memref.collapse_shape %9 [[0, 1, 2, 3, 4]] : memref<64x1x128x16x16xi8> into memref<2097152xi8>
    %13 = memref.cast %12 : memref<2097152xi8> to memref<?xi8>
    call @_memcpy_i8_i8(%11, %13, %c2097152) : (memref<?xi8>, memref<?xi8>, index) -> ()
    %14 = memref.alloc() : memref<64x1x128x18x18xi32>
    %15 = memref.cast %14 : memref<64x1x128x18x18xi32> to memref<?x?x?x?x?xi32>
    call @mcpuMemset5DInt32RandInt(%15, %c0_i16, %c0_i16, %c1_i32) : (memref<?x?x?x?x?xi32>, i16, i16, i32) -> ()
    %16 = memref.alloc() : memref<64x1x128x18x18xi32>
    %17 = memref.collapse_shape %14 [[0, 1, 2, 3, 4]] : memref<64x1x128x18x18xi32> into memref<2654208xi32>
    %18 = memref.cast %17 : memref<2654208xi32> to memref<?xi32>
    %19 = memref.collapse_shape %16 [[0, 1, 2, 3, 4]] : memref<64x1x128x18x18xi32> into memref<2654208xi32>
    %20 = memref.cast %19 : memref<2654208xi32> to memref<?xi32>
    call @_memcpy_i32_i32(%18, %20, %c2654208) : (memref<?xi32>, memref<?xi32>, index) -> ()
    call @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%0, %7, %14) : (memref<1x128x128x1x1xi8>, memref<64x1x128x16x16xi8>, memref<64x1x128x18x18xi32>) -> ()
    call @conv2d_cpu(%2, %9, %16) : (memref<1x128x128x1x1xi8>, memref<64x1x128x16x16xi8>, memref<64x1x128x18x18xi32>) -> ()
    call @miopen_conv2d_gkcyx_ngchw_ngkhw_0_verify(%14, %16) : (memref<64x1x128x18x18xi32>, memref<64x1x128x18x18xi32>) -> ()
    memref.dealloc %2 : memref<1x128x128x1x1xi8>
    memref.dealloc %9 : memref<64x1x128x16x16xi8>
    memref.dealloc %16 : memref<64x1x128x18x18xi32>
    memref.dealloc %0 : memref<1x128x128x1x1xi8>
    memref.dealloc %7 : memref<64x1x128x16x16xi8>
    memref.dealloc %14 : memref<64x1x128x18x18xi32>
    return
  }
  func private @mcpuMemset5DInt8RandInt(memref<?x?x?x?x?xi8>, i16, i16, i32)
  func @_memcpy_i8_i8(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %arg2 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<?xi8>
    memref.store %2, %arg1[%0] : memref<?xi8>
    %3 = arith.addi %0, %c1 : index
    cf.br ^bb1(%3 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func private @mcpuMemset5DInt32RandInt(memref<?x?x?x?x?xi32>, i16, i16, i32)
  func @_memcpy_i32_i32(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %arg2 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<?xi32>
    memref.store %2, %arg1[%0] : memref<?xi32>
    %3 = arith.addi %0, %c1 : index
    cf.br ^bb1(%3 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0_gpu(%arg0: memref<1x128x128x1x1xi8>, %arg1: memref<64x1x128x16x16xi8>, %arg2: memref<64x1x128x18x18xi32>) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c10368 = arith.constant 10368 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.cast %arg0 : memref<1x128x128x1x1xi8> to memref<?x?x?x?x?xi8>
    %1 = call @mgpuMemAlloc5DInt8(%0) : (memref<?x?x?x?x?xi8>) -> memref<?x?x?x?x?xi8>
    call @mgpuMemCopy5DInt8(%0, %1, %c1_i32) : (memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32) -> ()
    %2 = memref.cast %1 : memref<?x?x?x?x?xi8> to memref<1x128x128x1x1xi8>
    %3 = memref.cast %arg1 : memref<64x1x128x16x16xi8> to memref<?x?x?x?x?xi8>
    %4 = call @mgpuMemAlloc5DInt8(%3) : (memref<?x?x?x?x?xi8>) -> memref<?x?x?x?x?xi8>
    call @mgpuMemCopy5DInt8(%3, %4, %c1_i32) : (memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32) -> ()
    %5 = memref.cast %4 : memref<?x?x?x?x?xi8> to memref<64x1x128x16x16xi8>
    %6 = memref.cast %arg2 : memref<64x1x128x18x18xi32> to memref<?x?x?x?x?xi32>
    %7 = call @mgpuMemAlloc5DInt32(%6) : (memref<?x?x?x?x?xi32>) -> memref<?x?x?x?x?xi32>
    call @mgpuMemCopy5DInt32(%6, %7, %c1_i32) : (memref<?x?x?x?x?xi32>, memref<?x?x?x?x?xi32>, i32) -> ()
    %8 = memref.cast %7 : memref<?x?x?x?x?xi32> to memref<64x1x128x18x18xi32>
    gpu.launch_func  @miopen_conv2d_gkcyx_ngchw_ngkhw_0_module::@miopen_conv2d_gkcyx_ngchw_ngkhw_0 blocks in (%c10368, %c1, %c1) threads in (%c64, %c1, %c1) dynamic_shared_memory_size %c0_i32 args(%2 : memref<1x128x128x1x1xi8>, %5 : memref<64x1x128x16x16xi8>, %8 : memref<64x1x128x18x18xi32>)
    call @mgpuMemCopy5DInt8(%1, %0, %c2_i32) : (memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32) -> ()
    call @mgpuMemDealloc5DInt8(%1) : (memref<?x?x?x?x?xi8>) -> ()
    call @mgpuMemCopy5DInt8(%4, %3, %c2_i32) : (memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32) -> ()
    call @mgpuMemDealloc5DInt8(%4) : (memref<?x?x?x?x?xi8>) -> ()
    call @mgpuMemCopy5DInt32(%7, %6, %c2_i32) : (memref<?x?x?x?x?xi32>, memref<?x?x?x?x?xi32>, i32) -> ()
    call @mgpuMemDealloc5DInt32(%7) : (memref<?x?x?x?x?xi32>) -> ()
    return
  }
  func private @mgpuMemAlloc5DInt8(memref<?x?x?x?x?xi8>) -> memref<?x?x?x?x?xi8>
  func private @mgpuMemCopy5DInt8(memref<?x?x?x?x?xi8>, memref<?x?x?x?x?xi8>, i32)
  func private @mgpuMemAlloc5DInt32(memref<?x?x?x?x?xi32>) -> memref<?x?x?x?x?xi32>
  func private @mgpuMemCopy5DInt32(memref<?x?x?x?x?xi32>, memref<?x?x?x?x?xi32>, i32)
  func private @mgpuMemDealloc5DInt8(memref<?x?x?x?x?xi8>)
  func private @mgpuMemDealloc5DInt32(memref<?x?x?x?x?xi32>)
  func @conv2d_cpu(%arg0: memref<1x128x128x1x1xi8>, %arg1: memref<64x1x128x16x16xi8>, %arg2: memref<64x1x128x18x18xi32>) {
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c107_i8 = arith.constant 107 : i8
    %c99_i8 = arith.constant 99 : i8
    %c121_i8 = arith.constant 121 : i8
    %c120_i8 = arith.constant 120 : i8
    %c110_i8 = arith.constant 110 : i8
    %c104_i8 = arith.constant 104 : i8
    %c119_i8 = arith.constant 119 : i8
    %c103_i8 = arith.constant 103 : i8
    %0 = memref.cast %arg0 : memref<1x128x128x1x1xi8> to memref<*xi8>
    %1 = memref.cast %arg1 : memref<64x1x128x16x16xi8> to memref<*xi8>
    %2 = memref.cast %arg2 : memref<64x1x128x18x18xi32> to memref<*xi32>
    %3 = memref.alloca() : memref<5xi8>
    %4 = memref.alloca() : memref<5xi8>
    %5 = memref.alloca() : memref<5xi8>
    memref.store %c103_i8, %3[%c0] : memref<5xi8>
    memref.store %c107_i8, %3[%c1] : memref<5xi8>
    memref.store %c99_i8, %3[%c2] : memref<5xi8>
    memref.store %c121_i8, %3[%c3] : memref<5xi8>
    memref.store %c120_i8, %3[%c4] : memref<5xi8>
    memref.store %c110_i8, %4[%c0] : memref<5xi8>
    memref.store %c103_i8, %4[%c1] : memref<5xi8>
    memref.store %c99_i8, %4[%c2] : memref<5xi8>
    memref.store %c104_i8, %4[%c3] : memref<5xi8>
    memref.store %c119_i8, %4[%c4] : memref<5xi8>
    memref.store %c110_i8, %5[%c0] : memref<5xi8>
    memref.store %c103_i8, %5[%c1] : memref<5xi8>
    memref.store %c107_i8, %5[%c2] : memref<5xi8>
    memref.store %c104_i8, %5[%c3] : memref<5xi8>
    memref.store %c119_i8, %5[%c4] : memref<5xi8>
    %6 = memref.cast %3 : memref<5xi8> to memref<*xi8>
    %7 = memref.cast %4 : memref<5xi8> to memref<*xi8>
    %8 = memref.cast %5 : memref<5xi8> to memref<*xi8>
    call @mcpuConv2dInt8(%0, %1, %2, %6, %7, %8, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32, %c1_i32) : (memref<*xi8>, memref<*xi8>, memref<*xi32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
    return
  }
  func private @mcpuConv2dInt8(memref<*xi8>, memref<*xi8>, memref<*xi32>, memref<*xi8>, memref<*xi8>, memref<*xi8>, i32, i32, i32, i32, i32, i32, i32, i32, i32)
  func @miopen_conv2d_gkcyx_ngchw_ngkhw_0_verify(%arg0: memref<64x1x128x18x18xi32>, %arg1: memref<64x1x128x18x18xi32>) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c18 = arith.constant 18 : index
    %0 = memref.alloca() : memref<1xi32>
    memref.store %c1_i32, %0[%c0] : memref<1xi32>
    cf.br ^bb1(%c0 : index)
  ^bb1(%1: index):  // 2 preds: ^bb0, ^bb16
    %2 = arith.cmpi slt, %1, %c64 : index
    cf.cond_br %2, ^bb2, ^bb17
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0 : index)
  ^bb3(%3: index):  // 2 preds: ^bb2, ^bb15
    %4 = arith.cmpi slt, %3, %c1 : index
    cf.cond_br %4, ^bb4, ^bb16
  ^bb4:  // pred: ^bb3
    cf.br ^bb5(%c0 : index)
  ^bb5(%5: index):  // 2 preds: ^bb4, ^bb14
    %6 = arith.cmpi slt, %5, %c128 : index
    cf.cond_br %6, ^bb6, ^bb15
  ^bb6:  // pred: ^bb5
    cf.br ^bb7(%c0 : index)
  ^bb7(%7: index):  // 2 preds: ^bb6, ^bb13
    %8 = arith.cmpi slt, %7, %c18 : index
    cf.cond_br %8, ^bb8, ^bb14
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%c0 : index)
  ^bb9(%9: index):  // 2 preds: ^bb8, ^bb12
    %10 = arith.cmpi slt, %9, %c18 : index
    cf.cond_br %10, ^bb10, ^bb13
  ^bb10:  // pred: ^bb9
    %11 = memref.load %arg0[%1, %3, %5, %7, %9] : memref<64x1x128x18x18xi32>
    %12 = memref.load %arg1[%1, %3, %5, %7, %9] : memref<64x1x128x18x18xi32>
    %13 = arith.sitofp %11 : i32 to f32
    %14 = arith.sitofp %12 : i32 to f32
    %15 = arith.cmpi ne, %12, %11 : i32
    cf.cond_br %15, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    memref.store %c0_i32, %0[%c0] : memref<1xi32>
    call @mcpuPrintF32(%13, %14) : (f32, f32) -> ()
    cf.br ^bb12
  ^bb12:  // 2 preds: ^bb10, ^bb11
    %16 = arith.addi %9, %c1 : index
    cf.br ^bb9(%16 : index)
  ^bb13:  // pred: ^bb9
    %17 = arith.addi %7, %c1 : index
    cf.br ^bb7(%17 : index)
  ^bb14:  // pred: ^bb7
    %18 = arith.addi %5, %c1 : index
    cf.br ^bb5(%18 : index)
  ^bb15:  // pred: ^bb5
    %19 = arith.addi %3, %c1 : index
    cf.br ^bb3(%19 : index)
  ^bb16:  // pred: ^bb3
    %20 = arith.addi %1, %c1 : index
    cf.br ^bb1(%20 : index)
  ^bb17:  // pred: ^bb1
    %21 = memref.alloc() : memref<1xf32>
    %22 = memref.cast %0 : memref<1xi32> to memref<?xi32>
    %23 = memref.cast %21 : memref<1xf32> to memref<?xf32>
    call @_memcpy_i32_f32(%22, %23, %c1) : (memref<?xi32>, memref<?xf32>, index) -> ()
    %24 = memref.cast %21 : memref<1xf32> to memref<*xf32>
    call @print_memref_f32(%24) : (memref<*xf32>) -> ()
    memref.dealloc %21 : memref<1xf32>
    return
  }
  func private @mcpuPrintF32(f32, f32)
  func @_memcpy_i32_f32(%arg0: memref<?xi32>, %arg1: memref<?xf32>, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    cf.br ^bb1(%c0 : index)
  ^bb1(%0: index):  // 2 preds: ^bb0, ^bb2
    %1 = arith.cmpi slt, %0, %arg2 : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %2 = memref.load %arg0[%0] : memref<?xi32>
    %3 = arith.sitofp %2 : i32 to f32
    memref.store %3, %arg1[%0] : memref<?xf32>
    %4 = arith.addi %0, %c1 : index
    cf.br ^bb1(%4 : index)
  ^bb3:  // pred: ^bb1
    return
  }
  func private @print_memref_f32(memref<*xf32>)
  gpu.module @miopen_conv2d_gkcyx_ngchw_ngkhw_0_module {
    gpu.func @miopen_conv2d_gkcyx_ngchw_ngkhw_0(%arg0: memref<1x128x128x1x1xi8>, %arg1: memref<64x1x128x16x16xi8>, %arg2: memref<64x1x128x18x18xi32>) workgroup(%arg3 : memref<512xi8, 3>) private(%arg4 : memref<4xi8, 5>, %arg5 : memref<4xi8, 5>) kernel attributes {block_size = 64 : i32, grid_size = 10368 : i32} {
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      %c10368 = arith.constant 10368 : index
      %c4 = arith.constant 4 : index
      %cst = arith.constant dense<0> : vector<4xi32>
      %c324 = arith.constant 324 : index
      %c-1 = arith.constant -1 : index
      %c18 = arith.constant 18 : index
      %c0_i8 = arith.constant 0 : i8
      %c256 = arith.constant 256 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c7 = arith.constant 7 : index
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c8_i32 = arith.constant 8 : i32
      %c2 = arith.constant 2 : index
      %c16_i32 = arith.constant 16 : i32
      %c3 = arith.constant 3 : index
      %c24_i32 = arith.constant 24 : i32
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %0 = gpu.block_id  x
      %1 = gpu.thread_id  x
      %2 = arith.divui %0, %c10368 : index
      %3 = arith.remui %0, %c10368 : index
      %4 = arith.remui %3, %c8 : index
      %5 = arith.divui %3, %c8 : index
      %6 = arith.muli %4, %c16 : index
      %7 = arith.muli %5, %c16 : index
      %8 = arith.remui %1, %c16 : index
      %9 = arith.divui %1, %c16 : index
      %10 = arith.muli %9, %c4 : index
      %11 = arith.addi %6, %10 : index
      %12 = arith.divui %1, %c16 : index
      %13 = arith.remui %1, %c16 : index
      %14 = arith.muli %12, %c4 : index
      %15 = arith.addi %7, %13 : index
      %16 = memref.load %arg0[%2, %11, %8, %c0, %c0] : memref<1x128x128x1x1xi8>
      %17 = arith.addi %11, %c1 : index
      %18 = memref.load %arg0[%2, %17, %8, %c0, %c0] : memref<1x128x128x1x1xi8>
      %19 = arith.addi %11, %c2 : index
      %20 = memref.load %arg0[%2, %19, %8, %c0, %c0] : memref<1x128x128x1x1xi8>
      %21 = arith.addi %11, %c3 : index
      %22 = memref.load %arg0[%2, %21, %8, %c0, %c0] : memref<1x128x128x1x1xi8>
      %23 = arith.muli %8, %c16 : index
      %24 = arith.addi %23, %10 : index
      memref.store %16, %arg3[%24] : memref<512xi8, 3>
      %25 = arith.addi %24, %c1 : index
      memref.store %18, %arg3[%25] : memref<512xi8, 3>
      %26 = arith.addi %24, %c2 : index
      memref.store %20, %arg3[%26] : memref<512xi8, 3>
      %27 = arith.addi %24, %c3 : index
      memref.store %22, %arg3[%27] : memref<512xi8, 3>
      %28 = arith.cmpi slt, %15, %c0 : index
      %29 = arith.subi %c-1, %15 : index
      %30 = arith.select %28, %29, %15 : index
      %31 = arith.divsi %30, %c324 : index
      %32 = arith.subi %c-1, %31 : index
      %33 = arith.select %28, %32, %31 : index
      %34 = arith.remsi %15, %c324 : index
      %35 = arith.cmpi slt, %34, %c0 : index
      %36 = arith.addi %34, %c324 : index
      %37 = arith.select %35, %36, %34 : index
      %38 = arith.cmpi slt, %37, %c0 : index
      %39 = arith.subi %c-1, %37 : index
      %40 = arith.select %38, %39, %37 : index
      %41 = arith.divsi %40, %c18 : index
      %42 = arith.subi %c-1, %41 : index
      %43 = arith.select %38, %42, %41 : index
      %44 = arith.remsi %15, %c18 : index
      %45 = arith.cmpi slt, %44, %c0 : index
      %46 = arith.addi %44, %c18 : index
      %47 = arith.select %45, %46, %44 : index
      %48 = arith.addi %43, %c-1 : index
      %49 = arith.addi %47, %c-1 : index
      %50 = arith.cmpi sge, %48, %c0 : index
      %51 = arith.cmpi slt, %48, %c16 : index
      %52 = arith.andi %50, %51 : i1
      %53 = arith.cmpi sge, %49, %c0 : index
      %54 = arith.cmpi slt, %49, %c16 : index
      %55 = arith.andi %53, %54 : i1
      %56 = arith.andi %52, %55 : i1
      cf.cond_br %56, ^bb2(%48, %49 : index, index), ^bb2(%c0, %c0 : index, index)
    ^bb2(%57: index, %58: index):  // 2 preds: ^bb1, ^bb1
      cf.br ^bb3(%33, %2, %14, %57, %58 : index, index, index, index, index)
    ^bb3(%59: index, %60: index, %61: index, %62: index, %63: index):  // pred: ^bb2
      cf.br ^bb4
    ^bb4:  // pred: ^bb3
      %64 = memref.load %arg1[%59, %60, %61, %62, %63] : memref<64x1x128x16x16xi8>
      cf.cond_br %56, ^bb5(%64 : i8), ^bb5(%c0_i8 : i8)
    ^bb5(%65: i8):  // 2 preds: ^bb4, ^bb4
      cf.br ^bb6(%65 : i8)
    ^bb6(%66: i8):  // pred: ^bb5
      cf.br ^bb7
    ^bb7:  // pred: ^bb6
      %67 = arith.addi %14, %c1 : index
      %68 = arith.cmpi sge, %48, %c0 : index
      %69 = arith.cmpi slt, %48, %c16 : index
      %70 = arith.andi %68, %69 : i1
      %71 = arith.cmpi sge, %49, %c0 : index
      %72 = arith.cmpi slt, %49, %c16 : index
      %73 = arith.andi %71, %72 : i1
      %74 = arith.andi %70, %73 : i1
      cf.cond_br %74, ^bb8(%48, %49 : index, index), ^bb8(%c0, %c0 : index, index)
    ^bb8(%75: index, %76: index):  // 2 preds: ^bb7, ^bb7
      cf.br ^bb9(%33, %2, %67, %75, %76 : index, index, index, index, index)
    ^bb9(%77: index, %78: index, %79: index, %80: index, %81: index):  // pred: ^bb8
      cf.br ^bb10
    ^bb10:  // pred: ^bb9
      %82 = memref.load %arg1[%77, %78, %79, %80, %81] : memref<64x1x128x16x16xi8>
      cf.cond_br %74, ^bb11(%82 : i8), ^bb11(%c0_i8 : i8)
    ^bb11(%83: i8):  // 2 preds: ^bb10, ^bb10
      cf.br ^bb12(%83 : i8)
    ^bb12(%84: i8):  // pred: ^bb11
      cf.br ^bb13
    ^bb13:  // pred: ^bb12
      %85 = arith.addi %14, %c2 : index
      %86 = arith.cmpi sge, %48, %c0 : index
      %87 = arith.cmpi slt, %48, %c16 : index
      %88 = arith.andi %86, %87 : i1
      %89 = arith.cmpi sge, %49, %c0 : index
      %90 = arith.cmpi slt, %49, %c16 : index
      %91 = arith.andi %89, %90 : i1
      %92 = arith.andi %88, %91 : i1
      cf.cond_br %92, ^bb14(%48, %49 : index, index), ^bb14(%c0, %c0 : index, index)
    ^bb14(%93: index, %94: index):  // 2 preds: ^bb13, ^bb13
      cf.br ^bb15(%33, %2, %85, %93, %94 : index, index, index, index, index)
    ^bb15(%95: index, %96: index, %97: index, %98: index, %99: index):  // pred: ^bb14
      cf.br ^bb16
    ^bb16:  // pred: ^bb15
      %100 = memref.load %arg1[%95, %96, %97, %98, %99] : memref<64x1x128x16x16xi8>
      cf.cond_br %92, ^bb17(%100 : i8), ^bb17(%c0_i8 : i8)
    ^bb17(%101: i8):  // 2 preds: ^bb16, ^bb16
      cf.br ^bb18(%101 : i8)
    ^bb18(%102: i8):  // pred: ^bb17
      cf.br ^bb19
    ^bb19:  // pred: ^bb18
      %103 = arith.addi %14, %c3 : index
      %104 = arith.cmpi sge, %48, %c0 : index
      %105 = arith.cmpi slt, %48, %c16 : index
      %106 = arith.andi %104, %105 : i1
      %107 = arith.cmpi sge, %49, %c0 : index
      %108 = arith.cmpi slt, %49, %c16 : index
      %109 = arith.andi %107, %108 : i1
      %110 = arith.andi %106, %109 : i1
      cf.cond_br %110, ^bb20(%48, %49 : index, index), ^bb20(%c0, %c0 : index, index)
    ^bb20(%111: index, %112: index):  // 2 preds: ^bb19, ^bb19
      cf.br ^bb21(%33, %2, %103, %111, %112 : index, index, index, index, index)
    ^bb21(%113: index, %114: index, %115: index, %116: index, %117: index):  // pred: ^bb20
      cf.br ^bb22
    ^bb22:  // pred: ^bb21
      %118 = memref.load %arg1[%113, %114, %115, %116, %117] : memref<64x1x128x16x16xi8>
      cf.cond_br %110, ^bb23(%118 : i8), ^bb23(%c0_i8 : i8)
    ^bb23(%119: i8):  // 2 preds: ^bb22, ^bb22
      cf.br ^bb24(%119 : i8)
    ^bb24(%120: i8):  // pred: ^bb23
      cf.br ^bb25
    ^bb25:  // pred: ^bb24
      %121 = arith.muli %14, %c16 : index
      %122 = arith.addi %121, %13 : index
      %123 = arith.addi %122, %c256 : index
      memref.store %66, %arg3[%123] : memref<512xi8, 3>
      %124 = arith.addi %123, %c16 : index
      memref.store %84, %arg3[%124] : memref<512xi8, 3>
      %125 = arith.addi %123, %c32 : index
      memref.store %102, %arg3[%125] : memref<512xi8, 3>
      %126 = arith.addi %123, %c48 : index
      memref.store %120, %arg3[%126] : memref<512xi8, 3>
      %127 = arith.divui %1, %c64 : index
      %128 = arith.muli %127, %c16 : index
      cf.br ^bb26(%c0, %8, %14, %cst : index, index, index, vector<4xi32>)
    ^bb26(%129: index, %130: index, %131: index, %132: vector<4xi32>):  // 2 preds: ^bb25, ^bb60
      %133 = arith.cmpi slt, %129, %c7 : index
      cf.cond_br %133, ^bb27, ^bb61
    ^bb27:  // pred: ^bb26
      %134 = arith.addi %130, %c16 : index
      %135 = memref.load %arg0[%2, %11, %134, %c0, %c0] : memref<1x128x128x1x1xi8>
      %136 = arith.addi %11, %c1 : index
      %137 = memref.load %arg0[%2, %136, %134, %c0, %c0] : memref<1x128x128x1x1xi8>
      %138 = arith.addi %11, %c2 : index
      %139 = memref.load %arg0[%2, %138, %134, %c0, %c0] : memref<1x128x128x1x1xi8>
      %140 = arith.addi %11, %c3 : index
      %141 = memref.load %arg0[%2, %140, %134, %c0, %c0] : memref<1x128x128x1x1xi8>
      %142 = arith.addi %131, %c16 : index
      %143 = arith.cmpi slt, %15, %c0 : index
      %144 = arith.subi %c-1, %15 : index
      %145 = arith.select %143, %144, %15 : index
      %146 = arith.divsi %145, %c324 : index
      %147 = arith.subi %c-1, %146 : index
      %148 = arith.select %143, %147, %146 : index
      %149 = arith.remsi %15, %c324 : index
      %150 = arith.cmpi slt, %149, %c0 : index
      %151 = arith.addi %149, %c324 : index
      %152 = arith.select %150, %151, %149 : index
      %153 = arith.cmpi slt, %152, %c0 : index
      %154 = arith.subi %c-1, %152 : index
      %155 = arith.select %153, %154, %152 : index
      %156 = arith.divsi %155, %c18 : index
      %157 = arith.subi %c-1, %156 : index
      %158 = arith.select %153, %157, %156 : index
      %159 = arith.remsi %15, %c18 : index
      %160 = arith.cmpi slt, %159, %c0 : index
      %161 = arith.addi %159, %c18 : index
      %162 = arith.select %160, %161, %159 : index
      %163 = arith.addi %158, %c-1 : index
      %164 = arith.addi %162, %c-1 : index
      %165 = arith.cmpi sge, %163, %c0 : index
      %166 = arith.cmpi slt, %163, %c16 : index
      %167 = arith.andi %165, %166 : i1
      %168 = arith.cmpi sge, %164, %c0 : index
      %169 = arith.cmpi slt, %164, %c16 : index
      %170 = arith.andi %168, %169 : i1
      %171 = arith.andi %167, %170 : i1
      cf.cond_br %171, ^bb28(%163, %164 : index, index), ^bb28(%c0, %c0 : index, index)
    ^bb28(%172: index, %173: index):  // 2 preds: ^bb27, ^bb27
      cf.br ^bb29(%148, %2, %142, %172, %173 : index, index, index, index, index)
    ^bb29(%174: index, %175: index, %176: index, %177: index, %178: index):  // pred: ^bb28
      cf.br ^bb30
    ^bb30:  // pred: ^bb29
      %179 = memref.load %arg1[%174, %175, %176, %177, %178] : memref<64x1x128x16x16xi8>
      cf.cond_br %171, ^bb31(%179 : i8), ^bb31(%c0_i8 : i8)
    ^bb31(%180: i8):  // 2 preds: ^bb30, ^bb30
      cf.br ^bb32(%180 : i8)
    ^bb32(%181: i8):  // pred: ^bb31
      cf.br ^bb33
    ^bb33:  // pred: ^bb32
      %182 = arith.addi %142, %c1 : index
      %183 = arith.cmpi sge, %163, %c0 : index
      %184 = arith.cmpi slt, %163, %c16 : index
      %185 = arith.andi %183, %184 : i1
      %186 = arith.cmpi sge, %164, %c0 : index
      %187 = arith.cmpi slt, %164, %c16 : index
      %188 = arith.andi %186, %187 : i1
      %189 = arith.andi %185, %188 : i1
      cf.cond_br %189, ^bb34(%163, %164 : index, index), ^bb34(%c0, %c0 : index, index)
    ^bb34(%190: index, %191: index):  // 2 preds: ^bb33, ^bb33
      cf.br ^bb35(%148, %2, %182, %190, %191 : index, index, index, index, index)
    ^bb35(%192: index, %193: index, %194: index, %195: index, %196: index):  // pred: ^bb34
      cf.br ^bb36
    ^bb36:  // pred: ^bb35
      %197 = memref.load %arg1[%192, %193, %194, %195, %196] : memref<64x1x128x16x16xi8>
      cf.cond_br %189, ^bb37(%197 : i8), ^bb37(%c0_i8 : i8)
    ^bb37(%198: i8):  // 2 preds: ^bb36, ^bb36
      cf.br ^bb38(%198 : i8)
    ^bb38(%199: i8):  // pred: ^bb37
      cf.br ^bb39
    ^bb39:  // pred: ^bb38
      %200 = arith.addi %142, %c2 : index
      %201 = arith.cmpi sge, %163, %c0 : index
      %202 = arith.cmpi slt, %163, %c16 : index
      %203 = arith.andi %201, %202 : i1
      %204 = arith.cmpi sge, %164, %c0 : index
      %205 = arith.cmpi slt, %164, %c16 : index
      %206 = arith.andi %204, %205 : i1
      %207 = arith.andi %203, %206 : i1
      cf.cond_br %207, ^bb40(%163, %164 : index, index), ^bb40(%c0, %c0 : index, index)
    ^bb40(%208: index, %209: index):  // 2 preds: ^bb39, ^bb39
      cf.br ^bb41(%148, %2, %200, %208, %209 : index, index, index, index, index)
    ^bb41(%210: index, %211: index, %212: index, %213: index, %214: index):  // pred: ^bb40
      cf.br ^bb42
    ^bb42:  // pred: ^bb41
      %215 = memref.load %arg1[%210, %211, %212, %213, %214] : memref<64x1x128x16x16xi8>
      cf.cond_br %207, ^bb43(%215 : i8), ^bb43(%c0_i8 : i8)
    ^bb43(%216: i8):  // 2 preds: ^bb42, ^bb42
      cf.br ^bb44(%216 : i8)
    ^bb44(%217: i8):  // pred: ^bb43
      cf.br ^bb45
    ^bb45:  // pred: ^bb44
      %218 = arith.addi %142, %c3 : index
      %219 = arith.cmpi sge, %163, %c0 : index
      %220 = arith.cmpi slt, %163, %c16 : index
      %221 = arith.andi %219, %220 : i1
      %222 = arith.cmpi sge, %164, %c0 : index
      %223 = arith.cmpi slt, %164, %c16 : index
      %224 = arith.andi %222, %223 : i1
      %225 = arith.andi %221, %224 : i1
      cf.cond_br %225, ^bb46(%163, %164 : index, index), ^bb46(%c0, %c0 : index, index)
    ^bb46(%226: index, %227: index):  // 2 preds: ^bb45, ^bb45
      cf.br ^bb47(%148, %2, %218, %226, %227 : index, index, index, index, index)
    ^bb47(%228: index, %229: index, %230: index, %231: index, %232: index):  // pred: ^bb46
      cf.br ^bb48
    ^bb48:  // pred: ^bb47
      %233 = memref.load %arg1[%228, %229, %230, %231, %232] : memref<64x1x128x16x16xi8>
      cf.cond_br %225, ^bb49(%233 : i8), ^bb49(%c0_i8 : i8)
    ^bb49(%234: i8):  // 2 preds: ^bb48, ^bb48
      cf.br ^bb50(%234 : i8)
    ^bb50(%235: i8):  // pred: ^bb49
      cf.br ^bb51
    ^bb51:  // pred: ^bb50
      "gpu.lds_barrier"() : () -> ()
      %236 = gpu.thread_id  x
      %237 = arith.remui %236, %c64 : index
      %238 = arith.divui %237, %c16 : index
      %239 = arith.remui %237, %c16 : index
      cf.br ^bb52(%c0 : index)
    ^bb52(%240: index):  // 2 preds: ^bb51, ^bb53
      %241 = arith.cmpi slt, %240, %c4 : index
      cf.cond_br %241, ^bb53, ^bb54
    ^bb53:  // pred: ^bb52
      %242 = arith.muli %240, %c4 : index
      %243 = arith.addi %242, %238 : index
      %244 = arith.muli %243, %c16 : index
      %245 = arith.addi %244, %239 : index
      %246 = arith.addi %128, %245 : index
      %247 = memref.load %arg3[%246] : memref<512xi8, 3>
      memref.store %247, %arg5[%240] : memref<4xi8, 5>
      %248 = arith.muli %240, %c4 : index
      %249 = arith.addi %248, %238 : index
      %250 = arith.muli %249, %c16 : index
      %251 = arith.addi %250, %239 : index
      %252 = arith.addi %251, %c256 : index
      %253 = memref.load %arg3[%252] : memref<512xi8, 3>
      memref.store %253, %arg4[%240] : memref<4xi8, 5>
      %254 = arith.addi %240, %c1 : index
      cf.br ^bb52(%254 : index)
    ^bb54:  // pred: ^bb52
      cf.br ^bb55(%c0, %132 : index, vector<4xi32>)
    ^bb55(%255: index, %256: vector<4xi32>):  // 2 preds: ^bb54, ^bb59
      %257 = arith.cmpi slt, %255, %c4 : index
      cf.cond_br %257, ^bb56, ^bb60
    ^bb56:  // pred: ^bb55
      %258 = arith.muli %255, %c4 : index
      cf.br ^bb57(%c0, %256 : index, vector<4xi32>)
    ^bb57(%259: index, %260: vector<4xi32>):  // 2 preds: ^bb56, ^bb58
      %261 = arith.cmpi slt, %259, %c4 : index
      cf.cond_br %261, ^bb58, ^bb59
    ^bb58:  // pred: ^bb57
      %262 = arith.addi %258, %259 : index
      %263 = vector.transfer_read %arg5[%262], %c0_i8 : memref<4xi8, 5>, vector<4xi8>
      %264 = vector.transfer_read %arg4[%262], %c0_i8 : memref<4xi8, 5>, vector<4xi8>
      %265 = vector.extractelement %263[%c0 : index] : vector<4xi8>
      %266 = arith.extui %265 : i8 to i32
      %267 = arith.shli %266, %c0_i32 : i32
      %268 = vector.extractelement %263[%c1 : index] : vector<4xi8>
      %269 = arith.extui %268 : i8 to i32
      %270 = arith.shli %269, %c8_i32 : i32
      %271 = arith.ori %267, %270 : i32
      %272 = vector.extractelement %263[%c2 : index] : vector<4xi8>
      %273 = arith.extui %272 : i8 to i32
      %274 = arith.shli %273, %c16_i32 : i32
      %275 = arith.ori %271, %274 : i32
      %276 = vector.extractelement %263[%c3 : index] : vector<4xi8>
      %277 = arith.extui %276 : i8 to i32
      %278 = arith.shli %277, %c24_i32 : i32
      %279 = arith.ori %275, %278 : i32
      %280 = vector.extractelement %264[%c0 : index] : vector<4xi8>
      %281 = arith.extui %280 : i8 to i32
      %282 = arith.shli %281, %c0_i32 : i32
      %283 = vector.extractelement %264[%c1 : index] : vector<4xi8>
      %284 = arith.extui %283 : i8 to i32
      %285 = arith.shli %284, %c8_i32 : i32
      %286 = arith.ori %282, %285 : i32
      %287 = vector.extractelement %264[%c2 : index] : vector<4xi8>
      %288 = arith.extui %287 : i8 to i32
      %289 = arith.shli %288, %c16_i32 : i32
      %290 = arith.ori %286, %289 : i32
      %291 = vector.extractelement %264[%c3 : index] : vector<4xi8>
      %292 = arith.extui %291 : i8 to i32
      %293 = arith.shli %292, %c24_i32 : i32
      %294 = arith.ori %290, %293 : i32
      %295 = gpu.mfma(%279, %294, %260) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_16x16x16i8"} : i32, vector<4xi32>
      %296 = arith.addi %259, %c4 : index
      cf.br ^bb57(%296, %295 : index, vector<4xi32>)
    ^bb59:  // pred: ^bb57
      %297 = arith.addi %255, %c4 : index
      cf.br ^bb55(%297, %260 : index, vector<4xi32>)
    ^bb60:  // pred: ^bb55
      "gpu.lds_barrier"() : () -> ()
      %298 = arith.muli %8, %c16 : index
      %299 = arith.addi %298, %10 : index
      memref.store %135, %arg3[%299] : memref<512xi8, 3>
      %300 = arith.addi %299, %c1 : index
      memref.store %137, %arg3[%300] : memref<512xi8, 3>
      %301 = arith.addi %299, %c2 : index
      memref.store %139, %arg3[%301] : memref<512xi8, 3>
      %302 = arith.addi %299, %c3 : index
      memref.store %141, %arg3[%302] : memref<512xi8, 3>
      %303 = arith.muli %14, %c16 : index
      %304 = arith.addi %303, %13 : index
      %305 = arith.addi %304, %c256 : index
      memref.store %181, %arg3[%305] : memref<512xi8, 3>
      %306 = arith.addi %305, %c16 : index
      memref.store %199, %arg3[%306] : memref<512xi8, 3>
      %307 = arith.addi %305, %c32 : index
      memref.store %217, %arg3[%307] : memref<512xi8, 3>
      %308 = arith.addi %305, %c48 : index
      memref.store %235, %arg3[%308] : memref<512xi8, 3>
      %309 = arith.addi %129, %c1 : index
      cf.br ^bb26(%309, %134, %142, %256 : index, index, index, vector<4xi32>)
    ^bb61:  // pred: ^bb26
      "gpu.lds_barrier"() : () -> ()
      %310 = gpu.thread_id  x
      %311 = arith.remui %310, %c64 : index
      %312 = arith.divui %311, %c16 : index
      %313 = arith.remui %311, %c16 : index
      cf.br ^bb62(%c0 : index)
    ^bb62(%314: index):  // 2 preds: ^bb61, ^bb63
      %315 = arith.cmpi slt, %314, %c4 : index
      cf.cond_br %315, ^bb63, ^bb64
    ^bb63:  // pred: ^bb62
      %316 = arith.muli %314, %c4 : index
      %317 = arith.addi %316, %312 : index
      %318 = arith.muli %317, %c16 : index
      %319 = arith.addi %318, %313 : index
      %320 = arith.addi %128, %319 : index
      %321 = memref.load %arg3[%320] : memref<512xi8, 3>
      memref.store %321, %arg5[%314] : memref<4xi8, 5>
      %322 = arith.muli %314, %c4 : index
      %323 = arith.addi %322, %312 : index
      %324 = arith.muli %323, %c16 : index
      %325 = arith.addi %324, %313 : index
      %326 = arith.addi %325, %c256 : index
      %327 = memref.load %arg3[%326] : memref<512xi8, 3>
      memref.store %327, %arg4[%314] : memref<4xi8, 5>
      %328 = arith.addi %314, %c1 : index
      cf.br ^bb62(%328 : index)
    ^bb64:  // pred: ^bb62
      cf.br ^bb65(%c0, %132 : index, vector<4xi32>)
    ^bb65(%329: index, %330: vector<4xi32>):  // 2 preds: ^bb64, ^bb69
      %331 = arith.cmpi slt, %329, %c4 : index
      cf.cond_br %331, ^bb66, ^bb70
    ^bb66:  // pred: ^bb65
      %332 = arith.muli %329, %c4 : index
      cf.br ^bb67(%c0, %330 : index, vector<4xi32>)
    ^bb67(%333: index, %334: vector<4xi32>):  // 2 preds: ^bb66, ^bb68
      %335 = arith.cmpi slt, %333, %c4 : index
      cf.cond_br %335, ^bb68, ^bb69
    ^bb68:  // pred: ^bb67
      %336 = arith.addi %332, %333 : index
      %337 = vector.transfer_read %arg5[%336], %c0_i8 : memref<4xi8, 5>, vector<4xi8>
      %338 = vector.transfer_read %arg4[%336], %c0_i8 : memref<4xi8, 5>, vector<4xi8>
      %339 = vector.extractelement %337[%c0 : index] : vector<4xi8>
      %340 = arith.extui %339 : i8 to i32
      %341 = arith.shli %340, %c0_i32 : i32
      %342 = vector.extractelement %337[%c1 : index] : vector<4xi8>
      %343 = arith.extui %342 : i8 to i32
      %344 = arith.shli %343, %c8_i32 : i32
      %345 = arith.ori %341, %344 : i32
      %346 = vector.extractelement %337[%c2 : index] : vector<4xi8>
      %347 = arith.extui %346 : i8 to i32
      %348 = arith.shli %347, %c16_i32 : i32
      %349 = arith.ori %345, %348 : i32
      %350 = vector.extractelement %337[%c3 : index] : vector<4xi8>
      %351 = arith.extui %350 : i8 to i32
      %352 = arith.shli %351, %c24_i32 : i32
      %353 = arith.ori %349, %352 : i32
      %354 = vector.extractelement %338[%c0 : index] : vector<4xi8>
      %355 = arith.extui %354 : i8 to i32
      %356 = arith.shli %355, %c0_i32 : i32
      %357 = vector.extractelement %338[%c1 : index] : vector<4xi8>
      %358 = arith.extui %357 : i8 to i32
      %359 = arith.shli %358, %c8_i32 : i32
      %360 = arith.ori %356, %359 : i32
      %361 = vector.extractelement %338[%c2 : index] : vector<4xi8>
      %362 = arith.extui %361 : i8 to i32
      %363 = arith.shli %362, %c16_i32 : i32
      %364 = arith.ori %360, %363 : i32
      %365 = vector.extractelement %338[%c3 : index] : vector<4xi8>
      %366 = arith.extui %365 : i8 to i32
      %367 = arith.shli %366, %c24_i32 : i32
      %368 = arith.ori %364, %367 : i32
      %369 = gpu.mfma(%353, %368, %334) {imm = [0 : i32, 0 : i32, 0 : i32], instr = "mfma_i32_16x16x16i8"} : i32, vector<4xi32>
      %370 = arith.addi %333, %c4 : index
      cf.br ^bb67(%370, %369 : index, vector<4xi32>)
    ^bb69:  // pred: ^bb67
      %371 = arith.addi %329, %c4 : index
      cf.br ^bb65(%371, %334 : index, vector<4xi32>)
    ^bb70:  // pred: ^bb65
      %372 = arith.remui %1, %c64 : index
      %373 = arith.divui %372, %c16 : index
      %374 = arith.remui %372, %c16 : index
      %375 = arith.andi %372, %c2 : index
      %376 = arith.cmpi ne, %375, %c0 : index
      %377 = vector.extractelement %330[%c0 : index] : vector<4xi32>
      %378 = vector.extractelement %330[%c1 : index] : vector<4xi32>
      %379 = vector.extractelement %330[%c2 : index] : vector<4xi32>
      %380 = vector.extractelement %330[%c3 : index] : vector<4xi32>
      %381 = arith.select %376, %377, %379 : i32
      %382 = vector.insertelement %381, %330[%c2 : index] : vector<4xi32>
      %383 = arith.select %376, %378, %380 : i32
      %384 = vector.insertelement %383, %382[%c3 : index] : vector<4xi32>
      %385 = arith.select %376, %379, %377 : i32
      %386 = vector.insertelement %385, %384[%c0 : index] : vector<4xi32>
      %387 = arith.select %376, %380, %378 : i32
      %388 = vector.insertelement %387, %386[%c1 : index] : vector<4xi32>
      %389 = arith.andi %372, %c1 : index
      %390 = arith.cmpi ne, %389, %c0 : index
      %391 = vector.extractelement %388[%c0 : index] : vector<4xi32>
      %392 = vector.extractelement %388[%c1 : index] : vector<4xi32>
      %393 = vector.extractelement %388[%c2 : index] : vector<4xi32>
      %394 = vector.extractelement %388[%c3 : index] : vector<4xi32>
      %395 = arith.select %390, %391, %392 : i32
      %396 = vector.insertelement %395, %388[%c1 : index] : vector<4xi32>
      %397 = arith.select %390, %392, %393 : i32
      %398 = vector.insertelement %397, %396[%c2 : index] : vector<4xi32>
      %399 = arith.select %390, %393, %394 : i32
      %400 = vector.insertelement %399, %398[%c3 : index] : vector<4xi32>
      %401 = arith.select %390, %394, %391 : i32
      %402 = vector.insertelement %401, %400[%c0 : index] : vector<4xi32>
      %403 = vector.extractelement %402[%c0 : index] : vector<4xi32>
      %404 = vector.extractelement %402[%c1 : index] : vector<4xi32>
      %405 = vector.extractelement %402[%c2 : index] : vector<4xi32>
      %406 = vector.extractelement %402[%c3 : index] : vector<4xi32>
      %407 = gpu.warp_swizzle {selector = [0 : i32, 3 : i32, 2 : i32, 1 : i32]} %403 : i32
      %408 = gpu.warp_swizzle {selector = [1 : i32, 0 : i32, 3 : i32, 2 : i32]} %404 : i32
      %409 = gpu.warp_swizzle {selector = [2 : i32, 1 : i32, 0 : i32, 3 : i32]} %405 : i32
      %410 = gpu.warp_swizzle {selector = [3 : i32, 2 : i32, 1 : i32, 0 : i32]} %406 : i32
      %411 = vector.insertelement %407, %402[%c0 : index] : vector<4xi32>
      %412 = vector.insertelement %408, %411[%c1 : index] : vector<4xi32>
      %413 = vector.insertelement %409, %412[%c2 : index] : vector<4xi32>
      %414 = vector.insertelement %410, %413[%c3 : index] : vector<4xi32>
      %415 = arith.andi %372, %c1 : index
      %416 = arith.cmpi ne, %415, %c0 : index
      %417 = vector.extractelement %414[%c0 : index] : vector<4xi32>
      %418 = vector.extractelement %414[%c1 : index] : vector<4xi32>
      %419 = vector.extractelement %414[%c2 : index] : vector<4xi32>
      %420 = vector.extractelement %414[%c3 : index] : vector<4xi32>
      %421 = arith.select %416, %417, %420 : i32
      %422 = vector.insertelement %421, %414[%c3 : index] : vector<4xi32>
      %423 = arith.select %416, %418, %417 : i32
      %424 = vector.insertelement %423, %422[%c0 : index] : vector<4xi32>
      %425 = arith.select %416, %419, %418 : i32
      %426 = vector.insertelement %425, %424[%c1 : index] : vector<4xi32>
      %427 = arith.select %416, %420, %419 : i32
      %428 = vector.insertelement %427, %426[%c2 : index] : vector<4xi32>
      %429 = arith.andi %372, %c2 : index
      %430 = arith.cmpi ne, %429, %c0 : index
      %431 = vector.extractelement %428[%c0 : index] : vector<4xi32>
      %432 = vector.extractelement %428[%c1 : index] : vector<4xi32>
      %433 = vector.extractelement %428[%c2 : index] : vector<4xi32>
      %434 = vector.extractelement %428[%c3 : index] : vector<4xi32>
      %435 = arith.select %430, %431, %433 : i32
      %436 = vector.insertelement %435, %428[%c2 : index] : vector<4xi32>
      %437 = arith.select %430, %432, %434 : i32
      %438 = vector.insertelement %437, %436[%c3 : index] : vector<4xi32>
      %439 = arith.select %430, %433, %431 : i32
      %440 = vector.insertelement %439, %438[%c0 : index] : vector<4xi32>
      %441 = arith.select %430, %434, %432 : i32
      %442 = vector.insertelement %441, %440[%c1 : index] : vector<4xi32>
      %443 = arith.divui %374, %c4 : index
      %444 = arith.muli %443, %c4 : index
      %445 = arith.muli %373, %c4 : index
      %446 = arith.remui %374, %c4 : index
      %447 = arith.addi %445, %446 : index
      %448 = arith.muli %127, %c16 : index
      %449 = arith.addi %448, %447 : index
      %450 = arith.addi %6, %449 : index
      %451 = arith.addi %7, %444 : index
      %452 = arith.divui %450, %c16 : index
      %453 = arith.remui %450, %c16 : index
      %454 = arith.divui %453, %c4 : index
      %455 = arith.remui %450, %c4 : index
      %456 = arith.divui %451, %c4 : index
      %457 = arith.remui %451, %c4 : index
      %458 = arith.muli %452, %c16 : index
      %459 = arith.muli %454, %c4 : index
      %460 = arith.addi %458, %459 : index
      %461 = arith.addi %460, %455 : index
      %462 = arith.muli %456, %c4 : index
      %463 = arith.addi %462, %457 : index
      %464 = arith.cmpi slt, %463, %c0 : index
      %465 = arith.subi %c-1, %463 : index
      %466 = arith.select %464, %465, %463 : index
      %467 = arith.divsi %466, %c324 : index
      %468 = arith.subi %c-1, %467 : index
      %469 = arith.select %464, %468, %467 : index
      %470 = arith.remsi %463, %c324 : index
      %471 = arith.cmpi slt, %470, %c0 : index
      %472 = arith.addi %470, %c324 : index
      %473 = arith.select %471, %472, %470 : index
      %474 = arith.cmpi slt, %473, %c0 : index
      %475 = arith.subi %c-1, %473 : index
      %476 = arith.select %474, %475, %473 : index
      %477 = arith.divsi %476, %c18 : index
      %478 = arith.subi %c-1, %477 : index
      %479 = arith.select %474, %478, %477 : index
      %480 = arith.remsi %463, %c18 : index
      %481 = arith.cmpi slt, %480, %c0 : index
      %482 = arith.addi %480, %c18 : index
      %483 = arith.select %481, %482, %480 : index
      %484 = vector.extractelement %442[%c0 : index] : vector<4xi32>
      %485 = vector.insertelement %484, %cst[%c0 : index] : vector<4xi32>
      %486 = vector.extractelement %442[%c1 : index] : vector<4xi32>
      %487 = vector.insertelement %486, %485[%c1 : index] : vector<4xi32>
      %488 = vector.extractelement %442[%c2 : index] : vector<4xi32>
      %489 = vector.insertelement %488, %487[%c2 : index] : vector<4xi32>
      %490 = vector.extractelement %442[%c3 : index] : vector<4xi32>
      %491 = vector.insertelement %490, %489[%c3 : index] : vector<4xi32>
      %492 = arith.index_cast %469 : index to i32
      %493 = arith.index_cast %2 : index to i32
      %494 = arith.index_cast %461 : index to i32
      %495 = arith.index_cast %479 : index to i32
      %496 = arith.index_cast %483 : index to i32
      gpu.raw_buffer_store(%491, %arg2, %c0_i32, %492, %493, %494, %495, %496) : vector<4xi32>, memref<64x1x128x18x18xi32>, i32, i32, i32, i32, i32, i32
      gpu.return
    }
  }
}
