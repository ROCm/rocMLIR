// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --test-side-effects --verify-diagnostics

func.func @rock_conv(%filter : memref<?x?x?x?x?xf32>,
                     %input : memref<?x?x?x?x?xf32>,
                     %output : memref<?x?x?x?x?xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource 'transform.mapping'}}
  rock.conv(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}

func.func @rock_conv_bwd_data(%filter : memref<?x?x?x?x?xf32>,
                              %input : memref<?x?x?x?x?xf32>,
                              %output : memref<?x?x?x?x?xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource 'transform.mapping'}}
  rock.conv_bwd_data(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    kernelId = 0 : index,
    input_layout = ["n", "gi", "c", "0i", "1i"],
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}

func.func @rock_conv_bwd_weight(%filter : memref<?x?x?x?x?xf32>,
                                %input : memref<?x?x?x?x?xf32>,
                                %output : memref<?x?x?x?x?xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'write' on op operand 0, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource 'transform.mapping'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource 'transform.mapping'}}
  rock.conv_bwd_weight(%filter, %input, %output) {
    filter_layout = ["g", "k", "c", "0", "1"],
    input_layout = ["n", "gi", "c", "0i", "1i"],
    numCU = 64 : i32,
    output_layout = ["n", "go", "k", "0o", "1o"],
    dilations = [1 : index,  1 : index],
    strides = [1 : index,  1 : index],
    padding = [0 : index,  0 : index,  0 : index,  0 : index]
  } : memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>, memref<?x?x?x?x?xf32>
  return
}

func.func @rock_gemm(%a : memref<32x64xf16>,
                     %b : memref<1x32x128xf16>,
                     %c : memref<64x128xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.gemm %c = tr %a * %b storeMethod = set
  : memref<64x128xf32> = memref<32x64xf16> * memref<1x32x128xf16>
  func.return
}

func.func @rock_gridwise_gemm(%A : memref<2x72x128xf32>,
                              %B : memref<2x72x256xf32>,
                              %C : memref<2x128x256xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.gridwise_gemm %C = %A * %B storeMethod(set) {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    numCU = 64 : i32,
    params = #rock.general_gemm_params<
      blockSize = 128,
      kPerBlock = 8,
      kPerThread = 1,
      kpack = 1,
      mPerBlock = 128,
      mPerThread = 4,
      nPerBlock = 128,
      nPerThread = 4,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2>
  } : memref<2x128x256xf32> = memref<2x72x128xf32> * memref<2x72x256xf32>
  return
}

func.func @rock_gridwise_gemm_accel(%A : memref<2x1024x1024xf32>,
                                    %B : memref<2x1024x2048xf32>,
                                    %C : memref<2x1024x2048xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.gridwise_gemm_accel(%A, %B, %C) storeMethod(set) {
    blockSize = 256 : i32,
    gridSize = 1 : i32,
    params = #rock.xdlops_gemm_derived_params<
      kpackPerBlock = 4,
      kpack = 4,
      mPerBlock = 128,
      mPerWave = 64,
      nPerBlock = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<2x1024x1024xf32>, memref<2x1024x2048xf32>, memref<2x1024x2048xf32>
  return
}

func.func @rock_global_load(%source : memref<?x?x?x?x?xf32>, %valid : i1)  attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{operation has no memory effects}}
  %c1 = arith.constant 1 : index
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %loaded = rock.global_load
    %source[%c1, %c1, %c1, %c1, %c1] if %valid
    : memref<?x?x?x?x?xf32> -> vector<8xf32>
  return 
}

func.func @rock_global_load_to_lds(%mem: memref<192xf32>) attributes {arch = "##TOKEN_ARCH##"} {
    // expected-remark @below {{operation has no memory effects}}
    %c0 = arith.constant 0 : index
    // expected-remark @below {{operation has no memory effects}}
    %true = arith.constant true
    // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
    %lds = rock.alloc() : memref<64xi8, #gpu.address_space<workgroup>>
    // expected-remark @below {{operation has no memory effects}}
    %lds_view = memref.view %lds[%c0][] : memref<64xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
    // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
    // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
    rock.global_load_to_lds %mem[%c0] -> %lds_view[%c0]  if %true {transferType = f32} : memref<192xf32> -> memref<4xf32, #gpu.address_space<workgroup>>
    return
}

func.func @rock_in_bounds_load(%buffer: memref<128x128xf32, 3>,
                               %idx0: index,
                               %idx1: index) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  %ret = rock.in_bounds_load %buffer[%idx0, %idx1]
    : memref<128x128xf32, 3>, index, index -> vector<4xf32>
  return
}

func.func @rock_in_bounds_store(%buffer: memref<128x128xf32, 3>,
                                %data: vector<4xf32>,
                                %idx0: index,
                                %idx1: index) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  rock.in_bounds_store %data -> %buffer[%idx0, %idx1]
  : vector<4xf32> -> memref<128x128xf32, 3>, index, index
  return
}

#transform_map0 = #rock.transform_map<affine_map<(d0, d1) -> (0, d1, d0)> by [<Merge{1, 32} ["nr"] at [0] -> ["1", "z"] at [0, 2]>, <PassThrough ["r"] at [1] -> ["0"] at [1]>] bounds = [32, 20] -> [1, 20, 32]>
#transform_map1 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0 * 4 + d1, d2)> by [<Unmerge{8, 4} ["bid", "nr_per_bid"] at [0, 1] -> ["nr"] at [0]>, <PassThrough ["r"] at [2] -> ["r"] at [1]>] bounds = [8, 4, 20] -> [32, 20]>
#transform_map2 = #rock.transform_map<affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["bid"] at [0] -> ["bid"] at [0]>, <PassThrough ["tid"] at [1] -> ["r"] at [2]>, <PassThrough ["iter"] at [2] -> ["nr_per_bid"] at [1]>] bounds = [8, 20, 4] -> [8, 4, 20]>
#transform_map3 = #rock.transform_map<affine_map<(d0, d1) -> (d1, d0)> by [<PassThrough ["tid"] at [0] -> ["r"] at [1]>, <PassThrough ["iter"] at [1] -> ["nr_per_bid"] at [0]>] bounds = [20, 4] -> [4, 20]>
func.func @rock_threadwise_ops(%input : memref<1x20x32xf32>, 
                               %output : memref<1x20x32xf32>) attributes{arch = "##TOKEN_ARCH##", block_size = 20 : i32, grid_size = 8 : i32, kernel} {
  // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
  %input_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
  %output_reg = rock.alloc() : memref<4xf32, #gpu.address_space<private>>
  // expected-remark @below {{found an instance of 'allocate' on op result 0, on resource '<Default>'}}
  %ws_lds = rock.alloc() : memref<4x20xf32, #gpu.address_space<workgroup>>
  // expected-remark @below {{operation has no memory effects}}
  %bid = rock.workgroup_id : index
  // expected-remark @below {{operation has no memory effects}}
  %tid = rock.workitem_id : index
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  rock.threadwise_read_into {forceUnroll, useIndexDiffs}
    [#transform_map2, #transform_map1, #transform_map0](%input)[%bid, %tid] -> %input_reg : memref<1x20x32xf32> ->  memref<4xf32, #gpu.address_space<private>>
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  rock.threadwise_write_all {forceUnroll, useIndexDiffs} %input_reg -> [#transform_map3](%ws_lds)[%tid] by set : memref<4xf32, #gpu.address_space<private>> -> memref<4x20xf32, #gpu.address_space<workgroup>>
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  rock.threadwise_copy %input_reg -> %output_reg : memref<4xf32, #gpu.address_space<private>> -> memref<4xf32, #gpu.address_space<private>>
  return
}

func.func @rock_blockwise_gemm(%A : memref<8x128x1xf32, 3>,
                               %B : memref<8x128x1xf32, 3>,
                               %C : memref<8x8xf32, 5>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.blockwise_gemm %C += %A * %B {
    inMPerThread = 2 : i32,
    inNPerThread = 2 : i32,
    params = #rock.general_gemm_params<
    blockSize = 256,
    kPerBlock = 8,
    mPerBlock = 128,
    nPerBlock = 128,
    kpack = 1,
    kPerThread = 1,
    mPerThread = 4,
    nPerThread = 4,
    splitKFactor = 1, 
    scheduleVersion = 1, 
    outputSwizzle = 2>
  } :  memref<8x8xf32, 5> += memref<8x128x1xf32, 3> * memref<8x128x1xf32, 3>
  return
}

func.func @rock_threadwise_gemm(%lhs : memref<4x8x1xf32, 5>,
                                %rhs : memref<4x8x1xf32, 5>,
                                %output : memref<8x8xf32, 5>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.threadwise_gemm %output += %lhs * %rhs
  : memref<8x8xf32, 5> += memref<4x8x1xf32, 5> * memref<4x8x1xf32, 5>
  return
}

func.func @rock_accel_gemm(%matrixA : memref<1x16xf32, 5>,
                           %matrixB : memref<1x16xf32, 5>,
                           %matrixC : memref<1x1xvector<32xf32>, 5>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{operation has no memory effects}}
  %c0 = arith.constant 0 : index
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 2, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  rock.threadwise_accel_gemm %matrixC += %matrixA * %matrixB at [%c0, %c0, %c0] {
    params = #rock.xdlops_gemm_derived_params<
      mPerBlock = 256,
      nPerBlock = 256,
      kpackPerBlock = 16,
      mPerWave = 128,
      nPerWave = 64,
      mnPerXdl = 32,
      kpack = 1,
      splitKFactor = 1, 
      scheduleVersion = 1, 
      outputSwizzle = 2,
      forceUnroll = true>
  } : memref<1x1xvector<32xf32>, 5> += memref<1x16xf32, 5> * memref<1x16xf32, 5>
  return
}

func.func @rock_gridwise_attn(%arg0: memref<1x384x64xf32>,
                              %arg1: memref<1x64x384xf32>,
                              %arg2: memref<1x384x64xf32>,
                              %arg3: memref<1x384x64xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{operation has no memory effects}}
  %0 = rock.transform %arg0 by <affine_map<(d0, d1, d2) -> (d0, d2, d1)> by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [1, 64, 384] -> [1, 384, 64]> : memref<1x384x64xf32> to memref<1x64x384xf32>
  // expected-remark @below {{found an instance of 'read' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  rock.gridwise_attention_accel(%0, %arg1, %arg2, %arg3) preSoftmaxOps = {} {
    blockSize = 64 : i32,
    gridSize = 24 : i32,
    params0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    params1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, forceUnroll = true>,
    firstGemmIndices = array<i64: 0>,
    operand_segment_sizes = array<i32: 1, 1, 1, 0, 0, 1, 0>,
    splitKV = 1 : i32
  } : memref<1x64x384xf32>, memref<1x64x384xf32>, memref<1x384x64xf32>, memref<1x384x64xf32>
  return
}

func.func @rock_reduce(%arg0: memref<2x12x12xf32>,
                       %arg1: memref<2x12x1xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 1, on resource '<Default>'}}
  rock.reduce sum %arg0 into %arg1 {axis = 2 : index, blockSize = 64 : i32, gridSize = 2 : i32} : memref<2x12x12xf32> into memref<2x12x1xf32>
  func.return
}

#xldops_attn_params_g0 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 1, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
#xldops_attn_params_g1 = #rock.xdlops_gemm_derived_params<kpackPerBlock = 8, mPerBlock = 32, nPerBlock = 32, kpack = 4, mPerWave = 32, nPerWave = 32, mnPerXdl = 32, forceUnroll = true, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2>
func.func @rock_gemmelementwisegemm_simple(%arg0: memref<1x64x1024xf32>,
                                           %arg1: memref<1x64x1024xf32>,
                                           %arg2: memref<1x1024x64xf32>,
                                           %arg3: memref<1x1024x64xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  rock.gemm_elementwise_gemm{
     ab = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = ab * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>
  }
  return
}

func.func @rock_conv_gemm(%arg0: memref<1x128x256x1x1xf16>,
                          %arg1: memref<2x1x256x32x32xf16>,
                          %arg2: memref<1x128x64xf16>,
                          %arg3: memref<1x2048x64xf16>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  rock.conv_elementwise_gemm{
   ab = conv(%arg0, %arg1) : memref<1x128x256x1x1xf16>, memref<2x1x256x32x32xf16>
   %arg3 = ab * %arg2 : memref<1x128x64xf16> -> memref<1x2048x64xf16>
  } {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], firstGemmIndices = array<i64: 0>, input_layout = ["ni", "gi", "ci", "0i", "1i"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]}
  return
}

func.func @rock_attention(%arg0: memref<1x64x1024xf32>,
                          %arg1: memref<1x64x1024xf32>,
                          %arg2: memref<1x1024x64xf32>,
                          %arg3: memref<1x1024x64xf32>) attributes {arch = "##TOKEN_ARCH##"} {
  // expected-remark @below {{found an instance of 'read' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'write' on op operand 3, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 0, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 1, on resource '<Default>'}}
  // expected-remark @below {{found an instance of 'read' on op operand 2, on resource '<Default>'}}
  rock.attention{
     qk = tr %arg0 * %arg1 : memref<1x64x1024xf32>, memref<1x64x1024xf32>
     %arg3 = softmax(qk) * %arg2 : memref<1x1024x64xf32> -> memref<1x1024x64xf32>
  } { 
    params0 = #xldops_attn_params_g0,
    params1 = #xldops_attn_params_g1,
    firstGemmIndices = array<i64: 0>,
    splitKV = 1 : i32
  }
  return
}

