// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
// CHECK-LABEL: @test_matmul_t_block_scaled_basic
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}
// Test basic tosa.matmul_t_block_scaled lowering to rock.gemm with scales
// A: [1, 128, 256] f4, A_scale: [1, 128, 8] f8 (K/32 = 256/32 = 8)
// B: [1, 512, 256] f4 (transposed, N=512, K=256), B_scale: [1, 512, 8] f8
// Output: [1, 128, 512] f32
func.func @test_matmul_t_block_scaled_basic(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                             %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                             %b_data: tensor<1x512x256xf4E2M1FN>, 
                                             %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                             -> tensor<1x128x512xf32> attributes {kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}

// CHECK-LABEL: @test_matmul_t_block_scaled_batched
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_batched(%a_data: tensor<4x128x256xf4E2M1FN>, 
                                               %a_scale: tensor<4x128x8xf8E8M0FNU>,
                                               %b_data: tensor<4x512x256xf4E2M1FN>, 
                                               %b_scale: tensor<4x512x8xf8E8M0FNU>) 
                                               -> tensor<4x128x512xf32> attributes {kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<4x128x256xf4E2M1FN>, tensor<4x128x8xf8E8M0FNU>, tensor<4x512x256xf4E2M1FN>, tensor<4x512x8xf8E8M0FNU>) 
      -> tensor<4x128x512xf32>
  return %result : tensor<4x128x512xf32>
}

// CHECK-LABEL: @test_matmul_t_block_scaled_large_k
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_large_k(%a_data: tensor<1x64x512xf4E2M1FN>, 
                                               %a_scale: tensor<1x64x16xf8E8M0FNU>,
                                               %b_data: tensor<1x256x512xf4E2M1FN>, 
                                               %b_scale: tensor<1x256x16xf8E8M0FNU>) 
                                               -> tensor<1x64x256xf32> attributes {kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x64x512xf4E2M1FN>, tensor<1x64x16xf8E8M0FNU>, tensor<1x256x512xf4E2M1FN>, tensor<1x256x16xf8E8M0FNU>) 
      -> tensor<1x64x256xf32>
  return %result : tensor<1x64x256xf32>
}

// Test transpose on A data input
// A is transposed 
// Using symmetric shape M=K=256 so scale shape remains valid after transpose fusion
// This should result in rock.gemm with `tr` on A data
// A scale is NOT transposed (no transpose on scale input)
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = tr %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_a(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                   %a_scale: tensor<1x256x8xf8E8M0FNU>,
                                                   %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                   %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                   -> tensor<1x256x512xf32> attributes {kernel} {
  // Transpose A from [1, 256, 256] to [1, 256, 256] (shape unchanged due to M=K)
  // The transpose flag will still be set to exercise the code path
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  return %result : tensor<1x256x512xf32>
}

// Test transpose on B data input
// B is transposed 
// Using symmetric shape N=K=256 so scale shape remains valid after transpose fusion
// Since matmul_t_block_scaled already expects B transposed, this toggles the transpose
// Note: B data transpose is independent of B scale transpose. B scale remains in its
// default transposed state (matching matmul_t_block_scaled's B layout expectation)
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_b(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                   %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                   %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                   %b_scale: tensor<1x256x8xf8E8M0FNU>) 
                                                   -> tensor<1x128x256xf32> attributes {kernel} {
  // Transpose B from [1, 256, 256] to [1, 256, 256] (shape unchanged due to N=K)
  // The transpose flag will still be toggled to exercise the code path
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  return %result : tensor<1x128x256xf32>
}

// Test transpose on A scale only (without A data transpose)
// A scale: [batch, M, K/blockSize] -> transposed to [batch, K/blockSize, M]
// Using symmetric shape M = K/blockSize = 8 so transpose doesn't change shape
// A scale transpose is independent of A data transpose
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_scale
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_a_scale(%a_data: tensor<1x8x256xf4E2M1FN>, 
                                                         %a_scale: tensor<1x8x8xf8E8M0FNU>,
                                                         %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                         %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                         -> tensor<1x8x512xf32> attributes {kernel} {
  // Transpose A scale from [1, 8, 8] to [1, 8, 8] (shape unchanged due to M = K/32 = 8)
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale_tr, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x8x512xf32>
  return %result : tensor<1x8x512xf32>
}

// Test transpose on B scale only (toggles from default transposed state)
// B scale: [batch, N, K/blockSize] (default transposed) -> toggled to [batch, K/blockSize, N]
// Using symmetric shape N = K/blockSize = 8 so transpose doesn't change shape
// B scale transpose is independent of B data transpose
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_scale
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_b_scale(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                         %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                         %b_data: tensor<1x8x256xf4E2M1FN>, 
                                                         %b_scale: tensor<1x8x8xf8E8M0FNU>) 
                                                         -> tensor<1x128x8xf32> attributes {kernel} {
  // Transpose B scale from [1, 8, 8] to [1, 8, 8] (shape unchanged due to N = K/32 = 8)
  // This toggles B scale from its default transposed state to non-transposed
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>) 
      -> tensor<1x128x8xf32>
  return %result : tensor<1x128x8xf32>
}

// Test transpose on both A scale and B scale
// Both scales have symmetric shapes for valid transpose fusion
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_both_scales
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_both_scales(%a_data: tensor<1x8x256xf4E2M1FN>, 
                                                             %a_scale: tensor<1x8x8xf8E8M0FNU>,
                                                             %b_data: tensor<1x8x256xf4E2M1FN>, 
                                                             %b_scale: tensor<1x8x8xf8E8M0FNU>) 
                                                             -> tensor<1x8x8xf32> attributes {kernel} {
  // Transpose both A scale and B scale (shapes unchanged due to symmetric dimensions)
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf8E8M0FNU>) -> tensor<1x8x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale_tr, %b_data, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>, tensor<1x8x256xf4E2M1FN>, tensor<1x8x8xf8E8M0FNU>) 
      -> tensor<1x8x8xf32>
  return %result : tensor<1x8x8xf32>
}

// Test transpose on A data AND A scale together
// A data: [batch, M, K] with symmetric M=K=256
// A scale: [batch, M, K/32] with M=256, K/32=8 -> after transpose [batch, K/32, M] = [batch, 8, 256]
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_data_and_scale
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = tr %{{.*}} scaled by tr %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_a_data_and_scale(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                                  %a_scale: tensor<1x8x256xf8E8M0FNU>,
                                                                  %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                                  %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                                  -> tensor<1x256x512xf32> attributes {kernel} {
  // Transpose A data (symmetric M=K=256) 
  // Transpose A scale from [1, 8, 256] to [1, 256, 8] which gives valid [batch, M=256, K/32=8]
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %a_scale_tr = "tosa.transpose"(%a_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x256xf8E8M0FNU>) -> tensor<1x256x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale_tr, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  return %result : tensor<1x256x512xf32>
}

// Test transpose on B data AND B scale together
// B data: [batch, N, K] with symmetric N=K=256
// B scale pre-transpose [1, 8, 256] 
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_data_and_scale
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_b_data_and_scale(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                                  %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                                  %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                                  %b_scale: tensor<1x8x256xf8E8M0FNU>) 
                                                                  -> tensor<1x128x256xf32> attributes {kernel} {
  // Transpose B data (symmetric N=K=256)
  // Transpose B scale from [1, 8, 256] to [1, 256, 8] which gives valid [batch, N=256, K/32=8]
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %b_scale_tr = "tosa.transpose"(%b_scale) {perms = array<i32: 0, 2, 1>} : (tensor<1x8x256xf8E8M0FNU>) -> tensor<1x256x8xf8E8M0FNU>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale_tr {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  return %result : tensor<1x128x256xf32>
}

// Test output transpose (transpose_c) on matmul_t_block_scaled
// Output: [batch, M, N] = [1, 128, 512] transposed to [1, 512, 128]
// The transpose on the output should be fused into the gemm as cTransposed
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_c
// CHECK: rock.transform
// CHECK: rock.gemm tr %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_c(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                    %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                    %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                    %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                    -> tensor<1x512x128xf32> attributes {kernel} {
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x128x512xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x128x512xf32>) -> tensor<1x512x128xf32>
  return %result_tr : tensor<1x512x128xf32>
}

// Test output transpose combined with input transpose on A
// A data is transposed, output is transposed
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_a_and_c
// CHECK: rock.transform
// CHECK: rock.gemm tr %{{.*}} = tr %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_a_and_c(%a_data: tensor<1x256x256xf4E2M1FN>, 
                                                          %a_scale: tensor<1x256x8xf8E8M0FNU>,
                                                          %b_data: tensor<1x512x256xf4E2M1FN>, 
                                                          %b_scale: tensor<1x512x8xf8E8M0FNU>) 
                                                          -> tensor<1x512x256xf32> attributes {kernel} {
  %a_tr = "tosa.transpose"(%a_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_tr, %a_scale, %b_data, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>, tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>) 
      -> tensor<1x256x512xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x512xf32>) -> tensor<1x512x256xf32>
  return %result_tr : tensor<1x512x256xf32>
}

// Test output transpose combined with B data transpose (toggles B's default transpose)
// CHECK-LABEL: @test_matmul_t_block_scaled_transpose_b_and_c
// CHECK: rock.transform
// CHECK: rock.gemm tr %{{.*}} = %{{.*}} scaled by %{{.*}} * %{{.*}} scaled by tr %{{.*}}

func.func @test_matmul_t_block_scaled_transpose_b_and_c(%a_data: tensor<1x128x256xf4E2M1FN>, 
                                                          %a_scale: tensor<1x128x8xf8E8M0FNU>,
                                                          %b_data: tensor<1x256x256xf4E2M1FN>, 
                                                          %b_scale: tensor<1x256x8xf8E8M0FNU>) 
                                                          -> tensor<1x256x128xf32> attributes {kernel} {
  %b_tr = "tosa.transpose"(%b_data) {perms = array<i32: 0, 2, 1>} : (tensor<1x256x256xf4E2M1FN>) -> tensor<1x256x256xf4E2M1FN>
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_tr, %b_scale {block_size = #tosa.block_size<BLOCK_SIZE_32>} 
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>, tensor<1x256x256xf4E2M1FN>, tensor<1x256x8xf8E8M0FNU>) 
      -> tensor<1x128x256xf32>
  %result_tr = "tosa.transpose"(%result) {perms = array<i32: 0, 2, 1>} : (tensor<1x128x256xf32>) -> tensor<1x256x128xf32>
  return %result_tr : tensor<1x256x128xf32>
}
}
