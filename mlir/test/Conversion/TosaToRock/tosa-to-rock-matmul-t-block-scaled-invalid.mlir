// RUN: rocmlir-opt --tosa-to-rock -verify-diagnostics --split-input-file %s

// Test: K dimension not a multiple of block_size after transpose_b changes which dim is K.
// B data shape is [1, 100, 256] which passes TOSA verifier (K=256 is divisible by 32).
// But transpose_b toggles the default B transpose, making K=100 which is not divisible by 32.
module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
func.func @k_not_multiple_of_block_size_via_transpose_b(
    %a_data: tensor<1x128x256xf4E2M1FN>,
    %a_scale: tensor<1x128x8xf8E8M0FNU>,
    %b_data: tensor<1x100x256xf4E2M1FN>,
    %b_scale: tensor<1x100x8xf8E8M0FNU>) -> tensor<1x128x100xf32> attributes {kernel} {
  // expected-error @+2 {{'tosa.matmul_t_block_scaled' op K dimension (100) must be a multiple of block_size (32)}}
  // expected-error @+1 {{failed to legalize operation 'tosa.matmul_t_block_scaled' that was explicitly marked illegal}}
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale
      {block_size = #tosa.block_size<BLOCK_SIZE_32>, transpose_b = true}
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>,
         tensor<1x100x256xf4E2M1FN>, tensor<1x100x8xf8E8M0FNU>)
      -> tensor<1x128x100xf32>
  return %result : tensor<1x128x100xf32>
}
}

// -----

// Test: A scale K dimension mismatch after transpose_a_scale.
// A scale shape [1, 128, 8] passes TOSA verifier. But transpose_a_scale makes
// the converter read dim 1 (128) as the K-scale dimension instead of dim 2 (8),
// causing a mismatch with expected K/blockSize = 256/32 = 8.
module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
func.func @a_scale_k_mismatch_via_transpose_a_scale(
    %a_data: tensor<1x128x256xf4E2M1FN>,
    %a_scale: tensor<1x128x8xf8E8M0FNU>,
    %b_data: tensor<1x512x256xf4E2M1FN>,
    %b_scale: tensor<1x512x8xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {kernel} {
  // expected-error @+2 {{'tosa.matmul_t_block_scaled' op A scale K dimension (128) does not match K / block_size (8)}}
  // expected-error @+1 {{failed to legalize operation 'tosa.matmul_t_block_scaled' that was explicitly marked illegal}}
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale
      {block_size = #tosa.block_size<BLOCK_SIZE_32>, transpose_a_scale = true}
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>,
         tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>)
      -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}
}

// -----

// Test: B scale K dimension mismatch after transpose_b_scale.
// B scale shape [1, 512, 8] passes TOSA verifier. But transpose_b_scale makes
// the converter read dim 1 (512) as the K-scale dimension instead of dim 2 (8),
// causing a mismatch with expected K/blockSize = 256/32 = 8.
module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx950"} {
func.func @b_scale_k_mismatch_via_transpose_b_scale(
    %a_data: tensor<1x128x256xf4E2M1FN>,
    %a_scale: tensor<1x128x8xf8E8M0FNU>,
    %b_data: tensor<1x512x256xf4E2M1FN>,
    %b_scale: tensor<1x512x8xf8E8M0FNU>) -> tensor<1x128x512xf32> attributes {kernel} {
  // expected-error @+2 {{'tosa.matmul_t_block_scaled' op B scale K dimension (512) does not match K / block_size (8)}}
  // expected-error @+1 {{failed to legalize operation 'tosa.matmul_t_block_scaled' that was explicitly marked illegal}}
  %result = tosa.matmul_t_block_scaled %a_data, %a_scale, %b_data, %b_scale
      {block_size = #tosa.block_size<BLOCK_SIZE_32>, transpose_b_scale = true}
      : (tensor<1x128x256xf4E2M1FN>, tensor<1x128x8xf8E8M0FNU>,
         tensor<1x512x256xf4E2M1FN>, tensor<1x512x8xf8E8M0FNU>)
      -> tensor<1x128x512xf32>
  return %result : tensor<1x128x512xf32>
}
}
