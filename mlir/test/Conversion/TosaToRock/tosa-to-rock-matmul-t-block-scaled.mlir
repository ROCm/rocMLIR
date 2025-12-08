// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

// Test basic tosa.matmul_t_block_scaled lowering to rock.gemm with scales
// A: [1, 128, 256] f4, A_scale: [1, 128, 8] f8 (K/32 = 256/32 = 8)
// B: [1, 512, 256] f4 (transposed, N=512, K=256), B_scale: [1, 512, 8] f8
// Output: [1, 128, 512] f32

module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx942"} {

// CHECK-LABEL: @test_matmul_t_block_scaled_basic
// CHECK: rock.transform
// CHECK: rock.gemm %{{.*}} = %{{.*}} scaled by %{{.*}} * tr %{{.*}} scaled by tr %{{.*}}

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

}
