// RUN: mlir-translate -mlir-to-miopen-cflags %s | FileCheck %s

func @basic_parsing(%filter : memref<?x?x?xf32>, %input : memref<?x?x?xf32>, %output : memref<?x?x?xf32>) {
  miopen.gridwise_gemm(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    kernel_algorithm = "backward_weight_v4r4",
    filter_dimension = [1, 128, 128, 3, 3],
    filter_layout = ["g", "k", "c", "y", "x"],
    input_dimension = [128, 1, 128, 14, 14],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_dimension = [128, 1, 128, 14, 14],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    padding = [[1, 1], [1, 1]],
    strides = [1, 1]
  } : memref<?x?x?xf32>,
      memref<?x?x?xf32>,
      memref<?x?x?xf32>
  return
}
// CHECK-LABEL: basic_parsing
// CHECK: -DCK_PARAM_PROBLEM_C=128
// CHECK: -DCK_PARAM_PROBLEM_CONV_DILATION_H=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_DILATION_W=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_STRIDE_H=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_STRIDE_W=1
// CHECK: -DCK_PARAM_PROBLEM_HI=14
// CHECK: -DCK_PARAM_PROBLEM_HO=14
// CHECK: -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=1
// CHECK: -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=1
// CHECK: -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=1
// CHECK: -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=1
// CHECK: -DCK_PARAM_PROBLEM_K=128
// CHECK: -DCK_PARAM_PROBLEM_N=128
// CHECK: -DCK_PARAM_PROBLEM_WI=14
// CHECK: -DCK_PARAM_PROBLEM_WO=14
// CHECK: -DCK_PARAM_PROBLEM_X=3
// CHECK: -DCK_PARAM_PROBLEM_Y=3

func @all_params(%filter : memref<?x?x?xf32>, %input : memref<?x?x?xf32>, %output : memref<?x?x?xf32>) {
  miopen.gridwise_gemm(%filter, %input, %output) {
    arch = "gfx906",
    num_cu = 64,
    kernel_algorithm = "backward_weight_v4r4",
    filter_dimension = [1, 128, 128, 3, 3],
    filter_layout = ["g", "k", "c", "y", "x"],
    input_dimension = [128, 1, 128, 14, 14],
    input_layout = ["ni", "gi", "ci", "hi", "wi"],
    output_dimension = [128, 1, 128, 14, 14],
    output_layout = ["no", "go", "ko", "ho", "wo"],
    dilations = [1, 1],
    padding = [[1, 1], [1, 1]],
    strides = [1, 1]
  } : memref<?x?x?xf32>,
      memref<?x?x?xf32>,
      memref<?x?x?xf32>
  return
}
// CHECK-LABEL: all_params
// CHECK: -DCK_PARAM_DEPENDENT_GRID_SIZE=9
// CHECK: -DCK_PARAM_PROBLEM_C=128
// CHECK: -DCK_PARAM_PROBLEM_CONV_DILATION_H=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_DILATION_W=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_STRIDE_H=1
// CHECK: -DCK_PARAM_PROBLEM_CONV_STRIDE_W=1
// CHECK: -DCK_PARAM_PROBLEM_HI=14
// CHECK: -DCK_PARAM_PROBLEM_HO=14
// CHECK: -DCK_PARAM_PROBLEM_IN_LEFT_PAD_H=1
// CHECK: -DCK_PARAM_PROBLEM_IN_LEFT_PAD_W=1
// CHECK: -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_H=1
// CHECK: -DCK_PARAM_PROBLEM_IN_RIGHT_PAD_W=1
// CHECK: -DCK_PARAM_PROBLEM_K=128
// CHECK: -DCK_PARAM_PROBLEM_N=128
// CHECK: -DCK_PARAM_PROBLEM_WI=14
// CHECK: -DCK_PARAM_PROBLEM_WO=14
// CHECK: -DCK_PARAM_PROBLEM_X=3
// CHECK: -DCK_PARAM_PROBLEM_Y=3
// CHECK: -DCK_PARAM_TUNABLE_BLOCK_SIZE=256
// CHECK: -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=16
// CHECK: -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M=16
// CHECK: -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M=1
// CHECK: -DCK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM=1
// CHECK: -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K=16
// CHECK: -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N=16
// CHECK: -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N=1
// CHECK: -DCK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM=1
// CHECK: -DCK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_K_PER_BLOCK=16
// CHECK: -DCK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_M_PER_BLOCK=128
// CHECK: -DCK_PARAM_TUNABLE_GEMM_M_PER_THREAD=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER=4
// CHECK: -DCK_PARAM_TUNABLE_GEMM_N_PER_BLOCK=128
// CHECK: -DCK_PARAM_TUNABLE_GEMM_N_PER_THREAD=4
// CHECK: -DCK_THREADWISE_GEMM_USE_AMD_INLINE_ASM=1
// CHECK: -DMIOPEN_USE_BFP16=0
// CHECK: -DMIOPEN_USE_FP16=0
// CHECK: -DMIOPEN_USE_FP32=1
// CHECK: -D__HIP_PLATFORM_HCC__=1
// CHECK: -std=c++14
