// RUN: rocmlir-gen --arch %arch  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 --operation=conv2d_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2

// RUN: rocmlir-gen --arch %arch -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv2d_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2_GKYXC

// This config requires a zero initialization utility kernel.
// Check at the top-level there is a utility kernel with gemm_id=-1.
// RUN: rocmlir-gen --arch %arch  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv2d_bwd_data | FileCheck %s --check-prefix=STRIDE2_1x1_TOP_LEVEL
// Check after -rock-lowering, only gridwise_gemm with gemm_id=0 exists.
// RUN: rocmlir-gen --arch %arch  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv2d_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2_1x1_LOWERING

// STRIDE2: {{rock.gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{rock.gemm.*gemm_id = 1 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{rock.gemm.*gemm_id = 2 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{rock.gemm.*gemm_id = 3 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}

// STRIDE2_GKYXC: {{rock.gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{rock.gemm.*gemm_id = 1 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{rock.gemm.*gemm_id = 2 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{rock.gemm.*gemm_id = 3 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}

// STRIDE2_1x1_TOP_LEVEL: rock.conv2d_bwd_data(%arg0, %arg1, %arg2) features = {{.*}} {arch = {{.*}}, dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = -1 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 64 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : memref<1x32x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x32x8x8xf32>
// STRIDE2_1x1_TOP_LEVEL: rock.conv2d_bwd_data(%arg0, %arg1, %arg2) features = {{.*}} {arch = {{.*}}, dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 64 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : memref<1x32x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x32x8x8xf32>

// STRIDE2_1x1_LOWERING-NOT: {{rock.gemm.*gemm_id = -1 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2_1x1_LOWERING: {{rock.gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
