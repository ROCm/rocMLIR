// RUN: mlir-miopen-driver  -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 -pv  --operation=conv2d_bwd_data  -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s --check-prefix=STRIDE2

// RUN: mlir-miopen-driver -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 -pv  --operation=conv2d_bwd_data  -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s --check-prefix=STRIDE2_GKYXC


// RUN: mlir-miopen-driver  -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 -pv  --operation=conv2d_bwd_data  -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s --check-prefix=STRIDE2_1x1


// STRIDE2: {{miopen.gridwise_gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{miopen.gridwise_gemm.*gemm_id = 1 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{miopen.gridwise_gemm.*gemm_id = 2 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
// STRIDE2: {{miopen.gridwise_gemm.*gemm_id = 3 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}


// STRIDE2_GKYXC: {{miopen.gridwise_gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{miopen.gridwise_gemm.*gemm_id = 1 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{miopen.gridwise_gemm.*gemm_id = 2 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}
// STRIDE2_GKYXC: {{miopen.gridwise_gemm.*gemm_id = 3 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 1 : i32.*}}


// STRIDE2_1x1: {{miopen.gridwise_gemm.*gemm_id = 0 : i32.*matrix_a_source_data_per_read = 1 : i32, matrix_a_source_vector_read_dim = 2 : i32.*matrix_b_source_data_per_read = 1 : i32, matrix_b_source_vector_read_dim = 2 : i32.*}}
