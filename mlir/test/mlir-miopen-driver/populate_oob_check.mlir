// RUN: miopen-gen  -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KCYX
// CHECK_KCYX: miopen.global_load %arg1{{.*}}leftOobDims = [3 : i32, 4 : i32]{{.*}}rightOobDims = [3 : i32, 4 : i32]

// RUN: miopen-gen  -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KYXC
// CHECK_KYXC: miopen.global_load %arg1{{.*}}leftOobDims = [1 : i32, 2 : i32]{{.*}}rightOobDims = [1 : i32, 2 : i32]

// RUN: miopen-gen  -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h_l=1 --padding_w_l=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KCYX_LEFT
// CHECK_KCYX_LEFT: miopen.global_load %arg1{{.*}}leftOobDims = [3 : i32, 4 : i32]{{.*}}rightOobDims = []

// RUN: miopen-gen  -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h_l=1 --padding_w_l=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KYXC_LEFT
// CHECK_KYXC_LEFT: miopen.global_load %arg1{{.*}}leftOobDims = [1 : i32, 2 : i32]{{.*}}rightOobDims = []

// RUN: miopen-gen -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1  --operation=conv2d_bwd_data | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KCYX_BWD
// CHECK_KCYX_BWD: miopen.global_load %arg2{{.*}}leftOobDims = [3 : i32, 4 : i32]{{.*}}rightOobDims = [3 : i32, 4 : i32]

// RUN: miopen-gen -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1  --operation=conv2d_bwd_data | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gemm-to-gridwise -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=CHECK_KYXC_BWD
// CHECK_KYXC_BWD: miopen.global_load %arg2{{.*}}leftOobDims = [1 : i32, 2 : i32]{{.*}}rightOobDims = [1 : i32, 2 : i32]

