// RUN: mlir-miopen-driver -pv   -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -miopen-lowering | FileCheck %s --check-prefix=CHECK_KCYX
// CHECK_KCYX: bound_check = [0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32]

// RUN: mlir-miopen-driver -pv   -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3| FileCheck %s --check-prefix=CHECK_KCYX_STEP3
// CHECK_KCYX_STEP3: {{.*bound_check = \[0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32\].*metadata.*operand = 0 : i32.*}}

// RUN: mlir-miopen-driver -pv   -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -miopen-lowering | FileCheck %s --check-prefix=CHECK_KYXC
// CHECK_KYXC: bound_check = [0 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32]

// RUN: mlir-miopen-driver -pv   -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3| FileCheck %s --check-prefix=CHECK_KYXC_STEP3
// CHECK_KYXC_STEP3: {{.*bound_check = \[0 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32\].*metadata.*operand = 0 : i32.*}}

// RUN: mlir-miopen-driver -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv  --operation=conv2d_bwd_data -miopen-lowering | FileCheck %s --check-prefix=CHECK_KCYX_BWD
// CHECK_KCYX_BWD: bound_check = [0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32]

// RUN: mlir-miopen-driver -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv  --operation=conv2d_bwd_data -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3| FileCheck %s --check-prefix=CHECK_KCYX_BWD_STEP3
// CHECK_KCYX_BWD_STEP3: {{.*bound_check = \[0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32\].*metadata.*operand = 0 : i32.*}}

// RUN: mlir-miopen-driver -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv  --operation=conv2d_bwd_data -miopen-lowering | FileCheck %s --check-prefix=CHECK_KYXC_BWD
// CHECK_KYXC_BWD: bound_check = [0 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32]

// RUN: mlir-miopen-driver -p=false  -fil_layout=gkcyx -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv  --operation=conv2d_bwd_data -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 -miopen-lowering-step3| FileCheck %s --check-prefix=CHECK_KYXC_BWD_STEP3
// CHECK_KYXC_BWD_STEP3: {{.*bound_check = \[0 : i32, 1 : i32, 1 : i32, 0 : i32, 0 : i32\].*metadata.*operand = 0 : i32.*}}