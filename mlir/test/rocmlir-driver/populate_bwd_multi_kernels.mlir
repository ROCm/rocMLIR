// RUN: rocmlir-gen --arch gfx906  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1 --operation=conv_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2

// RUN: rocmlir-gen --arch gfx906 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2_GKYXC

// This config requires a zero initialization utility kernel.
// Check at the top-level there is a utility kernel.
// RUN: rocmlir-gen --arch gfx906  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data | FileCheck %s --check-prefix=STRIDE2_1x1_TOP_LEVEL
// Check after -rock-lowering, only gemm with corresponding kernel IDs exists.
// RUN: rocmlir-gen --arch gfx906  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data | rocmlir-opt -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STRIDE2_1x1_LOWERING

// STRIDE2: {{rock.gemm.*kernelId = 0 : index.*}}
// STRIDE2: {{rock.gemm.*kernelId = 1 : index.*}}
// STRIDE2: {{rock.gemm.*kernelId = 2 : index.*}}
// STRIDE2: {{rock.gemm.*kernelId = 3 : index.*}}

// STRIDE2_GKYXC: {{rock.gemm.*kernelId = 0 : index.*}}
// STRIDE2_GKYXC: {{rock.gemm.*kernelId = 1 : index.*}}
// STRIDE2_GKYXC: {{rock.gemm.*kernelId = 2 : index.*}}
// STRIDE2_GKYXC: {{rock.gemm.*kernelId = 3 : index.*}}

// STRIDE2_1x1_TOP_LEVEL: rock.init_kernel %arg1 features = {{.*}} : memref<200704xf32>
// STRIDE2_1x1_TOP_LEVEL: [[exp0:%.+]] = rock.transform %arg0 by {{.*}} : memref<1024xf32> to memref<1x32x32x1x1xf32>
// STRIDE2_1x1_TOP_LEVEL: [[exp1:%.+]] = rock.transform %arg1 by {{.*}} : memref<200704xf32> to memref<32x1x32x14x14xf32>
// STRIDE2_1x1_TOP_LEVEL: [[exp2:%.+]] = rock.transform %arg2 by {{.*}} : memref<65536xf32> to memref<32x1x32x8x8xf32>
// STRIDE2_1x1_TOP_LEVEL: rock.conv_bwd_data([[exp0]], [[exp1]], [[exp2]]) features = {{.*}} {arch = {{.*}}, dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 0 : index, numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index]} : memref<1x32x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x32x8x8xf32>

// STRIDE2_1x1_LOWERING-NOT: rock.init_kernel
// STRIDE2_1x1_LOWERING: {{rock.gemm.*kernelId = 0 : index.*}}
