// RUN: rocmlir-lib-test --args " --operation conv2d --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_nchw_kcyx_nkhw --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN
// RUN: rocmlir-lib-test --args " --operation conv2d --arch amdgcn-amd-amdhsa:gfx906 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_fwd --groupsize 1" --option tuningparams | FileCheck %s --check-prefix=TUNING
// RUN: rocmlir-gen --conv-config "--operation conv2d --arch amdgcn-amd-amdhsa:gfx906 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_fwd --groupsize 1" | FileCheck %s --check-prefix=DRIVER

// BIN: ELF
// TUNING: globalSize{{.*}}localSize{{.*}}

// DRIVER: rock.conv2d(%arg0, %arg1, %arg2) features = dot {arch = "gfx906", dilations = [1 : i32, 1 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], output_layout = ["no", "go", "ko", "ho", "wo"], padding = [0 : i32, 0 : i32, 0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>


