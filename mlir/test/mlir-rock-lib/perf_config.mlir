// RUN: rocmlir-lib-test --args " --operation conv2d --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_nchw_kcyx_nkhw --groupsize 1 --perf_config v2:64,32,32,4,2,2,1" --option bin | FileCheck %s --check-prefix=BIN
// RUN: rocmlir-gen --conv-config " --operation conv2d --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_nchw_kcyx_nkhw --groupsize 1 --perf_config v2:64,32,32,4,2,2,1" | rocmlir-driver -rock-affix-params | FileCheck %s --check-prefix=Tuning
// RUN: rocmlir-gen --conv-config " --operation conv2d --arch amdgcn-amd-amdhsa:gfx908 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv2d_nchw_kcyx_nkhw --groupsize 1 --x2 1 --perf_config v2:128,128,8,64,64,4,1,1,1" | rocmlir-driver -rock-affix-params | FileCheck %s --check-prefix=Tuning-xdlops

// BIN: ELF
// Tuning: blockSize = 64
// Tuning: mPerBlock = 32
// Tuning: nPerBlock = 32
// Tuning: mPerThread = 2
// Tuning: nPerThread = 2
// Tuning-xdlops: kpackPerBlock = 8
// Tuning-xdlops: mPerBlock = 128
// Tuning-xdlops: nPerBlock = 128
// Tuning-xdlops: kpack = 4
// Tuning-xdlops: mPerWave = 64
// Tuning-xdlops: nPerWave = 64
