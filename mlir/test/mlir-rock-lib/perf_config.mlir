// RUN: rocmlir-lib-test --args " --operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1 --perf_config v3:64,64,8,16,16,4,1,1,2,1,1" --option bin | FileCheck %s --check-prefix=BIN
// RUN: rocmlir-gen --conv-config " --operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1 --perf_config v3:64,64,8,16,16,4,1,1,2,1,1" | rocmlir-driver -rock-affix-params | FileCheck %s --check-prefix=Tuning
// RUN: rocmlir-gen --conv-config " --operation conv --arch amdgcn-amd-amdhsa:gfx908 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1 --x2 1 --perf_config v3:128,128,8,64,64,4,1,1,2,1,1" | rocmlir-driver -rock-affix-params | FileCheck %s --check-prefix=Tuning-xdlops

// BIN: ELF
// Tuning: kpackPerBlock = 8
// Tuning: mPerBlock = 64
// Tuning: nPerBlock = 64
// Tuning: mPerWave = 16 
// Tuning: nPerWave = 16
// Tuning-xdlops: kpackPerBlock = 8
// Tuning-xdlops: mPerBlock = 128
// Tuning-xdlops: nPerBlock = 128
// Tuning-xdlops: kpack = 4
// Tuning-xdlops: mPerWave = 64
// Tuning-xdlops: nPerWave = 64
