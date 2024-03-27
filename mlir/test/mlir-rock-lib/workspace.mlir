/////////////////////////////////////
// No-padding along Gemm M/N/K cases.
/////////////////////////////////////

// Forward convolution
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO

// Backward data convolution
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO

// Backward weight convolution
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 0 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO
// fp16+XDLOPS+backward weight. This would require a workspace.
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_NONZERO

/////////////////////////////////////
// Padding along Gemm M/N/K cases.
/////////////////////////////////////

// Forward convolution
// Omitted here. Existing logic would never trigger a kernel which requires a workspace.
// Test cases above are already sufficient.

// Backward data convolution
// Omitted here. Existing logic would never trigger a kernel which requires a workspace.
// Test cases above are already sufficient.

// Backward weight convolution
// For cases which involves padding along Gemm M/N/K dimension, a kernel which requires a workspace would not be generated.
// fp16+XDLOPS+backward weight, but with padding along GemmK dimension, this would NOT require a workspace.
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 20 --in_channels 3 --out_channels 6 --in_h 32 --in_w 32 --out_h 16 --out_w 16 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 3 --padding_w 3 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO

// fp16+XDLOPS+backward weight, but with padding along GemmN dimension, this would NOT require a workspace.
// RUN: rocmlir-lib-test --args " --x2 1 --operation conv_bwd_weight --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --num_cu 120 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 256 --in_channels 3 --out_channels 64 --in_h 224 --in_w 224 --out_h 112 --out_w 112 --fil_h 7 --fil_w 7 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 3 --padding_w 3 --kernel_name conv_nchw_kcyx_nkhw --groupsize 1" --option workspace | FileCheck %s --check-prefix=WORKSPACE_ZERO

// WORKSPACE_ZERO: Workspace=0
// WORKSPACE_NONZERO: Workspace=4194304
