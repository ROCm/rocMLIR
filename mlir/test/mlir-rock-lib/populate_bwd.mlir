////////////////////////////////////////////
// Test case which depends on 1 GPU kernel.
////////////////////////////////////////////

// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name foo --groupsize 1" --option kernelcount | FileCheck %s --check-prefix=KERNELCOUNT1
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name foo --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN1
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name foo --groupsize 1 --kernel_id 0" --option tuningparams | FileCheck %s --check-prefix=TUNING1
// RUN: rocmlir-gen --conv-config "--operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name foo --groupsize 1 " | FileCheck %s --check-prefix=DRIVER1

// KERNELCOUNT1: Kernel count=1
// BIN1: ELF
// TUNING1: globalSize{{.*}}localSize{{.*}}
// DRIVER1-COUNT-3: rock.transform %{{.+}} by
// DRIVER1: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 0 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index], usesV4R1 = true} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

////////////////////////////////////////////
// Test case which depends on 4 GPU kernels.
////////////////////////////////////////////

// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1" --option kernelcount | FileCheck %s --check-prefix=KERNELCOUNT4
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1" --option bin | FileCheck %s --check-prefix=BIN4
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1 --kernel_id 0" --option tuningparams | FileCheck %s --check-prefix=TUNING4_0
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1 --kernel_id 1" --option tuningparams | FileCheck %s --check-prefix=TUNING4_1
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1 --kernel_id 2" --option tuningparams | FileCheck %s --check-prefix=TUNING4_2
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1 --kernel_id 3" --option tuningparams | FileCheck %s --check-prefix=TUNING4_3
// RUN: rocmlir-gen --conv-config " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp32 --fil_type fp32 --out_type fp32 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 6 --out_w 6 --fil_h 3 --fil_w 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name bar --groupsize 1 " | FileCheck %s --check-prefix=DRIVER4

// KERNELCOUNT4: Kernel count=4
// BIN4: ELF
// BIN4: ELF
// BIN4: ELF
// BIN4: ELF
// TUNING4_0: globalSize{{.*}}localSize{{.*}}
// TUNING4_1: globalSize{{.*}}localSize{{.*}}
// TUNING4_2: globalSize{{.*}}localSize{{.*}}
// TUNING4_3: globalSize{{.*}}localSize{{.*}}
// DRIVER4-COUNT-3: rock.transform %{{.+}} by
// DRIVER4-NEXT: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 0 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index], usesV4R1 = true} : memref<1x1024x1024x3x3xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x6x6xf32>
// DRIVER4-COUNT-3: rock.transform %{{.+}} by
// DRIVER4-NEXT: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 1 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index], usesV4R1 = true} : memref<1x1024x1024x3x3xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x6x6xf32>
// DRIVER4-COUNT-3: rock.transform %{{.+}} by
// DRIVER4-NEXT: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 2 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index], usesV4R1 = true} : memref<1x1024x1024x3x3xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x6x6xf32>
// DRIVER4-COUNT-3: rock.transform %{{.+}} by
// DRIVER4-NEXT: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 3 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index], usesV4R1 = true} : memref<1x1024x1024x3x3xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x6x6xf32>

////////////////////////////////////////////
// Test case which depends on 1 GPU kernel
// that requires an argument to be
// zero-initialized.
////////////////////////////////////////////
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --out_h 7 --out_w 7 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name baz --groupsize 1 " --option kernelcount | FileCheck %s --check-prefix=ZEROINIT_KERNELCOUNT
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --out_h 7 --out_w 7 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name baz --groupsize 1 " --option bin | FileCheck %s --check-prefix=ZEROINIT_BIN
// RUN: rocmlir-lib-test --args " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --out_h 7 --out_w 7 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name baz --groupsize 1 --kernel_id 0" --option tuningparams | FileCheck %s --check-prefix=ZEROINIT_TUNING
// RUN: rocmlir-gen --conv-config " --operation conv_bwd_data --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --in_type fp16 --fil_type fp16 --out_type fp16 --fil_layout GNCHW --in_layout NGCHW --out_layout NGCHW --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --out_h 7 --out_w 7 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0 --kernel_name baz --groupsize 1 " | FileCheck %s --check-prefix=ZEROINIT_DRIVER

// ZEROINIT_KERNELCOUNT: Kernel count=1
// ZEROINIT_BIN: ELF
// ZEROINIT_TUNING: globalSize=100352, localSize=64
// ZEROINIT_DRIVER: %arg1: memref<{{.*}}xf16> {rock.prefill = 0.000000e+00 : f16}
// ZEROINIT_DRIVER-COUNT-3: rock.transform %{{.+}} by
// ZEROINIT_DRIVER-NEXT: rock.conv_bwd_data(%{{.+}}, %{{.+}}, %{{.+}}) features = dot {dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], kernelId = 0 : index, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [2 : index, 2 : index], usesV4R1 = true} : memref<1x2048x1024x1x1xf16>, memref<256x1x1024x14x14xf16>, memref<256x1x2048x7x7xf16>
