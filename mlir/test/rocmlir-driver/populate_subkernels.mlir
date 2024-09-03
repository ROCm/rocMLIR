// RUN: rocmlir-gen -conv-config "--operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_fwd --groupsize 1 --kernel_id 0" | FileCheck %s --check-prefix=KERNEL0
// RUN: rocmlir-gen -conv-config "--operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_fwd --groupsize 1 --kernel_id 1" | FileCheck %s --check-prefix=KERNEL1
// RUN: rocmlir-gen -conv-config "--operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_fwd --groupsize 1 --kernel_id 2" | FileCheck %s --check-prefix=KERNEL2
// RUN: rocmlir-gen -conv-config "--operation conv --arch amdgcn-amd-amdhsa:gfx906 --num_cu 64 --fil_layout GNCHW --in_type fp32 --fil_type fp32 --out_type fp32 --in_layout NGCHW --out_layout NGCHW --batchsize 64 --in_channels 1024 --out_channels 1024 --in_h 14 --in_w 14 --out_h 14 --out_w 14 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --kernel_name conv_fwd --groupsize 1 --kernel_id 3" | FileCheck %s --check-prefix=KERNEL3

// KERNEL0-LABEL: module
// KERNEL0-NEXT:  func.func @conv_fwd_0(%arg0: memref<1048576xf32>, %arg1: memref<12845056xf32>, %arg2: memref<12845056xf32>) attributes {enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// KERNEL0-NEXT: %[[exp0:.*]] = rock.transform %arg0 {{.*}} : memref<1048576xf32> to memref<1x1024x1024x1x1xf32>
// KERNEL0-NEXT: %[[exp1:.*]] = rock.transform %arg1 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL0-NEXT: %[[exp2:.*]] = rock.transform %arg2 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL0-NEXT: rock.conv(%[[exp0]], %[[exp1]], %[[exp2]]) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 64 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

// KERNEL1-LABEL: module
// KERNEL1-NEXT:  func.func @conv_fwd_1(%arg0: memref<1048576xf32>, %arg1: memref<12845056xf32>, %arg2: memref<12845056xf32>) attributes {enable_splitk_for_tuning, kernel = 1 : i32, mhal.arch = "{{.*}}"} {
// KERNEL1-NEXT: %[[exp0:.*]] = rock.transform %arg0 {{.*}} : memref<1048576xf32> to memref<1x1024x1024x1x1xf32>
// KERNEL1-NEXT: %[[exp1:.*]] = rock.transform %arg1 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL1-NEXT: %[[exp2:.*]] = rock.transform %arg2 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL1-NEXT: rock.conv(%[[exp0]], %[[exp1]], %[[exp2]]) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 64 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

// KERNEL2-LABEL: module
// KERNEL2-NEXT:  func.func @conv_fwd_2(%arg0: memref<1048576xf32>, %arg1: memref<12845056xf32>, %arg2: memref<12845056xf32>) attributes {enable_splitk_for_tuning, kernel = 2 : i32, mhal.arch = "{{.*}}"} {
// KERNEL2-NEXT: %[[exp0:.*]] = rock.transform %arg0 {{.*}} : memref<1048576xf32> to memref<1x1024x1024x1x1xf32>
// KERNEL2-NEXT: %[[exp1:.*]] = rock.transform %arg1 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL2-NEXT: %[[exp2:.*]] = rock.transform %arg2 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL2-NEXT: rock.conv(%[[exp0]], %[[exp1]], %[[exp2]]) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 64 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>

// KERNEL3-LABEL: module
// KERNEL3-NEXT:  func.func @conv_fwd_3(%arg0: memref<1048576xf32>, %arg1: memref<12845056xf32>, %arg2: memref<12845056xf32>) attributes {enable_splitk_for_tuning, kernel = 3 : i32, mhal.arch = "{{.*}}"} {
// KERNEL3-NEXT: %[[exp0:.*]] = rock.transform %arg0 {{.*}} : memref<1048576xf32> to memref<1x1024x1024x1x1xf32>
// KERNEL3-NEXT: %[[exp1:.*]] = rock.transform %arg1 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL3-NEXT: %[[exp2:.*]] = rock.transform %arg2 {{.*}} : memref<12845056xf32> to memref<64x1x1024x14x14xf32>
// KERNEL3-NEXT: rock.conv(%[[exp0]], %[[exp1]], %[[exp2]]) features = dot {arch = "amdgcn-amd-amdhsa:gfx906", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 64 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x1024x1024x1x1xf32>, memref<64x1x1024x14x14xf32>, memref<64x1x1024x14x14xf32>
