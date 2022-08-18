// RUN: miopen-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -pv | FileCheck %s --check-prefix=PV

//PV: [[FIL1:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PV: [[FIL2:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PV: call @mlir_gen_igemm_conv2d_v4r4_wrw_xdlops_0_verify([[FIL1]], [[FIL2]])
//PV: %{{.*}} = memref.cast %{{.*}} : memref<256x1x2048x8x8xf32> to memref<*xf32>
//PV-NEXT: [[S1:%.*]] = arith.constant 2 : i32
//PV-NEXT: [[S2:%.*]] = arith.constant 2 : i32
//PV-NEXT: [[P1:%.*]] = arith.constant 1 : i32
//PV-NEXT: [[P2:%.*]] = arith.constant 1 : i32
//PV-NEXT: [[P3:%.*]] = arith.constant 1 : i32
//PV-NEXT: [[P4:%.*]] = arith.constant 1 : i32
//PV-NEXT: [[D1:%.*]] = arith.constant 2 : i32
//PV-NEXT: [[D2:%.*]] = arith.constant 2 : i32
//PV: [[K:%.*]] = arith.constant 107 : i8
//PV: [[C:%.*]] = arith.constant 99 : i8
//PV: [[Y:%.*]] = arith.constant 121 : i8
//PV: [[X:%.*]] = arith.constant 120 : i8
//PV: [[N:%.*]] = arith.constant 110 : i8
//PV: [[H:%.*]] = arith.constant 104 : i8
//PV: [[W:%.*]] = arith.constant 119 : i8
//PV: [[G:%.*]] = arith.constant 103 : i8
//PV: [[X2:%.*]] = arith.constant 1 : i32
//PV: memref.store [[G]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[Y]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[X]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PV-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PV: call @mcpuConv2dBwdWeightFloat(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, [[S1]], [[S2]], [[P1]], [[P2]], [[P3]], [[P4]], [[D1]], [[D2]], [[X2]]) : {{.*}}

// RUN: miopen-gen --conv-config "--x2 1 --operation conv2d_bwd_weight --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -pv_with_gpu | FileCheck %s --check-prefix=PVGPU

//PVGPU: miopen.conv2d_bwd_weight(%arg0, %arg1, %arg2) {arch = "gfx908", dilations = [2 : i32, 2 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32], xdlopsV2 = true} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>
//PVGPU: miopen.conv2d_bwd_weight(%{{.*}}, %{{.*}}, %{{.*}}) {arch = "gfx908", dilations = [2 : i32, 2 : i32], filter_layout = ["g", "k", "c", "y", "x"], gemm_id = 0 : i32, input_layout = ["ni", "gi", "ci", "hi", "wi"], numCu = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>

// RUN: miopen-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -prc -pr | FileCheck %s --check-prefix=PRC

//PRC:[[FIL:%.*]] = memref.cast %{{.*}} : memref<1x2048x1024x1x1xf32> to memref<*xf32>
//PRC: %{{.*}} = memref.cast %{{.*}} : memref<256x1x2048x8x8xf32> to memref<*xf32>
//PRC-NEXT: [[S1:%.*]] = arith.constant 2 : i32
//PRC-NEXT: [[S2:%.*]] = arith.constant 2 : i32
//PRC-NEXT: [[P1:%.*]] = arith.constant 1 : i32
//PRC-NEXT: [[P2:%.*]] = arith.constant 1 : i32
//PRC-NEXT: [[P3:%.*]] = arith.constant 1 : i32
//PRC-NEXT: [[P4:%.*]] = arith.constant 1 : i32
//PRC-NEXT: [[D1:%.*]] = arith.constant 2 : i32
//PRC-NEXT: [[D2:%.*]] = arith.constant 2 : i32
//PRC: [[K:%.*]] = arith.constant 107 : i8
//PRC: [[C:%.*]] = arith.constant 99 : i8
//PRC: [[Y:%.*]] = arith.constant 121 : i8
//PRC: [[X:%.*]] = arith.constant 120 : i8
//PRC: [[N:%.*]] = arith.constant 110 : i8
//PRC: [[H:%.*]] = arith.constant 104 : i8
//PRC: [[W:%.*]] = arith.constant 119 : i8
//PRC: [[G:%.*]] = arith.constant 103 : i8
//PRC: [[X2:%.*]] = arith.constant 1 : i32
//PRC: memref.store [[G]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[Y]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[X]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PRC-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PRC: call @mcpuConv2dBwdWeightFloat(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, [[S1]], [[S2]], [[P1]], [[P2]], [[P3]], [[P4]], [[D1]], [[D2]], [[X2]]) : {{.*}}

// RUN:  miopen-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -ph -pr  | FileCheck %s --check-prefix=PH

//PH: [[FIL:%.*]] = memref.cast %{{.*}} : memref<1x2048x1024x1x1xf32> to memref<*xf32>
//PH: call @printMemrefF32([[FIL]]) : (memref<*xf32>) -> ()
