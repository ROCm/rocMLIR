//RUN: rocmlir-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -pv --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PV

//PV: [[FIL1:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PV: [[FIL2:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PV: call @mlir_gen_igemm_conv2d_v4r4_wrw_xdlops_0_verify0([[FIL1]], [[FIL2]])
//PV: @conv2d_bwd_weight_cpu
//PV: %[[f32_0:.*]] = arith.constant 0.000000e+00 : f32
//PV: vector.insertelement %[[f32_0]]
//PV-NEXT: %[[filterFlat:.*]] = memref.collapse_shape {{.*}} : memref<{{.*}}> into memref<[[GKCYX:.*]]xf32>
//PV-NEXT: affine.for %{{.*}} = 0 to [[GKCYX]] {
//PV-NEXT: affine.apply
//PV-NEXT: vector.extractelement
//PV-NEXT: memref.store %{{.*}}, %[[filterFlat]]
//PV-NEXT: }
//PV-NEXT: affine.for [[ARG3:%.*]] = 0 to 1 {
//PV-NEXT:      affine.for [[ARG4:%.*]] = 0 to 2048 {
//PV-NEXT:        affine.for [[ARG5:%.*]] = 0 to 1024 {
//PV-NEXT:          affine.for [[ARG6:%.*]] = 0 to 1 {
//PV-NEXT:            affine.for [[ARG7:%.*]] = 0 to 1 {
//PV-NEXT:              affine.for [[ARG8:%.*]] = 0 to 256 {
//PV-NEXT:                affine.for [[ARG9:%.*]] = 0 to 8 {
//PV-NEXT:                  affine.for [[ARG10:%.*]] = 0 to 8 {
//PV:                         [[INH:%.*]] = affine.apply #map2([[ARG9]], [[ARG6]])
//PV:                         [[INW:%.*]] = affine.apply #map2([[ARG10]], [[ARG7]])
//PV:                         [[C14:%.*]] = arith.constant 14 : index
//PV:                         [[C14_0:%.*]] = arith.constant 14 : index
//PV:                        affine.if #set([[INH]], [[INW]])[[[C14]], [[C14_0]]] {
//PV-NEXT:                      [[OUT:%.*]] = memref.load %arg2[[[ARG8]], [[ARG3]], [[ARG4]], [[ARG9]], [[ARG10]]] : memref<256x1x2048x8x8xf32>
//PV-NEXT:                      [[IN:%.*]] = memref.load %arg1[[[ARG8]], [[ARG3]], [[ARG5]], [[INH]], [[INW]]] : memref<256x1x1024x14x14xf32>
//PV-NEXT:                      [[FIL:%.*]] = memref.load %arg0[[[ARG3]], [[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]]] : memref<1x2048x1024x1x1xf32>
//PV-NEXT:                      [[PRD:%.*]] = arith.mulf [[OUT]], [[IN]] : f32
//PV-NEXT:                      [[ACC:%.*]] = arith.addf [[FIL]], [[PRD]] : f32
//PV-NEXT:                      memref.store [[ACC]], %arg0[[[ARG3]], [[ARG4]], [[ARG5]], [[ARG6]], [[ARG7]]] : memref<1x2048x1024x1x1xf32>

// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -pv_with_cpp --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PVCPP

//PVCPP: [[FIL1:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PVCPP: [[FIL2:%.*]] = memref.alloc() : memref<1x2048x1024x1x1xf32>
//PVCPP: call @mlir_gen_igemm_conv2d_v4r4_wrw_xdlops_0_verify0([[FIL1]], [[FIL2]])
//PVCPP: %{{.*}} = memref.cast %{{.*}} : memref<256x1x2048x8x8xf32> to memref<*xf32>
//PVCPP-NEXT: [[S1:%.*]] = arith.constant 2 : i32
//PVCPP-NEXT: [[S2:%.*]] = arith.constant 2 : i32
//PVCPP-NEXT: [[P1:%.*]] = arith.constant 1 : i32
//PVCPP-NEXT: [[P2:%.*]] = arith.constant 1 : i32
//PVCPP-NEXT: [[P3:%.*]] = arith.constant 1 : i32
//PVCPP-NEXT: [[P4:%.*]] = arith.constant 1 : i32
//PVCPP-NEXT: [[D1:%.*]] = arith.constant 2 : i32
//PVCPP-NEXT: [[D2:%.*]] = arith.constant 2 : i32
//PVCPP: [[K:%.*]] = arith.constant 107 : i8
//PVCPP: [[C:%.*]] = arith.constant 99 : i8
//PVCPP: [[Y:%.*]] = arith.constant 121 : i8
//PVCPP: [[X:%.*]] = arith.constant 120 : i8
//PVCPP: [[N:%.*]] = arith.constant 110 : i8
//PVCPP: [[H:%.*]] = arith.constant 104 : i8
//PVCPP: [[W:%.*]] = arith.constant 119 : i8
//PVCPP: [[G:%.*]] = arith.constant 103 : i8
//PVCPP: [[X2:%.*]] = arith.constant 1 : i32
//PVCPP: memref.store [[G]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[Y]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[X]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[C]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[N]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[G]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[K]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[H]], %{{.*}} : memref<5xi8>
//PVCPP-NEXT: memref.store [[W]], %{{.*}} : memref<5xi8>
//PVCPP: call @mcpuConv2dBwdWeightFloat(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, [[S1]], [[S2]], [[P1]], [[P2]], [[P3]], [[P4]], [[D1]], [[D2]], [[X2]]) : {{.*}}

// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv2d_bwd_weight --kernel_id 1 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -pv_with_gpu --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PVGPU

//PVGPU: rock.conv2d_bwd_weight(%arg0, %arg1, %arg2) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [2 : i32, 2 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>
//PVGPU: rock.conv2d_bwd_weight(%{{.*}}, %{{.*}}, %{{.*}}) features = dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [2 : i32, 2 : i32], filter_layout = ["g", "k", "c", "y", "x"], input_layout = ["ni", "gi", "ci", "hi", "wi"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "ho", "wo"], padding = [1 : i32, 1 : i32, 1 : i32, 1 : i32], strides = [2 : i32, 2 : i32]} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>

// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -prc -pv_with_cpp --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PRC

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

// RUN:  rocmlir-gen --conv-config "--x2 1 --operation conv2d_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv2d_v4r4_wrw_xdlops" -ph -pr --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PH

//PH: [[FIL:%.*]] = memref.cast %{{.*}} : memref<1x2048x1024x1x1xf32> to memref<*xf32>
//PH: call @printMemrefF32([[FIL]]) : (memref<*xf32>) -> ()
