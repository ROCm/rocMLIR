//RUN: rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv_v4r4_wrw_xdlops" -pv --apply-bufferization-pipeline=false --mlir-print-local-scope | FileCheck %s --check-prefix=PV

//PV: [[FIL1:%.*]] = memref.alloc() : memref<2097152xf32>
//PV: [[FIL2:%.*]] = memref.alloc() : memref<2097152xf32>
//PV: call @mlir_gen_igemm_conv_v4r4_wrw_xdlops_0_verify0([[FIL1]], [[FIL2]])
//PV-LABEL: func @conv_bwd_weight_cpu
//PV-SAME: ([[ARG0:%.+]]: memref<2097152xf32>, [[ARG1:%.+]]: memref<51380224xf32>, [[ARG2:%.+]]: memref<33554432xf32>)
//PV: %[[f32_0:.*]] = arith.constant 0.000000e+00 : f32
//PV: vector.insertelement %[[f32_0]]
//PV-NEXT: affine.for %{{.*}} = 0 to 2097152 {
//PV-NEXT: affine.apply
//PV-NEXT: vector.extractelement
//PV-NEXT: memref.store %{{.*}}, [[ARG0]]
//PV-NEXT: }
//PV-NEXT: affine.for [[ARG3:%.*]] = 0 to 1 {
//PV-NEXT:      affine.for [[ARG4:%.*]] = 0 to 2048 {
//PV-NEXT:        affine.for [[ARG5:%.*]] = 0 to 1024 {
//PV-NEXT:          affine.for [[ARG6:%.*]] = 0 to 1 {
//PV-NEXT:            affine.for [[ARG7:%.*]] = 0 to 1 {
//PV-NEXT:              affine.for [[ARG8:%.*]] = 0 to 256 {
//PV-NEXT:                affine.for [[ARG9:%.*]] = 0 to 8 {
//PV-NEXT:                  affine.for [[ARG10:%.*]] = 0 to 8 {
//PV:                         [[INH:%.*]] = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1 * 2 - 1)>([[ARG9]], [[ARG6]])
//PV:                         [[INW:%.*]] = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1 * 2 - 1)>([[ARG10]], [[ARG7]])
//PV:                        affine.if affine_set<(d0, d1) : (d0 >= 0, -d0 + 13 >= 0, d1 >= 0, -d1 + 13 >= 0)>([[INH]], [[INW]]) {
//PV-NEXT:                      [[OUT:%.*]] = affine.load [[ARG2]][((([[ARG8]] + [[ARG3]]) * 2048 + [[ARG4]]) * 8 + [[ARG9]]) * 8 + [[ARG10]]]
//PV-NEXT:                      [[IN:%.*]] = affine.load [[ARG1]][((([[ARG8]] + [[ARG3]]) * 1024 + [[ARG5]]) * 14 + [[INH]]) * 14 + [[INW]]]
//PV-NEXT:                      [[FIL:%.*]] = affine.load [[ARG0]][([[ARG3]] * 2048 + [[ARG4]]) * 1024 + [[ARG5]] + [[ARG6]] + [[ARG7]]]
//PV-NEXT:                      [[PRD:%.*]] = arith.mulf [[OUT]], [[IN]] : f32
//PV-NEXT:                      [[ACC:%.*]] = arith.addf [[FIL]], [[PRD]] : f32
//PV-NEXT:                      affine.store [[ACC]], [[ARG0]][([[ARG3]] * 2048 + [[ARG4]]) * 1024 + [[ARG5]] + [[ARG6]] + [[ARG7]]]

// RUN: rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight --kernel_id 1 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv_v4r4_wrw_xdlops" -pv_with_gpu --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PVGPU

//PVGPU: rock.conv_bwd_weight(%{{.*}}, %{{.*}}, %{{.*}}) features = mfma|dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [2 : index, 2 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index]} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>
//PVGPU: rock.conv_bwd_weight(%{{.*}}, %{{.*}}, %{{.*}}) features = dot|atomic_add {arch = "amdgcn-amd-amdhsa:gfx908:sramecc+:xnack-", dilations = [2 : index, 2 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = 120 : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [1 : index, 1 : index, 1 : index, 1 : index], strides = [2 : index, 2 : index]} : memref<1x2048x1024x1x1xf32>, memref<256x1x1024x14x14xf32>, memref<256x1x2048x8x8xf32>

// RUN:  rocmlir-gen --conv-config "--x2 1 --operation conv_bwd_weight  --kernel_id 0 --num_cu 120 --arch amdgcn-amd-amdhsa:gfx908:sramecc+:xnack- --groupsize 1 --fil_layout GNCHW --fil_type fp32 --in_layout NGCHW --out_layout NGCHW --in_type fp32 --out_type fp32 --batchsize 256 --in_channels 1024 --out_channels 2048 --in_h 14 --in_w 14 --fil_h 1 --fil_w 1 --out_h 8 --out_w 8 --dilation_h 2 --dilation_w 2 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --kernel_name mlir_gen_igemm_conv_v4r4_wrw_xdlops" -ph -pr --apply-bufferization-pipeline=false | FileCheck %s --check-prefix=PH

//PH: [[FIL:%.*]] = memref.cast %{{.*}} : memref<2097152xf32> to memref<*xf32>
//PH: call @printMemrefF32([[FIL]]) : (memref<*xf32>) -> ()
