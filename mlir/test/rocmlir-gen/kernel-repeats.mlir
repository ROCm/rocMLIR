// RUN: rocmlir-gen --arch gfx900 --operation gemm -p -ph --kernel-repeats=5 | FileCheck %s --check-prefix=GEMM
// RUN: rocmlir-gen --arch gfx942 -pv --operation conv_bwd_weight -t f32 --fil_layout k01c --in_layout n01c --out_layout n01k --batchsize 64 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 256 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --kernel-repeats 5 | FileCheck %s --check-prefix=CONV_WRW
// RUN: rocmlir-gen --arch gfx942 -pv_with_gpu --operation conv_bwd_weight -t f32 --fil_layout k01c --in_layout n01c --out_layout n01k --batchsize 64 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 256 --fil_h 1 --fil_w 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 --groupsize 1 --kernel-repeats 5 | FileCheck %s --check-prefix=CONV_WRW_GPU

// GEMM-LABEL: @rock_gemm_gpu
// GEMM-DAG: %[[zero:.*]] = arith.constant 0 : index
// GEMM-DAG: %[[one:.*]] = arith.constant 1 : index
// GEMM-DAG: %[[five:.*]] = arith.constant 5 : index
// GEMM: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// GEMM-NEXT: func.call @rock_gemm
// GEMM-NEXT: }

// CONV_WRW-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_0
// CONV_WRW: rock.init_kernel
// CONV_WRW-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_1
// CONV_WRW: rock.conv_bwd_weight
// CONV_WRW-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_gpu
// CONV_WRW-DAG: %[[one:.*]] = arith.constant 1 : index
// CONV_WRW-DAG: %[[five:.*]] = arith.constant 5 : index
// CONV_WRW-DAG: %[[zero:.*]] = arith.constant 0 : index
// CONV_WRW: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// CONV_WRW-NEXT: func.call @rock_conv_bwd_weight_gk01c_ng01c_ng01k_0
// CONV_WRW-NEXT: func.call @rock_conv_bwd_weight_gk01c_ng01c_ng01k_1
// CONV_WRW-NEXT: }

// CONV_WRW_GPU-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_0
// CONV_WRW_GPU: rock.init_kernel
// CONV_WRW_GPU-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_1
// CONV_WRW_GPU: rock.conv_bwd_weight
// CONV_WRW_GPU-LABEL: func.func @rock_conv_bwd_weight_gk01c_ng01c_ng01k_gpu
// CONV_WRW_GPU-DAG: %[[zero:.*]] = arith.constant 0 : index
// CONV_WRW_GPU-DAG: %[[one:.*]] = arith.constant 1 : index
// CONV_WRW_GPU-DAG: %[[five:.*]] = arith.constant 5 : index
// CONV_WRW_GPU: scf.for %{{.*}} = %[[zero]] to %[[five]] step %[[one]] {
// CONV_WRW_GPU-NEXT: func.call @rock_conv_bwd_weight_gk01c_ng01c_ng01k_0
// CONV_WRW_GPU-NEXT: func.call @rock_conv_bwd_weight_gk01c_ng01c_ng01k_1
// CONV_WRW_GPU-NEXT: }
