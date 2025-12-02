// RUN: rocmlir-gen --operation conv_bwd_data -t f16 --arch %arch --fil_layout gkc01 --in_layout ngc01 --out_layout ngk01 --batchsize 1 --in_channels 192 --in_h 64 --in_w 64 --out_channels 384 --fil_h 4 --fil_w 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 1 --padding_w 1 --groupsize 1 --perf_config 'v3:128,64,8,128,64,16,1,1,2,1,1' | rocmlir-driver --kernel-pipeline=full | FileCheck %s

// CHECK: gpu.binary {{.*}} rock.blocks_per_cu = {{.*}} : i32

