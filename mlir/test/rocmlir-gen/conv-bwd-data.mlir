// RUN: rocmlir-gen --arch gfx906:sramecc+:xnack- -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data -v4r1 0 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=NOV4R1
// RUN: rocmlir-gen --arch gfx906:sramecc+:xnack- -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data -v4r1 1 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=V4R1

// NOV4R1: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// NOV4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_0({{.*}} kernel = 0 : i32
// NOV4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 0 : index{{.*}} usesV4R1 = false
// NOV4R1-NOT: kernel = 1: i32

// V4R1: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_0({{.*}} kernel = 0 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 0 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_1({{.*}} kernel = 1 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 1 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_2({{.*}} kernel = 2 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 2 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_3({{.*}} kernel = 3 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 3 : index{{.*}} usesV4R1 = true
