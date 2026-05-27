// RUN: rocmlir-gen --arch gfx906:sramecc+:xnack- -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data -v4r1 0 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=NOV4R1
// RUN: rocmlir-gen --arch gfx906:sramecc+:xnack- -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=32 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --groupsize=1  --operation=conv_bwd_data -v4r1 1 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefix=V4R1

// NOV4R1: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// NOV4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_0({{.*}} rock.kernel = 0 : i32
// NOV4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 0 : index{{.*}} usesV4R1 = false
// NOV4R1-NOT: rock.kernel = 1: i32

// V4R1: module attributes {mhal.arch = "[[$ARCH:.*]]"}
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_0({{.*}} rock.kernel = 0 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 0 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_1({{.*}} rock.kernel = 1 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 1 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_2({{.*}} rock.kernel = 2 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 2 : index{{.*}} usesV4R1 = true
// V4R1: @rock_conv_bwd_data_gk01c_n01gc_n01gk_3({{.*}} rock.kernel = 3 : i32
// V4R1: rock.conv_bwd_data(%0, %1, %2) {{.*}} kernelId = 3 : index{{.*}} usesV4R1 = true

// Test mixed dtype support: verify that CPU validation function uses correct types
// when fil_dtype, in_dtype, and out_dtype are all different
// RUN: rocmlir-gen --arch gfx942 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=1 -in_channels=32 -out_channels=32 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --operation=conv_bwd_data -fil_dtype f16 -in_dtype f32 -out_dtype f16 -pv | FileCheck %s --check-prefix=MIXED_DTYPE

// MIXED_DTYPE: func.func @rock_conv_bwd_data{{.*}}(%arg0: memref<{{[0-9]+}}xf16>, %arg1: memref<{{[0-9]+}}xf32>, %arg2: memref<{{[0-9]+}}xf16>)
// MIXED_DTYPE: func.func @conv_bwd_data_cpu(%arg0: memref<{{[0-9]+}}xf16>, %arg1: memref<{{[0-9]+}}xf32>, %arg2: memref<{{[0-9]+}}xf16>)
