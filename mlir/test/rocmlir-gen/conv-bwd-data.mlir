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

// Forward conv also accepts mixed (filter, input) -> output dtype combos as
// long as the filter/input dtypes agree -- the output type can be widened
// to f32 explicitly via `-out_dtype`.
// RUN: rocmlir-gen --arch gfx942 --operation conv -p -fil_dtype f16 -in_dtype f16 -out_dtype f32 | FileCheck %s --check-prefix=FWD_F16_F32
// FWD_F16_F32: func.func @rock_conv{{.*}}(%{{.*}}: memref<{{[0-9]+}}xf16>, %{{.*}}: memref<{{[0-9]+}}xf16>, %{{.*}}: memref<{{[0-9]+}}xf32>)
// FWD_F16_F32: rock.conv(%{{.*}}, %{{.*}}, %{{.*}})
// FWD_F16_F32-SAME: memref<{{.*}}xf16>, memref<{{.*}}xf16>, memref<{{.*}}xf32>
// RUN: rocmlir-gen --arch gfx942 --operation conv -p -fil_dtype bf16 -in_dtype bf16 -out_dtype f32 | FileCheck %s --check-prefix=FWD_BF16_F32
// FWD_BF16_F32: func.func @rock_conv{{.*}}(%{{.*}}: memref<{{[0-9]+}}xbf16>, %{{.*}}: memref<{{[0-9]+}}xbf16>, %{{.*}}: memref<{{[0-9]+}}xf32>)

// 3-D backward-data: kernel args are ordered (filter, input, output) like the
// forward kernel; the func returns void. With matching `padding_d/h/w` the
// padding attribute is `[h_l, h_r, w_l, w_r, d_l, d_r]`.
// RUN: rocmlir-gen --arch gfx942 --operation conv_bwd_data -t f32 -fil_layout=gkc012 -in_layout=ngc012 -out_layout=ngk012 -batchsize=2 -groupsize=1 -in_channels=4 -out_channels=4 -in_d=4 -in_h=4 -in_w=4 -fil_d=3 -fil_h=3 -fil_w=3 --conv_stride_d=1 --conv_stride_h=1 --conv_stride_w=1 --dilation_d=1 --dilation_h=1 --dilation_w=1 --padding_d=1 --padding_h=1 --padding_w=1 | FileCheck %s --check-prefix=BWD_DATA_3D
// BWD_DATA_3D-LABEL: func.func @rock_conv_bwd_data_gkc012_ngc012_ngk012
// BWD_DATA_3D-SAME: (%[[fil:.*]]: memref<432xf32>, %[[in:.*]]: memref<512xf32>, %[[out:.*]]: memref<512xf32>)
// BWD_DATA_3D: rock.transform %[[fil]] {{.*}} : memref<432xf32> to memref<1x4x4x3x3x3xf32>
// BWD_DATA_3D: rock.transform %[[in]] {{.*}} : memref<512xf32> to memref<2x1x4x4x4x4xf32>
// BWD_DATA_3D: rock.conv_bwd_data
// BWD_DATA_3D-SAME: filter_layout = ["g", "k", "c", "0", "1", "2"]
// BWD_DATA_3D-SAME: input_layout = ["ni", "gi", "ci", "0i", "1i", "2i"]
// BWD_DATA_3D-SAME: output_layout = ["no", "go", "ko", "0o", "1o", "2o"]
// BWD_DATA_3D-SAME: padding = [1 : index, 1 : index, 1 : index, 1 : index, 1 : index, 1 : index]
// BWD_DATA_3D-SAME: strides = [1 : index, 1 : index, 1 : index]
