// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=1 -in_channels=3 -in_h=224 -in_w=224 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=3 --padding_w=3 -t f32 --operation=conv2d %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_Resnext101_f32_23_1
// CHECK_Resnext101_f32_23_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_Resnext101_f32_23_1: [1]

// RUN: miopen-gen -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=1 -in_channels=3 -in_h=224 -in_w=224 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=3 --padding_w=3 -t f32 --operation=conv2d %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_Resnext101_f32_23_2
// CHECK_Resnext101_f32_23_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_Resnext101_f32_23_2: [1]

// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=1 -in_channels=3 -in_h=224 -in_w=224 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=3 --padding_w=3 -t f32 --operation=conv2d_bwd_weight %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_Resnext101_f32_23_3
// CHECK_Resnext101_f32_23_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_Resnext101_f32_23_3: [1]

// RUN: miopen-gen -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=1 -in_channels=3 -in_h=224 -in_w=224 -out_channels=64 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=3 --padding_w=3 -t f32 --operation=conv2d_bwd_weight %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_Resnext101_f32_23_4
// CHECK_Resnext101_f32_23_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_Resnext101_f32_23_4: [1]
