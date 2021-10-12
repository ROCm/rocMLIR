//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=1024 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_1

// CHECK_NCHW_XDLOPS_BWD_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_1: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -in_h=7 -in_w=7 -out_channels=1024 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_2

// CHECK_NCHW_XDLOPS_BWD_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_2: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=128 -in_h=56 -in_w=56 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_3

// CHECK_NCHW_XDLOPS_BWD_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_3: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -in_h=28 -in_w=28 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_4

// CHECK_NCHW_XDLOPS_BWD_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_4: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -in_h=56 -in_w=56 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_5

// CHECK_NCHW_XDLOPS_BWD_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_5: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -in_h=14 -in_w=14 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_6

// CHECK_NCHW_XDLOPS_BWD_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_6: [1]

//bwd xdlops ngchw
// RUN: mlir-miopen-driver -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -in_h=28 -in_w=28 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=1 --padding_w=1 -t f32 --operation=conv2d_bwd_data %pv -c %random_data %xdlops | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_XDLOPS_BWD_7

// CHECK_NCHW_XDLOPS_BWD_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_XDLOPS_BWD_7: [1]


