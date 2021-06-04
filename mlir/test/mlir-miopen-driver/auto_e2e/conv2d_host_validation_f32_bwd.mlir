// COM: bwd_weight
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=32 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BWD_WEIGHT1

// CHECK_BWD_WEIGHT1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_WEIGHT1: [1]

// COM: bwd_data
// FIXME: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 --operation=conv2d_bwd_data -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BWD_DATA1

// CHECK_BWD_DATA1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_DATA1: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/127
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_5
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_6
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_7
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_8

// CHECK_ISSUE_127_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_5: [1]
// CHECK_ISSUE_127_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_6: [1]
// CHECK_ISSUE_127_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_7: [1]
// CHECK_ISSUE_127_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_8: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/71
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_4

// CHECK_ISSUE_71_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_71_1: [1]
// CHECK_ISSUE_71_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_71_2: [1]
// CHECK_ISSUE_71_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_71_3: [1]
// CHECK_ISSUE_71_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_71_4: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/70
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=64 -out_channels=64 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_3

// CHECK_ISSUE_70_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_1: [1]
// CHECK_ISSUE_70_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_2: [1]
// CHECK_ISSUE_70_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_3: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/136
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=32 -out_channels=32 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_136_1

// CHECK_ISSUE_136_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_136_1: [1]
