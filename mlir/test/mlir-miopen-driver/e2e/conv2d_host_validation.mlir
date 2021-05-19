///////////////////////////////////////////////////////////////////////////////////////////
// Resnet101 NCHW
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_FWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_BWD

// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32  -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_BWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG3_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG3_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG3_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG4_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG4_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG4_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG5_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG5_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG5_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG6_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG6_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG6_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG7_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG7_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG7_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG8_FWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG8_BWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG8_WRW


// CHECK_RESNET101_NCHW_CONFIG1_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG1_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG1_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG1_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG1_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG1_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG2_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG2_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG2_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG2_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG2_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG2_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG3_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG3_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG3_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG3_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG3_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG3_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG4_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG4_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG4_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG4_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG4_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG4_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG5_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG5_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG5_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG5_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG5_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG5_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG6_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG6_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG6_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG6_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG6_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG6_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG7_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG7_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG7_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG7_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG7_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG7_WRW: [1]
// CHECK_RESNET101_NCHW_CONFIG8_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG8_FWD: [1]
// CHECK_RESNET101_NCHW_CONFIG8_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG8_BWD: [1]
// CHECK_RESNET101_NCHW_CONFIG8_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NCHW_CONFIG8_WRW: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Resnet101 NHWC
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_FWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_BWD

// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32  -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_BWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG3_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG3_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=128 -out_channels=128 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG3_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG4_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG4_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG4_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG5_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG5_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG5_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG6_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG6_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG6_WRW

// FIXME: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG7_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG7_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=512 -out_channels=512 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG7_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG8_FWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG8_BWD

// RUN: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=128 -in_channels=64 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --groupsize=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG8_WRW


// CHECK_RESNET101_NHWC_CONFIG1_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG1_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG1_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG1_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG1_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG1_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG2_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG2_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG2_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG2_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG2_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG2_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG3_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG3_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG3_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG3_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG3_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG3_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG4_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG4_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG4_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG4_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG4_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG4_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG5_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG5_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG5_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG5_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG5_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG5_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG6_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG6_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG6_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG6_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG6_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG6_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG7_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG7_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG7_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG7_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG7_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG7_WRW: [1]
// CHECK_RESNET101_NHWC_CONFIG8_FWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG8_FWD: [1]
// CHECK_RESNET101_NHWC_CONFIG8_BWD: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG8_BWD: [1]
// CHECK_RESNET101_NHWC_CONFIG8_WRW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET101_NHWC_CONFIG8_WRW: [1]
