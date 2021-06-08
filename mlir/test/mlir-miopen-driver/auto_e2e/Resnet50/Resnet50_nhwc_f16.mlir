// kyxc/nhwc/nhwk f16 fwd
// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_1

// CHECK_NHWC_FWD_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_1: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_2

// CHECK_NHWC_FWD_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_2: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_3

// CHECK_NHWC_FWD_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_3: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_4

// CHECK_NHWC_FWD_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_4: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_5

// CHECK_NHWC_FWD_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_5: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=58 -in_w=58 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_6

// CHECK_NHWC_FWD_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_6: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_7

// CHECK_NHWC_FWD_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_7: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=14 -in_w=14 -out_channels=1024 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_8

// CHECK_NHWC_FWD_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_8: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_9

// CHECK_NHWC_FWD_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_9: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=30 -in_w=30 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_10

// CHECK_NHWC_FWD_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_10: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=128 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_11

// CHECK_NHWC_FWD_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_11: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_12

// CHECK_NHWC_FWD_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_12: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=64 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_13

// CHECK_NHWC_FWD_13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_13: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_14

// CHECK_NHWC_FWD_14: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_14: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=1024 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_15

// CHECK_NHWC_FWD_15: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_15: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=128 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_16

// CHECK_NHWC_FWD_16: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_16: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_17

// CHECK_NHWC_FWD_17: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_17: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=7 -in_w=7 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_18

// CHECK_NHWC_FWD_18: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_18: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_19

// CHECK_NHWC_FWD_19: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_19: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_20

// CHECK_NHWC_FWD_20: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_20: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_21

// CHECK_NHWC_FWD_21: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_21: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_FWD_22

// CHECK_NHWC_FWD_22: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_FWD_22: [1]

// kyxc/nhwc/nhwk f16 bwd_weight
// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_1

// CHECK_NHWC_WRW_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_1: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_2

// CHECK_NHWC_WRW_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_2: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=1024 -in_h=14 -in_w=14 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_3

// CHECK_NHWC_WRW_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_3: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_4

// CHECK_NHWC_WRW_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_4: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_5

// CHECK_NHWC_WRW_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_5: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=128 -in_h=58 -in_w=58 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_6

// CHECK_NHWC_WRW_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_6: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=2048 -in_h=7 -in_w=7 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_7

// CHECK_NHWC_WRW_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_7: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=14 -in_w=14 -out_channels=1024 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_8

// CHECK_NHWC_WRW_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_8: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_9

// CHECK_NHWC_WRW_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_9: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=30 -in_w=30 -out_channels=256 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_10

// CHECK_NHWC_WRW_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_10: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=128 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_11

// CHECK_NHWC_WRW_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_11: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_12

// CHECK_NHWC_WRW_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_12: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=256 -in_h=56 -in_w=56 -out_channels=64 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_13

// CHECK_NHWC_WRW_13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_13: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=16 -in_w=16 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_14

// CHECK_NHWC_WRW_14: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_14: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=1024 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=2 --conv_stride_w=2 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_15

// CHECK_NHWC_WRW_15: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_15: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=128 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_16

// CHECK_NHWC_WRW_16: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_16: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=28 -in_w=28 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_17

// CHECK_NHWC_WRW_17: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_17: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=7 -in_w=7 -out_channels=2048 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_18

// CHECK_NHWC_WRW_18: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_18: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_19

// CHECK_NHWC_WRW_19: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_19: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=256 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_20

// CHECK_NHWC_WRW_20: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_20: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_21

// CHECK_NHWC_WRW_21: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_21: [1]

// RUN: mlir-miopen-driver -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=256 -in_channels=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d_bwd_weight -pv -c %random_data %xdlop | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NHWC_WRW_22

// CHECK_NHWC_WRW_22: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NHWC_WRW_22: [1]

