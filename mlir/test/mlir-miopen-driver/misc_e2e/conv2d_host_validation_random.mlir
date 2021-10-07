// filter 1x1 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_5

// CHECK_KCYX_NCHW_NKHW_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_5: [1]

// filter 1x1 dilation=2 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_6

// CHECK_KCYX_NCHW_NKHW_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_6: [1]

// filter 3x3 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_7

// CHECK_KCYX_NCHW_NKHW_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_7: [1]

// filter 3x3 padding=1 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_8

// CHECK_KCYX_NCHW_NKHW_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_8: [1]


// filter 1x1 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1  -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_4

// CHECK_KYXC_NHWC_NHWK_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_4: [1]

// filter 1x1 dilation=2 f16
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1  -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_5

// CHECK_KYXC_NHWC_NHWK_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_5: [1]

// group test
// RUN: mlir-miopen-driver -pv -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -groupsize=4  -batchsize=256 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GROUP_NHWC

// GROUP_NHWC: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// GROUP_NHWC: [1]

// RUN: mlir-miopen-driver -pv -p -rand 1 -rand_side filter -c| mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=FWD_FILTER

// FWD_FILTER: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// FWD_FILTER: [1]

// RUN: mlir-miopen-driver -pv -p -rand 1 -rand_side input -c| mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=FWD_INPUT

// FWD_INPUT: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// FWD_INPUT: [1]


// Use random float numbers

// RUN: mlir-miopen-driver -pv -p -rand 1 -rand_type float -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RAND_FLOAT_1

// CHECK_RAND_FLOAT_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RAND_FLOAT_1: [1]

// RUN: mlir-miopen-driver -pv -p -t f16 -rand 1 -rand_type float -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RAND_FLOAT_2

// CHECK_RAND_FLOAT_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RAND_FLOAT_2: [1]
