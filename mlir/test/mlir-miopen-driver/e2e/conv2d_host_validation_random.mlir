// kcyx_nchw_nkhw
// filter 3x3 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_4

// CHECK_KCYX_NCHW_NKHW_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_4: [1]

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


// group test
// RUN: mlir-miopen-driver -pv -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=4  -batchsize=256 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -rand 1 -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GROUP_NCHW

// GROUP_NCHW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// GROUP_NCHW: [1]

// kyxc_nhwc_nhwk
// filter 1x1 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c -rand 1| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_1

// CHECK_KYXC_NHWC_NHWK_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_1: [1]

// filter 1x1 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -rand 1  -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_2

// CHECK_KYXC_NHWC_NHWK_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_2: [1]

// filter 1x1 dilation=2 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -rand 1  -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_3

// CHECK_KYXC_NHWC_NHWK_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_3: [1]

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

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/146
///////////////////////////////////////////////////////////////////////////////////////////

// COM: yxck/nhwc/nhwk
// 3x3 stride=2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=2 --conv_stride_w=2 -pv -rand 1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_1

// 3x3 stride=3
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=3 --conv_stride_w=3 -pv -rand 1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_2

// 3x3 dilation=2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -pv -rand 1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_3

//5x5 dilation=2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -pv -rand 1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_4

// 3x3 stride_h=2 stride_w=3
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -conv_stride_h=2 -conv_stride_w=3 -dilation_h=1 -dilation_w=2 -pv -rand 1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_5

// COM: kyxc/nhwc/nhwk
// 3x3 dilation=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_6

// 3x3 stride=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_7

// 3x3 dilation=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_8

// 3x3 dilation=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_9

// 5x5 dilation_h=2 dilation_w=1 stride_h=1 stride_w=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=2 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_10

// COM: kcyx/nchw/nkhw
// 3x3 dilation=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_11

//5x5 dilation=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_12

//3x3 stride=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_13

// 3x3 dilation=2 stride=2
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_14

// 3x3 dilation_h=1 dilation_w=2 stride_h=2 stridw_w=3
// RUN: mlir-miopen-driver -pv -rand 1 -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=3 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_146_15

// CHECK_ISSUE_146_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_1: [1]
// CHECK_ISSUE_146_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_2: [1]
// CHECK_ISSUE_146_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_3: [1]
// CHECK_ISSUE_146_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_4: [1]
// CHECK_ISSUE_146_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_5: [1]
// CHECK_ISSUE_146_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_6: [1]
// CHECK_ISSUE_146_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_7: [1]
// CHECK_ISSUE_146_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_8: [1]
// CHECK_ISSUE_146_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_9: [1]
// CHECK_ISSUE_146_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_10: [1]
// CHECK_ISSUE_146_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_11: [1]
// CHECK_ISSUE_146_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_12: [1]
// CHECK_ISSUE_146_13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_13: [1]
// CHECK_ISSUE_146_14: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_14: [1]
// CHECK_ISSUE_146_15: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_146_15: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/155
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=64 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -c -rand 1| mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_155_1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -c -rand 1| mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_155_2

// CHECK_ISSUE_155_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_155_1: [1]
// CHECK_ISSUE_155_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_155_2: [1]
