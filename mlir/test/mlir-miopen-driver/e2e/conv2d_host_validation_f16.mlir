// COM: kyxc/nhwc/nhwk, fwd, filter=3x3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -p -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK1

// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK2

// COM: dilation=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK3

// COM: stride=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK4

// COM: dilation=2 stride=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK5

// COM: fil_h=2 fil_w=2
// RUN: mlir-miopen-driver --operation conv2d -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK6

// COM: fil_h=4 fil_w=4
// RUN: mlir-miopen-driver --operation conv2d -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK7


// COM: kcyx/nchw/nkhw
// RUN: mlir-miopen-driver -pv -p -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW1

// COM: dilation=2
// COM: filter 3x3
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW2

// COM: dilation=2
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW3

// COM: stride=2 
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW4

// COM: dilation=2 stride=2 
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW5

// COM: fil_h=2 fil_w=2
// RUN: mlir-miopen-driver --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW6

// COM: fil_h=4 fil_w=4
// RUN: mlir-miopen-driver --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW7


// CHECK_KYXC_NHWC_NHWK1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK1: [1]
// CHECK_KYXC_NHWC_NHWK2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK2: [1]
// CHECK_KYXC_NHWC_NHWK3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK3: [1]
// CHECK_KYXC_NHWC_NHWK4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK4: [1]
// CHECK_KYXC_NHWC_NHWK5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK5: [1]
// CHECK_KYXC_NHWC_NHWK6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK6: [1]
// CHECK_KYXC_NHWC_NHWK7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK7: [1]

// CHECK_KCYX_NCHW_NKHW1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW1: [1]
// CHECK_KCYX_NCHW_NKHW2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW2: [1]
// CHECK_KCYX_NCHW_NKHW3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW3: [1]
// CHECK_KCYX_NCHW_NKHW4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW4: [1]
// CHECK_KCYX_NCHW_NKHW5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW5: [1]
// CHECK_KCYX_NCHW_NKHW6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW6: [1]
// CHECK_KCYX_NCHW_NKHW7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW7: [1]

// COM: padding=1
// RUN:  mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING1

// CHECK_PADDING1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING1: [1]

// COM: padding=1
// RUN:  mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING2

// CHECK_PADDING2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING2: [1]

// COM: padding=1
// RUN: mlir-miopen-driver --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 4 --in_h 4 --in_w 4 --out_channels 32 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING3

// CHECK_PADDING3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING3: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER1

// CHECK_FILTER1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER1: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER2

// CHECK_FILTER2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER2: [1]

// COM: bwd_weight
// FIXME: this config suffers from deterministically producing 2048, instead of desired 12544.
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64  -in_h=7 -in_w=7 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16  --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BWD_WEIGHT1

// CHECK_BWD_WEIGHT1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_WEIGHT1: [1]

// COM: bwd_data
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64  -in_h=7 -in_w=7 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16  --operation=conv2d_bwd_data -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7

// CHECK_BWD_DATA1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_DATA1: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/41
///////////////////////////////////////////////////////////////////////////////////////////

// Various dimension values
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_1
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_2
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_3
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_4
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=16 -batchsize=64 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_5
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=64 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_6

// Various padding values
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_7
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_8

// Various dilation values
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_9
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_10
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_11
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils.so --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_12

// CHECK_ISSUE_41_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_1: [1]
// CHECK_ISSUE_41_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_2: [1]
// CHECK_ISSUE_41_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_3: [1]
// CHECK_ISSUE_41_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_4: [1]
// CHECK_ISSUE_41_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_5: [1]
// CHECK_ISSUE_41_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_6: [1]
// CHECK_ISSUE_41_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_7: [1]
// CHECK_ISSUE_41_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_8: [1]
// CHECK_ISSUE_41_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_9: [1]
// CHECK_ISSUE_41_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_10: [1]
// CHECK_ISSUE_41_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_11: [1]
// CHECK_ISSUE_41_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_12: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/114
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_4
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_5
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_6
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_7
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -t f16 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_8

// CHECK_ISSUE_114_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_1: [1]
// CHECK_ISSUE_114_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_2: [1]
// CHECK_ISSUE_114_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_3: [1]
// CHECK_ISSUE_114_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_4: [1]
// CHECK_ISSUE_114_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_5: [1]
// CHECK_ISSUE_114_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_6: [1]
// CHECK_ISSUE_114_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_7: [1]
// CHECK_ISSUE_114_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_114_8: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/40
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3  -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_1
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_2
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_3
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_4

// CHECK_ISSUE_40_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_40_1: [1]
// CHECK_ISSUE_40_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_40_2: [1]
// CHECK_ISSUE_40_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_40_3: [1]
// CHECK_ISSUE_40_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_40_4: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/127
///////////////////////////////////////////////////////////////////////////////////////////

// Foward cases.

// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 256 --in_channels 256 --in_h 30 --in_w 30 --out_channels 256 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_1
// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 256 --in_channels 512 --in_h 16 --in_w 16 --out_channels 512 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_2
// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 58 --in_w 58 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_3
// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 58 --in_w 58 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_3
// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 256 --in_h 30 --in_w 30 --out_channels 256 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_4
// RUN: mlir-miopen-driver -pv -c --operation conv2d -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 256 --in_h 14 --in_w 14 --out_channels 256 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_5

// Backward weight cases.
// RUN: mlir-miopen-driver -pv -c --operation conv2d_bwd_weight -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 256 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 2048 --fil_w 1 --fil_h 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_6
// RUN: mlir-miopen-driver -pv -c --operation conv2d_bwd_weight -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 256 --in_channels 1024 --in_h 14 --in_w 14 --out_channels 256 --fil_w 1 --fil_h 1 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_7
// FIXME: mlir-miopen-driver -pv -c --operation conv2d_bwd_weight -t f16 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 256 --in_channels 128 --in_h 58 --in_w 58 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_8
// RUN: mlir-miopen-driver -pv -c --operation conv2d_bwd_weight -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 28 --in_w 28 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_9
// RUN: mlir-miopen-driver -pv -c --operation conv2d_bwd_weight -t f16 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 256 --in_channels 128 --in_h 58 --in_w 58 --out_channels 128 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 2 --conv_stride_w 2 --padding_h 0 --padding_w 0  | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_10

// CHECK_ISSUE_127_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_1: [1]
// CHECK_ISSUE_127_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_2: [1]
// CHECK_ISSUE_127_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_3: [1]
// CHECK_ISSUE_127_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_4: [1]
// CHECK_ISSUE_127_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_5: [1]
// CHECK_ISSUE_127_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_6: [1]
// CHECK_ISSUE_127_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_7: [1]
// CHECK_ISSUE_127_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_8: [1]
// CHECK_ISSUE_127_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_9: [1]
// CHECK_ISSUE_127_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_10: [1]

// NHWC 3x3 padding cases.

// RUN: mlir-miopen-driver -pv --operation conv2d -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_11
// RUN: mlir-miopen-driver -pv --operation conv2d -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_12
// RUN: mlir-miopen-driver -pv --operation conv2d -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_13
// RUN: mlir-miopen-driver -pv --operation conv2d -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_14
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_15
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_16
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_17
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f16 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_18

// CHECK_ISSUE_127_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_11: [1]
// CHECK_ISSUE_127_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_12: [1]
// CHECK_ISSUE_127_13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_13: [1]
// CHECK_ISSUE_127_14: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_14: [1]
// CHECK_ISSUE_127_15: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_15: [1]
// CHECK_ISSUE_127_16: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_16: [1]
// CHECK_ISSUE_127_17: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_17: [1]
// CHECK_ISSUE_127_18: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_18: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/71
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=32 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_1
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=7 -in_w=7 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_2
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_3
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_71_4

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

// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=32 -out_channels=32 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_1
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_2
// RUN: mlir-miopen-driver -t f16 -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=64 -out_channels=64 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_70_3

// CHECK_ISSUE_70_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_1: [1]
// CHECK_ISSUE_70_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_2: [1]
// CHECK_ISSUE_70_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_70_3: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases remained to be studied.
///////////////////////////////////////////////////////////////////////////////////////////

// FIXME: mlir-miopen-driver -t f16 -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_5
