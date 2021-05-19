// COM: yxck/nhwc/nhwk
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -p -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK1
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK3
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK4
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK5
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=5 -fil_w=5 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK6
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK7
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK8
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --conv_stride_h=2 --conv_stride_w=2 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK9
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128  -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=3 --conv_stride_w=3 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK10
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK11
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK12
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -conv_stride_h=2 -conv_stride_w=3 -dilation_h=1 -dilation_w=2 -pv %random_data %xdlops -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK13
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK13
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK14

// COM: kyxc/nhwc/nhwk
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK4
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK5
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK6
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK7
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK9

// COM: kcyx/nchw/nkhw
// RUN: mlir-miopen-driver -pv %random_data %xdlops -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW4
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW5
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW6
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW7
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=3 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW9
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW10

// CHECK_YXCK_NHWC_NHWK1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK1: [1]
// CHECK_YXCK_NHWC_NHWK2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK2: [1]
// CHECK_YXCK_NHWC_NHWK3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK3: [1]
// CHECK_YXCK_NHWC_NHWK4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK4: [1]
// CHECK_YXCK_NHWC_NHWK5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK5: [1]
// CHECK_YXCK_NHWC_NHWK6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK6: [1]
// CHECK_YXCK_NHWC_NHWK7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK7: [1]
// CHECK_YXCK_NHWC_NHWK8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK8: [1]
// CHECK_YXCK_NHWC_NHWK9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK9: [1]
// CHECK_YXCK_NHWC_NHWK10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK10: [1]
// CHECK_YXCK_NHWC_NHWK11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK11: [1]
// CHECK_YXCK_NHWC_NHWK12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK12: [1]
// CHECK_YXCK_NHWC_NHWK13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK13: [1]
// CHECK_YXCK_NHWC_NHWK14: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_YXCK_NHWC_NHWK14: [1]

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
// CHECK_KYXC_NHWC_NHWK8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK8: [1]
// CHECK_KYXC_NHWC_NHWK9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK9: [1]

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
// CHECK_KCYX_NCHW_NKHW8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW8: [1]
// CHECK_KCYX_NCHW_NKHW9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW9: [1]
// CHECK_KCYX_NCHW_NKHW10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW10: [1]

// kcyx_nchw_nkhw
// filter 3x3 f32
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=256 -in_channels=64 -out_channels=64 -in_h=56 -in_w=56 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW_4

// CHECK_KCYX_NCHW_NKHW_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KCYX_NCHW_NKHW_4: [1]

// COM: padding=1
// RUN:  mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING1

// CHECK_PADDING1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING1: [1]

// filter 1x1 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false   -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_2

// CHECK_KYXC_NHWC_NHWK_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_2: [1]

// filter 1x1 dilation=2 f32
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false   -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK_3

// CHECK_KYXC_NHWC_NHWK_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_KYXC_NHWC_NHWK_3: [1]

// COM: padding=1
// RUN:  mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING2

// CHECK_PADDING2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING2: [1]

// COM: padding=1
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 4 --in_h 4 --in_w 4 --out_channels 32 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING3

// CHECK_PADDING3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING3: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER1

// CHECK_FILTER1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER1: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER2

// CHECK_FILTER2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER2: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/41
///////////////////////////////////////////////////////////////////////////////////////////

// Various dimension values
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_4
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=16 -batchsize=64 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_5
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=64 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_6

// Various padding values
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_7
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_8

// Various dilation values
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_11
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_12

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
// CHECK_ISSUE_41_11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_11: [1]
// CHECK_ISSUE_41_12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_41_12: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/114
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_4
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_5
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_6
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_7
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_8

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

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3  -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_3
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_4

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

// NHWC 3x3 padding cases.

// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_3
// RUN: mlir-miopen-driver -pv %random_data %xdlops --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_4

// CHECK_ISSUE_127_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_1: [1]
// CHECK_ISSUE_127_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_2: [1]
// CHECK_ISSUE_127_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_3: [1]
// CHECK_ISSUE_127_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_127_4: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/40
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_5

// CHECK_ISSUE_40_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_40_5: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/136
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=32 -in_channels=1 -out_channels=64 -in_h=14 -in_w=14 -fil_h=14 -fil_w=14 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_136_2
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=32 -in_channels=1 -out_channels=32 -in_h=14 -in_w=14 -fil_h=14 -fil_w=14 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_136_3
// FIXME: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=16 -in_channels=1 -out_channels=16 -in_h=14 -in_w=14 -fil_h=14 -fil_w=14 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_136_4

// CHECK_ISSUE_136_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_136_2: [1]
// CHECK_ISSUE_136_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_136_3: [1]
// CHECK_ISSUE_136_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_136_4: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/155
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=4 -out_channels=64 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_155_1
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=4 -in_w=4 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_155_2

// CHECK_ISSUE_155_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_155_1: [1]
// CHECK_ISSUE_155_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_ISSUE_155_2: [1]

// group test
// RUN: mlir-miopen-driver -pv %random_data %xdlops -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=4  -batchsize=256 -in_channels=256 -out_channels=256 -in_h=56 -in_w=56 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16  -c | mlir-rocm-runner  --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=GROUP_NCHW

// GROUP_NCHW: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// GROUP_NCHW: [1]
