// COM: yxck/nhwc/nhwk
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -p -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK1
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK3
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK4
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK5
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=5 -fil_w=5 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK6
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK7
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK_YXCK_NHWC_NHWK8
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --conv_stride_h=2 --conv_stride_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK9
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128  -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=3 --conv_stride_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK10
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK11
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK12
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -conv_stride_h=2 -conv_stride_w=3 -dilation_h=1 -dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK13
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK13
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK14

// COM: kyxc/nhwc/nhwk
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK4
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK5
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK6
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK7
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK9

// COM: kcyx/nchw/nkhw
// RUN: mlir-miopen-driver -pv -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW1
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW2
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW3
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW4
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW5
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW6
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW7
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=3 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW9
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --c onv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-ro cm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW10

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

