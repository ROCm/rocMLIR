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
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK13
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout yxck --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_YXCK_NHWC_NHWK14

// COM: kyxc/nhwc/nhwk
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK4
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK5
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK6
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK7
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kyxc --in_layout nhwc --out_layout nhwk --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KYXC_NHWC_NHWK9

// COM: kcyx/nchw/nkhw
// RUN: mlir-miopen-driver -pv -p  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW1
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW2
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW3
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW4
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW5
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW6
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW7
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=3 -p=false  -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW8
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 2 --fil_h 2 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW9
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 128 --in_channels 8 --in_h 32 --in_w 32 --out_channels 128 --fil_w 4 --fil_h 4 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 0 --padding_w 0 -p=false -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_KCYX_NCHW_NKHW10

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

// COM: padding=1
// RUN:  mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING1

// CHECK_PADDING1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING1: [1]

// COM: padding=1
// RUN:  mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING2

// CHECK_PADDING2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING2: [1]

// COM: padding=1
// RUN: mlir-miopen-driver --operation conv2d -t f32 --fil_layout kcyx --in_layout nchw --out_layout nkhw --batchsize 4 --in_channels 4 --in_h 4 --in_w 4 --out_channels 32 --fil_w 3 --fil_h 3 --dilation_h 1 --dilation_w 1 --conv_stride_h 1 --conv_stride_w 1 --padding_h 1 --padding_w 1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING3

// CHECK_PADDING3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING3: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER1

// CHECK_FILTER1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER1: [1]

// COM: filter 1x1
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_FILTER2

// CHECK_FILTER2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_FILTER2: [1]

// COM: bwd_weight
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=32 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BWD_WEIGHT1

// CHECK_BWD_WEIGHT1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_WEIGHT1: [1]

// COM: bwd_data
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f32 --operation=conv2d_bwd_data -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_BWD_DATA1

// CHECK_BWD_DATA1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_BWD_DATA1: [1]

///////////////////////////////////////////////////////////////////////////////////////////
// Cases reported in https://github.com/ROCmSoftwarePlatform/llvm-project-private/issues/41
///////////////////////////////////////////////////////////////////////////////////////////

// Various dimension values
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_1
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_2
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_3
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=16 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_4
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=16 -batchsize=64 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_5
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=64 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_6

// Various padding values
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_7
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_8

// Various dilation values
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_11
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_41_12

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

// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_1
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_4
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_5
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_6
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_7
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_114_8

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

// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -in_channels=8 -batchsize=128 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3  -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_1
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=16 -fil_h=3 -fil_w=3 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_2
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=16 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_3
// RUN: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=5 -fil_w=5 --padding_h=1 --padding_w=1 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_4

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

// RUN: mlir-miopen-driver -pv --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_1
// RUN: mlir-miopen-driver -pv --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_2
// RUN: mlir-miopen-driver -pv --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_3
// RUN: mlir-miopen-driver -pv --operation conv2d -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_4
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_5
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=256 -in_h=14 -in_w=14 -out_channels=256 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_6
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=512 -in_h=7 -in_w=7 -out_channels=512 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_7
// RUN: mlir-miopen-driver -pv --operation conv2d_bwd_weight -t f32 -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -in_channels=256 -batchsize=64 -in_h=56 -in_w=56 -out_channels=64 -fil_h=3 -fil_w=3 -dilation_h=1 -dilation_w=1 -conv_stride_h=1 -conv_stride_w=1 -padding_h=1 -padding_w=1 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_127_8

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

///////////////////////////////////////////////////////////////////////////////////////////
// Cases remained to be studied.
///////////////////////////////////////////////////////////////////////////////////////////

// FIXME: mlir-miopen-driver -pv -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=5 -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_ISSUE_40_5

///////////////////////////////////////////////////////////////////////////////////////////
// Resnet101 NCHW
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_FWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_BWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32  -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG1_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NCHW_CONFIG2_WRW

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

///////////////////////////////////////////////////////////////////////////////////////////
// Resnet101 NHWC
///////////////////////////////////////////////////////////////////////////////////////////

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_FWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_BWD

// FIXME: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32  -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG1_WRW

// RUN: mlir-miopen-driver --operation conv2d -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_FWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_data -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_BWD

// FIXME: mlir-miopen-driver --operation conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=32 -in_channels=1024 -out_channels=1024 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET101_NHWC_CONFIG2_WRW

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
