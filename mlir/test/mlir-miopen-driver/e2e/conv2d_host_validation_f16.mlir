// COM: kyxc/nhwc/nhwk, fwd, filter=3x3
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -p -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK1

// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2

// COM: dilation=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK3

// COM: stride=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK4

// COM: dilation=2 stride=2
// RUN: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK5


// COM: kcyx/nchw/nkhw
// RUN: mlir-miopen-driver -pv -p -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK6

// COM: dilation=2
// COM: filter 3x3
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7

// COM: dilation=2
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK8

// COM: stride=2 
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK9

// COM: dilation=2 stride=2 
// RUN: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK10

// CHECK1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK1: [1]
// CHECK2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK2: [1]
// CHECK3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK3: [1]
// CHECK4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK4: [1]
// CHECK5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK5: [1]
// CHECK6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK6: [1]
// CHECK7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK7: [1]
// CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK8: [1]
// CHECK9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK9: [1]
// CHECK10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK10: [1]

// COM: padding=1
// FIXME:  mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK

// COM: padding=1
// FIXME:  mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=14 -in_w=14 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK

// COM: filter 1x1
// FIXME: mlir-miopen-driver -pv -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2

// COM: filter 1x1
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7

// COM: bwd_weight
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 --operation=conv2d_bwd_weight -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7

// COM: bwd_data
// FIXME: mlir-miopen-driver -pv -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=32 -in_w=32 -fil_h=1 -fil_w=1   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -t f16 --operation=conv2d_bwd_data -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7

