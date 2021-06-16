// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=20 -groupsize=1 -in_channels=3 -out_channels=6 -in_h=32 -in_w=32 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2 -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG1

// CHECK_RESNET50_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET50_CONFIG1: [1]

// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f16 -p=false  -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=20 -groupsize=1 -in_channels=3 -out_channels=6 -in_h=32 -in_w=32 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2 -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG2

// CHECK_RESNET50_CONFIG2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET50_CONFIG2: [1]

// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f32 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=20 -groupsize=1 -in_channels=3 -out_channels=6 -in_h=32 -in_w=32 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2 -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG3

// CHECK_RESNET50_CONFIG3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET50_CONFIG3: [1]

// RUN: mlir-miopen-driver --operation=conv2d_bwd_weight -t f16 -p=false  -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=20 -groupsize=1 -in_channels=3 -out_channels=6 -in_h=32 -in_w=32 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2 -pv %random_data %xdlops -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG4

// CHECK_RESNET50_CONFIG4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_RESNET50_CONFIG4: [1]


