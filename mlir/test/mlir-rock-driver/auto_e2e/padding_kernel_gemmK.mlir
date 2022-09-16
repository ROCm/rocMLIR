// RUN: rock-gen --operation conv2d -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=1 -in_channels=3 -out_channels=64 -in_h=230 -in_w=230 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data %xdlops | mlir-rock-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG2

// CHECK_RESNET50_CONFIG2: [1 1 1]

// RUN: rock-gen --operation conv2d -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=1 -in_channels=3 -out_channels=64 -in_h=230 -in_w=230 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data %xdlops | mlir-rock-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_CONFIG4

// CHECK_RESNET50_CONFIG4: [1 1 1]

// RUN: rock-gen --operation conv2d -t f16 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=1 -in_channels=3 -out_channels=64 -in_h=224 -in_w=224 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=3 --padding_w=3 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data %xdlops --rand_type float | mlir-rock-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_F16_CONFIG1

// CHECK_RESNET50_F16_CONFIG1: [1 1 1]

// RUN: rock-gen --operation conv2d -t f16 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=256 -groupsize=1 -in_channels=3 -out_channels=64 -in_h=230 -in_w=230 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data %xdlops --rand_type float | mlir-rock-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_F16_CONFIG2

// CHECK_RESNET50_F16_CONFIG2: [1 1 1]

// RUN: rock-gen --operation conv2d -t f16 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=256 -groupsize=1 -in_channels=3 -out_channels=64 -in_h=230 -in_w=230 -fil_h=7 -fil_w=7 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data %xdlops --rand_type float | mlir-rock-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_RESNET50_F16_CONFIG4

// CHECK_RESNET50_F16_CONFIG4: [1 1 1]
