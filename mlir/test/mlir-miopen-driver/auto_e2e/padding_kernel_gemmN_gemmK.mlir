// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=1 -batchsize=5 --padding_h=0 --padding_w=0 -in_channels=13 -out_channels=11 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 -p=false --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --target=%targets --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_GEMMK_CONFIG1

// CHECK_PADDING_GEMMN_GEMMK_CONFIG1: [1 1 1]

// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=1 -batchsize=15 --padding_h=1 --padding_w=1 -in_channels=13 -out_channels=11 -in_h=7 -in_w=7 -fil_h=3 -fil_w=3 -p=false --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --target=%targets --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_GEMMK_CONFIG2

// CHECK_PADDING_GEMMN_GEMMK_CONFIG2: [1 1 1]

// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=1 -batchsize=64 -in_channels=15 -out_channels=64 --padding_h=0 --padding_w=0 -in_h=5 -in_w=5 -fil_h=3 -fil_w=3 -p=false --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --target=%targets --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_GEMMK_CONFIG3

// CHECK_PADDING_GEMMN_GEMMK_CONFIG3: [1 1 1]

// RUN: miopen-gen -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -groupsize=1 -batchsize=15 -in_channels=64 -out_channels=64 --padding_h=0 --padding_w=0 -in_h=5 -in_w=5 -fil_h=3 -fil_w=3 -p=false --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data %pv %random_data %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --target=%targets --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_GEMMK_CONFIG4

// CHECK_PADDING_GEMMN_GEMMK_CONFIG4: [1 1 1]
