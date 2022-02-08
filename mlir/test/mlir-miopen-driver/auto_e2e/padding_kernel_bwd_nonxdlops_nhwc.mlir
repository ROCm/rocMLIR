// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=64 -groupsize=1 -in_channels=7 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_CONFIG1

// CHECK_PADDING_GEMMM_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=19 -groupsize=1 -in_channels=64 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_CONFIG1

// CHECK_PADDING_GEMMN_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMN_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=64 -groupsize=1 -in_channels=64 -out_channels=19 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMK_CONFIG1

// CHECK_PADDING_GEMMK_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMK_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=19 -groupsize=1 -in_channels=7 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_GEMMN_CONFIG1

// CHECK_PADDING_GEMMM_GEMMN_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_GEMMN_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=64 -groupsize=1 -in_channels=7 -out_channels=19 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_GEMMK_CONFIG1

// CHECK_PADDING_GEMMM_GEMMK_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_GEMMK_CONFIG1: [1]
