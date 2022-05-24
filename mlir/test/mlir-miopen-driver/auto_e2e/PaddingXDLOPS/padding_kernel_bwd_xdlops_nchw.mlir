// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=7 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_STRIDE1_CONFIG1

// CHECK_PADDING_GEMMM_STRIDE1_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_STRIDE1_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=17 -groupsize=1 -in_channels=64 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_STRIDE1_CONFIG1

// CHECK_PADDING_GEMMN_STRIDE1_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMN_STRIDE1_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=64 -out_channels=17 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMK_STRIDE1_CONFIG1

// CHECK_PADDING_GEMMK_STRIDE1_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMK_STRIDE1_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=17 -groupsize=1 -in_channels=13 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_GEMMN_STRIDE1_CONFIG1

// CHECK_PADDING_GEMMM_GEMMN_STRIDE1_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_GEMMN_STRIDE1_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=13 -out_channels=17 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_GEMMK_STRIDE1_CONFIG1

// CHECK_PADDING_GEMMM_GEMMK_STRIDE1_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_GEMMK_STRIDE1_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=13 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_STRIDE2_CONFIG1

// CHECK_PADDING_GEMMM_STRIDE2_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_STRIDE2_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=64 -out_channels=13 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMK_STRIDE2_CONFIG1

// CHECK_PADDING_GEMMK_STRIDE2_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMK_STRIDE2_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=13 -groupsize=1 -in_channels=64 -out_channels=64 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMN_STRIDE2_CONFIG1

// CHECK_PADDING_GEMMN_STRIDE2_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMN_STRIDE2_CONFIG1: [1]

// RUN: miopen-gen --operation conv2d_bwd_data -t f32 -p=false -fil_layout=gkcyx -in_layout=ngchw -out_layout=ngkhw -batchsize=64 -groupsize=1 -in_channels=13 -out_channels=17 -in_h=11 -in_w=11 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 %pv %random_data -x2 | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_PADDING_GEMMM_GEMMK_STRIDE2_CONFIG1

// CHECK_PADDING_GEMMM_GEMMK_STRIDE2_CONFIG1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_PADDING_GEMMM_GEMMK_STRIDE2_CONFIG1: [1]
