
// nchw f16 fwd
// -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_1

// CHECK_NCHW_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_1: [1]

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_2

// CHECK_NCHW_2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_2: [1]

// nchw f16 bwd_weight
// -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f16 --operation conv2d_bwd_weight %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_3

// CHECK_NCHW_3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_3: [1]

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=64 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f16 --operation conv2d_bwd_weight %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_4

// CHECK_NCHW_4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_4: [1]


// nchw f32 fwd
// -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation conv2d %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_5

// CHECK_NCHW_5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_5: [1]

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f32 --operation conv2d %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_6

// CHECK_NCHW_6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_6: [1]

// nchw f32 wrw
// -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation conv2d_bwd_weight %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_7

// CHECK_NCHW_7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_7: [1]

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f32 --operation conv2d_bwd_weight %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_8

// CHECK_NCHW_8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_8: [1]

// nchw f32 bwd
// -n 256 -c 128 -H 28 -W 28 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=128 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=1 --padding_w=1 -t f32 --operation conv2d_bwd_data %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_9

// CHECK_NCHW_9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_9: [1]

// -n 256 -c 128 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1
// RUN: miopen-gen -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=256 -in_channels=128 -in_h=28 -in_w=28 -out_channels=512 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --conv_stride_h=1 --conv_stride_w=1 --padding_h=0 --padding_w=0 -t f32 --operation conv2d_bwd_data %pv %xdlops | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_NCHW_10

// CHECK_NCHW_10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_NCHW_10: [1]
