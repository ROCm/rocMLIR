// COM: gkcyx/ngchw/ngkhw, bwd_data,g=1, n=1, c=2, k=2, input=6x6
// COM: filter=1x1, dilation=1, padding=0, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK1
// CHECK1: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK1-NEXT{LITERAL}:[[[[[2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]],
// CHECK1-NEXT{LITERAL}:   [[2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK1-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]]]]]

// COM: filter=1x1, dilation=2, padding=0, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2
// CHECK2: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK2-NEXT{LITERAL}:[[[[[2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]],
// CHECK2-NEXT{LITERAL}:   [[2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK2-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]]]]]

// COM: filter=1x1, dilation=1, padding=1, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK3
// CHECK3: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK3-NEXT{LITERAL}:[[[[[2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]],
// CHECK3-NEXT{LITERAL}:   [[2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK3-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]]]]]

// COM: filter=1x1, dilation=2, padding=1, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK4
// CHECK4: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK4-NEXT{LITERAL}:[[[[[2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]],
// CHECK4-NEXT{LITERAL}:   [[2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK4-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]]]]]

// COM: filter=1x1, dilation=1, padding=0, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK5
// CHECK5: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK5-NEXT{LITERAL}:[[[[[2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK5-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK5-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]],
// CHECK5-NEXT{LITERAL}:   [[2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK5-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK5-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK5-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]]]]]

// COM: filter=1x1, dilation=2, padding=0, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK6
// CHECK6: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK6-NEXT{LITERAL}:[[[[[2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK6-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK6-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]],
// CHECK6-NEXT{LITERAL}:   [[2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK6-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK6-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK6-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]]]]]

// COM: filter=1x1, dilation=1, padding=1, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7
// CHECK7: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK7-NEXT{LITERAL}:[[[[[0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK7-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK7-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2]],
// CHECK7-NEXT{LITERAL}:   [[0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK7-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK7-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK7-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2]]]]]

// COM: filter=1x1, dilation=2, padding=1, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK8
// CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK8-NEXT{LITERAL}:[[[[[0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2]],
// CHECK8-NEXT{LITERAL}:   [[0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2],
// CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK8-NEXT{LITERAL}:    [0,     2,     0,     2,     0,     2]]]]]

// COM: filter=3x3, dilation=1, padding=0, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK9
// CHECK9: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK9-NEXT{LITERAL}:[[[[[2,     4,     6,     6,     4,     2],
// CHECK9-NEXT{LITERAL}:    [4,     8,     12,     12,     8,     4],
// CHECK9-NEXT{LITERAL}:    [6,     12,     18,     18,     12,     6],
// CHECK9-NEXT{LITERAL}:    [6,     12,     18,     18,     12,     6],
// CHECK9-NEXT{LITERAL}:    [4,     8,     12,     12,     8,     4],
// CHECK9-NEXT{LITERAL}:    [2,     4,     6,     6,     4,     2]],
// CHECK9-NEXT{LITERAL}:   [[2,     4,     6,     6,     4,     2],
// CHECK9-NEXT{LITERAL}:    [4,     8,     12,     12,     8,     4],
// CHECK9-NEXT{LITERAL}:    [6,     12,     18,     18,     12,     6],
// CHECK9-NEXT{LITERAL}:    [6,     12,     18,     18,     12,     6],
// CHECK9-NEXT{LITERAL}:    [4,     8,     12,     12,     8,     4],
// CHECK9-NEXT{LITERAL}:    [2,     4,     6,     6,     4,     2]]]]]

// COM: filter=3x3, dilation=2, padding=0, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK10
// CHECK10: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK10-NEXT{LITERAL}:[[[[[2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]],
// CHECK10-NEXT{LITERAL}:   [[2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2],
// CHECK10-NEXT{LITERAL}:    [2,     2,     2,     2,     2,     2]]]]]

// COM: filter=3x3, dilation=1, padding=1, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK11
// CHECK11: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK11-NEXT{LITERAL}:[[[[[8,     12,     12,     12,     12,     8],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [8,     12,     12,     12,     12,     8]],
// CHECK11-NEXT{LITERAL}:   [[8,     12,     12,     12,     12,     8],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [12,     18,     18,     18,     18,     12],
// CHECK11-NEXT{LITERAL}:    [8,     12,     12,     12,     12,     8]]]]]

// COM: filter=3x3, dilation=2, padding=1, stride=1
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK12
// CHECK12: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK12-NEXT{LITERAL}:[[[[[2,     4,     4,     4,     4,     2],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [2,     4,     4,     4,     4,     2]],
// CHECK12-NEXT{LITERAL}:   [[2,     4,     4,     4,     4,     2],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [4,     8,     8,     8,     8,     4],
// CHECK12-NEXT{LITERAL}:    [2,     4,     4,     4,     4,     2]]]]]

// COM: filter=3x3, dilation=1, padding=0, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK13
// CHECK13: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK13-NEXT{LITERAL}:[[[[[2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [4,     4,     8,     4,     4,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]],
// CHECK13-NEXT{LITERAL}:   [[2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [4,     4,     8,     4,     4,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [2,     2,     4,     2,     2,     0],
// CHECK13-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]]]]]

// COM: filter=3x3, dilation=2, padding=0, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK14
// CHECK14: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK14-NEXT{LITERAL}:[[[[[2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK14-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK14-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]],
// CHECK14-NEXT{LITERAL}:   [[2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK14-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0],
// CHECK14-NEXT{LITERAL}:    [2,     0,     2,     0,     2,     0],
// CHECK14-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0]]]]]

// COM: filter=3x3, dilation=1, padding=1, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK15
// CHECK15: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK15-NEXT{LITERAL}:[[[[[2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [4,     8,     4,     8,     4,     4],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [4,     8,     4,     8,     4,     4],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2]],
// CHECK15-NEXT{LITERAL}:   [[2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [4,     8,     4,     8,     4,     4],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [4,     8,     4,     8,     4,     4],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2],
// CHECK15-NEXT{LITERAL}:    [2,     4,     2,     4,     2,     2]]]]]

// COM: filter=3x3, dilation=2, padding=1, stride=2
// RUN: miopen-gen -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=6 -in_w=6 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_data -p=false | mlir-miopen-driver -c | mlir-rocm-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK16
// CHECK16: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 1, 2, 6, 6] strides = [72, 72, 36, 6, 1] data =
// CHECK16-NEXT{LITERAY}:[[[[[0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     8,     0,     8,     0,     4],
// CHECK16-NEXT{LITERAY}:    [0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     8,     0,     8,     0,     4],
// CHECK16-NEXT{LITERAY}:    [0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     4,     0,     4,     0,     2]],
// CHECK16-NEXT{LITERAY}:   [[0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     8,     0,     8,     0,     4],
// CHECK16-NEXT{LITERAY}:    [0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     8,     0,     8,     0,     4],
// CHECK16-NEXT{LITERAY}:    [0,     0,     0,     0,     0,     0],
// CHECK16-NEXT{LITERAY}:    [0,     4,     0,     4,     0,     2]]]]]
