// COM: kcyx/nchw/nkhw, bwd_weight, n=1, c=2, k=2, input=8x8
// COM: filter=1x1, dilation=1, padding=0, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK1
// CHECK1: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK1-NEXT{LITERAL}:[[[[[64]],
// CHECK1-NEXT{LITERAL}:    [[64]]],
// CHECK1-NEXT{LITERAL}:  [[[64]],
// CHECK1-NEXT{LITERAL}:    [[64]]]]]

// COM: filter=1x1, dilation=2, padding=0, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2
// CHECK2: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK2-NEXT{LITERAL}:[[[[[64]],
// CHECK2-NEXT{LITERAL}:    [[64]]],
// CHECK2-NEXT{LITERAL}:  [[[64]],
// CHECK2-NEXT{LITERAL}:    [[64]]]]]

// COM: filter=1x1, dilation=1, padding=1, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK3
// CHECK3: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK3-NEXT{LITERAL}:[[[[[64]],
// CHECK3-NEXT{LITERAL}:    [[64]]],
// CHECK3-NEXT{LITERAL}:  [[[64]],
// CHECK3-NEXT{LITERAL}:    [[64]]]]]

// COM: filter=1x1, dilation=2, padding=1, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK4
// CHECK4: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK4-NEXT{LITERAL}:[[[[[64]],
// CHECK4-NEXT{LITERAL}:    [[64]]],
// CHECK4-NEXT{LITERAL}:  [[[64]],
// CHECK4-NEXT{LITERAL}:    [[64]]]]]

// COM: filter=1x1, dilation=1, padding=0, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK5
// CHECK5: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK5-NEXT{LITERAL}:[[[[[16]],
// CHECK5-NEXT{LITERAL}:    [[16]]],
// CHECK5-NEXT{LITERAL}:  [[[16]],
// CHECK5-NEXT{LITERAL}:    [[16]]]]]

// COM: filter=1x1, dilation=2, padding=0, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK6
// CHECK6: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK6-NEXT{LITERAL}:[[[[[16]],
// CHECK6-NEXT{LITERAL}:    [[16]]],
// CHECK6-NEXT{LITERAL}:  [[[16]],
// CHECK6-NEXT{LITERAL}:    [[16]]]]]

// COM: filter=1x1, dilation=1, padding=1, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7
// CHECK7: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK7-NEXT{LITERAL}:[[[[[16]],
// CHECK7-NEXT{LITERAL}:    [[16]]],
// CHECK7-NEXT{LITERAL}:  [[[16]],
// CHECK7-NEXT{LITERAL}:    [[16]]]]]

// COM: filter=1x1, dilation=2, padding=1, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=1 -fil_w=1 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK8
// CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 1, 1] strides = [4, 2, 1, 1, 1] data =
// CHECK8-NEXT{LITERAL}:[[[[[16]],
// CHECK8-NEXT{LITERAL}:    [[16]]],
// CHECK8-NEXT{LITERAL}:  [[[16]],
// CHECK8-NEXT{LITERAL}:    [[16]]]]]

// COM: filter=3x3, dilation=1, padding=0, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK9
// CHECK9: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK9-NEXT{LITERAL}:[[[[[36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36]],
// CHECK9-NEXT{LITERAL}:    [[36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36]]],
// CHECK9-NEXT{LITERAL}:  [[[36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36]],
// CHECK9-NEXT{LITERAL}:    [[36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36],
// CHECK9-NEXT{LITERAL}:    [36,     36,     36]]]]]

// COM: filter=3x3, dilation=2, padding=0, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK10
// CHECK10: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK10-NEXT{LITERAL}:[[[[[16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16]],
// CHECK10-NEXT{LITERAL}:    [[16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16]]],
// CHECK10-NEXT{LITERAL}:  [[[16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16]],
// CHECK10-NEXT{LITERAL}:    [[16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16],
// CHECK10-NEXT{LITERAL}:    [16,     16,     16]]]]]

// COM: filter=3x3, dilation=1, padding=1, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK11
// CHECK11: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK11-NEXT{LITERAL}:[[[[[49,     56,     49],
// CHECK11-NEXT{LITERAL}:    [56,     64,     56],
// CHECK11-NEXT{LITERAL}:    [49,     56,     49]],
// CHECK11-NEXT{LITERAL}:    [[49,     56,     49],
// CHECK11-NEXT{LITERAL}:    [56,     64,     56],
// CHECK11-NEXT{LITERAL}:    [49,     56,     49]]],
// CHECK11-NEXT{LITERAL}:  [[[49,     56,     49],
// CHECK11-NEXT{LITERAL}:    [56,     64,     56],
// CHECK11-NEXT{LITERAL}:    [49,     56,     49]],
// CHECK11-NEXT{LITERAL}:    [[49,     56,     49],
// CHECK11-NEXT{LITERAL}:    [56,     64,     56],
// CHECK11-NEXT{LITERAL}:    [49,     56,     49]]]]]

// COM: filter=3x3, dilation=2, padding=1, stride=1
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=1 --conv_stride_w=1 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK12
// CHECK12: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK12-NEXT{LITERAL}:[[[[[25,     30,     25],
// CHECK12-NEXT{LITERAL}:    [30,     36,     30],
// CHECK12-NEXT{LITERAL}:    [25,     30,     25]],
// CHECK12-NEXT{LITERAL}:    [[25,     30,     25],
// CHECK12-NEXT{LITERAL}:    [30,     36,     30],
// CHECK12-NEXT{LITERAL}:    [25,     30,     25]]],
// CHECK12-NEXT{LITERAL}:  [[[25,     30,     25],
// CHECK12-NEXT{LITERAL}:    [30,     36,     30],
// CHECK12-NEXT{LITERAL}:    [25,     30,     25]],
// CHECK12-NEXT{LITERAL}:    [[25,     30,     25],
// CHECK12-NEXT{LITERAL}:    [30,     36,     30],
// CHECK12-NEXT{LITERAL}:    [25,     30,     25]]]]]

// COM: filter=3x3, dilation=1, padding=0, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK13
// CHECK13: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK13-NEXT{LITERAL}:[[[[[9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9]],
// CHECK13-NEXT{LITERAL}:    [[9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9]]],
// CHECK13-NEXT{LITERAL}:  [[[9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9]],
// CHECK13-NEXT{LITERAL}:    [[9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9],
// CHECK13-NEXT{LITERAL}:    [9,     9,     9]]]]]

// COM: filter=3x3, dilation=2, padding=0, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK14
// CHECK14: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK14-NEXT{LITERAL}:[[[[[4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4]],
// CHECK14-NEXT{LITERAL}:    [[4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4]]],
// CHECK14-NEXT{LITERAL}:  [[[4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4]],
// CHECK14-NEXT{LITERAL}:    [[4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4],
// CHECK14-NEXT{LITERAL}:    [4,     4,     4]]]]]

// COM: filter=3x3, dilation=1, padding=1, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK15
// CHECK15: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK15-NEXT{LITERAL}:[[[[[9,     12,     12],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16]],
// CHECK15-NEXT{LITERAL}:    [[9,     12,     12],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16]]],
// CHECK15-NEXT{LITERAL}:  [[[9,     12,     12],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16]],
// CHECK15-NEXT{LITERAL}:    [[9,     12,     12],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16],
// CHECK15-NEXT{LITERAL}:    [12,     16,     16]]]]]

// COM: filter=3x3, dilation=2, padding=1, stride=2
// RUN: rocmlir-gen --arch %arch -prc -rand=none -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=1 --padding_w=1 --conv_stride_h=2 --conv_stride_w=2 --operation=conv2d_bwd_weight -p=false | rocmlir-driver --host-pipeline=runner | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK16
// CHECK16: Unranked Memref base@ = 0x{{.*}} rank = 5 offset = 0 sizes = [1, 2, 2, 3, 3] strides = [36, 18, 9, 3, 1] data =
// CHECK16-NEXT{LITERAL}:[[[[[4,     6,     6],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9]],
// CHECK16-NEXT{LITERAL}:    [[4,     6,     6],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9]]],
// CHECK16-NEXT{LITERAL}:  [[[4,     6,     6],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9]],
// CHECK16-NEXT{LITERAL}:    [[4,     6,     6],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9],
// CHECK16-NEXT{LITERAL}:    [6,     9,     9]]]]]
