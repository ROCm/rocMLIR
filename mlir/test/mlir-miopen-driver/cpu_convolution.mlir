//RUN: mlir-miopen-driver -prc -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK1
//CHECK1: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 2, 6, 6] strides = [72, 36, 6, 1] data =
//CHECK1-NEXT{LITERAL}: [[[[18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18]],
//CHECK1-NEXT{LITERAL}:   [[18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18],
//CHECK1-NEXT{LITERAL}:    [18,     18,     18,     18,     18,     18]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false --operation=conv2d_bwd_weight -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2
//CHECK2: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [2, 2, 3, 3] strides = [18, 9, 3, 1] data =
//CHECK2-NEXT{LITERAL}: [[[[36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36]],
//CHECK2-NEXT{LITERAL}:   [[36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36]]],
//CHECK2-NEXT{LITERAL}:  [[[36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36]],
//CHECK2-NEXT{LITERAL}:   [[36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36],
//CHECK2-NEXT{LITERAL}:    [36,     36,     36]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3   --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false --operation=conv2d_bwd_data -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK3
//CHECK3: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 2, 8, 8] strides = [128, 64, 8, 1] data =
//CHECK3-NEXT{LITERAL}: [[[[2,     4,     6,     6,     6,     6,     4,     2],
//CHECK3-NEXT{LITERAL}:    [4,     8,     12,     12,     12,     12,     8,     4],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [4,     8,     12,     12,     12,     12,     8,     4],
//CHECK3-NEXT{LITERAL}:    [2,     4,     6,     6,     6,     6,     4,     2]],
//CHECK3-NEXT{LITERAL}:   [[2,     4,     6,     6,     6,     6,     4,     2],
//CHECK3-NEXT{LITERAL}:    [4,     8,     12,     12,     12,     12,     8,     4],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [6,     12,     18,     18,     18,     18,     12,     6],
//CHECK3-NEXT{LITERAL}:    [4,     8,     12,     12,     12,     12,     8,     4],
//CHECK3-NEXT{LITERAL}:    [2,     4,     6,     6,     6,     6,     4,     2]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=0 --padding_w=0 --conv_stride_h=1 --conv_stride_w=1 -p=false --operation=conv2d_bwd_weight -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK4
//CHECK4: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [2, 3, 3, 2] strides = [18, 6, 2, 1] data =
//CHECK4-NEXT{LITERAL}: [[[[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]],
//CHECK4-NEXT{LITERAL}:   [[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]],
//CHECK4-NEXT{LITERAL}:   [[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]]],
//CHECK4-NEXT{LITERAL}:  [[[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]],
//CHECK4-NEXT{LITERAL}:   [[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]],
//CHECK4-NEXT{LITERAL}:   [[16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16],
//CHECK4-NEXT{LITERAL}:    [16,     16]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=1 --dilation_w=1 --padding_h=0 --padding_w=0 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK5
//CHECK5: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 3, 3, 2] strides = [18, 6, 2, 1] data =
//CHECK5-NEXT{LITERAL}:[[[[18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18]],
//CHECK5-NEXT{LITERAL}:  [[18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18]],
//CHECK5-NEXT{LITERAL}:  [[18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18],
//CHECK5-NEXT{LITERAL}:   [18,     18]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=2 --padding_w=2 --conv_stride_h=2 --conv_stride_w=2 -p=false  -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK6
//CHECK6: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 4, 4, 2] strides = [32, 8, 2, 1] data =
//CHECK6-NEXT{LITERAL}: [[[[8,     8],
//CHECK6-NEXT{LITERAL}:    [12,     12],
//CHECK6-NEXT{LITERAL}:    [12,     12],
//CHECK6-NEXT{LITERAL}:    [8,     8]],
//CHECK6-NEXT{LITERAL}:   [[12,     12],
//CHECK6-NEXT{LITERAL}:    [18,     18],
//CHECK6-NEXT{LITERAL}:    [18,     18],
//CHECK6-NEXT{LITERAL}:    [12,     12]],
//CHECK6-NEXT{LITERAL}:   [[12,     12],
//CHECK6-NEXT{LITERAL}:    [18,     18],
//CHECK6-NEXT{LITERAL}:    [18,     18],
//CHECK6-NEXT{LITERAL}:    [12,     12]],
//CHECK6-NEXT{LITERAL}:   [[8,     8],
//CHECK6-NEXT{LITERAL}:    [12,     12],
//CHECK6-NEXT{LITERAL}:    [12,     12],
//CHECK6-NEXT{LITERAL}:    [8,     8]]]]

//RUN: mlir-miopen-driver -prc -fil_layout=kyxc -in_layout=nhwc -out_layout=nhwk -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=2 --padding_w=2 --conv_stride_h=2 --conv_stride_w=2 -p=false --operation=conv2d_bwd_weight  -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK7
//CHECK7: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [2, 3, 3, 2] strides = [18, 6, 2, 1] data =
//CHECK7-NEXT{LITERAL}: [[[[9,     9],
//CHECK7-NEXT{LITERAL}:    [12,     12],
//CHECK7-NEXT{LITERAL}:    [9,     9]],
//CHECK7-NEXT{LITERAL}:   [[12,     12],
//CHECK7-NEXT{LITERAL}:    [16,     16],
//CHECK7-NEXT{LITERAL}:    [12,     12]],
//CHECK7-NEXT{LITERAL}:   [[9,     9],
//CHECK7-NEXT{LITERAL}:    [12,     12],
//CHECK7-NEXT{LITERAL}:    [9,     9]]],
//CHECK7-NEXT{LITERAL}:  [[[9,     9],
//CHECK7-NEXT{LITERAL}:    [12,     12],
//CHECK7-NEXT{LITERAL}:    [9,     9]],
//CHECK7-NEXT{LITERAL}:   [[12,     12],
//CHECK7-NEXT{LITERAL}:    [16,     16],
//CHECK7-NEXT{LITERAL}:    [12,     12]],
//CHECK7-NEXT{LITERAL}:   [[9,     9],
//CHECK7-NEXT{LITERAL}:    [12,     12],
//CHECK7-NEXT{LITERAL}:    [9,     9]]]]


//RUN: mlir-miopen-driver -prc -fil_layout=kcyx -in_layout=nchw -out_layout=nkhw -batchsize=1 -in_channels=2 -out_channels=2 -in_h=8 -in_w=8 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 --padding_h=2 --padding_w=2 --conv_stride_h=2 --conv_stride_w=2 -p=false --operation=conv2d_bwd_data  -c| mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK8
//CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 4 offset = 0 sizes = [1, 2, 8, 8] strides = [128, 64, 8, 1] data =
//CHECK8-NEXT{LITERAL}: [[[[8,     0,     12,     0,     12,     0,     8,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [12,     0,     18,     0,     18,     0,     12,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [12,     0,     18,     0,     18,     0,     12,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [8,     0,     12,     0,     12,     0,     8,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0]],
//CHECK8-NEXT{LITERAL}:   [[8,     0,     12,     0,     12,     0,     8,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [12,     0,     18,     0,     18,     0,     12,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [12,     0,     18,     0,     18,     0,     12,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0],
//CHECK8-NEXT{LITERAL}:    [8,     0,     12,     0,     12,     0,     8,     0],
//CHECK8-NEXT{LITERAL}:    [0,     0,     0,     0,     0,     0,     0,     0]]]]

