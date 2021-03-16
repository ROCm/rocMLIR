// COM: yxck/nhwc/nhwk
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK8
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --conv_stride_h=2 --conv_stride_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK9
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128  -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=3 --conv_stride_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK10
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --dilation_h=2 --dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK11
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 --dilation_h=2 --dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK12
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -conv_stride_h=2 -conv_stride_w=3 -dilation_h=1 -dilation_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK13

// CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK8: [1]
// CHECK9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK9: [1]
// CHECK10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK10: [1]
// CHECK11: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK11: [1]
// CHECK12: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK12: [1]
// CHECK13: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK13: [1]
