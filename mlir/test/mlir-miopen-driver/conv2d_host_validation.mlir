// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -pv -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK1 
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=5 -fil_w=5 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK2
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=32 -in_w=32 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK3
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK4
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK5
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=8 -out_channels=128 -in_h=16 -in_w=64 -fil_h=5 -fil_w=5 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK6
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=128 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK7
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s  --check-prefix=CHECK8
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk  -batchsize=64 -in_channels=4 -out_channels=64 -in_h=16 -in_w=16 -fil_h=3 -fil_w=3   --conv_stride_h=2 --conv_stride_w=2 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK9
// RUN: mlir-miopen-driver -fil_layout=yxck -in_layout=nhwc -out_layout=nhwk -batchsize=128 -in_channels=8 -out_channels=128  -in_h=16 -in_w=16 -fil_h=3 -fil_w=3 --conv_stride_h=3 --conv_stride_w=3 -pv -p=false -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK10

// CHECK1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK1: [1]
// CHECK2: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK2: [1]
// CHECK3: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK3: [1]
// CHECK4: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK4: [1]
// CHECK5: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK5: [1]
// CHECK6: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK6: [1]
// CHECK7: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK7: [1]
// CHECK8: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK8: [1]
// CHECK9: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK9: [1]
// CHECK10: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK10: [1]
