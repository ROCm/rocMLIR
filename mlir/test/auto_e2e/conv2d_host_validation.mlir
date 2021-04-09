// RUN: mlir-miopen-driver -p -pv %random_data %xdlopv2 -c | mlir-rocm-runner --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CHECK_1

// CHECK_1: Unranked Memref base@ = 0x{{.*}} rank = 1 offset = 0 sizes = [1] strides = [1] data =
// CHECK_1: [1]
