// RUN: mlir-miopen-driver  -p -x2 -pv_with_gpu | FileCheck %s

// CHECK: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32} {
// CHECK: miopen.conv2d({{.*}}) {[[PARMS:.*]], xdlopsV2 = true} : memref<[[FILTERDIMS:[x0-9]+]]xf32>, memref<[[INPUTDIMS:[x0-9]+]]xf32>, memref<[[OUTPUTDIMS:[x0-9]+]]xf32>
// CHECK: call @conv2d({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: call @conv2d_1({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: func @miopen_conv2d_gkcyx_ngchw_ngkhw_1000({{.*}}) attributes {kernel = 0 : i32} {
// CHECK: miopen.conv2d({{.*}}) {{{.*}}ignore_tuning = true{{.*}}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: mlir-miopen-driver  -p -t f16 -pv_with_gpu | FileCheck %s --check-prefix=F16-CHECK

// F16-CHECK: func @miopen_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32} {
// F16-CHECK: miopen.conv2d({{.*}}) {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf16>, memref<[[INPUTDIMS:[x0-9]+]]xf16>, memref<[[OUTPUTDIMS:[x0-9]+]]xf16>
// F16-CHECK: call @conv2d({{.*}}) : (memref<[[FILTERDIMS]]xf16>, memref<[[INPUTDIMS]]xf16>, memref<[[OUTPUTDIMS]]xf16>) -> ()
// F16-CHECK: call @conv2d_1({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// F16-CHECK: func @miopen_conv2d_gkcyx_ngchw_ngkhw_1000({{.*}}) attributes {kernel = 0 : i32} {
// F16-CHECK: miopen.conv2d({{.*}}) {{{.*}}ignore_tuning = true{{.*}}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>
