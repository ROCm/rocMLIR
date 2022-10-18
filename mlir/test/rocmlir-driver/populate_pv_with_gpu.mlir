// RUN: rocmlir-gen --arch %arch  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu | FileCheck %s

// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32, kernel.arch = "{{.*}}"} {
// CHECK: rock.conv2d({{.*}}) features = mfma|dot|atomic_add, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf32>, memref<[[INPUTDIMS:[x0-9]+]]xf32>, memref<[[OUTPUTDIMS:[x0-9]+]]xf32>
// CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver({{.*}}) attributes {kernel = 0 : i32, kernel.arch = "{{.*}}"} {
// CHECK: rock.conv2d({{.*}}) features = dot|atomic_add, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {{{.*}}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: rocmlir-gen --arch %arch -mfma=off -atomic_add=off -dot=on  -p -t f16 -pv_with_gpu | FileCheck %s --check-prefix=F16-CHECK

// F16-CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32, kernel.arch = "{{.*}}"} {
// F16-CHECK: rock.conv2d({{.*}}) features = dot, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf16>, memref<[[INPUTDIMS:[x0-9]+]]xf16>, memref<[[OUTPUTDIMS:[x0-9]+]]xf16>
// F16-CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf16>, memref<[[INPUTDIMS]]xf16>, memref<[[OUTPUTDIMS]]xf16>) -> ()
// F16-CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// F16-CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver({{.*}}) attributes {kernel = 0 : i32, kernel.arch = "{{.*}}"} {
// F16-CHECK: rock.conv2d({{.*}}) features = dot, convParams = {padding : [0, 0, 0, 0], stride : [1, 1], dilation : [1, 1]} {{{.*}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>
