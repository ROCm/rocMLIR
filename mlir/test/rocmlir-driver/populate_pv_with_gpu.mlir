// RUN: rocmlir-gen --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu | FileCheck %s

// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// CHECK: rock.conv2d({{.*}}) features = mfma|dot|atomic_add {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf32>, memref<[[INPUTDIMS:[x0-9]+]]xf32>, memref<[[OUTPUTDIMS:[x0-9]+]]xf32>
// CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// CHECK: rock.conv2d({{.*}}) features = dot|atomic_add {{{.*}}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu | FileCheck %s --check-prefix=GEMM-CHECK

// GEMM-CHECK: func.func @rock_gemm({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = mfma|dot|atomic_add storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
// GEMM-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[ADIMS]]xf32>, memref<[[BDIMS]]xf32>, memref<[[CDIMS]]xf32>) -> ()
// GEMM-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[ADIMS]]xf32>, memref<[[BDIMS]]xf32>, memref<[[CDIMS]]xf32>) -> ()
// GEMM-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot|atomic_add storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS]]xf32> = memref<[[ADIMS]]xf32> * memref<[[BDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu --verifier-keep-perf-config=false --perf_config=32,64,4,32,64,1,1,1 | FileCheck %s --check-prefix=GEMM-HEURISTIC-CHECK

// GEMM-HEURISTIC-CHECK: func.func @rock_gemm({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-HEURISTIC-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = mfma|dot|atomic_add storeMethod = {{.*}}perf_config = "32,64,4,32,64,1,1,1"{{.*}} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
// GEMM-HEURISTIC-CHECK-NOT: perf_config
// GEMM-HEURISTIC-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[ADIMS]]xf32>, memref<[[BDIMS]]xf32>, memref<[[CDIMS]]xf32>) -> ()
// GEMM-HEURISTIC-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[ADIMS]]xf32>, memref<[[BDIMS]]xf32>, memref<[[CDIMS]]xf32>) -> ()
// GEMM-HEURISTIC-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-HEURISTIC-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot|atomic_add storeMethod = {{.*}} : memref<[[CDIMS]]xf32> = memref<[[ADIMS]]xf32> * memref<[[BDIMS]]xf32>

// RUN: rocmlir-gen --arch gfx908 -mfma=off -atomic_add=off -dot=on  -p -t f16 -pv_with_gpu | FileCheck %s --check-prefix=F16-CHECK

// F16-CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// F16-CHECK: rock.conv2d({{.*}}) features = dot {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf16>, memref<[[INPUTDIMS:[x0-9]+]]xf16>, memref<[[OUTPUTDIMS:[x0-9]+]]xf16>
// F16-CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf16>, memref<[[INPUTDIMS]]xf16>, memref<[[OUTPUTDIMS]]xf16>) -> ()
// F16-CHECK: call @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver_gpu({{.*}}) : (memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>) -> ()
// F16-CHECK: func.func @rock_conv2d_gkcyx_ngchw_ngkhw_0_ver({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// F16-CHECK: rock.conv2d({{.*}}) features = dot {{{.*}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908 -mfma=off -atomic_add=off -dot=on -p -t f16 -pv_with_gpu | FileCheck %s --check-prefix=GEMM-F16-CHECK

// GEMM-F16-CHECK: func.func @rock_gemm({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-F16-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf16> = memref<[[ADIMS:[x0-9]+]]xf16> * memref<[[BDIMS:[x0-9]+]]xf16>
// GEMM-F16-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[ADIMS]]xf16>, memref<[[BDIMS]]xf16>, memref<[[CDIMS]]xf16>) -> ()
// GEMM-F16-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[ADIMS]]xf32>, memref<[[BDIMS]]xf32>, memref<[[CDIMS]]xf32>) -> ()
// GEMM-F16-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-F16-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
