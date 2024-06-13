// TODO[split-K]: remove `-disable-split-k-for-tuning` after integrating split-K into MIGraphX
// RUN: rocmlir-gen --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu -disable-split-k-for-tuning | FileCheck %s

// CHECK: func.func @rock_conv_gkc01_ngc01_ngk01_0({{.*}}: memref<[[NFILTER:[0-9]+]]xf32>, {{.*}}: memref<[[NINPUT:[0-9]+]]xf32>, {{.*}}: memref<[[NOUTPUT:[0-9]+]]xf32>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// CHECK: rock.conv({{.*}}) features = mfma|dot|atomic_add {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf32>, memref<[[INPUTDIMS:[x0-9]+]]xf32>, memref<[[OUTPUTDIMS:[x0-9]+]]xf32>
// CHECK: call @rock_conv_gkc01_ngc01_ngk01_0_gpu({{.*}}) : (memref<[[NFILTER]]xf32>, memref<[[NINPUT]]xf32>, memref<[[NOUTPUT]]xf32>) -> ()
// CHECK: call @rock_conv_gkc01_ngc01_ngk01_0_ver_gpu({{.*}}) : (memref<[[NFILTER]]xf32>, memref<[[NINPUT]]xf32>, memref<[[NOUTPUT]]xf32>) -> ()
// CHECK: func.func @rock_conv_gkc01_ngc01_ngk01_0_ver({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// CHECK: rock.conv({{.*}}) features = dot|atomic_add {{{.*}}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu -disable-split-k-for-tuning | FileCheck %s --check-prefix=GEMM-CHECK

// GEMM-CHECK: func.func @rock_gemm({{.*}}: memref<[[NA:[0-9]+]]xf32>, {{.*}}: memref<[[NB:[0-9]+]]xf32>, {{.*}}: memref<[[NC:[0-9]+]]xf32>) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = mfma|dot|atomic_add storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
// GEMM-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[NA]]xf32>, memref<[[NB]]xf32>, memref<[[NC]]xf32>) -> ()
// GEMM-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[NA]]xf32>, memref<[[NB]]xf32>, memref<[[NC]]xf32>) -> ()
// GEMM-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot|atomic_add storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS]]xf32> = memref<[[ADIMS]]xf32> * memref<[[BDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908  -p -mfma=on -dot=on -atomic_add=on -pv_with_gpu --verifier-keep-perf-config=false --perf_config="v2:32,64,4,32,64,1,1,1,1" -disable-split-k-for-tuning | FileCheck %s --check-prefix=GEMM-HEURISTIC-CHECK

// GEMM-HEURISTIC-CHECK: func.func @rock_gemm({{.*}}: memref<[[NA:[0-9]+]]xf32>, {{.*}}: memref<[[NB:[0-9]+]]xf32>, {{.*}}: memref<[[NC:[0-9]+]]xf32>) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-HEURISTIC-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = mfma|dot|atomic_add storeMethod = {{.*}}perf_config = "v2:32,64,4,32,64,1,1,1,1"{{.*}} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
// GEMM-HEURISTIC-CHECK-NOT: perf_config
// GEMM-HEURISTIC-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[NA]]xf32>, memref<[[NB]]xf32>, memref<[[NC]]xf32>) -> ()
// GEMM-HEURISTIC-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[NA]]xf32>, memref<[[NB]]xf32>, memref<[[NC]]xf32>) -> ()
// GEMM-HEURISTIC-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-HEURISTIC-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot|atomic_add storeMethod = {{.*}} : memref<[[CDIMS]]xf32> = memref<[[ADIMS]]xf32> * memref<[[BDIMS]]xf32>

// RUN: rocmlir-gen --arch gfx908 -mfma=off -atomic_add=off -dot=on  -p -t f16 -pv_with_gpu -disable-split-k-for-tuning | FileCheck %s --check-prefix=F16-CHECK

// F16-CHECK: func.func @rock_conv_gkc01_ngc01_ngk01_0({{.*}}: memref<[[NFILTER:[0-9]+]]xf16>, {{.*}}: memref<[[NINPUT:[0-9]+]]xf16>, {{.*}}: memref<[[NOUTPUT:[0-9]+]]xf16>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// F16-CHECK: rock.conv({{.*}}) features = dot {[[PARMS:.*]]} : memref<[[FILTERDIMS:[x0-9]+]]xf16>, memref<[[INPUTDIMS:[x0-9]+]]xf16>, memref<[[OUTPUTDIMS:[x0-9]+]]xf16>
// F16-CHECK: call @rock_conv_gkc01_ngc01_ngk01_0_gpu({{.*}}) : (memref<[[NFILTER]]xf16>, memref<[[NINPUT]]xf16>, memref<[[NOUTPUT]]xf16>) -> ()
// F16-CHECK: call @rock_conv_gkc01_ngc01_ngk01_0_ver_gpu({{.*}}) : (memref<[[NFILTER]]xf32>, memref<[[NINPUT]]xf32>, memref<[[NOUTPUT]]xf32>) -> ()
// F16-CHECK: func.func @rock_conv_gkc01_ngc01_ngk01_0_ver({{.*}}) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"} {
// F16-CHECK: rock.conv({{.*}}) features = dot {{{.*}} : memref<[[FILTERDIMS]]xf32>, memref<[[INPUTDIMS]]xf32>, memref<[[OUTPUTDIMS]]xf32>

// RUN: rocmlir-gen --operation gemm --arch gfx908 -mfma=off -atomic_add=off -dot=on -p -t f16 -pv_with_gpu -disable-split-k-for-tuning | FileCheck %s --check-prefix=GEMM-F16-CHECK

// GEMM-F16-CHECK: func.func @rock_gemm({{.*}}: memref<[[NA:[0-9]+]]xf16>, {{.*}}: memref<[[NB:[0-9]+]]xf16>, {{.*}}: memref<[[NC:[0-9]+]]xf16>) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-F16-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf16> = memref<[[ADIMS:[x0-9]+]]xf16> * memref<[[BDIMS:[x0-9]+]]xf16>
// GEMM-F16-CHECK: call @rock_gemm_gpu({{.*}}) : (memref<[[NA]]xf16>, memref<[[NB]]xf16>, memref<[[NC]]xf16>) -> ()
// GEMM-F16-CHECK: call @rock_gemm_ver_gpu({{.*}}) : (memref<[[NA]]xf32>, memref<[[NB]]xf32>, memref<[[NC]]xf32>) -> ()
// GEMM-F16-CHECK: func.func @rock_gemm_ver({{.*}}) attributes {kernel, mhal.arch = "{{.*}}"} {
// GEMM-F16-CHECK: rock.gemm {{.*}} = {{.*}} * {{.*}} features = dot storeMethod = {{.*}} {[[PARMS:.*]]} : memref<[[CDIMS:[x0-9]+]]xf32> = memref<[[ADIMS:[x0-9]+]]xf32> * memref<[[BDIMS:[x0-9]+]]xf32>
