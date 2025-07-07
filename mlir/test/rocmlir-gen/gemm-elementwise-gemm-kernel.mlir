// RUN: rocmlir-gen --arch gfx942:sramecc+:xnack- --operation gemm_gemm -m 1024 -n 1024 -k 32 -gemmO 32 -t f32 -pv --apply-bufferization-pipeline=false | rocmlir-opt | FileCheck %s --enable-var-scope

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_gemm_gemm
// CHECK-SAME: (%[[aRaw:.*0]]: memref<32768xf32>,
// CHECK-SAME: %[[bRaw:.*1]]: memref<32768xf32>,
// CHECK-SAME: %[[cRaw:.*2]]: memref<32768xf32>,
// CHECK-SAME: %[[outputRaw:.*3]]: memref<32768xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}
// CHECK-NEXT: %[[a:.*]] = rock.transform %[[aRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK-NEXT: %[[b:.*]] = rock.transform %[[bRaw]] {{.*}} : memref<32768xf32> to memref<1x32x1024xf32>
// CHECK-NEXT: %[[c:.*]] = rock.transform %[[cRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>
// CHECK-NEXT: %[[output:.*]] = rock.transform %[[outputRaw]] {{.*}} : memref<32768xf32> to memref<1x1024x32xf32>

// CHECK-NEXT: rock.gemm_elementwise_gemm
// CHECK-NEXT: ab = %[[a]] * %[[b]]
// CHECK: %[[output]] = ab * %[[c]]
// CHECK: return

// CHECK-LABEL: func.func @host_naive_gemm_gemm
// CHECK: %[[abTensor:.*]] = tosa.matmul %[[aTensor:.*]], %[[bTensor:.*]], %{{.*}}, %{{.*}} : ([[aShape:tensor<.*>]], [[bShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[squareShape:tensor<.*>]]
// CHECK-DAG: %[[resultTensor:.*]] = tosa.matmul %[[abTensor]], %[[cTensor:.*]], %{{.*}}, %{{.*}} {acc_type = f32} : ([[squareShape]], [[cShape:tensor<.*>]], tensor<1xf32>, tensor<1xf32>) -> [[cShape]]
// CHECK: return
