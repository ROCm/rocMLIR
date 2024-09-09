// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on --mlir-print-local-scope | FileCheck %s --enable-var-scope=false -D\$ITYPE=f32 -D\$OTYPE=f32
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -t f16 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on --mlir-print-local-scope | FileCheck %s -D\$ITYPE=f16 -D\$OTYPE=f16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -t bf16 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on --mlir-print-local-scope | FileCheck %s -D\$ITYPE=bf16 -D\$OTYPE=bf16
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -p -t i8 -mfma=off -wmma=off -atomic_add=off -atomic_fmax_f32=off -dot=on --mlir-print-local-scope | FileCheck %s -D\$ITYPE=i8 -D\$OTYPE=i32

// CHECK-LABEL: module
// CHECK-NEXT: func.func @rock_conv_gkc01_ngc01_ngk01_0
// CHECK-SAME: ([[arg0:%.+]]: memref<9216x[[$ITYPE]]>, [[arg1:%.+]]: memref<1048576x[[$ITYPE]]>, [[arg2:%.+]]: memref<14745600x[[$OTYPE]]>)
// CHECK-SAME: attributes {enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "{{.*}}"}
// CHECK-NEXT: [[fil:%.+]] = rock.transform [[arg0]]
// CHECK-SAME: ["g", "k", "c", "0", "1"]
// CHECK-NEXT: [[$In:%.+]] = rock.transform [[arg1]]
// CHECK-SAME: ["ni", "gi", "ci", "0i", "1i"]
// CHECK-NEXT: [[$Out:%.+]] = rock.transform [[arg2]]
// CHECK-SAME: ["no", "go", "ko", "0o", "1o"]
// CHECK-NEXT: rock.conv([[fil]], [[$In]], [[$Out]])  features = dot {arch = "{{.*}}", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 0 : index, 0 : index], strides = [1 : index, 1 : index]} : memref<1x128x8x3x3x[[$ITYPE]]>, memref<128x1x8x32x32x[[$ITYPE]]>, memref<128x1x128x30x30x[[$OTYPE]]>
