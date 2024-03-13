// RUN: rocmlir-gen --arch %arch -mfma=off -atomic_add=off -p=false --padding_h=0 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 | FileCheck %s --check-prefix=Padding_One
// RUN: rocmlir-gen --arch %arch -mfma=off -atomic_add=off -p=false --padding_h=3 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 | FileCheck %s --check-prefix=Padding_Two

// Padding_One-LABEL: module
// Padding_One-NEXT: func.func @rock_conv_gkc01_ngc01_ngk01_0({{.*}}: memref<1x256x32x1x1xf32>, {{.*}}: memref<32x1x32x14x14xf32>, {{.*}}: memref<32x1x256x14x17xf32>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"}
// Padding_One-NEXT: rock.conv({{.*}}, {{.*}}, {{.*}}) features = {{.*}} {arch = "{{.*}}", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x14x17xf32>

// Padding_Two-LABEL: module
// Padding_Two-NEXT: func.func @rock_conv_gkc01_ngc01_ngk01_0({{.*}}: memref<1x256x32x1x1xf32>, {{.*}}: memref<32x1x32x14x14xf32>, {{.*}}: memref<32x1x256x20x17xf32>) attributes {kernel = 0 : i32, mhal.arch = "{{.*}}"}
// Padding_Two-NEXT: rock.conv({{.*}}, {{.*}}, {{.*}}) features = {{.*}} {arch = "{{.*}}", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [3 : index, 3 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x20x17xf32>
