// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -mfma=off -atomic_add=off -p=false --padding_h=0 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 --mlir-print-local-scope | FileCheck %s --check-prefix=Padding_One
// RUN: rocmlir-gen --arch gfx90a:sramecc+:xnack- -mfma=off -atomic_add=off -p=false --padding_h=3 -batchsize=32 -in_channels=32 -out_channels=256 -in_h=14 -in_w=14 -fil_h=1 -fil_w=1  --padding_w_l=1 --padding_w_r=2 --mlir-print-local-scope | FileCheck %s --check-prefix=Padding_Two

// Padding_One-LABEL: func.func @rock_conv_gkc01_ngc01_ngk01_0
// Padding_One-SAME: ([[arg0:%.+]]: memref<8192xf32>, [[arg1:%.+]]: memref<200704xf32>, [[arg2:%.+]]: memref<1949696xf32>)
// Padding_One-SAME: attributes {enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "{{.*}}"}
// Padding_One-NEXT: [[exp0:%.+]] = rock.transform [[arg0]] by
// Padding_One-SAME: Unmerge{1, 256, 32, 1, 1}
// Padding_One-NEXT: [[exp1:%.+]] = rock.transform [[arg1]] by
// Padding_One-SAME: Unmerge{32, 1, 32, 14, 14}
// Padding_One-NEXT: [[exp2:%.+]] = rock.transform [[arg2]] by
// Padding_One-SAME: Unmerge{32, 1, 256, 14, 17}
// Padding_One-NEXT: rock.conv([[exp0]], [[exp1]], [[exp2]]) features = {{.*}} {arch = "{{.*}}", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [0 : index, 0 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x14x17xf32>

// Padding_Two-LABEL: func.func @rock_conv_gkc01_ngc01_ngk01_0
// Padding_Two-SAME: ([[arg0:%.+]]: memref<8192xf32>, [[arg1:%.+]]: memref<200704xf32>, [[arg2:%.+]]: memref<2785280xf32>)
// Padding_Two-SAME: attributes {enable_splitk_for_tuning, kernel = 0 : i32, mhal.arch = "{{.*}}"}
// Padding_Two-NEXT: [[exp0:%.+]] = rock.transform [[arg0]] by
// Padding_Two-SAME: Unmerge{1, 256, 32, 1, 1}
// Padding_Two-NEXT: [[exp1:%.+]] = rock.transform [[arg1]] by
// Padding_Two-SAME: Unmerge{32, 1, 32, 14, 14}
// Padding_Two-NEXT: [[exp2:%.+]] = rock.transform [[arg2]] by
// Padding_Two-SAME: Unmerge{32, 1, 256, 20, 17}
// Padding_Two-NEXT: rock.conv([[exp0]], [[exp1]], [[exp2]]) features = {{.*}} {arch = "{{.*}}", dilations = [1 : index, 1 : index], filter_layout = ["g", "k", "c", "0", "1"], input_layout = ["ni", "gi", "ci", "0i", "1i"], numCU = {{.*}} : i32, output_layout = ["no", "go", "ko", "0o", "1o"], padding = [3 : index, 3 : index, 1 : index, 2 : index], strides = [1 : index, 1 : index]} : memref<1x256x32x1x1xf32>, memref<32x1x32x14x14xf32>, memref<32x1x256x20x17xf32>
