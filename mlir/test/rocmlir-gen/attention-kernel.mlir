// RUN: rocmlir-gen --arch %arch --operation attention -seq_len 1024 -num_heads 32 -t f32 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK

// CHECK: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK-LABEL: func.func @rock_attention
// CHECK-SAME: (%[[queries:.*0]]: memref<1024x32xf32>,
// CHECK-SAME: %[[keys:.*1]]: memref<32x1024xf32>,
// CHECK-SAME: %[[values:.*2]]: memref<1024x32xf32>,
// CHECK-SAME: %[[scale:.*3]]: memref<1024x1024xf32>,
// CHECK-SAME: %[[output:.*4]]: memref<1024x32xf32>)
// CHECK-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK-NEXT: rock.attention(%[[queries]], %[[keys]], %[[values]], %[[scale]], %[[output]])
// CHECK: return
