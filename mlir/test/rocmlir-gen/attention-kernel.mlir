// RUN: rocmlir-gen --arch %arch --operation attention -seq_len 1024 -num_heads 32 --with-attn-scale -t f32 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_SCALE
// RUN: rocmlir-gen --arch %arch --operation attention -seq_len 1024 -num_heads 32 -t f32 | rocmlir-opt | FileCheck %s --enable-var-scope --check-prefixes=CHECK_NO_SCALE

// CHECK_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_SCALE-LABEL: func.func @rock_attention
// CHECK_SCALE-SAME: (%[[queries:.*0]]: memref<1x1024x32xf32>,
// CHECK_SCALE-SAME: %[[keys:.*1]]: memref<1x32x1024xf32>,
// CHECK_SCALE-SAME: %[[values:.*2]]: memref<1x1024x32xf32>,
// CHECK_SCALE-SAME: %[[scale:.*3]]: memref<1x1024x1024xf32>,
// CHECK_SCALE-SAME: %[[output:.*4]]: memref<1x1024x32xf32>)
// CHECK_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK_SCALE-NEXT: rock.attention(%[[queries]], %[[keys]], %[[values]], %[[scale]], %[[output]])
// CHECK_SCALE: return

// CHECK_NO_SCALE: module attributes {mhal.arch = "[[$ARCH:.*]]"}

// CHECK_NO_SCALE-LABEL: func.func @rock_attention
// CHECK_NO_SCALE-SAME: (%[[queries:.*0]]: memref<1x1024x32xf32>,
// CHECK_NO_SCALE-SAME: %[[keys:.*1]]: memref<1x32x1024xf32>,
// CHECK_NO_SCALE-SAME: %[[values:.*2]]: memref<1x1024x32xf32>,
// CHECK_NO_SCALE-SAME: %[[output:.*3]]: memref<1x1024x32xf32>)
// CHECK_NO_SCALE-SAME: attributes {kernel, mhal.arch = "[[$ARCH]]"}

// CHECK_NO_SCALE-NEXT: rock.attention(%[[queries]], %[[keys]], %[[values]], %[[output]])
// CHECK_NO_SCALE: return
