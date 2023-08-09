// RUN: rocmlir-gen --arch %arch --operation attention -seq_len 1024 -num_heads 32 -t f32 | FileCheck %s --enable-var-scope --check-prefixes=CHECK

// CHECK:      func.func
// CHECK-SAME: function_type
// CHECK-SAME: memref<1024x32xf32>, memref<1024x32xf32>, memref<1024x32xf32>, memref<1024x1024xf32>, memref<1024x32xf32>
// CHECK-SAME: sym_name = "rock_attention"

// CHECK:       "rock.attention"
// CHECK-SAME:   memref<1024x32xf32>, memref<1024x32xf32>, memref<1024x32xf32>, memref<1024x1024xf32>, memref<1024x32xf32>

// CHECK: {kernel, mhal.arch = "[[$ARCH:.*]]"}
// CHECK-NEXT: {mhal.arch = "[[$ARCH]]"}
