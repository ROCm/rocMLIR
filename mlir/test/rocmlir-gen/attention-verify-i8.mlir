// RUN: rocmlir-gen -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --with-attn-bias --operation attention --transK=true  --operation attention -t i8 --arch gfx90a:sramecc+:xnack- -pv   -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_FULL
// RUN: rocmlir-gen -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-bias --operation attention --transK=true  --operation attention -t i8 --arch gfx90a:sramecc+:xnack- -pv   -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_NOSCALE
// RUN: rocmlir-gen -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --with-attn-scale --operation attention --transK=true  --operation attention -t i8 --arch gfx90a:sramecc+:xnack- -pv   -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_NOBIAS
// RUN: rocmlir-gen -seq_len_q 384 -seq_len_k 384 -head_dim_qk 64 -head_dim_v 64 --operation attention --transK=true  --operation attention -t i8 --arch gfx90a:sramecc+:xnack- -pv   -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_SIMPLE
// RUN: rocmlir-gen -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-scale --with-attn-bias --operation attention --transK=true -t i8 --arch gfx90a:sramecc+:xnack- -pv -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_CPU_FMA_REF
// RUN: rocmlir-gen -seq_len_q 16 -seq_len_k 16 -head_dim_qk 16 -head_dim_v 16 --with-attn-scale --with-attn-bias --operation attention --transK=false -t i8 --arch gfx90a:sramecc+:xnack- -pv -RMS_threshold 0.003 | FileCheck %s --enable-var-scope --check-prefixes=CHECK_I8_CPU_FMA_REF

// CHECK_I8_FULL-LABEL: @main
// CHECK_I8_FULL: call @rock_attention_gpu(%[[alloc0:.+]], %[[alloc1:.+]], %[[alloc2:.+]], %[[alloc3:.+]], %[[alloc4:.+]], %[[alloc5:.+]], %[[alloc6:.+]], %[[gpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_FULL: call @host_naive_attention(%[[alloc7:.+]], %[[alloc8:.+]], %[[alloc9:.+]], %[[alloc10:.+]], %[[alloc11:.+]], %[[alloc12:.+]], %[[alloc13:.+]], %[[cpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_FULL: call @rock_attention_verify7(%[[gpu_out]], %[[cpu_out]]) : (memref<24576xf16>, memref<24576xf16>) -> ()

// CHECK_I8_NOSCALE-LABEL: @main
// CHECK_I8_NOSCALE: call @rock_attention_gpu(%[[alloc0:.+]], %[[alloc1:.+]], %[[alloc2:.+]], %[[alloc3:.+]], %[[alloc4:.+]], %[[alloc5:.+]], %[[gpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_NOSCALE: call @host_naive_attention(%[[alloc7:.+]], %[[alloc8:.+]], %[[alloc9:.+]], %[[alloc10:.+]], %[[alloc11:.+]], %[[alloc12:.+]], %[[cpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_NOSCALE: call @rock_attention_verify6(%[[gpu_out]], %[[cpu_out]]) : (memref<24576xf16>, memref<24576xf16>) -> ()

// CHECK_I8_NOBIAS-LABEL: @main
// CHECK_I8_NOBIAS: call @rock_attention_gpu(%[[alloc0:.+]], %[[alloc1:.+]], %[[alloc2:.+]], %[[alloc3:.+]], %[[alloc4:.+]], %[[alloc5:.+]], %[[gpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_NOBIAS: call @host_naive_attention(%[[alloc7:.+]], %[[alloc8:.+]], %[[alloc9:.+]], %[[alloc10:.+]], %[[alloc11:.+]], %[[alloc12:.+]], %[[cpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<147456xf16>, memref<24576xf16>) -> ()
// CHECK_I8_NOBIAS: call @rock_attention_verify6(%[[gpu_out]], %[[cpu_out]]) : (memref<24576xf16>, memref<24576xf16>) -> ()

// CHECK_I8_SIMPLE-LABEL: @main
// CHECK_I8_SIMPLE: call @rock_attention_gpu(%[[alloc0:.+]], %[[alloc1:.+]], %[[alloc2:.+]], %[[alloc3:.+]], %[[alloc4:.+]], %[[gpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<24576xf16>) -> ()
// CHECK_I8_SIMPLE: call @host_naive_attention(%[[alloc7:.+]], %[[alloc8:.+]], %[[alloc9:.+]], %[[alloc10:.+]], %[[alloc11:.+]], %[[cpu_out:.+]]) : (memref<24576xi8>, memref<24576xi8>, memref<24576xf16>, memref<1xi8>, memref<1xf16>, memref<24576xf16>) -> ()
// CHECK_I8_SIMPLE: call @rock_attention_verify5(%[[gpu_out]], %[[cpu_out]]) : (memref<24576xf16>, memref<24576xf16>) -> ()

// CHECK_I8_CPU_FMA_REF-LABEL: func.func @rock_attention
// CHECK_I8_CPU_FMA_REF:      %[[QK_F16:.*]] = arith.sitofp {{.*}} : i32 to f16
// CHECK_I8_CPU_FMA_REF-NEXT: %[[QSCALE:.*]] = arith.mulf %[[QK_F16]], {{.*}} : f16
// CHECK_I8_CPU_FMA_REF-NEXT: %[[ATTN_SCALE:.*]] = arith.mulf %[[QSCALE]], {{.*}} : f16
// CHECK_I8_CPU_FMA_REF-NEXT: %[[BIASED:.*]] = arith.addf %[[ATTN_SCALE]], {{.*}} : f16
// CHECK_I8_CPU_FMA_REF-NEXT: linalg.yield %[[BIASED]] : f16
// CHECK_I8_CPU_FMA_REF-LABEL: func.func @host_naive_attention
// CHECK_I8_CPU_FMA_REF:      linalg.generic
// CHECK_I8_CPU_FMA_REF:      ^bb0(%[[IN_I32:.*]]: i32, %[[ZERO_POINT:.*]]: i8, %[[QUANT_SCALE:.*]]: f16, %[[ATTN_SCALE_IN:.*]]: f16, %[[BIAS_IN:.*]]: f16, %{{.*}}: f32):
// CHECK_I8_CPU_FMA_REF-NEXT: %[[BIAS_F32:.*]] = arith.extf %[[BIAS_IN]] : f16 to f32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[SCALE_F32:.*]] = arith.extf %[[ATTN_SCALE_IN]] : f16 to f32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[ZERO_POINT_I32:.*]] = arith.extsi %[[ZERO_POINT]] : i8 to i32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[QK_I32:.*]] = arith.subi %[[IN_I32]], %[[ZERO_POINT_I32]] : i32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[QK_REF_F16:.*]] = arith.sitofp %[[QK_I32]] : i32 to f16
// CHECK_I8_CPU_FMA_REF-NEXT: %[[QSCALE_REF:.*]] = arith.mulf %[[QK_REF_F16]], %[[QUANT_SCALE]] : f16
// CHECK_I8_CPU_FMA_REF-NEXT: %[[SCORE_F32:.*]] = arith.extf %[[QSCALE_REF]] : f16 to f32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[SCALED_F32:.*]] = arith.mulf %[[SCORE_F32]], %[[SCALE_F32]] : f32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[BIASED_F32:.*]] = arith.addf %[[SCALED_F32]], %[[BIAS_F32]] : f32
// CHECK_I8_CPU_FMA_REF-NEXT: %[[BIASED_F16:.*]] = arith.truncf %[[BIASED_F32]] : f32 to f16
// CHECK_I8_CPU_FMA_REF-NEXT: arith.extf %[[BIASED_F16]] : f16 to f32
