// Smart tuning ranks the applicable pool with the learned per-(arch, op) model
// (SmartTuningDb) and returns at most ROCMLIR_SMART_TUNING_LIST_MAX configs.
// A gfx942 gemm model is embedded, so the space is truncated to the cap (3).
// RUN: env ROCMLIR_SMART_TUNING_LIST_MAX=3 rocmlir-gen --arch gfx942 --operation=gemm -t f16 -m 1024 -k 768 -n 512 --emit-tuning-space=smart | FileCheck %s --check-prefix=SMART
// SMART-COUNT-3: {{^v[0-9]+:}}
// SMART-NOT: {{^v[0-9]+:}}

// A gfx942 conv model is embedded too; smart ranks the exhaustive conv pool and
// truncates to the cap.
// RUN: env ROCMLIR_SMART_TUNING_LIST_MAX=5 rocmlir-gen --arch gfx942 --operation conv -t f32 -fil_layout=gkyxc -in_layout=nhwgc -out_layout=nhwgk -batchsize=1 -in_channels=64 -out_channels=64 -in_h=28 -in_w=28 -fil_h=3 -fil_w=3 --emit-tuning-space=smart | FileCheck %s --check-prefix=CONV
// CONV-COUNT-5: {{^v[0-9]+:}}
// CONV-NOT: {{^v[0-9]+:}}

// A gfx942 attention model is embedded (configs are emitted as "attn:vN:...").
// RUN: env ROCMLIR_SMART_TUNING_LIST_MAX=4 rocmlir-gen --arch gfx942 --operation=attention -t f16 -g 1 -head_dim_qk 64 -head_dim_v 64 -num_heads_q 8 -num_heads_kv 8 -seq_len_q 1024 -seq_len_k 1024 --emit-tuning-space=smart | FileCheck %s --check-prefix=ATTN
// ATTN-COUNT-4: {{^attn:v[0-9]+:}}
// ATTN-NOT: {{^attn:v[0-9]+:}}

// When no model is embedded for the (arch, op), smart tuning is a hard error
// (no silent fallback to another space).
// RUN: not --crash rocmlir-gen --arch gfx90a --operation=gemm -t f16 -m 1024 -k 768 -n 512 --emit-tuning-space=smart 2>&1 | FileCheck %s --check-prefix=NOMODEL
// NOMODEL: no model is embedded for arch '{{.*}}gfx90a'
