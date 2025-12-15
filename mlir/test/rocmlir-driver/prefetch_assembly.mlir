// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX1200
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX1100
// RUN: rocmlir-gen --arch gfx1250 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX1250

// Only gfx1250 supports global_prefetch_b8 instruction

// GFX942-NOT: global_prefetch_b8
// GFX950-NOT: global_prefetch_b8
// GFX1100-NOT: global_prefetch_b8
// GFX1200-NOT: global_prefetch_b8
// GFX1250: global_prefetch_b8
