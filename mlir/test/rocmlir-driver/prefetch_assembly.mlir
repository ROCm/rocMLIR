// RUN: rocmlir-gen --arch gfx942 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX942
// RUN: rocmlir-gen --arch gfx950 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX950
// RUN: rocmlir-gen --arch gfx1200 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX1200
// RUN: rocmlir-gen --arch gfx1100 --operation gemm -t f16 -p | rocmlir-driver -c --debug-only=serialize-to-isa 2>&1 | FileCheck %s --check-prefix=GFX1100

// GFX942-NOT: global_prefetch_b8
// GFX950-NOT: global_prefetch_b8
// GFX1100-NOT: global_prefetch_b8
// GFX1200-NOT: global_prefetch_b8

// TODO(gfx1250): add gfx1250 when it doesn't fail compilation
