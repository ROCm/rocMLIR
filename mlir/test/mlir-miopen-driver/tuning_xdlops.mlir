// Check the naming of tuning parameters for xdlops

// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform | FileCheck %s --check-prefix=STEP1
// RUN: mlir-miopen-driver -p -x2 -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | FileCheck %s --check-prefix=STEP2

// STEP1: m_per_wave
// STEP1: n_per_wave
// STEP1-NOT: m_per_thread
// STEP1-NOT: n_per_thread

// STEP2: m_per_wave
// STEP2: n_per_wave
// STEP2-NOT: m_per_thread
// STEP2-NOT: n_per_thread
