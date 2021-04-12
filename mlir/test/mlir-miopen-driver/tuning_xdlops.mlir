// Check the naming of tuning parameters for xdlops

// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params | FileCheck %s --check-prefix=STEP1
// RUN: mlir-miopen-driver -p -x2 -miopen-lowering -miopen-affine-transform -miopen-affix-params -miopen-lowering-step2 | FileCheck %s --check-prefix=STEP2

//STEP1: m_per_wave
//STEP1: n_per_wave
//STEP1-NOT: m_per_thread
//STEP1-NOT: n_per_thread

//STEP2: m_per_wave
//STEP2: n_per_wave
//STEP1-NOT: m_per_thread
//STEP1-NOT: n_per_thread
