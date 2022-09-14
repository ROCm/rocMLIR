// Check the naming of tuning parameters for xdlops and matrix c vectorization values

// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm | FileCheck %s --check-prefix=STEP1
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-conv-to-gemm -miopen-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=STEP2

// STEP1: mPerWave
// STEP1: nPerWave
// STEP1-NOT: m_per_thread
// STEP1-NOT: n_per_thread

// STEP2: mPerWave
// STEP2: nPerWave
// STEP2-NOT: m_per_thread
// STEP2-NOT: n_per_thread
