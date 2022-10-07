// Check the naming of tuning parameters for xdlops and matrix c vectorization values

// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm | FileCheck %s --check-prefix=STEP1
// RUN: rocmlir-gen --arch %arch -p -mfma=on | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gridwise-gemm-to-blockwise | FileCheck %s --check-prefix=STEP2

// STEP1: mPerWave
// STEP1: nPerWave
// STEP1-NOT: m_per_thread
// STEP1-NOT: n_per_thread

// STEP2: mPerWave
// STEP2: nPerWave
// STEP2-NOT: m_per_thread
// STEP2-NOT: n_per_thread
