// Check the naming of tuning parameters for xdlops and matrix c vectorization values

// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-affine-transform | FileCheck %s --check-prefix=STEP1
// RUN: miopen-gen -p -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-affine-transform -miopen-lowering-step2 | FileCheck %s --check-prefix=STEP2
// RUN: miopen-gen -p --fil_layout=kyxc --in_layout=nhwc --out_layout=nhwk -x2 | mlir-miopen-driver -miopen-affix-params -miopen-lowering -miopen-affine-transform | FileCheck %s --check-prefix=NHWC

// STEP1: m_per_wave
// STEP1: matrix_c_data_per_copy = 4
// STEP1: matrix_c_dest_vector_write_dim = 4
// STEP1: matrix_c_source_vector_read_dim = 2
// STEP1: n_per_wave
// STEP1-NOT: m_per_thread
// STEP1-NOT: n_per_thread

// STEP2: m_per_wave
// STEP2: n_per_wave
// STEP2-NOT: m_per_thread
// STEP2-NOT: n_per_thread

// NHWC: matrix_c_data_per_copy = 4
// NHWC: matrix_c_dest_vector_write_dim = 4
// NHWC: matrix_c_source_vector_read_dim = 1
