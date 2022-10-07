// Check the guards of tensor layouts in rock-driver 

// RUN: not rocmlir-gen --arch %arch -p -fil_layout ykcx 2>&1 | FileCheck %s --check-prefix=ERR1
// RUN: not rocmlir-gen --arch %arch -p -fil_layout kycx 2>&1 | FileCheck %s --check-prefix=ERR2
// RUN: not rocmlir-gen --arch %arch -p -in_layout nhcw  2>&1 | FileCheck %s --check-prefix=ERR3
// RUN: not rocmlir-gen --arch %arch -p -in_layout chnw  2>&1 | FileCheck %s --check-prefix=ERR4

ERR1: Unsupported filter layout
ERR2: Unsupported filter layout
ERR3: Unsupported input layout
ERR4: Unsupported input layout
