// Check the guards of tensor layouts in miopen-driver 

// XFAIL:* 
// RUN: mlir-miopen-driver -p -fil_layout ykcx 2>&1 | FileCheck %s
// RUN: mlir-miopen-driver -p -fil_layout kycx 2>&1 | FileCheck %s
// RUN: mlir-miopen-driver -p -in_layout chnw 2>&1 | FileCheck %s
// RUN: mlir-miopen-driver -p -in_layout nhcw 2>&1 | FileCheck %s
