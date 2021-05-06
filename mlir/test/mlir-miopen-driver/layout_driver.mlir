// Check the guards of tensor layouts in miopen-driver 

// XFAIL:* 
// RUN: mlir-miopen-driver -p -fil_layout ykcx || mlir-miopen-driver -p -fil_layout kycx || mlir-miopen-driver -p -in_layout nhcw || mlir-miopen-driver -p -in_layout chnw
