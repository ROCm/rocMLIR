
// RUN: rocmlir-gen --arch %arch -p -fil_layout ykcx 2>&1 | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -fil_layout kycx 2>&1 | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -in_layout nhcw  2>&1 | FileCheck %s
// RUN: rocmlir-gen --arch %arch -p -in_layout chnw  2>&1 | FileCheck %s

CHECK: rock.conv
