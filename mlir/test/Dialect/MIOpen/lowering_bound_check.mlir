// RUN: mlir-miopen-driver -batchsize=1024 -in_channels=1024 -out_channels=1024 -fil_w=1 -fil_h=1 -in_h=14 -in_w=14 -padding_h=1 -padding_w=1 -miopen-affix-params -miopen-lowering |FileCheck %s --check-prefix=CHECK
// RUN: mlir-miopen-driver -batchsize=1024 -in_channels=1024 -out_channels=1024 -fil_w=1 -fil_h=1 -in_h=14 -in_w=14 -padding_h=1 -padding_w=1 -miopen-affix-params -miopen-lowering -miopen-lowering-step2 -miopen-lowering-step3|FileCheck %s --check-prefix=CHECK_STEP3
// CHECK: bound_check = [0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32]
// CHECK_STEP3: {{.*bound_check = \[0 : i32, 0 : i32, 0 : i32, 1 : i32, 1 : i32\].*metadata.*operand = 0.*}}
