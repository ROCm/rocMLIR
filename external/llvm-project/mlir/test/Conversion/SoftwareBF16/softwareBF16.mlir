// RUN: mlir-opt --llvm-software-bf16 %s| FileCheck %s

module attributes {llvm.data_layout = ""} {
  llvm.func @verify_bf16_f32(%arg0: bf16, %arg1: f32) -> i32 {
//CHECK:  llvm.func @verify_bf16_f32(%arg0: i16, %arg1: f32) -> i32 {

    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[V0:.*]] = llvm.mlir.constant(32767 : i32) : i32
//CHECK:    %[[V1:.*]] = llvm.mlir.constant(16 : i32) : i32
//CHECK:    %[[V2:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK:    %[[V3:.*]] = llvm.mlir.constant(0 : i32) : i32

    %2 = llvm.mlir.constant(1.503910e-01 : bf16) : bf16
//CHECK:    %[[V4:.*]] = llvm.mlir.constant(15898 : i16) : i16

    %3 = llvm.fptrunc %arg1 : f32 to bf16
//CHECK:    %[[V5:.*]] = llvm.bitcast %arg1 : f32 to i32
//CHECK-NEXT:    %[[V6:.*]] = llvm.lshr %[[V5]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V7:.*]] = llvm.and %[[V6]], %[[V2]]  : i32
//CHECK-NEXT:    %[[V8:.*]] = llvm.add %[[V5]], %[[V0]]  : i32
//CHECK-NEXT:    %[[V9:.*]] = llvm.add %[[V7]], %[[V8]]  : i32
//CHECK-NEXT:    %[[V10:.*]] = llvm.lshr %[[V9]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V11:.*]] = llvm.trunc %[[V10]] : i32 to i16

    %4 = llvm.fsub %arg0, %3  : bf16
//CHECK:    %[[V12:.*]] = llvm.zext %arg0 : i16 to i32
//CHECK-NEXT:    %[[V13:.*]] = llvm.shl %[[V12]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V14:.*]] = llvm.bitcast %[[V13]] : i32 to f32
//CHECK:    %[[V15:.*]] = llvm.zext %[[V11]] : i16 to i32
//CHECK-NEXT:    %[[V16:.*]] = llvm.shl %[[V15]], %1  : i32
//CHECK-NEXT:    %[[V17:.*]] = llvm.bitcast %[[V16]] : i32 to f32
//CHECK:    %[[V18:.*]] = llvm.fsub %[[V14]], %[[V17]]  : f32
//CHECK-NEXT:    %[[V19:.*]] = llvm.bitcast %[[V18]] : f32 to i32
//CHECK-NEXT:    %[[V20:.*]] = llvm.lshr %[[V19]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V21:.*]] = llvm.and %[[V20]], %[[V2]]  : i32
//CHECK-NEXT:    %[[V22:.*]] = llvm.add %[[V19]], %[[V0]]  : i32
//CHECK-NEXT:    %[[V23:.*]] = llvm.add %[[V21]], %[[V22]]  : i32
//CHECK-NEXT:    %[[V24:.*]] = llvm.lshr %[[V23]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V25:.*]] = llvm.trunc %[[V24]] : i32 to i16

    %5 = llvm.fcmp "ugt" %4, %2 : bf16
//CHECK:    %[[V26:.*]] = llvm.zext %[[V25]] : i16 to i32
//CHECK-NEXT:    %[[V27:.*]] = llvm.shl %[[V26]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V28:.*]] = llvm.bitcast %[[V27]] : i32 to f32
//CHECK:    %[[V29:.*]] = llvm.zext %[[V4]] : i16 to i32
//CHECK-NEXT:    %[[V30:.*]] = llvm.shl %[[V29]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V31:.*]] = llvm.bitcast %[[V30]] : i32 to f32
//CHECK:    %{{.*}} = llvm.fcmp "ugt" %[[V28]], %[[V31]] : f32

    llvm.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %1 : i32
  ^bb2:  // pred: ^bb0
    llvm.return %0 : i32
  }
}
