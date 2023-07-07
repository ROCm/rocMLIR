// RUN: mlir-opt --llvm-software-bf16 %s| FileCheck %s

module attributes {llvm.data_layout = ""} {
  llvm.func @verify_bf16_f32(%arg0: bf16, %arg1: f32) -> i32 {
//CHECK:  llvm.func @verify_bf16_f32(%arg0: i16, %arg1: f32) -> i32 {

    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
//CHECK-DAG:    %[[V4:.*]] = llvm.mlir.constant(1041891328 : i32) : i32
//CHECK-DAG:    %[[V1:.*]] = llvm.mlir.constant(16 : i32) : i32
//CHECK-DAG:    %[[V2:.*]] = llvm.mlir.constant(1 : i32) : i32
//CHECK-DAG:    %[[V0:.*]] = llvm.mlir.constant(32767 : i32) : i32
//CHECK-DAG:    %[[V3:.*]] = llvm.mlir.constant(0 : i32) : i32

    %2 = llvm.mlir.constant(1.503910e-01 : bf16) : bf16

    %3 = llvm.fptrunc %arg1 : f32 to bf16
//CHECK:    %[[V5:.*]] = llvm.bitcast %arg1 : f32 to i32
//CHECK-NEXT:    %[[V6:.*]] = llvm.lshr %[[V5]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V7:.*]] = llvm.and %[[V6]], %[[V2]]  : i32
//CHECK-NEXT:    %[[V8:.*]] = llvm.add %[[V5]], %[[V0]]  : i32
//CHECK-NEXT:    %[[V9:.*]] = llvm.lshr %[[V8]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V10:.*]] = llvm.add %[[V7]], %[[V9]]  : i32
//CHECK-NEXT:    %[[V12:.*]] = llvm.trunc %[[V10]] : i32 to i16

    %4 = llvm.fsub %arg0, %3  : bf16
//CHECK:    %[[V13:.*]] = llvm.zext %arg0 : i16 to i32
//CHECK-NEXT:    %[[V14:.*]] = llvm.shl %[[V13]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V15:.*]] = llvm.bitcast %[[V14]] : i32 to f32
//CHECK:    %[[V16:.*]] = llvm.zext %[[V12]] : i16 to i32
//CHECK-NEXT:    %[[V17:.*]] = llvm.shl %[[V16]], %1  : i32
//CHECK-NEXT:    %[[V18:.*]] = llvm.bitcast %[[V17]] : i32 to f32
//CHECK:    %[[V19:.*]] = llvm.fsub %[[V15]], %[[V18]]  : f32
//CHECK-NEXT:    %[[V20:.*]] = llvm.bitcast %[[V19]] : f32 to i32
//CHECK-NEXT:    %[[V21:.*]] = llvm.lshr %[[V20]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V22:.*]] = llvm.and %[[V21]], %[[V2]]  : i32
//CHECK-NEXT:    %[[V23:.*]] = llvm.add %[[V20]], %[[V0]]  : i32
//CHECK-NEXT:    %[[V24:.*]] = llvm.lshr %[[V23]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V25:.*]] = llvm.add %[[V22]], %[[V24]]  : i32
//CHECK-NEXT:    %[[V27:.*]] = llvm.trunc %[[V25]] : i32 to i16

    %5 = llvm.fcmp "ugt" %4, %2 : bf16
//CHECK:    %[[V28:.*]] = llvm.zext %[[V27]] : i16 to i32
//CHECK-NEXT:    %[[V29:.*]] = llvm.shl %[[V28]], %[[V1]]  : i32
//CHECK-NEXT:    %[[V30:.*]] = llvm.bitcast %[[V29]] : i32 to f32
//CHECK-NEXT:    %[[V33:.*]] = llvm.bitcast %[[V4]] : i32 to f32
//CHECK:    %{{.*}} = llvm.fcmp "ugt" %[[V30]], %[[V33]] : f32

    llvm.cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.return %1 : i32
  ^bb2:  // pred: ^bb0
    llvm.return %0 : i32
  }
}
