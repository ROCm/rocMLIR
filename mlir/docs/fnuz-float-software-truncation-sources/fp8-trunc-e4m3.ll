define linkonce_odr dso_local noundef zeroext i8 @e4m3(float)(float noundef %f_x) local_unnamed_addr #2 comdat {
entry:
  %0 = bitcast float %f_x to i32
  %and1 = and i32 %0, 8388607
  %shr = lshr i32 %0, 23
  %and2 = and i32 %shr, 255
  %1 = lshr i32 %0, 24
  %shl = and i32 %1, 128
  %and4 = and i32 %0, 2139095040
  %cmp = icmp eq i32 %and4, 2139095040
  br i1 %cmp, label %cleanup91, label %if.end

if.end:
  %cmp5 = icmp eq i32 %and2, 0
  br i1 %cmp5, label %cleanup91, label %if.end7

if.end7:
  %sub = add nsw i32 %and2, -127
  %cmp8 = icmp ult i32 %and2, 121
  %sub10 = sub nsw i32 120, %and2
  %exponent_diff.0 = select i1 %cmp8, i32 %sub10, i32 0
  %add12 = or i32 %and1, 8388608
  %2 = tail call i32 @llvm.umin.i32(i32 %exponent_diff.0, i32 11)
  %notmask = shl nsw i32 -1048576, %2
  %sub16 = xor i32 %notmask, -1
  %and17 = and i32 %add12, %sub16
  %sub21 = add nsw i32 %exponent_diff.0, 19
  %.sroa.speculated140 = tail call i32 @llvm.umin.i32(i32 %sub21, i32 31)
  %shl23 = shl nuw i32 1, %.sroa.speculated140
  %cmp24 = icmp eq i32 %and17, %shl23
  %cmp25 = icmp sgt i32 %exponent_diff.0, 0
  %.sroa.speculated = tail call i32 @llvm.umin.i32(i32 %exponent_diff.0, i32 31)
  %shr30 = select i1 %cmp25, i32 %.sroa.speculated, i32 0
  %mantissa.0 = lshr i32 %add12, %shr30
  %add40 = add nsw i32 %sub, %exponent_diff.0
  %and38 = lshr i32 %mantissa.0, 23
  %3 = or i32 %and38, -2
  %add41 = add nsw i32 %add40, %3
  %sub43 = add nsw i32 %add41, 9
  %and44 = lshr i32 %mantissa.0, 20
  %4 = and i32 %and44, 1
  %sext = add nuw nsw i32 %4, 1048575
  %cond51 = select i1 %cmp24, i32 %sext, i32 0
  %cond54 = add nuw nsw i32 %cond51, %mantissa.0
  %and55 = and i32 %cond54, 1048575
  %add56 = add nuw nsw i32 %and55, %mantissa.0
  %cmp57 = icmp ne i32 %sub43, 0
  %and58 = and i32 %add56, 8388608
  %tobool59.not = icmp eq i32 %and58, 0
  %or.cond133 = select i1 %cmp57, i1 true, i1 %tobool59.not
  br i1 %or.cond133, label %if.else61, label %if.end76

if.else61:
  %tobool63.not = icmp ugt i32 %add56, 16777215
  %inc = add nsw i32 %add41, 10
  %f8_exponent.0 = select i1 %tobool63.not, i32 %inc, i32 %sub43
  %cmp69 = icmp sgt i32 %f8_exponent.0, 15
  br i1 %cmp69, label %if.then70, label %if.end71

if.then70:
  %5 = trunc i32 %1 to i8
  %conv = or i8 %5, 127
  br label %cleanup91

if.end71:
  %shr65 = zext i1 %tobool63.not to i32
  %mantissa.1 = lshr i32 %add56, %shr65
  %cmp72 = icmp eq i32 %f8_exponent.0, 0
  %cmp74 = icmp ult i32 %mantissa.1, 1048576
  %or.cond = select i1 %cmp72, i1 %cmp74, i1 false
  br i1 %or.cond, label %cleanup91, label %if.end76

if.end76:
  %f8_exponent.0150160 = phi i32 [ %f8_exponent.0, %if.end71 ], [ 1, %if.end7 ]
  %shr68152159.in = phi i32 [ %mantissa.1, %if.end71 ], [ %add56, %if.end7 ]
  %shr68152159 = lshr i32 %shr68152159.in, 20
  %and77 = and i32 %shr68152159, 7
  %shl79 = shl nsw i32 %f8_exponent.0150160, 3
  %or = or i32 %shl79, %shl
  %or80 = or i32 %or, %and77
  %conv81 = trunc i32 %or80 to i8
  br label %cleanup91

cleanup91:
  %retval.1 = phi i8 [ -128, %entry ], [ 0, %if.end ], [ %conv, %if.then70 ], [ %conv81, %if.end76 ], [ 0, %if.end71 ]
  ret i8 %retval.1
}

declare i32 @llvm.umin.i32(i32, i32)
