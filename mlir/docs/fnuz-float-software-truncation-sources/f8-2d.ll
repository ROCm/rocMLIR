; ModuleID = 'f8-2d.cpp'
source_filename = "f8-2d.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$_Z10cast_to_f8IfLb0ELb0EEhT_jjbj = comdat any

@e4m3 = external dso_local local_unnamed_addr global i8, align 1
@e5m2 = external dso_local local_unnamed_addr global i8, align 1

; Function Attrs: mustprogress uwtable
define dso_local void @_Z3foof(float noundef %v) local_unnamed_addr #0 {
entry:
  %call = tail call noundef zeroext i8 @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(float noundef %v, i32 noundef 3, i32 noundef 4, i1 noundef zeroext false, i32 noundef 0)
  store i8 %call, ptr @e4m3, align 1, !tbaa !3
  %call1 = tail call noundef zeroext i8 @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(float noundef %v, i32 noundef 2, i32 noundef 5, i1 noundef zeroext false, i32 noundef 0)
  store i8 %call1, ptr @e5m2, align 1, !tbaa !3
  ret void
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local noundef zeroext i8 @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(float noundef %f_x, i32 noundef %Wm, i32 noundef %We, i1 noundef zeroext %stoch, i32 noundef %rng) local_unnamed_addr #0 comdat {
entry:
  %0 = bitcast float %f_x to i32
  %and1 = and i32 %0, 8388607
  %shr = lshr i32 %0, 23
  %and2 = and i32 %shr, 255
  %1 = lshr i32 %0, 24
  %shl = and i32 %1, 128
  %shl4 = shl nuw i32 1, %We
  %sub = add nsw i32 %shl4, -1
  %shl5 = shl i32 %sub, %Wm
  %add = add i32 %shl5, %shl
  %notmask = shl nsw i32 -1, %Wm
  %sub11 = xor i32 %notmask, -1
  %add13 = add i32 %add, %sub11
  %cmp = icmp eq i32 %Wm, 2
  %and16 = and i32 %0, 2139095040
  %cmp17 = icmp eq i32 %and16, 2139095040
  br i1 %cmp17, label %if.then, label %if.end26

if.then:                                          ; preds = %entry
  %cmp18 = icmp eq i32 %Wm, 3
  br i1 %cmp18, label %if.then19, label %if.end

if.then19:                                        ; preds = %if.then
  %conv = trunc i32 %add13 to i8
  br label %cleanup165

if.end:                                           ; preds = %if.then
  %sub20 = sub i32 23, %Wm
  %shr21 = lshr i32 %and1, %sub20
  %and24 = and i32 %shr21, %sub11
  %or = or i32 %add, %and24
  %conv25 = trunc i32 %or to i8
  br label %cleanup165

if.end26:                                         ; preds = %entry
  switch i32 %0, label %if.end32 [
    i32 0, label %cleanup165
    i32 -2147483648, label %if.then30
  ]

if.then30:                                        ; preds = %if.end26
  br label %cleanup165

if.end32:                                         ; preds = %if.end26
  %sub33 = add i32 %We, -1
  %notmask244 = shl nsw i32 -1, %sub33
  %cmp38 = icmp eq i32 %and2, 0
  %cmp39 = icmp ne i32 %and1, 0
  %or.cond = and i1 %cmp39, %cmp38
  br i1 %or.cond, label %if.then40, label %if.else43

if.then40:                                        ; preds = %if.end32
  %sub42 = add nsw i32 %notmask244, 128
  br label %if.end51

if.else43:                                        ; preds = %if.end32
  %sub37 = add nsw i32 %notmask244, 2
  %sub44 = add nsw i32 %and2, -127
  %cmp45.not = icmp sgt i32 %sub44, %sub37
  %sub47 = sub nsw i32 %sub37, %sub44
  %add50 = or i32 %and1, 8388608
  %spec.select260 = select i1 %cmp45.not, i32 0, i32 %sub47
  br label %if.end51

if.end51:                                         ; preds = %if.else43, %if.then40
  %act_exponent.0 = phi i32 [ -126, %if.then40 ], [ %sub44, %if.else43 ]
  %exponent_diff.1 = phi i32 [ %sub42, %if.then40 ], [ %spec.select260, %if.else43 ]
  %mantissa.0 = phi i32 [ %and1, %if.then40 ], [ %add50, %if.else43 ]
  %sub53 = sub i32 23, %Wm
  %add54 = add i32 %exponent_diff.1, %sub53
  %.sroa.speculated256 = tail call i32 @llvm.umin.i32(i32 %add54, i32 31)
  %notmask245 = shl nsw i32 -1, %.sroa.speculated256
  %sub57 = xor i32 %notmask245, -1
  %and58 = and i32 %mantissa.0, %sub57
  %sub63 = add i32 %add54, -1
  %.sroa.speculated253 = tail call i32 @llvm.umin.i32(i32 %sub63, i32 31)
  %shl65 = shl nuw i32 1, %.sroa.speculated253
  %cmp66 = icmp eq i32 %and58, %shl65
  %cmp68 = icmp sgt i32 %exponent_diff.1, 0
  br i1 %cmp68, label %if.then69, label %if.else74

if.then69:                                        ; preds = %if.end51
  %.sroa.speculated = tail call i32 @llvm.umin.i32(i32 %exponent_diff.1, i32 31)
  %shr73 = lshr i32 %mantissa.0, %.sroa.speculated
  br label %if.end80

if.else74:                                        ; preds = %if.end51
  %cmp75 = icmp eq i32 %exponent_diff.1, -1
  %shl78 = zext i1 %cmp75 to i32
  %spec.select = shl nuw nsw i32 %mantissa.0, %shl78
  br label %if.end80

if.end80:                                         ; preds = %if.else74, %if.then69
  %mantissa.1 = phi i32 [ %shr73, %if.then69 ], [ %spec.select, %if.else74 ]
  %and81 = lshr i32 %mantissa.1, 23
  %2 = or i32 %and81, -2
  %add83 = sub i32 %act_exponent.0, %notmask244
  %add84 = add i32 %add83, %exponent_diff.1
  %sub87 = add i32 %add84, %2
  %shl89 = shl nuw i32 1, %sub53
  %sub90 = add i32 %shl89, -1
  %and93 = and i32 %mantissa.1, %shl89
  %tobool94.not = icmp eq i32 %and93, 0
  %narrow = select i1 %cmp66, i1 %tobool94.not, i1 false
  %cond106 = sext i1 %narrow to i32
  %rng.mux = add nsw i32 %mantissa.1, %cond106
  %cond111 = select i1 %stoch, i32 %rng, i32 %rng.mux
  %and112 = and i32 %cond111, %sub90
  %add113 = add i32 %and112, %mantissa.1
  %cmp114 = icmp ne i32 %sub87, 0
  %and116 = and i32 %add113, 8388608
  %tobool117.not = icmp eq i32 %and116, 0
  %or.cond246 = select i1 %cmp114, i1 true, i1 %tobool117.not
  br i1 %or.cond246, label %if.else119, label %if.end125

if.else119:                                       ; preds = %if.end80
  %and120 = and i32 %add113, 16777216
  %tobool121.not = icmp eq i32 %and120, 0
  br i1 %tobool121.not, label %if.end125, label %if.then122

if.then122:                                       ; preds = %if.else119
  %shr123 = lshr i32 %add113, 1
  %inc = add nsw i32 %sub87, 1
  br label %if.end125

if.end125:                                        ; preds = %if.end80, %if.else119, %if.then122
  %f8_exponent.0 = phi i32 [ %inc, %if.then122 ], [ %sub87, %if.else119 ], [ 1, %if.end80 ]
  %mantissa.2 = phi i32 [ %shr123, %if.then122 ], [ %add113, %if.else119 ], [ %add113, %if.end80 ]
  %shr127 = lshr i32 %mantissa.2, %sub53
  %cmp129 = icmp eq i32 %Wm, 3
  %cond130.neg = select i1 %cmp129, i32 -1, i32 -2
  %sub131 = add i32 %cond130.neg, %shl4
  %cmp132 = icmp sgt i32 %f8_exponent.0, %sub131
  br i1 %cmp132, label %if.then133, label %if.end140

if.then133:                                       ; preds = %if.end125
  %cond138 = select i1 %cmp, i32 %add, i32 %add13
  br label %cleanup

if.end140:                                        ; preds = %if.end125
  %cmp141 = icmp eq i32 %f8_exponent.0, 0
  %cmp143 = icmp eq i32 %shr127, 0
  %or.cond177 = select i1 %cmp141, i1 %cmp143, i1 false
  br i1 %or.cond177, label %cleanup, label %if.end147

if.end147:                                        ; preds = %if.end140
  %and150 = and i32 %shr127, %sub11
  %shl152 = shl i32 %f8_exponent.0, %Wm
  %3 = or i32 %shl152, %and150
  %or154 = or i32 %3, %shl
  br label %cleanup

cleanup:                                          ; preds = %if.end140, %if.end147, %if.then133
  %retval.0.in = phi i32 [ %cond138, %if.then133 ], [ %or154, %if.end147 ], [ %shl, %if.end140 ]
  %retval.0 = trunc i32 %retval.0.in to i8
  br label %cleanup165

cleanup165:                                       ; preds = %if.end26, %cleanup, %if.then30, %if.end, %if.then19
  %retval.1 = phi i8 [ %conv, %if.then19 ], [ %conv25, %if.end ], [ -128, %if.then30 ], [ %retval.0, %cleanup ], [ 0, %if.end26 ]
  ret i8 %retval.1
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.umin.i32(i32, i32) #1

attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"uwtable", i32 2}
!2 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.1 24154 f53cd7e03908085f4932f7329464cd446426436a)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
