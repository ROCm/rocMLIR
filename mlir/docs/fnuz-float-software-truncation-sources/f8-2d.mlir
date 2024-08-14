#tbaa_root = #llvm.tbaa_root<id = "Simple C++ TBAA">
#tbaa_type_desc = #llvm.tbaa_type_desc<id = "omnipotent char", members = {<#tbaa_root, 0>}>
#tbaa_tag = #llvm.tbaa_tag<base_type = #tbaa_type_desc, access_type = #tbaa_type_desc, offset = 0>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  llvm.comdat @__llvm_global_comdat {
    llvm.comdat_selector @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj any
  }
  llvm.mlir.global external local_unnamed_addr @e4m3() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
  llvm.mlir.global external local_unnamed_addr @e5m2() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
  llvm.func local_unnamed_addr @_Z3foof(%arg0: f32 {llvm.noundef}) attributes {passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(3 : i32) : i32
    %1 = llvm.mlir.constant(4 : i32) : i32
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.addressof @e4m3 : !llvm.ptr
    %5 = llvm.mlir.constant(2 : i32) : i32
    %6 = llvm.mlir.constant(5 : i32) : i32
    %7 = llvm.mlir.addressof @e5m2 : !llvm.ptr
    %8 = llvm.call @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(%arg0, %0, %1, %2, %3) : (f32, i32, i32, i1, i32) -> i8
    llvm.store %8, %4 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    %9 = llvm.call @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(%arg0, %5, %6, %2, %3) : (f32, i32, i32, i1, i32) -> i8
    llvm.store %9, %7 {alignment = 1 : i64, tbaa = [#tbaa_tag]} : i8, !llvm.ptr
    llvm.return
  }
  llvm.func linkonce_odr local_unnamed_addr @_Z10cast_to_f8IfLb0ELb0EEhT_jjbj(%arg0: f32 {llvm.noundef}, %arg1: i32 {llvm.noundef}, %arg2: i32 {llvm.noundef}, %arg3: i1 {llvm.noundef, llvm.zeroext}, %arg4: i32 {llvm.noundef}) -> (i8 {llvm.noundef, llvm.zeroext}) comdat(@__llvm_global_comdat::@_Z10cast_to_f8IfLb0ELb0EEhT_jjbj) attributes {passthrough = ["mustprogress", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(8388607 : i32) : i32
    %1 = llvm.mlir.constant(23 : i32) : i32
    %2 = llvm.mlir.constant(255 : i32) : i32
    %3 = llvm.mlir.constant(24 : i32) : i32
    %4 = llvm.mlir.constant(128 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.mlir.constant(-1 : i32) : i32
    %7 = llvm.mlir.constant(2 : i32) : i32
    %8 = llvm.mlir.constant(2139095040 : i32) : i32
    %9 = llvm.mlir.constant(0 : i8) : i8
    %10 = llvm.mlir.constant(-128 : i8) : i8
    %11 = llvm.mlir.constant(0 : i32) : i32
    %12 = llvm.mlir.constant(-127 : i32) : i32
    %13 = llvm.mlir.constant(8388608 : i32) : i32
    %14 = llvm.mlir.constant(-126 : i32) : i32
    %15 = llvm.mlir.constant(31 : i32) : i32
    %16 = llvm.mlir.constant(-2 : i32) : i32
    %17 = llvm.mlir.constant(false) : i1
    %18 = llvm.mlir.constant(true) : i1
    %19 = llvm.mlir.constant(16777216 : i32) : i32
    %20 = llvm.mlir.constant(3 : i32) : i32
    %21 = llvm.bitcast %arg0 : f32 to i32
    %22 = llvm.and %21, %0  : i32
    %23 = llvm.lshr %21, %1  : i32
    %24 = llvm.and %23, %2  : i32
    %25 = llvm.lshr %21, %3  : i32
    %26 = llvm.and %25, %4  : i32
    %27 = llvm.shl %5, %arg2 overflow<nuw> : i32
    %28 = llvm.add %27, %6 overflow<nsw> : i32
    %29 = llvm.shl %28, %arg1 : i32
    %30 = llvm.add %29, %26 : i32
    %31 = llvm.shl %6, %arg1 overflow<nsw> : i32
    %32 = llvm.xor %31, %6  : i32
    %33 = llvm.add %30, %32 : i32
    %34 = llvm.icmp "eq" %arg1, %7 : i32
    %35 = llvm.and %21, %8  : i32
    %36 = llvm.icmp "eq" %35, %8 : i32
    llvm.cond_br %36, ^bb1, ^bb4
  ^bb1:  // pred: ^bb0
    %37 = llvm.icmp "eq" %arg1, %20 : i32
    llvm.cond_br %37, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %38 = llvm.trunc %33 : i32 to i8
    llvm.br ^bb20(%38 : i8)
  ^bb3:  // pred: ^bb1
    %39 = llvm.sub %1, %arg1 : i32
    %40 = llvm.lshr %22, %39  : i32
    %41 = llvm.and %40, %32  : i32
    %42 = llvm.add %30, %41 : i32
    %43 = llvm.trunc %42 : i32 to i8
    llvm.br ^bb20(%43 : i8)
  ^bb4:  // pred: ^bb0
    llvm.switch %21 : i32, ^bb6 [
      0: ^bb20(%9 : i8),
      2147483648: ^bb5
    ]
  ^bb5:  // pred: ^bb4
    llvm.br ^bb20(%10 : i8)
  ^bb6:  // pred: ^bb4
    %44 = llvm.add %arg2, %6 : i32
    %45 = llvm.shl %6, %44 overflow<nsw> : i32
    %46 = llvm.icmp "eq" %24, %11 : i32
    %47 = llvm.icmp "ne" %22, %11 : i32
    %48 = llvm.and %47, %46  : i1
    llvm.cond_br %48, ^bb7, ^bb8
  ^bb7:  // pred: ^bb6
    %49 = llvm.add %45, %4 overflow<nsw> : i32
    llvm.br ^bb9(%14, %49, %22 : i32, i32, i32)
  ^bb8:  // pred: ^bb6
    %50 = llvm.add %45, %7 overflow<nsw> : i32
    %51 = llvm.add %24, %12 overflow<nsw> : i32
    %52 = llvm.icmp "sgt" %51, %50 : i32
    %53 = llvm.sub %50, %51 overflow<nsw> : i32
    %54 = llvm.or %22, %13  : i32
    %55 = llvm.select %52, %11, %53 : i1, i32
    llvm.br ^bb9(%51, %55, %54 : i32, i32, i32)
  ^bb9(%56: i32, %57: i32, %58: i32):  // 2 preds: ^bb7, ^bb8
    %59 = llvm.sub %1, %arg1 : i32
    %60 = llvm.add %57, %59 : i32
    %61 = llvm.intr.umin(%60, %15)  : (i32, i32) -> i32
    %62 = llvm.shl %6, %61 overflow<nsw> : i32
    %63 = llvm.xor %62, %6  : i32
    %64 = llvm.and %58, %63  : i32
    %65 = llvm.add %60, %6 : i32
    %66 = llvm.intr.umin(%65, %15)  : (i32, i32) -> i32
    %67 = llvm.shl %5, %66 overflow<nuw> : i32
    %68 = llvm.icmp "eq" %64, %67 : i32
    %69 = llvm.icmp "sgt" %57, %11 : i32
    llvm.cond_br %69, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %70 = llvm.intr.umin(%57, %15)  : (i32, i32) -> i32
    %71 = llvm.lshr %58, %70  : i32
    llvm.br ^bb12(%71 : i32)
  ^bb11:  // pred: ^bb9
    %72 = llvm.icmp "eq" %57, %6 : i32
    %73 = llvm.zext %72 : i1 to i32
    %74 = llvm.shl %58, %73 overflow<nsw, nuw> : i32
    llvm.br ^bb12(%74 : i32)
  ^bb12(%75: i32):  // 2 preds: ^bb10, ^bb11
    %76 = llvm.lshr %75, %1  : i32
    %77 = llvm.or %76, %16  : i32
    %78 = llvm.sub %56, %45 : i32
    %79 = llvm.add %78, %57 : i32
    %80 = llvm.add %79, %77 : i32
    %81 = llvm.shl %5, %59 overflow<nuw> : i32
    %82 = llvm.add %81, %6 : i32
    %83 = llvm.and %75, %81  : i32
    %84 = llvm.icmp "eq" %83, %11 : i32
    %85 = llvm.select %68, %84, %17 : i1, i1
    %86 = llvm.sext %85 : i1 to i32
    %87 = llvm.add %75, %86 overflow<nsw> : i32
    %88 = llvm.select %arg3, %arg4, %87 : i1, i32
    %89 = llvm.and %88, %82  : i32
    %90 = llvm.add %89, %75 : i32
    %91 = llvm.icmp "ne" %80, %11 : i32
    %92 = llvm.and %90, %13  : i32
    %93 = llvm.icmp "eq" %92, %11 : i32
    %94 = llvm.select %91, %18, %93 : i1, i1
    llvm.cond_br %94, ^bb13, ^bb15(%5, %90 : i32, i32)
  ^bb13:  // pred: ^bb12
    %95 = llvm.and %90, %19  : i32
    %96 = llvm.icmp "eq" %95, %11 : i32
    llvm.cond_br %96, ^bb15(%80, %90 : i32, i32), ^bb14
  ^bb14:  // pred: ^bb13
    %97 = llvm.lshr %90, %5  : i32
    %98 = llvm.add %80, %5 overflow<nsw> : i32
    llvm.br ^bb15(%98, %97 : i32, i32)
  ^bb15(%99: i32, %100: i32):  // 3 preds: ^bb12, ^bb13, ^bb14
    %101 = llvm.lshr %100, %59  : i32
    %102 = llvm.icmp "eq" %arg1, %20 : i32
    %103 = llvm.select %102, %6, %16 : i1, i32
    %104 = llvm.add %103, %27 : i32
    %105 = llvm.icmp "sgt" %99, %104 : i32
    llvm.cond_br %105, ^bb16, ^bb17
  ^bb16:  // pred: ^bb15
    %106 = llvm.select %34, %30, %33 : i1, i32
    llvm.br ^bb19(%106 : i32)
  ^bb17:  // pred: ^bb15
    %107 = llvm.icmp "eq" %99, %11 : i32
    %108 = llvm.icmp "eq" %101, %11 : i32
    %109 = llvm.select %107, %108, %17 : i1, i1
    llvm.cond_br %109, ^bb19(%26 : i32), ^bb18
  ^bb18:  // pred: ^bb17
    %110 = llvm.and %101, %32  : i32
    %111 = llvm.shl %99, %arg1 : i32
    %112 = llvm.or %111, %110  : i32
    %113 = llvm.or %112, %26  : i32
    llvm.br ^bb19(%113 : i32)
  ^bb19(%114: i32):  // 3 preds: ^bb16, ^bb17, ^bb18
    %115 = llvm.trunc %114 : i32 to i8
    llvm.br ^bb20(%115 : i8)
  ^bb20(%116: i8):  // 5 preds: ^bb2, ^bb3, ^bb4, ^bb5, ^bb19
    llvm.return %116 : i8
  }
}
