// This test will fail as long as the backend bug (case filed here:
// https://ontrack-internal.amd.com/browse/SWDEV-559105) remains unimplemented.
// When this passes, we can go ahead and remove this test and update the
// gfx12 workaround that was added in GridwiseGemmToBlockwise
// (see the PR here https://github.com/ROCm/rocMLIR/pull/1990)

// XFAIL: *
// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -c | mlir-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext --entry-point-result=void | FileCheck %s
// CHECK: [1 1 1]

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1) -> (d0, d1)>
#map8 = affine_map<(d0, d1) -> (d0)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map10 = affine_map<(d0, d1) -> (d1)>
#map11 = affine_map<(d0, d1, d2) -> (d1)>
module attributes {mhal.arch = "##TOKEN_ARCH##"} {
  memref.global "private" constant @__constant_384xi32 : memref<384xi32> = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000"> {alignment = 64 : i64}
  memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[0, 1, 2, 3]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_12xi32 : memref<12xi32> = dense<[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]> {alignment = 64 : i64}
  func.func @rock_attention(%arg0: memref<768xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 1536 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg1: memref<147456xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 294912 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg2: memref<147456xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 294912 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg3: memref<4608xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 9216 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg4: memref<4608xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 9216 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg5: memref<3xi32> {llvm.align = 16 : i64, llvm.dereferenceable = 12 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.readonly}, %arg6: memref<48xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 96 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.writeonly}, %arg7: memref<3072xf16> {llvm.align = 16 : i64, llvm.dereferenceable = 6144 : i64, llvm.inreg, llvm.noalias, llvm.nocapture, llvm.nofree, llvm.nonnull, llvm.noundef, llvm.writeonly}) attributes {block_size = 32 : i32, features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, grid_size = 24 : i32, kernel, mhal.arch = "##TOKEN_ARCH##"} {
    %cst = arith.constant dense<0.000000e+00> : vector<8xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf16>
    %false = arith.constant false
    %c752 = arith.constant 752 : index
    %c720 = arith.constant 720 : index
    %c688 = arith.constant 688 : index
    %c656 = arith.constant 656 : index
    %c624 = arith.constant 624 : index
    %c592 = arith.constant 592 : index
    %c560 = arith.constant 560 : index
    %c528 = arith.constant 528 : index
    %c736 = arith.constant 736 : index
    %c704 = arith.constant 704 : index
    %c672 = arith.constant 672 : index
    %c640 = arith.constant 640 : index
    %c608 = arith.constant 608 : index
    %c576 = arith.constant 576 : index
    %c544 = arith.constant 544 : index
    %c512 = arith.constant 512 : index
    %c240 = arith.constant 240 : index
    %c208 = arith.constant 208 : index
    %c176 = arith.constant 176 : index
    %c144 = arith.constant 144 : index
    %c112 = arith.constant 112 : index
    %c80 = arith.constant 80 : index
    %c31 = arith.constant 31 : index
    %c30 = arith.constant 30 : index
    %c29 = arith.constant 29 : index
    %c27 = arith.constant 27 : index
    %c26 = arith.constant 26 : index
    %c25 = arith.constant 25 : index
    %c23 = arith.constant 23 : index
    %c22 = arith.constant 22 : index
    %c21 = arith.constant 21 : index
    %c19 = arith.constant 19 : index
    %c18 = arith.constant 18 : index
    %c17 = arith.constant 17 : index
    %c15 = arith.constant 15 : index
    %c14 = arith.constant 14 : index
    %c13 = arith.constant 13 : index
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    %c480 = arith.constant 480 : index
    %c416 = arith.constant 416 : index
    %c352 = arith.constant 352 : index
    %c288 = arith.constant 288 : index
    %c224 = arith.constant 224 : index
    %c160 = arith.constant 160 : index
    %c96 = arith.constant 96 : index
    %c448 = arith.constant 448 : index
    %c7 = arith.constant 7 : index
    %c28 = arith.constant 28 : index
    %c6 = arith.constant 6 : index
    %c320 = arith.constant 320 : index
    %c5 = arith.constant 5 : index
    %c20 = arith.constant 20 : index
    %c256 = arith.constant 256 : index
    %c192 = arith.constant 192 : index
    %c12 = arith.constant 12 : index
    %c128 = arith.constant 128 : index
    %c1152 = arith.constant 1152 : index
    %c24 = arith.constant 24 : index
    %c0 = arith.constant 0 : index
    %c48 = arith.constant 48 : index
    %c3072 = arith.constant 3072 : index
    %c4608 = arith.constant 4608 : index
    %c768 = arith.constant 768 : index
    %c16 = arith.constant 16 : index
    %c384 = arith.constant 384 : index
    %c64 = arith.constant 64 : index
    %cst_1 = arith.constant dense<1.44269502> : vector<2xf32>
    %c8 = arith.constant 8 : index
    %cst_2 = arith.constant 0xFF800000 : f32
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %cst_4 = arith.constant dense<0.000000e+00> : vector<8xf32>
    %c2 = arith.constant 2 : index
    %cst_5 = arith.constant 6.933590e-01 : f16
    %0 = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
    %1 = rock.alloc() : memref<2048xi8, #gpu.address_space<workgroup>>
    %2 = rock.workgroup_id : index
    %3 = rock.workitem_id : index
    %4 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %5 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %6 = rock.alloc() : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %7 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %8 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
    %9 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
    %10 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
    %11 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
    %12 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %13 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %14 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %15 = rock.alloc() : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %16 = rock.alloc() : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %17 = rock.alloc() : memref<32xf32, #gpu.address_space<private>>
    %18 = rock.alloc() : memref<2x32xf32, #gpu.address_space<private>>
    %19 = rock.alloc() : memref<2x32xf16, #gpu.address_space<private>>
    %20 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    %21 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    cf.br ^bb1(%c0 : index)
  ^bb1(%22: index):  // 2 preds: ^bb0, ^bb2
    %23 = arith.cmpi slt, %22, %c2 : index
    cf.cond_br %23, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    memref.store %cst_2, %20[%22] : memref<2xf32, #gpu.address_space<private>>
    %24 = arith.addi %22, %c1 : index
    cf.br ^bb1(%24 : index)
  ^bb3:  // pred: ^bb1
    %25 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    cf.br ^bb4(%c0 : index)
  ^bb4(%26: index):  // 2 preds: ^bb3, ^bb5
    %27 = arith.cmpi slt, %26, %c2 : index
    cf.cond_br %27, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    memref.store %cst_3, %25[%26] : memref<2xf32, #gpu.address_space<private>>
    %28 = arith.addi %26, %c1 : index
    cf.br ^bb4(%28 : index)
  ^bb6:  // pred: ^bb4
    %29 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    cf.br ^bb7(%c0 : index)
  ^bb7(%30: index):  // 2 preds: ^bb6, ^bb11
    %31 = arith.cmpi slt, %30, %c2 : index
    cf.cond_br %31, ^bb8, ^bb12
  ^bb8:  // pred: ^bb7
    cf.br ^bb9(%c0 : index)
  ^bb9(%32: index):  // 2 preds: ^bb8, ^bb10
    %33 = arith.cmpi slt, %32, %c32 : index
    cf.cond_br %33, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    memref.store %cst_3, %18[%30, %32] : memref<2x32xf32, #gpu.address_space<private>>
    %34 = arith.addi %32, %c1 : index
    cf.br ^bb9(%34 : index)
  ^bb11:  // pred: ^bb9
    %35 = arith.addi %30, %c1 : index
    cf.br ^bb7(%35 : index)
  ^bb12:  // pred: ^bb7
    %36 = arith.divui %2, %c4 : index
    %37 = arith.remui %2, %c4 : index
    %38 = rock.alloc() : memref<1xi32, #gpu.address_space<private>>
    %39 = arith.muli %36, %c2 overflow<nsw> : index
    %40 = arith.divui %39, %c4 : index
    %memspacecast = memref.memory_space_cast %arg5 : memref<3xi32> to memref<3xi32, #gpu.address_space<global>>
    %41 = memref.load %memspacecast[%40] : memref<3xi32, #gpu.address_space<global>>
    memref.store %41, %38[%c0] : memref<1xi32, #gpu.address_space<private>>
    %42 = memref.load %38[%c0] : memref<1xi32, #gpu.address_space<private>>
    %43 = arith.index_cast %42 : i32 to index
    %44 = arith.addi %43, %c32 : index
    %45 = arith.divui %44, %c32 : index
    %46 = arith.addi %45, %c3 : index
    %47 = arith.divui %46, %c4 : index
    %48 = arith.muli %37, %47 : index
    %49 = arith.addi %37, %c1 : index
    %50 = arith.muli %49, %47 : index
    %51 = arith.minui %45, %50 : index
    %52 = arith.subi %51, %c1 : index
    %53 = arith.cmpi ugt, %51, %48 : index
    cf.cond_br %53, ^bb13, ^bb144
  ^bb13:  // pred: ^bb12
    %54 = arith.divui %3, %c4 : index
    %55 = arith.remui %3, %c4 : index
    %56 = arith.muli %36, %c64 overflow<nsw> : index
    %memspacecast_6 = memref.memory_space_cast %arg1 : memref<147456xf16> to memref<147456xf16, #gpu.address_space<global>>
    %57 = arith.divui %3, %c8 : index
    %58 = arith.remui %3, %c8 : index
    %59 = arith.muli %57, %c8 overflow<nsw> : index
    %60 = arith.addi %39, %59 : index
    %61 = arith.muli %60, %c64 overflow<nsw> : index
    %view = memref.view %1[%c0][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xf16, #gpu.address_space<workgroup>>
    %62 = arith.muli %55, %c8 overflow<nsw> : index
    %63 = arith.muli %54, %c128 overflow<nsw> : index
    %64 = arith.addi %63, %62 : index
    %view_7 = memref.view %0[%c0][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<1024xf16, #gpu.address_space<workgroup>>
    %65 = arith.muli %58, %c128 overflow<nsw> : index
    %66 = arith.addi %65, %59 : index
    %67 = arith.divui %3, %c16 : index
    %68 = arith.remui %3, %c16 : index
    %69 = arith.muli %67, %c512 overflow<nsw> : index
    %70 = arith.muli %67, %c8 overflow<nsw> : index
    %71 = arith.addi %39, %68 : index
    %72 = arith.muli %71, %c384 overflow<nsw> : index
    %view_8 = memref.view %0[%c0][] : memref<2048xi8, #gpu.address_space<workgroup>> to memref<256xi8, #gpu.address_space<workgroup>>
    %view_9 = memref.view %view_8[%c0][] : memref<256xi8, #gpu.address_space<workgroup>> to memref<64xf32, #gpu.address_space<workgroup>>
    %73 = arith.muli %3, %c2 overflow<nsw> : index
    %74 = arith.muli %67, %c256 overflow<nsw> : index
    %75 = arith.addi %74, %68 : index
    %76 = arith.muli %36, %c384 overflow<nsw> : index
    %memspacecast_10 = memref.memory_space_cast %arg2 : memref<147456xf16> to memref<147456xf16, #gpu.address_space<global>>
    cf.br ^bb14(%48 : index)
  ^bb14(%77: index):  // 2 preds: ^bb13, ^bb139
    %78 = arith.cmpi slt, %77, %51 : index
    cf.cond_br %78, ^bb15, ^bb140
  ^bb15:  // pred: ^bb14
    cf.br ^bb16(%c0 : index)
  ^bb16(%79: index):  // 2 preds: ^bb15, ^bb17
    %80 = arith.cmpi slt, %79, %c4 : index
    cf.cond_br %80, ^bb17, ^bb18
  ^bb17:  // pred: ^bb16
    memref.store %cst_4, %6[%79] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %81 = arith.addi %79, %c1 : index
    cf.br ^bb16(%81 : index)
  ^bb18:  // pred: ^bb16
    rock.lds_barrier
    %82 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %83 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %84 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %85 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %86 = arith.muli %77, %c4 overflow<nsw> : index
    %87 = arith.addi %86, %55 : index
    %88 = arith.muli %87, %c8 overflow<nsw> : index
    cf.br ^bb19(%c0 : index)
  ^bb19(%89: index):  // 2 preds: ^bb18, ^bb35
    %90 = arith.cmpi slt, %89, %c2 : index
    cf.cond_br %90, ^bb20, ^bb36
  ^bb20:  // pred: ^bb19
    %91 = arith.muli %89, %c8 overflow<nsw> : index
    %92 = arith.addi %91, %54 : index
    %93 = arith.muli %92, %c4 overflow<nsw> : index
    %94 = arith.addi %56, %93 : index
    %95 = arith.muli %94, %c384 overflow<nsw> : index
    %96 = arith.addi %95, %88 : index
    %97 = vector.load %memspacecast_6[%96] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %97, %84[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %98 = arith.addi %96, %c384 : index
    %99 = vector.load %memspacecast_6[%98] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %99, %84[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %100 = arith.addi %96, %c768 : index
    %101 = vector.load %memspacecast_6[%100] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %101, %84[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %102 = arith.addi %96, %c1152 : index
    %103 = vector.load %memspacecast_6[%102] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %103, %84[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %104 = arith.addi %91, %58 : index
    %105 = arith.muli %104, %c4 overflow<nsw> : index
    %106 = arith.addi %61, %105 : index
    %107 = arith.cmpi ult, %59, %c2 : index
    %108 = arith.select %107, %106, %c768 : index
    %109 = arith.index_cast %108 : index to i32
    %110 = amdgpu.raw_buffer_load %arg0[%109] : memref<768xf16>, i32 -> vector<4xf16>
    vector.store %110, %82[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    %111 = arith.addi %59, %c1 : index
    %112 = arith.cmpi ult, %111, %c2 : index
    %113 = arith.addi %106, %c64 : index
    %114 = arith.select %112, %113, %c768 : index
    %115 = arith.index_cast %114 : index to i32
    %116 = amdgpu.raw_buffer_load %arg0[%115] : memref<768xf16>, i32 -> vector<4xf16>
    vector.store %116, %82[%c4] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c12] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c20] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    vector.store %cst_0, %82[%c28] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    cf.br ^bb21(%c0 : index)
  ^bb21(%117: index):  // 2 preds: ^bb20, ^bb22
    %118 = arith.cmpi slt, %117, %c32 : index
    cf.cond_br %118, ^bb22, ^bb23
  ^bb22:  // pred: ^bb21
    %119 = arith.divui %117, %c8 : index
    %120 = arith.muli %119, %c8 overflow<nsw> : index
    %121 = arith.remui %117, %c8 : index
    %122 = arith.addi %120, %121 : index
    %123 = vector.load %84[%122] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %123, %85[%117] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %124 = arith.addi %117, %c8 : index
    cf.br ^bb21(%124 : index)
  ^bb23:  // pred: ^bb21
    %125 = vector.load %85[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %125, %view[%64] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %126 = arith.addi %64, %c32 : index
    %127 = vector.load %85[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %127, %view[%126] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %128 = arith.addi %64, %c64 : index
    %129 = vector.load %85[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %129, %view[%128] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %130 = arith.addi %64, %c96 : index
    %131 = vector.load %85[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %131, %view[%130] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    cf.br ^bb24(%c0 : index)
  ^bb24(%132: index):  // 2 preds: ^bb23, ^bb25
    %133 = arith.cmpi slt, %132, %c32 : index
    cf.cond_br %133, ^bb25, ^bb26
  ^bb25:  // pred: ^bb24
    %134 = arith.remui %132, %c8 : index
    %135 = arith.muli %134, %c4 overflow<nsw> : index
    %136 = arith.divui %132, %c8 : index
    %137 = arith.addi %135, %136 : index
    %138 = memref.load %82[%137] : memref<32xf16, #gpu.address_space<private>>
    memref.store %138, %83[%132] : memref<32xf16, #gpu.address_space<private>>
    %139 = arith.addi %132, %c1 : index
    cf.br ^bb24(%139 : index)
  ^bb26:  // pred: ^bb24
    %140 = vector.load %83[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %140, %view_7[%66] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %141 = arith.addi %66, %c32 : index
    %142 = vector.load %83[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %142, %view_7[%141] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %143 = arith.addi %66, %c64 : index
    %144 = vector.load %83[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %144, %view_7[%143] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %145 = arith.addi %66, %c96 : index
    %146 = vector.load %83[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %146, %view_7[%145] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    cf.br ^bb27(%c0 : index)
  ^bb27(%147: index):  // 2 preds: ^bb26, ^bb34
    %148 = arith.cmpi slt, %147, %c2 : index
    cf.cond_br %148, ^bb28, ^bb35
  ^bb28:  // pred: ^bb27
    %149 = arith.muli %147, %c16 overflow<nsw> : index
    %150 = arith.addi %149, %68 : index
    %151 = arith.addi %69, %150 : index
    %152 = memref.load %view[%151] : memref<1024xf16, #gpu.address_space<workgroup>>
    %153 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %154 = vector.insert %152, %153 [0] : f16 into vector<8xf16>
    memref.store %154, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %155 = arith.addi %151, %c32 : index
    %156 = memref.load %view[%155] : memref<1024xf16, #gpu.address_space<workgroup>>
    %157 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %158 = vector.insert %156, %157 [1] : f16 into vector<8xf16>
    memref.store %158, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %159 = arith.addi %151, %c64 : index
    %160 = memref.load %view[%159] : memref<1024xf16, #gpu.address_space<workgroup>>
    %161 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %162 = vector.insert %160, %161 [2] : f16 into vector<8xf16>
    memref.store %162, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %163 = arith.addi %151, %c96 : index
    %164 = memref.load %view[%163] : memref<1024xf16, #gpu.address_space<workgroup>>
    %165 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %166 = vector.insert %164, %165 [3] : f16 into vector<8xf16>
    memref.store %166, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %167 = arith.addi %151, %c128 : index
    %168 = memref.load %view[%167] : memref<1024xf16, #gpu.address_space<workgroup>>
    %169 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %170 = vector.insert %168, %169 [4] : f16 into vector<8xf16>
    memref.store %170, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %171 = arith.addi %151, %c160 : index
    %172 = memref.load %view[%171] : memref<1024xf16, #gpu.address_space<workgroup>>
    %173 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %174 = vector.insert %172, %173 [5] : f16 into vector<8xf16>
    memref.store %174, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %175 = arith.addi %151, %c192 : index
    %176 = memref.load %view[%175] : memref<1024xf16, #gpu.address_space<workgroup>>
    %177 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %178 = vector.insert %176, %177 [6] : f16 into vector<8xf16>
    memref.store %178, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %179 = arith.addi %151, %c224 : index
    %180 = memref.load %view[%179] : memref<1024xf16, #gpu.address_space<workgroup>>
    %181 = memref.load %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %182 = vector.insert %180, %181 [7] : f16 into vector<8xf16>
    memref.store %182, %4[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %183 = arith.addi %151, %c256 : index
    %184 = memref.load %view[%183] : memref<1024xf16, #gpu.address_space<workgroup>>
    %185 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %186 = vector.insert %184, %185 [0] : f16 into vector<8xf16>
    memref.store %186, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %187 = arith.addi %151, %c288 : index
    %188 = memref.load %view[%187] : memref<1024xf16, #gpu.address_space<workgroup>>
    %189 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %190 = vector.insert %188, %189 [1] : f16 into vector<8xf16>
    memref.store %190, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %191 = arith.addi %151, %c320 : index
    %192 = memref.load %view[%191] : memref<1024xf16, #gpu.address_space<workgroup>>
    %193 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %194 = vector.insert %192, %193 [2] : f16 into vector<8xf16>
    memref.store %194, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %195 = arith.addi %151, %c352 : index
    %196 = memref.load %view[%195] : memref<1024xf16, #gpu.address_space<workgroup>>
    %197 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %198 = vector.insert %196, %197 [3] : f16 into vector<8xf16>
    memref.store %198, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %199 = arith.addi %151, %c384 : index
    %200 = memref.load %view[%199] : memref<1024xf16, #gpu.address_space<workgroup>>
    %201 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %202 = vector.insert %200, %201 [4] : f16 into vector<8xf16>
    memref.store %202, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %203 = arith.addi %151, %c416 : index
    %204 = memref.load %view[%203] : memref<1024xf16, #gpu.address_space<workgroup>>
    %205 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %206 = vector.insert %204, %205 [5] : f16 into vector<8xf16>
    memref.store %206, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %207 = arith.addi %151, %c448 : index
    %208 = memref.load %view[%207] : memref<1024xf16, #gpu.address_space<workgroup>>
    %209 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %210 = vector.insert %208, %209 [6] : f16 into vector<8xf16>
    memref.store %210, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %211 = arith.addi %151, %c480 : index
    %212 = memref.load %view[%211] : memref<1024xf16, #gpu.address_space<workgroup>>
    %213 = memref.load %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %214 = vector.insert %212, %213 [7] : f16 into vector<8xf16>
    memref.store %214, %4[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %215 = arith.muli %147, %c2 overflow<nsw> : index
    cf.br ^bb29(%c0 : index)
  ^bb29(%216: index):  // 2 preds: ^bb28, ^bb33
    %217 = arith.cmpi slt, %216, %c2 : index
    cf.cond_br %217, ^bb30, ^bb34
  ^bb30:  // pred: ^bb29
    %218 = arith.muli %216, %c16 overflow<nsw> : index
    %219 = arith.addi %218, %68 : index
    %220 = arith.addi %69, %219 : index
    %221 = memref.load %view_7[%220] : memref<1024xf16, #gpu.address_space<workgroup>>
    %222 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %223 = vector.insert %221, %222 [0] : f16 into vector<8xf16>
    memref.store %223, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %224 = arith.addi %220, %c32 : index
    %225 = memref.load %view_7[%224] : memref<1024xf16, #gpu.address_space<workgroup>>
    %226 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %227 = vector.insert %225, %226 [1] : f16 into vector<8xf16>
    memref.store %227, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %228 = arith.addi %220, %c64 : index
    %229 = memref.load %view_7[%228] : memref<1024xf16, #gpu.address_space<workgroup>>
    %230 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %231 = vector.insert %229, %230 [2] : f16 into vector<8xf16>
    memref.store %231, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %232 = arith.addi %220, %c96 : index
    %233 = memref.load %view_7[%232] : memref<1024xf16, #gpu.address_space<workgroup>>
    %234 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %235 = vector.insert %233, %234 [3] : f16 into vector<8xf16>
    memref.store %235, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %236 = arith.addi %220, %c128 : index
    %237 = memref.load %view_7[%236] : memref<1024xf16, #gpu.address_space<workgroup>>
    %238 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %239 = vector.insert %237, %238 [4] : f16 into vector<8xf16>
    memref.store %239, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %240 = arith.addi %220, %c160 : index
    %241 = memref.load %view_7[%240] : memref<1024xf16, #gpu.address_space<workgroup>>
    %242 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %243 = vector.insert %241, %242 [5] : f16 into vector<8xf16>
    memref.store %243, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %244 = arith.addi %220, %c192 : index
    %245 = memref.load %view_7[%244] : memref<1024xf16, #gpu.address_space<workgroup>>
    %246 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %247 = vector.insert %245, %246 [6] : f16 into vector<8xf16>
    memref.store %247, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %248 = arith.addi %220, %c224 : index
    %249 = memref.load %view_7[%248] : memref<1024xf16, #gpu.address_space<workgroup>>
    %250 = memref.load %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %251 = vector.insert %249, %250 [7] : f16 into vector<8xf16>
    memref.store %251, %5[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %252 = arith.addi %220, %c256 : index
    %253 = memref.load %view_7[%252] : memref<1024xf16, #gpu.address_space<workgroup>>
    %254 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %255 = vector.insert %253, %254 [0] : f16 into vector<8xf16>
    memref.store %255, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %256 = arith.addi %220, %c288 : index
    %257 = memref.load %view_7[%256] : memref<1024xf16, #gpu.address_space<workgroup>>
    %258 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %259 = vector.insert %257, %258 [1] : f16 into vector<8xf16>
    memref.store %259, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %260 = arith.addi %220, %c320 : index
    %261 = memref.load %view_7[%260] : memref<1024xf16, #gpu.address_space<workgroup>>
    %262 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %263 = vector.insert %261, %262 [2] : f16 into vector<8xf16>
    memref.store %263, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %264 = arith.addi %220, %c352 : index
    %265 = memref.load %view_7[%264] : memref<1024xf16, #gpu.address_space<workgroup>>
    %266 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %267 = vector.insert %265, %266 [3] : f16 into vector<8xf16>
    memref.store %267, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %268 = arith.addi %220, %c384 : index
    %269 = memref.load %view_7[%268] : memref<1024xf16, #gpu.address_space<workgroup>>
    %270 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %271 = vector.insert %269, %270 [4] : f16 into vector<8xf16>
    memref.store %271, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %272 = arith.addi %220, %c416 : index
    %273 = memref.load %view_7[%272] : memref<1024xf16, #gpu.address_space<workgroup>>
    %274 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %275 = vector.insert %273, %274 [5] : f16 into vector<8xf16>
    memref.store %275, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %276 = arith.addi %220, %c448 : index
    %277 = memref.load %view_7[%276] : memref<1024xf16, #gpu.address_space<workgroup>>
    %278 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %279 = vector.insert %277, %278 [6] : f16 into vector<8xf16>
    memref.store %279, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %280 = arith.addi %220, %c480 : index
    %281 = memref.load %view_7[%280] : memref<1024xf16, #gpu.address_space<workgroup>>
    %282 = memref.load %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %283 = vector.insert %281, %282 [7] : f16 into vector<8xf16>
    memref.store %283, %5[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %284 = arith.addi %215, %216 : index
    cf.br ^bb31(%c0 : index)
  ^bb31(%285: index):  // 2 preds: ^bb30, ^bb32
    %286 = arith.cmpi slt, %285, %c2 : index
    cf.cond_br %286, ^bb32, ^bb33
  ^bb32:  // pred: ^bb31
    %287 = memref.load %4[%285] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %288 = memref.load %5[%285] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %289 = memref.load %6[%284] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %290 = amdgpu.wmma %287 * %288 + %289 {clamp} : vector<8xf16>, vector<8xf16>, vector<8xf32>
    memref.store %290, %6[%284] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %291 = arith.addi %285, %c1 : index
    cf.br ^bb31(%291 : index)
  ^bb33:  // pred: ^bb31
    %292 = arith.addi %216, %c1 : index
    cf.br ^bb29(%292 : index)
  ^bb34:  // pred: ^bb29
    %293 = arith.addi %147, %c1 : index
    cf.br ^bb27(%293 : index)
  ^bb35:  // pred: ^bb27
    %294 = arith.addi %89, %c1 : index
    cf.br ^bb19(%294 : index)
  ^bb36:  // pred: ^bb19
    %295 = memref.load %6[%c0] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %296 = arith.truncf %295 : vector<8xf32> to vector<8xf16>
    vector.store %296, %7[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %297 = memref.load %6[%c1] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %298 = arith.truncf %297 : vector<8xf32> to vector<8xf16>
    vector.store %298, %7[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %299 = memref.load %6[%c2] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %300 = arith.truncf %299 : vector<8xf32> to vector<8xf16>
    vector.store %300, %7[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %301 = memref.load %6[%c3] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %302 = arith.truncf %301 : vector<8xf32> to vector<8xf16>
    vector.store %302, %7[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %303 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %304 = arith.muli %77, %c32 overflow<nsw> : index
    %305 = arith.addi %304, %70 : index
    %306 = arith.addi %72, %305 : index
    %307 = arith.cmpi ult, %68, %c2 : index
    %308 = arith.select %307, %306, %c4608 : index
    %309 = arith.index_cast %308 : index to i32
    %310 = amdgpu.raw_buffer_load %arg3[%309] : memref<4608xf16>, i32 -> vector<8xf16>
    vector.store %310, %303[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %311 = arith.addi %68, %c16 : index
    vector.store %cst, %303[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %312 = arith.addi %306, %c16 : index
    %313 = arith.select %307, %312, %c4608 : index
    %314 = arith.index_cast %313 : index to i32
    %315 = amdgpu.raw_buffer_load %arg3[%314] : memref<4608xf16>, i32 -> vector<8xf16>
    vector.store %315, %303[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %cst, %303[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %316 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %317 = amdgpu.raw_buffer_load %arg4[%309] : memref<4608xf16>, i32 -> vector<8xf16>
    vector.store %317, %316[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %cst, %316[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %318 = amdgpu.raw_buffer_load %arg4[%314] : memref<4608xf16>, i32 -> vector<8xf16>
    vector.store %318, %316[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %cst, %316[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %319 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    cf.br ^bb37(%c0 : index)
  ^bb37(%320: index):  // 2 preds: ^bb36, ^bb38
    %321 = arith.cmpi slt, %320, %c32 : index
    cf.cond_br %321, ^bb38, ^bb39
  ^bb38:  // pred: ^bb37
    %322 = vector.load %7[%320] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    %323 = vector.load %303[%320] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    %324 = vector.load %316[%320] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    %325 = arith.mulf %322, %323 : vector<4xf16>
    %326 = arith.addf %325, %324 : vector<4xf16>
    vector.store %326, %319[%320] : memref<32xf16, #gpu.address_space<private>>, vector<4xf16>
    %327 = arith.addi %320, %c4 : index
    cf.br ^bb37(%327 : index)
  ^bb39:  // pred: ^bb37
    %328 = vector.load %319[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<32xf16>
    %329 = arith.extf %328 : vector<32xf16> to vector<32xf32>
    vector.store %329, %8[%c0] : memref<32xf32, #gpu.address_space<private>>, vector<32xf32>
    cf.br ^bb40(%c0 : index)
  ^bb40(%330: index):  // 2 preds: ^bb39, ^bb41
    %331 = arith.cmpi slt, %330, %c32 : index
    cf.cond_br %331, ^bb41, ^bb42
  ^bb41:  // pred: ^bb40
    %332 = vector.load %8[%330] : memref<32xf32, #gpu.address_space<private>>, vector<2xf32>
    %333 = arith.mulf %332, %cst_1 : vector<2xf32>
    vector.store %333, %8[%330] : memref<32xf32, #gpu.address_space<private>>, vector<2xf32>
    %334 = arith.addi %330, %c2 : index
    cf.br ^bb40(%334 : index)
  ^bb42:  // pred: ^bb40
    %335 = arith.cmpi eq, %307, %false : i1
    cf.cond_br %335, ^bb43, ^bb44
  ^bb43:  // pred: ^bb42
    memref.store %cst_2, %8[%c0] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c1] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c2] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c3] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c4] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c5] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c6] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c7] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb44
  ^bb44:  // 2 preds: ^bb42, ^bb43
    memref.store %cst_2, %8[%c8] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c9] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c10] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c11] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c12] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c13] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c14] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c15] : memref<32xf32, #gpu.address_space<private>>
    cf.cond_br %335, ^bb45, ^bb46
  ^bb45:  // pred: ^bb44
    memref.store %cst_2, %8[%c16] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c17] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c18] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c19] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c20] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c21] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c22] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c23] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb46
  ^bb46:  // 2 preds: ^bb44, ^bb45
    memref.store %cst_2, %8[%c24] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c25] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c26] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c27] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c28] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c29] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c30] : memref<32xf32, #gpu.address_space<private>>
    memref.store %cst_2, %8[%c31] : memref<32xf32, #gpu.address_space<private>>
    %336 = arith.cmpi eq, %77, %52 : index
    cf.cond_br %336, ^bb47, ^bb112
  ^bb47:  // pred: ^bb46
    %337 = arith.cmpi ugt, %305, %43 : index
    cf.cond_br %337, ^bb48, ^bb49
  ^bb48:  // pred: ^bb47
    memref.store %cst_2, %8[%c0] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb49
  ^bb49:  // 2 preds: ^bb47, ^bb48
    %338 = arith.addi %305, %c1 : index
    %339 = arith.cmpi ugt, %338, %43 : index
    cf.cond_br %339, ^bb50, ^bb51
  ^bb50:  // pred: ^bb49
    memref.store %cst_2, %8[%c1] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb51
  ^bb51:  // 2 preds: ^bb49, ^bb50
    %340 = arith.addi %305, %c2 : index
    %341 = arith.cmpi ugt, %340, %43 : index
    cf.cond_br %341, ^bb52, ^bb53
  ^bb52:  // pred: ^bb51
    memref.store %cst_2, %8[%c2] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb53
  ^bb53:  // 2 preds: ^bb51, ^bb52
    %342 = arith.addi %305, %c3 : index
    %343 = arith.cmpi ugt, %342, %43 : index
    cf.cond_br %343, ^bb54, ^bb55
  ^bb54:  // pred: ^bb53
    memref.store %cst_2, %8[%c3] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb55
  ^bb55:  // 2 preds: ^bb53, ^bb54
    %344 = arith.addi %305, %c4 : index
    %345 = arith.cmpi ugt, %344, %43 : index
    cf.cond_br %345, ^bb56, ^bb57
  ^bb56:  // pred: ^bb55
    memref.store %cst_2, %8[%c4] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb57
  ^bb57:  // 2 preds: ^bb55, ^bb56
    %346 = arith.addi %305, %c5 : index
    %347 = arith.cmpi ugt, %346, %43 : index
    cf.cond_br %347, ^bb58, ^bb59
  ^bb58:  // pred: ^bb57
    memref.store %cst_2, %8[%c5] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb59
  ^bb59:  // 2 preds: ^bb57, ^bb58
    %348 = arith.addi %305, %c6 : index
    %349 = arith.cmpi ugt, %348, %43 : index
    cf.cond_br %349, ^bb60, ^bb61
  ^bb60:  // pred: ^bb59
    memref.store %cst_2, %8[%c6] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb61
  ^bb61:  // 2 preds: ^bb59, ^bb60
    %350 = arith.addi %305, %c7 : index
    %351 = arith.cmpi ugt, %350, %43 : index
    cf.cond_br %351, ^bb62, ^bb63
  ^bb62:  // pred: ^bb61
    memref.store %cst_2, %8[%c7] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb63
  ^bb63:  // 2 preds: ^bb61, ^bb62
    cf.cond_br %337, ^bb64, ^bb65
  ^bb64:  // pred: ^bb63
    memref.store %cst_2, %8[%c8] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb65
  ^bb65:  // 2 preds: ^bb63, ^bb64
    cf.cond_br %339, ^bb66, ^bb67
  ^bb66:  // pred: ^bb65
    memref.store %cst_2, %8[%c9] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb67
  ^bb67:  // 2 preds: ^bb65, ^bb66
    cf.cond_br %341, ^bb68, ^bb69
  ^bb68:  // pred: ^bb67
    memref.store %cst_2, %8[%c10] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb69
  ^bb69:  // 2 preds: ^bb67, ^bb68
    cf.cond_br %343, ^bb70, ^bb71
  ^bb70:  // pred: ^bb69
    memref.store %cst_2, %8[%c11] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb71
  ^bb71:  // 2 preds: ^bb69, ^bb70
    cf.cond_br %345, ^bb72, ^bb73
  ^bb72:  // pred: ^bb71
    memref.store %cst_2, %8[%c12] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb73
  ^bb73:  // 2 preds: ^bb71, ^bb72
    cf.cond_br %347, ^bb74, ^bb75
  ^bb74:  // pred: ^bb73
    memref.store %cst_2, %8[%c13] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb75
  ^bb75:  // 2 preds: ^bb73, ^bb74
    cf.cond_br %349, ^bb76, ^bb77
  ^bb76:  // pred: ^bb75
    memref.store %cst_2, %8[%c14] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb77
  ^bb77:  // 2 preds: ^bb75, ^bb76
    cf.cond_br %351, ^bb78, ^bb79
  ^bb78:  // pred: ^bb77
    memref.store %cst_2, %8[%c15] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb79
  ^bb79:  // 2 preds: ^bb77, ^bb78
    %352 = arith.addi %305, %c16 : index
    %353 = arith.cmpi ugt, %352, %43 : index
    cf.cond_br %353, ^bb80, ^bb81
  ^bb80:  // pred: ^bb79
    memref.store %cst_2, %8[%c16] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb81
  ^bb81:  // 2 preds: ^bb79, ^bb80
    %354 = arith.addi %305, %c17 : index
    %355 = arith.cmpi ugt, %354, %43 : index
    cf.cond_br %355, ^bb82, ^bb83
  ^bb82:  // pred: ^bb81
    memref.store %cst_2, %8[%c17] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb83
  ^bb83:  // 2 preds: ^bb81, ^bb82
    %356 = arith.addi %305, %c18 : index
    %357 = arith.cmpi ugt, %356, %43 : index
    cf.cond_br %357, ^bb84, ^bb85
  ^bb84:  // pred: ^bb83
    memref.store %cst_2, %8[%c18] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb85
  ^bb85:  // 2 preds: ^bb83, ^bb84
    %358 = arith.addi %305, %c19 : index
    %359 = arith.cmpi ugt, %358, %43 : index
    cf.cond_br %359, ^bb86, ^bb87
  ^bb86:  // pred: ^bb85
    memref.store %cst_2, %8[%c19] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb87
  ^bb87:  // 2 preds: ^bb85, ^bb86
    %360 = arith.addi %305, %c20 : index
    %361 = arith.cmpi ugt, %360, %43 : index
    cf.cond_br %361, ^bb88, ^bb89
  ^bb88:  // pred: ^bb87
    memref.store %cst_2, %8[%c20] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb89
  ^bb89:  // 2 preds: ^bb87, ^bb88
    %362 = arith.addi %305, %c21 : index
    %363 = arith.cmpi ugt, %362, %43 : index
    cf.cond_br %363, ^bb90, ^bb91
  ^bb90:  // pred: ^bb89
    memref.store %cst_2, %8[%c21] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb91
  ^bb91:  // 2 preds: ^bb89, ^bb90
    %364 = arith.addi %305, %c22 : index
    %365 = arith.cmpi ugt, %364, %43 : index
    cf.cond_br %365, ^bb92, ^bb93
  ^bb92:  // pred: ^bb91
    memref.store %cst_2, %8[%c22] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb93
  ^bb93:  // 2 preds: ^bb91, ^bb92
    %366 = arith.addi %305, %c23 : index
    %367 = arith.cmpi ugt, %366, %43 : index
    cf.cond_br %367, ^bb94, ^bb95
  ^bb94:  // pred: ^bb93
    memref.store %cst_2, %8[%c23] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb95
  ^bb95:  // 2 preds: ^bb93, ^bb94
    cf.cond_br %353, ^bb96, ^bb97
  ^bb96:  // pred: ^bb95
    memref.store %cst_2, %8[%c24] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb97
  ^bb97:  // 2 preds: ^bb95, ^bb96
    cf.cond_br %355, ^bb98, ^bb99
  ^bb98:  // pred: ^bb97
    memref.store %cst_2, %8[%c25] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb99
  ^bb99:  // 2 preds: ^bb97, ^bb98
    cf.cond_br %357, ^bb100, ^bb101
  ^bb100:  // pred: ^bb99
    memref.store %cst_2, %8[%c26] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb101
  ^bb101:  // 2 preds: ^bb99, ^bb100
    cf.cond_br %359, ^bb102, ^bb103
  ^bb102:  // pred: ^bb101
    memref.store %cst_2, %8[%c27] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb103
  ^bb103:  // 2 preds: ^bb101, ^bb102
    cf.cond_br %361, ^bb104, ^bb105
  ^bb104:  // pred: ^bb103
    memref.store %cst_2, %8[%c28] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb105
  ^bb105:  // 2 preds: ^bb103, ^bb104
    cf.cond_br %363, ^bb106, ^bb107
  ^bb106:  // pred: ^bb105
    memref.store %cst_2, %8[%c29] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb107
  ^bb107:  // 2 preds: ^bb105, ^bb106
    cf.cond_br %365, ^bb108, ^bb109
  ^bb108:  // pred: ^bb107
    memref.store %cst_2, %8[%c30] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb109
  ^bb109:  // 2 preds: ^bb107, ^bb108
    cf.cond_br %367, ^bb110, ^bb111
  ^bb110:  // pred: ^bb109
    memref.store %cst_2, %8[%c31] : memref<32xf32, #gpu.address_space<private>>
    cf.br ^bb111
  ^bb111:  // 2 preds: ^bb109, ^bb110
    cf.br ^bb112
  ^bb112:  // 2 preds: ^bb46, ^bb111
    rock.lds_barrier
    %368 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    cf.br ^bb113(%c0 : index)
  ^bb113(%369: index):  // 2 preds: ^bb112, ^bb114
    %370 = arith.cmpi slt, %369, %c2 : index
    cf.cond_br %370, ^bb114, ^bb115
  ^bb114:  // pred: ^bb113
    memref.store %cst_2, %368[%369] : memref<2xf32, #gpu.address_space<private>>
    %371 = arith.addi %369, %c1 : index
    cf.br ^bb113(%371 : index)
  ^bb115:  // pred: ^bb113
    %372 = memref.load %8[%c0] : memref<32xf32, #gpu.address_space<private>>
    %373 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %374 = arith.maximumf %373, %372 : f32
    memref.store %374, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %375 = memref.load %8[%c1] : memref<32xf32, #gpu.address_space<private>>
    %376 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %377 = arith.maximumf %376, %375 : f32
    memref.store %377, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %378 = memref.load %8[%c2] : memref<32xf32, #gpu.address_space<private>>
    %379 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %380 = arith.maximumf %379, %378 : f32
    memref.store %380, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %381 = memref.load %8[%c3] : memref<32xf32, #gpu.address_space<private>>
    %382 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %383 = arith.maximumf %382, %381 : f32
    memref.store %383, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %384 = memref.load %8[%c4] : memref<32xf32, #gpu.address_space<private>>
    %385 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %386 = arith.maximumf %385, %384 : f32
    memref.store %386, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %387 = memref.load %8[%c5] : memref<32xf32, #gpu.address_space<private>>
    %388 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %389 = arith.maximumf %388, %387 : f32
    memref.store %389, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %390 = memref.load %8[%c6] : memref<32xf32, #gpu.address_space<private>>
    %391 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %392 = arith.maximumf %391, %390 : f32
    memref.store %392, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %393 = memref.load %8[%c7] : memref<32xf32, #gpu.address_space<private>>
    %394 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %395 = arith.maximumf %394, %393 : f32
    memref.store %395, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %396 = memref.load %8[%c8] : memref<32xf32, #gpu.address_space<private>>
    %397 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %398 = arith.maximumf %397, %396 : f32
    memref.store %398, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %399 = memref.load %8[%c9] : memref<32xf32, #gpu.address_space<private>>
    %400 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %401 = arith.maximumf %400, %399 : f32
    memref.store %401, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %402 = memref.load %8[%c10] : memref<32xf32, #gpu.address_space<private>>
    %403 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %404 = arith.maximumf %403, %402 : f32
    memref.store %404, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %405 = memref.load %8[%c11] : memref<32xf32, #gpu.address_space<private>>
    %406 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %407 = arith.maximumf %406, %405 : f32
    memref.store %407, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %408 = memref.load %8[%c12] : memref<32xf32, #gpu.address_space<private>>
    %409 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %410 = arith.maximumf %409, %408 : f32
    memref.store %410, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %411 = memref.load %8[%c13] : memref<32xf32, #gpu.address_space<private>>
    %412 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %413 = arith.maximumf %412, %411 : f32
    memref.store %413, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %414 = memref.load %8[%c14] : memref<32xf32, #gpu.address_space<private>>
    %415 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %416 = arith.maximumf %415, %414 : f32
    memref.store %416, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %417 = memref.load %8[%c15] : memref<32xf32, #gpu.address_space<private>>
    %418 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %419 = arith.maximumf %418, %417 : f32
    memref.store %419, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %420 = memref.load %8[%c16] : memref<32xf32, #gpu.address_space<private>>
    %421 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %422 = arith.maximumf %421, %420 : f32
    memref.store %422, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %423 = memref.load %8[%c17] : memref<32xf32, #gpu.address_space<private>>
    %424 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %425 = arith.maximumf %424, %423 : f32
    memref.store %425, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %426 = memref.load %8[%c18] : memref<32xf32, #gpu.address_space<private>>
    %427 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %428 = arith.maximumf %427, %426 : f32
    memref.store %428, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %429 = memref.load %8[%c19] : memref<32xf32, #gpu.address_space<private>>
    %430 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %431 = arith.maximumf %430, %429 : f32
    memref.store %431, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %432 = memref.load %8[%c20] : memref<32xf32, #gpu.address_space<private>>
    %433 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %434 = arith.maximumf %433, %432 : f32
    memref.store %434, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %435 = memref.load %8[%c21] : memref<32xf32, #gpu.address_space<private>>
    %436 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %437 = arith.maximumf %436, %435 : f32
    memref.store %437, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %438 = memref.load %8[%c22] : memref<32xf32, #gpu.address_space<private>>
    %439 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %440 = arith.maximumf %439, %438 : f32
    memref.store %440, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %441 = memref.load %8[%c23] : memref<32xf32, #gpu.address_space<private>>
    %442 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %443 = arith.maximumf %442, %441 : f32
    memref.store %443, %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %444 = memref.load %8[%c24] : memref<32xf32, #gpu.address_space<private>>
    %445 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %446 = arith.maximumf %445, %444 : f32
    memref.store %446, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %447 = memref.load %8[%c25] : memref<32xf32, #gpu.address_space<private>>
    %448 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %449 = arith.maximumf %448, %447 : f32
    memref.store %449, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %450 = memref.load %8[%c26] : memref<32xf32, #gpu.address_space<private>>
    %451 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %452 = arith.maximumf %451, %450 : f32
    memref.store %452, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %453 = memref.load %8[%c27] : memref<32xf32, #gpu.address_space<private>>
    %454 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %455 = arith.maximumf %454, %453 : f32
    memref.store %455, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %456 = memref.load %8[%c28] : memref<32xf32, #gpu.address_space<private>>
    %457 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %458 = arith.maximumf %457, %456 : f32
    memref.store %458, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %459 = memref.load %8[%c29] : memref<32xf32, #gpu.address_space<private>>
    %460 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %461 = arith.maximumf %460, %459 : f32
    memref.store %461, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %462 = memref.load %8[%c30] : memref<32xf32, #gpu.address_space<private>>
    %463 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %464 = arith.maximumf %463, %462 : f32
    memref.store %464, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %465 = memref.load %8[%c31] : memref<32xf32, #gpu.address_space<private>>
    %466 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %467 = arith.maximumf %466, %465 : f32
    memref.store %467, %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %468 = memref.load %368[%c0] : memref<2xf32, #gpu.address_space<private>>
    %469 = arith.muli %68, %c2 overflow<nsw> : index
    %470 = arith.addi %469, %67 : index
    memref.store %468, %view_9[%470] : memref<64xf32, #gpu.address_space<workgroup>>
    %471 = memref.load %368[%c1] : memref<2xf32, #gpu.address_space<private>>
    %472 = arith.muli %311, %c2 overflow<nsw> : index
    %473 = arith.addi %472, %67 : index
    memref.store %471, %view_9[%473] : memref<64xf32, #gpu.address_space<workgroup>>
    rock.lds_barrier
    %474 = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    memref.store %cst_2, %474[%c0] : memref<1xf32, #gpu.address_space<private>>
    %475 = vector.load %view_9[%73] : memref<64xf32, #gpu.address_space<workgroup>>, vector<2xf32>
    %476 = memref.load %474[%c0] : memref<1xf32, #gpu.address_space<private>>
    %477 = vector.reduction <maximumf>, %475 : vector<2xf32> into f32
    %478 = arith.maximumf %476, %477 : f32
    memref.store %478, %474[%c0] : memref<1xf32, #gpu.address_space<private>>
    memref.store %478, %view_9[%73] : memref<64xf32, #gpu.address_space<workgroup>>
    rock.lds_barrier
    %479 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %479, %9[%c0] : memref<32xf32, #gpu.address_space<private>>
    %480 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %480, %9[%c1] : memref<32xf32, #gpu.address_space<private>>
    %481 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %481, %9[%c2] : memref<32xf32, #gpu.address_space<private>>
    %482 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %482, %9[%c3] : memref<32xf32, #gpu.address_space<private>>
    %483 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %483, %9[%c4] : memref<32xf32, #gpu.address_space<private>>
    %484 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %484, %9[%c5] : memref<32xf32, #gpu.address_space<private>>
    %485 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %485, %9[%c6] : memref<32xf32, #gpu.address_space<private>>
    %486 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %486, %9[%c7] : memref<32xf32, #gpu.address_space<private>>
    %487 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %487, %9[%c8] : memref<32xf32, #gpu.address_space<private>>
    %488 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %488, %9[%c9] : memref<32xf32, #gpu.address_space<private>>
    %489 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %489, %9[%c10] : memref<32xf32, #gpu.address_space<private>>
    %490 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %490, %9[%c11] : memref<32xf32, #gpu.address_space<private>>
    %491 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %491, %9[%c12] : memref<32xf32, #gpu.address_space<private>>
    %492 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %492, %9[%c13] : memref<32xf32, #gpu.address_space<private>>
    %493 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %493, %9[%c14] : memref<32xf32, #gpu.address_space<private>>
    %494 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %494, %9[%c15] : memref<32xf32, #gpu.address_space<private>>
    %495 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %495, %9[%c16] : memref<32xf32, #gpu.address_space<private>>
    %496 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %496, %9[%c17] : memref<32xf32, #gpu.address_space<private>>
    %497 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %497, %9[%c18] : memref<32xf32, #gpu.address_space<private>>
    %498 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %498, %9[%c19] : memref<32xf32, #gpu.address_space<private>>
    %499 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %499, %9[%c20] : memref<32xf32, #gpu.address_space<private>>
    %500 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %500, %9[%c21] : memref<32xf32, #gpu.address_space<private>>
    %501 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %501, %9[%c22] : memref<32xf32, #gpu.address_space<private>>
    %502 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %502, %9[%c23] : memref<32xf32, #gpu.address_space<private>>
    %503 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %503, %9[%c24] : memref<32xf32, #gpu.address_space<private>>
    %504 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %504, %9[%c25] : memref<32xf32, #gpu.address_space<private>>
    %505 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %505, %9[%c26] : memref<32xf32, #gpu.address_space<private>>
    %506 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %506, %9[%c27] : memref<32xf32, #gpu.address_space<private>>
    %507 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %507, %9[%c28] : memref<32xf32, #gpu.address_space<private>>
    %508 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %508, %9[%c29] : memref<32xf32, #gpu.address_space<private>>
    %509 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %509, %9[%c30] : memref<32xf32, #gpu.address_space<private>>
    %510 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %510, %9[%c31] : memref<32xf32, #gpu.address_space<private>>
    %511 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %512 = memref.load %9[%c0] : memref<32xf32, #gpu.address_space<private>>
    %513 = arith.maximumf %511, %512 : f32
    %514 = memref.load %8[%c0] : memref<32xf32, #gpu.address_space<private>>
    %515 = arith.subf %514, %513 : f32
    %516 = math.exp2 %515 : f32
    memref.store %516, %10[%c0] : memref<32xf32, #gpu.address_space<private>>
    %517 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %518 = memref.load %9[%c1] : memref<32xf32, #gpu.address_space<private>>
    %519 = arith.maximumf %517, %518 : f32
    %520 = memref.load %8[%c1] : memref<32xf32, #gpu.address_space<private>>
    %521 = arith.subf %520, %519 : f32
    %522 = math.exp2 %521 : f32
    memref.store %522, %10[%c1] : memref<32xf32, #gpu.address_space<private>>
    %523 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %524 = memref.load %9[%c2] : memref<32xf32, #gpu.address_space<private>>
    %525 = arith.maximumf %523, %524 : f32
    %526 = memref.load %8[%c2] : memref<32xf32, #gpu.address_space<private>>
    %527 = arith.subf %526, %525 : f32
    %528 = math.exp2 %527 : f32
    memref.store %528, %10[%c2] : memref<32xf32, #gpu.address_space<private>>
    %529 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %530 = memref.load %9[%c3] : memref<32xf32, #gpu.address_space<private>>
    %531 = arith.maximumf %529, %530 : f32
    %532 = memref.load %8[%c3] : memref<32xf32, #gpu.address_space<private>>
    %533 = arith.subf %532, %531 : f32
    %534 = math.exp2 %533 : f32
    memref.store %534, %10[%c3] : memref<32xf32, #gpu.address_space<private>>
    %535 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %536 = memref.load %9[%c4] : memref<32xf32, #gpu.address_space<private>>
    %537 = arith.maximumf %535, %536 : f32
    %538 = memref.load %8[%c4] : memref<32xf32, #gpu.address_space<private>>
    %539 = arith.subf %538, %537 : f32
    %540 = math.exp2 %539 : f32
    memref.store %540, %10[%c4] : memref<32xf32, #gpu.address_space<private>>
    %541 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %542 = memref.load %9[%c5] : memref<32xf32, #gpu.address_space<private>>
    %543 = arith.maximumf %541, %542 : f32
    %544 = memref.load %8[%c5] : memref<32xf32, #gpu.address_space<private>>
    %545 = arith.subf %544, %543 : f32
    %546 = math.exp2 %545 : f32
    memref.store %546, %10[%c5] : memref<32xf32, #gpu.address_space<private>>
    %547 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %548 = memref.load %9[%c6] : memref<32xf32, #gpu.address_space<private>>
    %549 = arith.maximumf %547, %548 : f32
    %550 = memref.load %8[%c6] : memref<32xf32, #gpu.address_space<private>>
    %551 = arith.subf %550, %549 : f32
    %552 = math.exp2 %551 : f32
    memref.store %552, %10[%c6] : memref<32xf32, #gpu.address_space<private>>
    %553 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %554 = memref.load %9[%c7] : memref<32xf32, #gpu.address_space<private>>
    %555 = arith.maximumf %553, %554 : f32
    %556 = memref.load %8[%c7] : memref<32xf32, #gpu.address_space<private>>
    %557 = arith.subf %556, %555 : f32
    %558 = math.exp2 %557 : f32
    memref.store %558, %10[%c7] : memref<32xf32, #gpu.address_space<private>>
    %559 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %560 = memref.load %9[%c16] : memref<32xf32, #gpu.address_space<private>>
    %561 = arith.maximumf %559, %560 : f32
    %562 = memref.load %8[%c16] : memref<32xf32, #gpu.address_space<private>>
    %563 = arith.subf %562, %561 : f32
    %564 = math.exp2 %563 : f32
    memref.store %564, %10[%c16] : memref<32xf32, #gpu.address_space<private>>
    %565 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %566 = memref.load %9[%c17] : memref<32xf32, #gpu.address_space<private>>
    %567 = arith.maximumf %565, %566 : f32
    %568 = memref.load %8[%c17] : memref<32xf32, #gpu.address_space<private>>
    %569 = arith.subf %568, %567 : f32
    %570 = math.exp2 %569 : f32
    memref.store %570, %10[%c17] : memref<32xf32, #gpu.address_space<private>>
    %571 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %572 = memref.load %9[%c18] : memref<32xf32, #gpu.address_space<private>>
    %573 = arith.maximumf %571, %572 : f32
    %574 = memref.load %8[%c18] : memref<32xf32, #gpu.address_space<private>>
    %575 = arith.subf %574, %573 : f32
    %576 = math.exp2 %575 : f32
    memref.store %576, %10[%c18] : memref<32xf32, #gpu.address_space<private>>
    %577 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %578 = memref.load %9[%c19] : memref<32xf32, #gpu.address_space<private>>
    %579 = arith.maximumf %577, %578 : f32
    %580 = memref.load %8[%c19] : memref<32xf32, #gpu.address_space<private>>
    %581 = arith.subf %580, %579 : f32
    %582 = math.exp2 %581 : f32
    memref.store %582, %10[%c19] : memref<32xf32, #gpu.address_space<private>>
    %583 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %584 = memref.load %9[%c20] : memref<32xf32, #gpu.address_space<private>>
    %585 = arith.maximumf %583, %584 : f32
    %586 = memref.load %8[%c20] : memref<32xf32, #gpu.address_space<private>>
    %587 = arith.subf %586, %585 : f32
    %588 = math.exp2 %587 : f32
    memref.store %588, %10[%c20] : memref<32xf32, #gpu.address_space<private>>
    %589 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %590 = memref.load %9[%c21] : memref<32xf32, #gpu.address_space<private>>
    %591 = arith.maximumf %589, %590 : f32
    %592 = memref.load %8[%c21] : memref<32xf32, #gpu.address_space<private>>
    %593 = arith.subf %592, %591 : f32
    %594 = math.exp2 %593 : f32
    memref.store %594, %10[%c21] : memref<32xf32, #gpu.address_space<private>>
    %595 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %596 = memref.load %9[%c22] : memref<32xf32, #gpu.address_space<private>>
    %597 = arith.maximumf %595, %596 : f32
    %598 = memref.load %8[%c22] : memref<32xf32, #gpu.address_space<private>>
    %599 = arith.subf %598, %597 : f32
    %600 = math.exp2 %599 : f32
    memref.store %600, %10[%c22] : memref<32xf32, #gpu.address_space<private>>
    %601 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %602 = memref.load %9[%c23] : memref<32xf32, #gpu.address_space<private>>
    %603 = arith.maximumf %601, %602 : f32
    %604 = memref.load %8[%c23] : memref<32xf32, #gpu.address_space<private>>
    %605 = arith.subf %604, %603 : f32
    %606 = math.exp2 %605 : f32
    memref.store %606, %10[%c23] : memref<32xf32, #gpu.address_space<private>>
    %607 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %608 = memref.load %9[%c8] : memref<32xf32, #gpu.address_space<private>>
    %609 = arith.maximumf %607, %608 : f32
    %610 = memref.load %8[%c8] : memref<32xf32, #gpu.address_space<private>>
    %611 = arith.subf %610, %609 : f32
    %612 = math.exp2 %611 : f32
    memref.store %612, %10[%c8] : memref<32xf32, #gpu.address_space<private>>
    %613 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %614 = memref.load %9[%c9] : memref<32xf32, #gpu.address_space<private>>
    %615 = arith.maximumf %613, %614 : f32
    %616 = memref.load %8[%c9] : memref<32xf32, #gpu.address_space<private>>
    %617 = arith.subf %616, %615 : f32
    %618 = math.exp2 %617 : f32
    memref.store %618, %10[%c9] : memref<32xf32, #gpu.address_space<private>>
    %619 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %620 = memref.load %9[%c10] : memref<32xf32, #gpu.address_space<private>>
    %621 = arith.maximumf %619, %620 : f32
    %622 = memref.load %8[%c10] : memref<32xf32, #gpu.address_space<private>>
    %623 = arith.subf %622, %621 : f32
    %624 = math.exp2 %623 : f32
    memref.store %624, %10[%c10] : memref<32xf32, #gpu.address_space<private>>
    %625 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %626 = memref.load %9[%c11] : memref<32xf32, #gpu.address_space<private>>
    %627 = arith.maximumf %625, %626 : f32
    %628 = memref.load %8[%c11] : memref<32xf32, #gpu.address_space<private>>
    %629 = arith.subf %628, %627 : f32
    %630 = math.exp2 %629 : f32
    memref.store %630, %10[%c11] : memref<32xf32, #gpu.address_space<private>>
    %631 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %632 = memref.load %9[%c12] : memref<32xf32, #gpu.address_space<private>>
    %633 = arith.maximumf %631, %632 : f32
    %634 = memref.load %8[%c12] : memref<32xf32, #gpu.address_space<private>>
    %635 = arith.subf %634, %633 : f32
    %636 = math.exp2 %635 : f32
    memref.store %636, %10[%c12] : memref<32xf32, #gpu.address_space<private>>
    %637 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %638 = memref.load %9[%c13] : memref<32xf32, #gpu.address_space<private>>
    %639 = arith.maximumf %637, %638 : f32
    %640 = memref.load %8[%c13] : memref<32xf32, #gpu.address_space<private>>
    %641 = arith.subf %640, %639 : f32
    %642 = math.exp2 %641 : f32
    memref.store %642, %10[%c13] : memref<32xf32, #gpu.address_space<private>>
    %643 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %644 = memref.load %9[%c14] : memref<32xf32, #gpu.address_space<private>>
    %645 = arith.maximumf %643, %644 : f32
    %646 = memref.load %8[%c14] : memref<32xf32, #gpu.address_space<private>>
    %647 = arith.subf %646, %645 : f32
    %648 = math.exp2 %647 : f32
    memref.store %648, %10[%c14] : memref<32xf32, #gpu.address_space<private>>
    %649 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %650 = memref.load %9[%c15] : memref<32xf32, #gpu.address_space<private>>
    %651 = arith.maximumf %649, %650 : f32
    %652 = memref.load %8[%c15] : memref<32xf32, #gpu.address_space<private>>
    %653 = arith.subf %652, %651 : f32
    %654 = math.exp2 %653 : f32
    memref.store %654, %10[%c15] : memref<32xf32, #gpu.address_space<private>>
    %655 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %656 = memref.load %9[%c24] : memref<32xf32, #gpu.address_space<private>>
    %657 = arith.maximumf %655, %656 : f32
    %658 = memref.load %8[%c24] : memref<32xf32, #gpu.address_space<private>>
    %659 = arith.subf %658, %657 : f32
    %660 = math.exp2 %659 : f32
    memref.store %660, %10[%c24] : memref<32xf32, #gpu.address_space<private>>
    %661 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %662 = memref.load %9[%c25] : memref<32xf32, #gpu.address_space<private>>
    %663 = arith.maximumf %661, %662 : f32
    %664 = memref.load %8[%c25] : memref<32xf32, #gpu.address_space<private>>
    %665 = arith.subf %664, %663 : f32
    %666 = math.exp2 %665 : f32
    memref.store %666, %10[%c25] : memref<32xf32, #gpu.address_space<private>>
    %667 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %668 = memref.load %9[%c26] : memref<32xf32, #gpu.address_space<private>>
    %669 = arith.maximumf %667, %668 : f32
    %670 = memref.load %8[%c26] : memref<32xf32, #gpu.address_space<private>>
    %671 = arith.subf %670, %669 : f32
    %672 = math.exp2 %671 : f32
    memref.store %672, %10[%c26] : memref<32xf32, #gpu.address_space<private>>
    %673 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %674 = memref.load %9[%c27] : memref<32xf32, #gpu.address_space<private>>
    %675 = arith.maximumf %673, %674 : f32
    %676 = memref.load %8[%c27] : memref<32xf32, #gpu.address_space<private>>
    %677 = arith.subf %676, %675 : f32
    %678 = math.exp2 %677 : f32
    memref.store %678, %10[%c27] : memref<32xf32, #gpu.address_space<private>>
    %679 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %680 = memref.load %9[%c28] : memref<32xf32, #gpu.address_space<private>>
    %681 = arith.maximumf %679, %680 : f32
    %682 = memref.load %8[%c28] : memref<32xf32, #gpu.address_space<private>>
    %683 = arith.subf %682, %681 : f32
    %684 = math.exp2 %683 : f32
    memref.store %684, %10[%c28] : memref<32xf32, #gpu.address_space<private>>
    %685 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %686 = memref.load %9[%c29] : memref<32xf32, #gpu.address_space<private>>
    %687 = arith.maximumf %685, %686 : f32
    %688 = memref.load %8[%c29] : memref<32xf32, #gpu.address_space<private>>
    %689 = arith.subf %688, %687 : f32
    %690 = math.exp2 %689 : f32
    memref.store %690, %10[%c29] : memref<32xf32, #gpu.address_space<private>>
    %691 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %692 = memref.load %9[%c30] : memref<32xf32, #gpu.address_space<private>>
    %693 = arith.maximumf %691, %692 : f32
    %694 = memref.load %8[%c30] : memref<32xf32, #gpu.address_space<private>>
    %695 = arith.subf %694, %693 : f32
    %696 = math.exp2 %695 : f32
    memref.store %696, %10[%c30] : memref<32xf32, #gpu.address_space<private>>
    %697 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %698 = memref.load %9[%c31] : memref<32xf32, #gpu.address_space<private>>
    %699 = arith.maximumf %697, %698 : f32
    %700 = memref.load %8[%c31] : memref<32xf32, #gpu.address_space<private>>
    %701 = arith.subf %700, %699 : f32
    %702 = math.exp2 %701 : f32
    memref.store %702, %10[%c31] : memref<32xf32, #gpu.address_space<private>>
    rock.lds_barrier
    %703 = rock.alloc() : memref<2xf32, #gpu.address_space<private>>
    cf.br ^bb116(%c0 : index)
  ^bb116(%704: index):  // 2 preds: ^bb115, ^bb117
    %705 = arith.cmpi slt, %704, %c2 : index
    cf.cond_br %705, ^bb117, ^bb118
  ^bb117:  // pred: ^bb116
    memref.store %cst_3, %703[%704] : memref<2xf32, #gpu.address_space<private>>
    %706 = arith.addi %704, %c1 : index
    cf.br ^bb116(%706 : index)
  ^bb118:  // pred: ^bb116
    %707 = memref.load %10[%c0] : memref<32xf32, #gpu.address_space<private>>
    %708 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %709 = arith.addf %708, %707 : f32
    memref.store %709, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %710 = memref.load %10[%c1] : memref<32xf32, #gpu.address_space<private>>
    %711 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %712 = arith.addf %711, %710 : f32
    memref.store %712, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %713 = memref.load %10[%c2] : memref<32xf32, #gpu.address_space<private>>
    %714 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %715 = arith.addf %714, %713 : f32
    memref.store %715, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %716 = memref.load %10[%c3] : memref<32xf32, #gpu.address_space<private>>
    %717 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %718 = arith.addf %717, %716 : f32
    memref.store %718, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %719 = memref.load %10[%c4] : memref<32xf32, #gpu.address_space<private>>
    %720 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %721 = arith.addf %720, %719 : f32
    memref.store %721, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %722 = memref.load %10[%c5] : memref<32xf32, #gpu.address_space<private>>
    %723 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %724 = arith.addf %723, %722 : f32
    memref.store %724, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %725 = memref.load %10[%c6] : memref<32xf32, #gpu.address_space<private>>
    %726 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %727 = arith.addf %726, %725 : f32
    memref.store %727, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %728 = memref.load %10[%c7] : memref<32xf32, #gpu.address_space<private>>
    %729 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %730 = arith.addf %729, %728 : f32
    memref.store %730, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %731 = memref.load %10[%c8] : memref<32xf32, #gpu.address_space<private>>
    %732 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %733 = arith.addf %732, %731 : f32
    memref.store %733, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %734 = memref.load %10[%c9] : memref<32xf32, #gpu.address_space<private>>
    %735 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %736 = arith.addf %735, %734 : f32
    memref.store %736, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %737 = memref.load %10[%c10] : memref<32xf32, #gpu.address_space<private>>
    %738 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %739 = arith.addf %738, %737 : f32
    memref.store %739, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %740 = memref.load %10[%c11] : memref<32xf32, #gpu.address_space<private>>
    %741 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %742 = arith.addf %741, %740 : f32
    memref.store %742, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %743 = memref.load %10[%c12] : memref<32xf32, #gpu.address_space<private>>
    %744 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %745 = arith.addf %744, %743 : f32
    memref.store %745, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %746 = memref.load %10[%c13] : memref<32xf32, #gpu.address_space<private>>
    %747 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %748 = arith.addf %747, %746 : f32
    memref.store %748, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %749 = memref.load %10[%c14] : memref<32xf32, #gpu.address_space<private>>
    %750 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %751 = arith.addf %750, %749 : f32
    memref.store %751, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %752 = memref.load %10[%c15] : memref<32xf32, #gpu.address_space<private>>
    %753 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %754 = arith.addf %753, %752 : f32
    memref.store %754, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %755 = memref.load %10[%c16] : memref<32xf32, #gpu.address_space<private>>
    %756 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %757 = arith.addf %756, %755 : f32
    memref.store %757, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %758 = memref.load %10[%c17] : memref<32xf32, #gpu.address_space<private>>
    %759 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %760 = arith.addf %759, %758 : f32
    memref.store %760, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %761 = memref.load %10[%c18] : memref<32xf32, #gpu.address_space<private>>
    %762 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %763 = arith.addf %762, %761 : f32
    memref.store %763, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %764 = memref.load %10[%c19] : memref<32xf32, #gpu.address_space<private>>
    %765 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %766 = arith.addf %765, %764 : f32
    memref.store %766, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %767 = memref.load %10[%c20] : memref<32xf32, #gpu.address_space<private>>
    %768 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %769 = arith.addf %768, %767 : f32
    memref.store %769, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %770 = memref.load %10[%c21] : memref<32xf32, #gpu.address_space<private>>
    %771 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %772 = arith.addf %771, %770 : f32
    memref.store %772, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %773 = memref.load %10[%c22] : memref<32xf32, #gpu.address_space<private>>
    %774 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %775 = arith.addf %774, %773 : f32
    memref.store %775, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %776 = memref.load %10[%c23] : memref<32xf32, #gpu.address_space<private>>
    %777 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %778 = arith.addf %777, %776 : f32
    memref.store %778, %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    %779 = memref.load %10[%c24] : memref<32xf32, #gpu.address_space<private>>
    %780 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %781 = arith.addf %780, %779 : f32
    memref.store %781, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %782 = memref.load %10[%c25] : memref<32xf32, #gpu.address_space<private>>
    %783 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %784 = arith.addf %783, %782 : f32
    memref.store %784, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %785 = memref.load %10[%c26] : memref<32xf32, #gpu.address_space<private>>
    %786 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %787 = arith.addf %786, %785 : f32
    memref.store %787, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %788 = memref.load %10[%c27] : memref<32xf32, #gpu.address_space<private>>
    %789 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %790 = arith.addf %789, %788 : f32
    memref.store %790, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %791 = memref.load %10[%c28] : memref<32xf32, #gpu.address_space<private>>
    %792 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %793 = arith.addf %792, %791 : f32
    memref.store %793, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %794 = memref.load %10[%c29] : memref<32xf32, #gpu.address_space<private>>
    %795 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %796 = arith.addf %795, %794 : f32
    memref.store %796, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %797 = memref.load %10[%c30] : memref<32xf32, #gpu.address_space<private>>
    %798 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %799 = arith.addf %798, %797 : f32
    memref.store %799, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %800 = memref.load %10[%c31] : memref<32xf32, #gpu.address_space<private>>
    %801 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %802 = arith.addf %801, %800 : f32
    memref.store %802, %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    %803 = memref.load %703[%c0] : memref<2xf32, #gpu.address_space<private>>
    memref.store %803, %view_9[%470] : memref<64xf32, #gpu.address_space<workgroup>>
    %804 = memref.load %703[%c1] : memref<2xf32, #gpu.address_space<private>>
    memref.store %804, %view_9[%473] : memref<64xf32, #gpu.address_space<workgroup>>
    rock.lds_barrier
    %805 = rock.alloc() : memref<1xf32, #gpu.address_space<private>>
    memref.store %cst_3, %805[%c0] : memref<1xf32, #gpu.address_space<private>>
    %806 = vector.load %view_9[%73] : memref<64xf32, #gpu.address_space<workgroup>>, vector<2xf32>
    %807 = memref.load %805[%c0] : memref<1xf32, #gpu.address_space<private>>
    %808 = vector.reduction <add>, %806 : vector<2xf32> into f32
    %809 = arith.addf %807, %808 : f32
    memref.store %809, %805[%c0] : memref<1xf32, #gpu.address_space<private>>
    memref.store %809, %view_9[%73] : memref<64xf32, #gpu.address_space<workgroup>>
    rock.lds_barrier
    %810 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %810, %11[%c0] : memref<32xf32, #gpu.address_space<private>>
    %811 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %811, %11[%c1] : memref<32xf32, #gpu.address_space<private>>
    %812 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %812, %11[%c2] : memref<32xf32, #gpu.address_space<private>>
    %813 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %813, %11[%c3] : memref<32xf32, #gpu.address_space<private>>
    %814 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %814, %11[%c4] : memref<32xf32, #gpu.address_space<private>>
    %815 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %815, %11[%c5] : memref<32xf32, #gpu.address_space<private>>
    %816 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %816, %11[%c6] : memref<32xf32, #gpu.address_space<private>>
    %817 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %817, %11[%c7] : memref<32xf32, #gpu.address_space<private>>
    %818 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %818, %11[%c8] : memref<32xf32, #gpu.address_space<private>>
    %819 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %819, %11[%c9] : memref<32xf32, #gpu.address_space<private>>
    %820 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %820, %11[%c10] : memref<32xf32, #gpu.address_space<private>>
    %821 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %821, %11[%c11] : memref<32xf32, #gpu.address_space<private>>
    %822 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %822, %11[%c12] : memref<32xf32, #gpu.address_space<private>>
    %823 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %823, %11[%c13] : memref<32xf32, #gpu.address_space<private>>
    %824 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %824, %11[%c14] : memref<32xf32, #gpu.address_space<private>>
    %825 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %825, %11[%c15] : memref<32xf32, #gpu.address_space<private>>
    %826 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %826, %11[%c16] : memref<32xf32, #gpu.address_space<private>>
    %827 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %827, %11[%c17] : memref<32xf32, #gpu.address_space<private>>
    %828 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %828, %11[%c18] : memref<32xf32, #gpu.address_space<private>>
    %829 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %829, %11[%c19] : memref<32xf32, #gpu.address_space<private>>
    %830 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %830, %11[%c20] : memref<32xf32, #gpu.address_space<private>>
    %831 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %831, %11[%c21] : memref<32xf32, #gpu.address_space<private>>
    %832 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %832, %11[%c22] : memref<32xf32, #gpu.address_space<private>>
    %833 = memref.load %view_9[%469] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %833, %11[%c23] : memref<32xf32, #gpu.address_space<private>>
    %834 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %834, %11[%c24] : memref<32xf32, #gpu.address_space<private>>
    %835 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %835, %11[%c25] : memref<32xf32, #gpu.address_space<private>>
    %836 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %836, %11[%c26] : memref<32xf32, #gpu.address_space<private>>
    %837 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %837, %11[%c27] : memref<32xf32, #gpu.address_space<private>>
    %838 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %838, %11[%c28] : memref<32xf32, #gpu.address_space<private>>
    %839 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %839, %11[%c29] : memref<32xf32, #gpu.address_space<private>>
    %840 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %840, %11[%c30] : memref<32xf32, #gpu.address_space<private>>
    %841 = memref.load %view_9[%472] : memref<64xf32, #gpu.address_space<workgroup>>
    memref.store %841, %11[%c31] : memref<32xf32, #gpu.address_space<private>>
    %842 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %843 = memref.load %11[%c0] : memref<32xf32, #gpu.address_space<private>>
    %844 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %845 = memref.load %9[%c0] : memref<32xf32, #gpu.address_space<private>>
    %846 = arith.maximumf %844, %845 : f32
    %847 = arith.subf %844, %846 : f32
    %848 = math.exp2 %847 : f32
    memref.store %848, %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %849 = arith.mulf %848, %842 : f32
    %850 = arith.addf %849, %843 : f32
    memref.store %850, %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    memref.store %846, %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %851 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %852 = memref.load %11[%c8] : memref<32xf32, #gpu.address_space<private>>
    %853 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %854 = memref.load %9[%c8] : memref<32xf32, #gpu.address_space<private>>
    %855 = arith.maximumf %853, %854 : f32
    %856 = arith.subf %853, %855 : f32
    %857 = math.exp2 %856 : f32
    memref.store %857, %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %858 = arith.mulf %857, %851 : f32
    %859 = arith.addf %858, %852 : f32
    memref.store %859, %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    memref.store %855, %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %860 = vector.load %10[%c0] : memref<32xf32, #gpu.address_space<private>>, vector<32xf32>
    %861 = arith.truncf %860 : vector<32xf32> to vector<32xf16>
    vector.store %861, %12[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<32xf16>
    rock.lds_barrier
    cf.br ^bb119(%c0 : index)
  ^bb119(%862: index):  // 2 preds: ^bb118, ^bb120
    %863 = arith.cmpi slt, %862, %c32 : index
    cf.cond_br %863, ^bb120, ^bb121
  ^bb120:  // pred: ^bb119
    %864 = arith.divui %862, %c16 : index
    %865 = arith.remui %862, %c8 : index
    %866 = arith.divsi %865, %c8 : index
    %867 = arith.addi %864, %866 : index
    %868 = arith.muli %867, %c2 overflow<nsw> : index
    %869 = arith.remui %862, %c16 : index
    %870 = arith.divsi %869, %c8 : index
    %871 = arith.addi %868, %870 : index
    %872 = arith.muli %871, %c8 overflow<nsw> : index
    %873 = arith.addi %872, %865 : index
    %874 = vector.load %12[%873] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %874, %13[%862] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %875 = arith.addi %862, %c8 : index
    cf.br ^bb119(%875 : index)
  ^bb121:  // pred: ^bb119
    %876 = memref.load %13[%c0] : memref<32xf16, #gpu.address_space<private>>
    memref.store %876, %view_7[%75] : memref<1024xf16, #gpu.address_space<workgroup>>
    %877 = arith.addi %75, %c32 : index
    %878 = memref.load %13[%c1] : memref<32xf16, #gpu.address_space<private>>
    memref.store %878, %view_7[%877] : memref<1024xf16, #gpu.address_space<workgroup>>
    %879 = arith.addi %75, %c64 : index
    %880 = memref.load %13[%c2] : memref<32xf16, #gpu.address_space<private>>
    memref.store %880, %view_7[%879] : memref<1024xf16, #gpu.address_space<workgroup>>
    %881 = arith.addi %75, %c96 : index
    %882 = memref.load %13[%c3] : memref<32xf16, #gpu.address_space<private>>
    memref.store %882, %view_7[%881] : memref<1024xf16, #gpu.address_space<workgroup>>
    %883 = arith.addi %75, %c128 : index
    %884 = memref.load %13[%c4] : memref<32xf16, #gpu.address_space<private>>
    memref.store %884, %view_7[%883] : memref<1024xf16, #gpu.address_space<workgroup>>
    %885 = arith.addi %75, %c160 : index
    %886 = memref.load %13[%c5] : memref<32xf16, #gpu.address_space<private>>
    memref.store %886, %view_7[%885] : memref<1024xf16, #gpu.address_space<workgroup>>
    %887 = arith.addi %75, %c192 : index
    %888 = memref.load %13[%c6] : memref<32xf16, #gpu.address_space<private>>
    memref.store %888, %view_7[%887] : memref<1024xf16, #gpu.address_space<workgroup>>
    %889 = arith.addi %75, %c224 : index
    %890 = memref.load %13[%c7] : memref<32xf16, #gpu.address_space<private>>
    memref.store %890, %view_7[%889] : memref<1024xf16, #gpu.address_space<workgroup>>
    %891 = arith.addi %75, %c16 : index
    %892 = memref.load %13[%c8] : memref<32xf16, #gpu.address_space<private>>
    memref.store %892, %view_7[%891] : memref<1024xf16, #gpu.address_space<workgroup>>
    %893 = arith.addi %75, %c48 : index
    %894 = memref.load %13[%c9] : memref<32xf16, #gpu.address_space<private>>
    memref.store %894, %view_7[%893] : memref<1024xf16, #gpu.address_space<workgroup>>
    %895 = arith.addi %75, %c80 : index
    %896 = memref.load %13[%c10] : memref<32xf16, #gpu.address_space<private>>
    memref.store %896, %view_7[%895] : memref<1024xf16, #gpu.address_space<workgroup>>
    %897 = arith.addi %75, %c112 : index
    %898 = memref.load %13[%c11] : memref<32xf16, #gpu.address_space<private>>
    memref.store %898, %view_7[%897] : memref<1024xf16, #gpu.address_space<workgroup>>
    %899 = arith.addi %75, %c144 : index
    %900 = memref.load %13[%c12] : memref<32xf16, #gpu.address_space<private>>
    memref.store %900, %view_7[%899] : memref<1024xf16, #gpu.address_space<workgroup>>
    %901 = arith.addi %75, %c176 : index
    %902 = memref.load %13[%c13] : memref<32xf16, #gpu.address_space<private>>
    memref.store %902, %view_7[%901] : memref<1024xf16, #gpu.address_space<workgroup>>
    %903 = arith.addi %75, %c208 : index
    %904 = memref.load %13[%c14] : memref<32xf16, #gpu.address_space<private>>
    memref.store %904, %view_7[%903] : memref<1024xf16, #gpu.address_space<workgroup>>
    %905 = arith.addi %75, %c240 : index
    %906 = memref.load %13[%c15] : memref<32xf16, #gpu.address_space<private>>
    memref.store %906, %view_7[%905] : memref<1024xf16, #gpu.address_space<workgroup>>
    %907 = arith.addi %75, %c512 : index
    %908 = memref.load %13[%c16] : memref<32xf16, #gpu.address_space<private>>
    memref.store %908, %view_7[%907] : memref<1024xf16, #gpu.address_space<workgroup>>
    %909 = arith.addi %75, %c544 : index
    %910 = memref.load %13[%c17] : memref<32xf16, #gpu.address_space<private>>
    memref.store %910, %view_7[%909] : memref<1024xf16, #gpu.address_space<workgroup>>
    %911 = arith.addi %75, %c576 : index
    %912 = memref.load %13[%c18] : memref<32xf16, #gpu.address_space<private>>
    memref.store %912, %view_7[%911] : memref<1024xf16, #gpu.address_space<workgroup>>
    %913 = arith.addi %75, %c608 : index
    %914 = memref.load %13[%c19] : memref<32xf16, #gpu.address_space<private>>
    memref.store %914, %view_7[%913] : memref<1024xf16, #gpu.address_space<workgroup>>
    %915 = arith.addi %75, %c640 : index
    %916 = memref.load %13[%c20] : memref<32xf16, #gpu.address_space<private>>
    memref.store %916, %view_7[%915] : memref<1024xf16, #gpu.address_space<workgroup>>
    %917 = arith.addi %75, %c672 : index
    %918 = memref.load %13[%c21] : memref<32xf16, #gpu.address_space<private>>
    memref.store %918, %view_7[%917] : memref<1024xf16, #gpu.address_space<workgroup>>
    %919 = arith.addi %75, %c704 : index
    %920 = memref.load %13[%c22] : memref<32xf16, #gpu.address_space<private>>
    memref.store %920, %view_7[%919] : memref<1024xf16, #gpu.address_space<workgroup>>
    %921 = arith.addi %75, %c736 : index
    %922 = memref.load %13[%c23] : memref<32xf16, #gpu.address_space<private>>
    memref.store %922, %view_7[%921] : memref<1024xf16, #gpu.address_space<workgroup>>
    %923 = arith.addi %75, %c528 : index
    %924 = memref.load %13[%c24] : memref<32xf16, #gpu.address_space<private>>
    memref.store %924, %view_7[%923] : memref<1024xf16, #gpu.address_space<workgroup>>
    %925 = arith.addi %75, %c560 : index
    %926 = memref.load %13[%c25] : memref<32xf16, #gpu.address_space<private>>
    memref.store %926, %view_7[%925] : memref<1024xf16, #gpu.address_space<workgroup>>
    %927 = arith.addi %75, %c592 : index
    %928 = memref.load %13[%c26] : memref<32xf16, #gpu.address_space<private>>
    memref.store %928, %view_7[%927] : memref<1024xf16, #gpu.address_space<workgroup>>
    %929 = arith.addi %75, %c624 : index
    %930 = memref.load %13[%c27] : memref<32xf16, #gpu.address_space<private>>
    memref.store %930, %view_7[%929] : memref<1024xf16, #gpu.address_space<workgroup>>
    %931 = arith.addi %75, %c656 : index
    %932 = memref.load %13[%c28] : memref<32xf16, #gpu.address_space<private>>
    memref.store %932, %view_7[%931] : memref<1024xf16, #gpu.address_space<workgroup>>
    %933 = arith.addi %75, %c688 : index
    %934 = memref.load %13[%c29] : memref<32xf16, #gpu.address_space<private>>
    memref.store %934, %view_7[%933] : memref<1024xf16, #gpu.address_space<workgroup>>
    %935 = arith.addi %75, %c720 : index
    %936 = memref.load %13[%c30] : memref<32xf16, #gpu.address_space<private>>
    memref.store %936, %view_7[%935] : memref<1024xf16, #gpu.address_space<workgroup>>
    %937 = arith.addi %75, %c752 : index
    %938 = memref.load %13[%c31] : memref<32xf16, #gpu.address_space<private>>
    memref.store %938, %view_7[%937] : memref<1024xf16, #gpu.address_space<workgroup>>
    %939 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %940 = rock.alloc() : memref<32xf16, #gpu.address_space<private>>
    %941 = arith.muli %77, %c8 overflow<nsw> : index
    %942 = arith.addi %941, %54 : index
    %943 = arith.muli %942, %c4 overflow<nsw> : index
    %944 = arith.addi %76, %943 : index
    %945 = arith.muli %944, %c64 overflow<nsw> : index
    cf.br ^bb122(%c0 : index)
  ^bb122(%946: index):  // 2 preds: ^bb121, ^bb138
    %947 = arith.cmpi slt, %946, %c2 : index
    cf.cond_br %947, ^bb123, ^bb139
  ^bb123:  // pred: ^bb122
    %948 = arith.muli %946, %c4 overflow<nsw> : index
    %949 = arith.addi %948, %55 : index
    %950 = arith.muli %949, %c8 overflow<nsw> : index
    %951 = arith.addi %945, %950 : index
    %952 = vector.load %memspacecast_10[%951] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %952, %939[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %953 = arith.addi %951, %c64 : index
    %954 = vector.load %memspacecast_10[%953] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %954, %939[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %955 = arith.addi %951, %c128 : index
    %956 = vector.load %memspacecast_10[%955] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %956, %939[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %957 = arith.addi %951, %c192 : index
    %958 = vector.load %memspacecast_10[%957] : memref<147456xf16, #gpu.address_space<global>>, vector<8xf16>
    vector.store %958, %939[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    cf.br ^bb124(%c0 : index)
  ^bb124(%959: index):  // 2 preds: ^bb123, ^bb125
    %960 = arith.cmpi slt, %959, %c32 : index
    cf.cond_br %960, ^bb125, ^bb126
  ^bb125:  // pred: ^bb124
    %961 = arith.divui %959, %c8 : index
    %962 = arith.muli %961, %c8 overflow<nsw> : index
    %963 = arith.remui %959, %c8 : index
    %964 = arith.addi %962, %963 : index
    %965 = vector.load %939[%964] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %965, %940[%959] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    %966 = arith.addi %959, %c8 : index
    cf.br ^bb124(%966 : index)
  ^bb126:  // pred: ^bb124
    %967 = vector.load %940[%c0] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %967, %view[%64] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %968 = arith.addi %64, %c32 : index
    %969 = vector.load %940[%c8] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %969, %view[%968] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %970 = arith.addi %64, %c64 : index
    %971 = vector.load %940[%c16] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %971, %view[%970] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    %972 = arith.addi %64, %c96 : index
    %973 = vector.load %940[%c24] : memref<32xf16, #gpu.address_space<private>>, vector<8xf16>
    vector.store %973, %view[%972] : memref<1024xf16, #gpu.address_space<workgroup>>, vector<8xf16>
    cf.br ^bb127(%c0 : index)
  ^bb127(%974: index):  // 2 preds: ^bb126, ^bb128
    %975 = arith.cmpi slt, %974, %c4 : index
    cf.cond_br %975, ^bb128, ^bb129
  ^bb128:  // pred: ^bb127
    memref.store %cst_4, %16[%974] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %976 = arith.addi %974, %c1 : index
    cf.br ^bb127(%976 : index)
  ^bb129:  // pred: ^bb127
    cf.br ^bb130(%c0 : index)
  ^bb130(%977: index):  // 2 preds: ^bb129, ^bb137
    %978 = arith.cmpi slt, %977, %c2 : index
    cf.cond_br %978, ^bb131, ^bb138
  ^bb131:  // pred: ^bb130
    %979 = arith.muli %977, %c16 overflow<nsw> : index
    %980 = arith.addi %979, %68 : index
    %981 = arith.addi %69, %980 : index
    %982 = memref.load %view[%981] : memref<1024xf16, #gpu.address_space<workgroup>>
    %983 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %984 = vector.insert %982, %983 [0] : f16 into vector<8xf16>
    memref.store %984, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %985 = arith.addi %981, %c32 : index
    %986 = memref.load %view[%985] : memref<1024xf16, #gpu.address_space<workgroup>>
    %987 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %988 = vector.insert %986, %987 [1] : f16 into vector<8xf16>
    memref.store %988, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %989 = arith.addi %981, %c64 : index
    %990 = memref.load %view[%989] : memref<1024xf16, #gpu.address_space<workgroup>>
    %991 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %992 = vector.insert %990, %991 [2] : f16 into vector<8xf16>
    memref.store %992, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %993 = arith.addi %981, %c96 : index
    %994 = memref.load %view[%993] : memref<1024xf16, #gpu.address_space<workgroup>>
    %995 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %996 = vector.insert %994, %995 [3] : f16 into vector<8xf16>
    memref.store %996, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %997 = arith.addi %981, %c128 : index
    %998 = memref.load %view[%997] : memref<1024xf16, #gpu.address_space<workgroup>>
    %999 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1000 = vector.insert %998, %999 [4] : f16 into vector<8xf16>
    memref.store %1000, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1001 = arith.addi %981, %c160 : index
    %1002 = memref.load %view[%1001] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1003 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1004 = vector.insert %1002, %1003 [5] : f16 into vector<8xf16>
    memref.store %1004, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1005 = arith.addi %981, %c192 : index
    %1006 = memref.load %view[%1005] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1007 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1008 = vector.insert %1006, %1007 [6] : f16 into vector<8xf16>
    memref.store %1008, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1009 = arith.addi %981, %c224 : index
    %1010 = memref.load %view[%1009] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1011 = memref.load %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1012 = vector.insert %1010, %1011 [7] : f16 into vector<8xf16>
    memref.store %1012, %14[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1013 = arith.addi %981, %c256 : index
    %1014 = memref.load %view[%1013] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1015 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1016 = vector.insert %1014, %1015 [0] : f16 into vector<8xf16>
    memref.store %1016, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1017 = arith.addi %981, %c288 : index
    %1018 = memref.load %view[%1017] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1019 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1020 = vector.insert %1018, %1019 [1] : f16 into vector<8xf16>
    memref.store %1020, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1021 = arith.addi %981, %c320 : index
    %1022 = memref.load %view[%1021] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1023 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1024 = vector.insert %1022, %1023 [2] : f16 into vector<8xf16>
    memref.store %1024, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1025 = arith.addi %981, %c352 : index
    %1026 = memref.load %view[%1025] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1027 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1028 = vector.insert %1026, %1027 [3] : f16 into vector<8xf16>
    memref.store %1028, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1029 = arith.addi %981, %c384 : index
    %1030 = memref.load %view[%1029] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1031 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1032 = vector.insert %1030, %1031 [4] : f16 into vector<8xf16>
    memref.store %1032, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1033 = arith.addi %981, %c416 : index
    %1034 = memref.load %view[%1033] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1035 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1036 = vector.insert %1034, %1035 [5] : f16 into vector<8xf16>
    memref.store %1036, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1037 = arith.addi %981, %c448 : index
    %1038 = memref.load %view[%1037] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1039 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1040 = vector.insert %1038, %1039 [6] : f16 into vector<8xf16>
    memref.store %1040, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1041 = arith.addi %981, %c480 : index
    %1042 = memref.load %view[%1041] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1043 = memref.load %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1044 = vector.insert %1042, %1043 [7] : f16 into vector<8xf16>
    memref.store %1044, %14[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1045 = arith.muli %977, %c2 overflow<nsw> : index
    cf.br ^bb132(%c0 : index)
  ^bb132(%1046: index):  // 2 preds: ^bb131, ^bb136
    %1047 = arith.cmpi slt, %1046, %c2 : index
    cf.cond_br %1047, ^bb133, ^bb137
  ^bb133:  // pred: ^bb132
    %1048 = arith.muli %1046, %c16 overflow<nsw> : index
    %1049 = arith.addi %1048, %68 : index
    %1050 = arith.addi %69, %1049 : index
    %1051 = memref.load %view_7[%1050] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1052 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1053 = vector.insert %1051, %1052 [0] : f16 into vector<8xf16>
    memref.store %1053, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1054 = arith.addi %1050, %c32 : index
    %1055 = memref.load %view_7[%1054] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1056 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1057 = vector.insert %1055, %1056 [1] : f16 into vector<8xf16>
    memref.store %1057, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1058 = arith.addi %1050, %c64 : index
    %1059 = memref.load %view_7[%1058] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1060 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1061 = vector.insert %1059, %1060 [2] : f16 into vector<8xf16>
    memref.store %1061, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1062 = arith.addi %1050, %c96 : index
    %1063 = memref.load %view_7[%1062] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1064 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1065 = vector.insert %1063, %1064 [3] : f16 into vector<8xf16>
    memref.store %1065, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1066 = arith.addi %1050, %c128 : index
    %1067 = memref.load %view_7[%1066] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1068 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1069 = vector.insert %1067, %1068 [4] : f16 into vector<8xf16>
    memref.store %1069, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1070 = arith.addi %1050, %c160 : index
    %1071 = memref.load %view_7[%1070] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1072 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1073 = vector.insert %1071, %1072 [5] : f16 into vector<8xf16>
    memref.store %1073, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1074 = arith.addi %1050, %c192 : index
    %1075 = memref.load %view_7[%1074] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1076 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1077 = vector.insert %1075, %1076 [6] : f16 into vector<8xf16>
    memref.store %1077, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1078 = arith.addi %1050, %c224 : index
    %1079 = memref.load %view_7[%1078] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1080 = memref.load %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1081 = vector.insert %1079, %1080 [7] : f16 into vector<8xf16>
    memref.store %1081, %15[%c0] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1082 = arith.addi %1050, %c256 : index
    %1083 = memref.load %view_7[%1082] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1084 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1085 = vector.insert %1083, %1084 [0] : f16 into vector<8xf16>
    memref.store %1085, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1086 = arith.addi %1050, %c288 : index
    %1087 = memref.load %view_7[%1086] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1088 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1089 = vector.insert %1087, %1088 [1] : f16 into vector<8xf16>
    memref.store %1089, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1090 = arith.addi %1050, %c320 : index
    %1091 = memref.load %view_7[%1090] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1092 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1093 = vector.insert %1091, %1092 [2] : f16 into vector<8xf16>
    memref.store %1093, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1094 = arith.addi %1050, %c352 : index
    %1095 = memref.load %view_7[%1094] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1096 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1097 = vector.insert %1095, %1096 [3] : f16 into vector<8xf16>
    memref.store %1097, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1098 = arith.addi %1050, %c384 : index
    %1099 = memref.load %view_7[%1098] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1100 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1101 = vector.insert %1099, %1100 [4] : f16 into vector<8xf16>
    memref.store %1101, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1102 = arith.addi %1050, %c416 : index
    %1103 = memref.load %view_7[%1102] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1104 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1105 = vector.insert %1103, %1104 [5] : f16 into vector<8xf16>
    memref.store %1105, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1106 = arith.addi %1050, %c448 : index
    %1107 = memref.load %view_7[%1106] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1108 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1109 = vector.insert %1107, %1108 [6] : f16 into vector<8xf16>
    memref.store %1109, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1110 = arith.addi %1050, %c480 : index
    %1111 = memref.load %view_7[%1110] : memref<1024xf16, #gpu.address_space<workgroup>>
    %1112 = memref.load %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1113 = vector.insert %1111, %1112 [7] : f16 into vector<8xf16>
    memref.store %1113, %15[%c1] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1114 = arith.addi %1045, %1046 : index
    cf.br ^bb134(%c0 : index)
  ^bb134(%1115: index):  // 2 preds: ^bb133, ^bb135
    %1116 = arith.cmpi slt, %1115, %c2 : index
    cf.cond_br %1116, ^bb135, ^bb136
  ^bb135:  // pred: ^bb134
    %1117 = memref.load %14[%1115] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1118 = memref.load %15[%1115] : memref<2xvector<8xf16>, #gpu.address_space<private>>
    %1119 = memref.load %16[%1114] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %1120 = amdgpu.wmma %1117 * %1118 + %1119 {clamp} : vector<8xf16>, vector<8xf16>, vector<8xf32>
    memref.store %1120, %16[%1114] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    %1121 = arith.addi %1115, %c1 : index
    cf.br ^bb134(%1121 : index)
  ^bb136:  // pred: ^bb134
    %1122 = arith.addi %1046, %c1 : index
    cf.br ^bb132(%1122 : index)
  ^bb137:  // pred: ^bb132
    %1123 = arith.addi %977, %c1 : index
    cf.br ^bb130(%1123 : index)
  ^bb138:  // pred: ^bb130
    %1124 = memref.load %16[%c0] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    vector.store %1124, %17[%c0] : memref<32xf32, #gpu.address_space<private>>, vector<8xf32>
    %1125 = memref.load %16[%c1] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    vector.store %1125, %17[%c8] : memref<32xf32, #gpu.address_space<private>>, vector<8xf32>
    %1126 = memref.load %16[%c2] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    vector.store %1126, %17[%c16] : memref<32xf32, #gpu.address_space<private>>, vector<8xf32>
    %1127 = memref.load %16[%c3] : memref<4xvector<8xf32>, #gpu.address_space<private>>
    vector.store %1127, %17[%c24] : memref<32xf32, #gpu.address_space<private>>, vector<8xf32>
    %subview = memref.subview %18[%946, 0] [1, 32] [1, 1] : memref<2x32xf32, #gpu.address_space<private>> to memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1128 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1129 = memref.load %subview[%c0] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1130 = arith.mulf %1129, %1128 : f32
    %1131 = memref.load %17[%c0] : memref<32xf32, #gpu.address_space<private>>
    %1132 = arith.addf %1130, %1131 : f32
    memref.store %1132, %subview[%c0] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1133 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1134 = memref.load %subview[%c1] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1135 = arith.mulf %1134, %1133 : f32
    %1136 = memref.load %17[%c1] : memref<32xf32, #gpu.address_space<private>>
    %1137 = arith.addf %1135, %1136 : f32
    memref.store %1137, %subview[%c1] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1138 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1139 = memref.load %subview[%c2] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1140 = arith.mulf %1139, %1138 : f32
    %1141 = memref.load %17[%c2] : memref<32xf32, #gpu.address_space<private>>
    %1142 = arith.addf %1140, %1141 : f32
    memref.store %1142, %subview[%c2] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1143 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1144 = memref.load %subview[%c3] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1145 = arith.mulf %1144, %1143 : f32
    %1146 = memref.load %17[%c3] : memref<32xf32, #gpu.address_space<private>>
    %1147 = arith.addf %1145, %1146 : f32
    memref.store %1147, %subview[%c3] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1148 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1149 = memref.load %subview[%c4] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1150 = arith.mulf %1149, %1148 : f32
    %1151 = memref.load %17[%c4] : memref<32xf32, #gpu.address_space<private>>
    %1152 = arith.addf %1150, %1151 : f32
    memref.store %1152, %subview[%c4] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1153 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1154 = memref.load %subview[%c5] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1155 = arith.mulf %1154, %1153 : f32
    %1156 = memref.load %17[%c5] : memref<32xf32, #gpu.address_space<private>>
    %1157 = arith.addf %1155, %1156 : f32
    memref.store %1157, %subview[%c5] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1158 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1159 = memref.load %subview[%c6] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1160 = arith.mulf %1159, %1158 : f32
    %1161 = memref.load %17[%c6] : memref<32xf32, #gpu.address_space<private>>
    %1162 = arith.addf %1160, %1161 : f32
    memref.store %1162, %subview[%c6] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1163 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1164 = memref.load %subview[%c7] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1165 = arith.mulf %1164, %1163 : f32
    %1166 = memref.load %17[%c7] : memref<32xf32, #gpu.address_space<private>>
    %1167 = arith.addf %1165, %1166 : f32
    memref.store %1167, %subview[%c7] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1168 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1169 = memref.load %subview[%c16] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1170 = arith.mulf %1169, %1168 : f32
    %1171 = memref.load %17[%c16] : memref<32xf32, #gpu.address_space<private>>
    %1172 = arith.addf %1170, %1171 : f32
    memref.store %1172, %subview[%c16] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1173 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1174 = memref.load %subview[%c17] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1175 = arith.mulf %1174, %1173 : f32
    %1176 = memref.load %17[%c17] : memref<32xf32, #gpu.address_space<private>>
    %1177 = arith.addf %1175, %1176 : f32
    memref.store %1177, %subview[%c17] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1178 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1179 = memref.load %subview[%c18] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1180 = arith.mulf %1179, %1178 : f32
    %1181 = memref.load %17[%c18] : memref<32xf32, #gpu.address_space<private>>
    %1182 = arith.addf %1180, %1181 : f32
    memref.store %1182, %subview[%c18] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1183 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1184 = memref.load %subview[%c19] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1185 = arith.mulf %1184, %1183 : f32
    %1186 = memref.load %17[%c19] : memref<32xf32, #gpu.address_space<private>>
    %1187 = arith.addf %1185, %1186 : f32
    memref.store %1187, %subview[%c19] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1188 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1189 = memref.load %subview[%c20] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1190 = arith.mulf %1189, %1188 : f32
    %1191 = memref.load %17[%c20] : memref<32xf32, #gpu.address_space<private>>
    %1192 = arith.addf %1190, %1191 : f32
    memref.store %1192, %subview[%c20] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1193 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1194 = memref.load %subview[%c21] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1195 = arith.mulf %1194, %1193 : f32
    %1196 = memref.load %17[%c21] : memref<32xf32, #gpu.address_space<private>>
    %1197 = arith.addf %1195, %1196 : f32
    memref.store %1197, %subview[%c21] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1198 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1199 = memref.load %subview[%c22] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1200 = arith.mulf %1199, %1198 : f32
    %1201 = memref.load %17[%c22] : memref<32xf32, #gpu.address_space<private>>
    %1202 = arith.addf %1200, %1201 : f32
    memref.store %1202, %subview[%c22] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1203 = memref.load %21[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1204 = memref.load %subview[%c23] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1205 = arith.mulf %1204, %1203 : f32
    %1206 = memref.load %17[%c23] : memref<32xf32, #gpu.address_space<private>>
    %1207 = arith.addf %1205, %1206 : f32
    memref.store %1207, %subview[%c23] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1208 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1209 = memref.load %subview[%c8] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1210 = arith.mulf %1209, %1208 : f32
    %1211 = memref.load %17[%c8] : memref<32xf32, #gpu.address_space<private>>
    %1212 = arith.addf %1210, %1211 : f32
    memref.store %1212, %subview[%c8] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1213 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1214 = memref.load %subview[%c9] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1215 = arith.mulf %1214, %1213 : f32
    %1216 = memref.load %17[%c9] : memref<32xf32, #gpu.address_space<private>>
    %1217 = arith.addf %1215, %1216 : f32
    memref.store %1217, %subview[%c9] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1218 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1219 = memref.load %subview[%c10] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1220 = arith.mulf %1219, %1218 : f32
    %1221 = memref.load %17[%c10] : memref<32xf32, #gpu.address_space<private>>
    %1222 = arith.addf %1220, %1221 : f32
    memref.store %1222, %subview[%c10] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1223 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1224 = memref.load %subview[%c11] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1225 = arith.mulf %1224, %1223 : f32
    %1226 = memref.load %17[%c11] : memref<32xf32, #gpu.address_space<private>>
    %1227 = arith.addf %1225, %1226 : f32
    memref.store %1227, %subview[%c11] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1228 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1229 = memref.load %subview[%c12] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1230 = arith.mulf %1229, %1228 : f32
    %1231 = memref.load %17[%c12] : memref<32xf32, #gpu.address_space<private>>
    %1232 = arith.addf %1230, %1231 : f32
    memref.store %1232, %subview[%c12] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1233 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1234 = memref.load %subview[%c13] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1235 = arith.mulf %1234, %1233 : f32
    %1236 = memref.load %17[%c13] : memref<32xf32, #gpu.address_space<private>>
    %1237 = arith.addf %1235, %1236 : f32
    memref.store %1237, %subview[%c13] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1238 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1239 = memref.load %subview[%c14] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1240 = arith.mulf %1239, %1238 : f32
    %1241 = memref.load %17[%c14] : memref<32xf32, #gpu.address_space<private>>
    %1242 = arith.addf %1240, %1241 : f32
    memref.store %1242, %subview[%c14] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1243 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1244 = memref.load %subview[%c15] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1245 = arith.mulf %1244, %1243 : f32
    %1246 = memref.load %17[%c15] : memref<32xf32, #gpu.address_space<private>>
    %1247 = arith.addf %1245, %1246 : f32
    memref.store %1247, %subview[%c15] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1248 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1249 = memref.load %subview[%c24] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1250 = arith.mulf %1249, %1248 : f32
    %1251 = memref.load %17[%c24] : memref<32xf32, #gpu.address_space<private>>
    %1252 = arith.addf %1250, %1251 : f32
    memref.store %1252, %subview[%c24] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1253 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1254 = memref.load %subview[%c25] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1255 = arith.mulf %1254, %1253 : f32
    %1256 = memref.load %17[%c25] : memref<32xf32, #gpu.address_space<private>>
    %1257 = arith.addf %1255, %1256 : f32
    memref.store %1257, %subview[%c25] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1258 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1259 = memref.load %subview[%c26] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1260 = arith.mulf %1259, %1258 : f32
    %1261 = memref.load %17[%c26] : memref<32xf32, #gpu.address_space<private>>
    %1262 = arith.addf %1260, %1261 : f32
    memref.store %1262, %subview[%c26] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1263 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1264 = memref.load %subview[%c27] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1265 = arith.mulf %1264, %1263 : f32
    %1266 = memref.load %17[%c27] : memref<32xf32, #gpu.address_space<private>>
    %1267 = arith.addf %1265, %1266 : f32
    memref.store %1267, %subview[%c27] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1268 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1269 = memref.load %subview[%c28] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1270 = arith.mulf %1269, %1268 : f32
    %1271 = memref.load %17[%c28] : memref<32xf32, #gpu.address_space<private>>
    %1272 = arith.addf %1270, %1271 : f32
    memref.store %1272, %subview[%c28] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1273 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1274 = memref.load %subview[%c29] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1275 = arith.mulf %1274, %1273 : f32
    %1276 = memref.load %17[%c29] : memref<32xf32, #gpu.address_space<private>>
    %1277 = arith.addf %1275, %1276 : f32
    memref.store %1277, %subview[%c29] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1278 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1279 = memref.load %subview[%c30] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1280 = arith.mulf %1279, %1278 : f32
    %1281 = memref.load %17[%c30] : memref<32xf32, #gpu.address_space<private>>
    %1282 = arith.addf %1280, %1281 : f32
    memref.store %1282, %subview[%c30] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1283 = memref.load %21[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1284 = memref.load %subview[%c31] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1285 = arith.mulf %1284, %1283 : f32
    %1286 = memref.load %17[%c31] : memref<32xf32, #gpu.address_space<private>>
    %1287 = arith.addf %1285, %1286 : f32
    memref.store %1287, %subview[%c31] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1288 = arith.addi %946, %c1 : index
    cf.br ^bb122(%1288 : index)
  ^bb139:  // pred: ^bb122
    rock.dealloc %0 : memref<2048xi8, #gpu.address_space<workgroup>>
    rock.dealloc %1 : memref<2048xi8, #gpu.address_space<workgroup>>
    %1289 = arith.addi %77, %c1 : index
    cf.br ^bb14(%1289 : index)
  ^bb140:  // pred: ^bb14
    cf.br ^bb141(%c0 : index)
  ^bb141(%1290: index):  // 2 preds: ^bb140, ^bb142
    %1291 = arith.cmpi slt, %1290, %c2 : index
    cf.cond_br %1291, ^bb142, ^bb143
  ^bb142:  // pred: ^bb141
    %subview_11 = memref.subview %18[%1290, 0] [1, 32] [1, 1] : memref<2x32xf32, #gpu.address_space<private>> to memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1292 = memref.load %subview_11[%c0] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1293 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1294 = arith.divf %1292, %1293 : f32
    memref.store %1294, %subview_11[%c0] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1295 = memref.load %subview_11[%c1] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1296 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1297 = arith.divf %1295, %1296 : f32
    memref.store %1297, %subview_11[%c1] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1298 = memref.load %subview_11[%c2] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1299 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1300 = arith.divf %1298, %1299 : f32
    memref.store %1300, %subview_11[%c2] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1301 = memref.load %subview_11[%c3] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1302 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1303 = arith.divf %1301, %1302 : f32
    memref.store %1303, %subview_11[%c3] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1304 = memref.load %subview_11[%c4] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1305 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1306 = arith.divf %1304, %1305 : f32
    memref.store %1306, %subview_11[%c4] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1307 = memref.load %subview_11[%c5] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1308 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1309 = arith.divf %1307, %1308 : f32
    memref.store %1309, %subview_11[%c5] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1310 = memref.load %subview_11[%c6] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1311 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1312 = arith.divf %1310, %1311 : f32
    memref.store %1312, %subview_11[%c6] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1313 = memref.load %subview_11[%c7] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1314 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1315 = arith.divf %1313, %1314 : f32
    memref.store %1315, %subview_11[%c7] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1316 = memref.load %subview_11[%c16] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1317 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1318 = arith.divf %1316, %1317 : f32
    memref.store %1318, %subview_11[%c16] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1319 = memref.load %subview_11[%c17] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1320 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1321 = arith.divf %1319, %1320 : f32
    memref.store %1321, %subview_11[%c17] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1322 = memref.load %subview_11[%c18] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1323 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1324 = arith.divf %1322, %1323 : f32
    memref.store %1324, %subview_11[%c18] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1325 = memref.load %subview_11[%c19] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1326 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1327 = arith.divf %1325, %1326 : f32
    memref.store %1327, %subview_11[%c19] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1328 = memref.load %subview_11[%c20] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1329 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1330 = arith.divf %1328, %1329 : f32
    memref.store %1330, %subview_11[%c20] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1331 = memref.load %subview_11[%c21] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1332 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1333 = arith.divf %1331, %1332 : f32
    memref.store %1333, %subview_11[%c21] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1334 = memref.load %subview_11[%c22] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1335 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1336 = arith.divf %1334, %1335 : f32
    memref.store %1336, %subview_11[%c22] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1337 = memref.load %subview_11[%c23] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1338 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1339 = arith.divf %1337, %1338 : f32
    memref.store %1339, %subview_11[%c23] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1340 = memref.load %subview_11[%c8] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1341 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1342 = arith.divf %1340, %1341 : f32
    memref.store %1342, %subview_11[%c8] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1343 = memref.load %subview_11[%c9] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1344 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1345 = arith.divf %1343, %1344 : f32
    memref.store %1345, %subview_11[%c9] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1346 = memref.load %subview_11[%c10] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1347 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1348 = arith.divf %1346, %1347 : f32
    memref.store %1348, %subview_11[%c10] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1349 = memref.load %subview_11[%c11] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1350 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1351 = arith.divf %1349, %1350 : f32
    memref.store %1351, %subview_11[%c11] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1352 = memref.load %subview_11[%c12] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1353 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1354 = arith.divf %1352, %1353 : f32
    memref.store %1354, %subview_11[%c12] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1355 = memref.load %subview_11[%c13] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1356 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1357 = arith.divf %1355, %1356 : f32
    memref.store %1357, %subview_11[%c13] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1358 = memref.load %subview_11[%c14] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1359 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1360 = arith.divf %1358, %1359 : f32
    memref.store %1360, %subview_11[%c14] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1361 = memref.load %subview_11[%c15] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1362 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1363 = arith.divf %1361, %1362 : f32
    memref.store %1363, %subview_11[%c15] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1364 = memref.load %subview_11[%c24] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1365 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1366 = arith.divf %1364, %1365 : f32
    memref.store %1366, %subview_11[%c24] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1367 = memref.load %subview_11[%c25] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1368 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1369 = arith.divf %1367, %1368 : f32
    memref.store %1369, %subview_11[%c25] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1370 = memref.load %subview_11[%c26] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1371 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1372 = arith.divf %1370, %1371 : f32
    memref.store %1372, %subview_11[%c26] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1373 = memref.load %subview_11[%c27] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1374 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1375 = arith.divf %1373, %1374 : f32
    memref.store %1375, %subview_11[%c27] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1376 = memref.load %subview_11[%c28] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1377 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1378 = arith.divf %1376, %1377 : f32
    memref.store %1378, %subview_11[%c28] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1379 = memref.load %subview_11[%c29] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1380 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1381 = arith.divf %1379, %1380 : f32
    memref.store %1381, %subview_11[%c29] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1382 = memref.load %subview_11[%c30] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1383 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1384 = arith.divf %1382, %1383 : f32
    memref.store %1384, %subview_11[%c30] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1385 = memref.load %subview_11[%c31] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1386 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1387 = arith.divf %1385, %1386 : f32
    memref.store %1387, %subview_11[%c31] : memref<32xf32, strided<[1], offset: ?>, #gpu.address_space<private>>
    %1388 = arith.addi %1290, %c1 : index
    cf.br ^bb141(%1388 : index)
  ^bb143:  // pred: ^bb141
    %collapse_shape = memref.collapse_shape %18 [[0, 1]] : memref<2x32xf32, #gpu.address_space<private>> into memref<64xf32, #gpu.address_space<private>>
    %collapse_shape_12 = memref.collapse_shape %19 [[0, 1]] : memref<2x32xf16, #gpu.address_space<private>> into memref<64xf16, #gpu.address_space<private>>
    %1389 = vector.load %collapse_shape[%c0] : memref<64xf32, #gpu.address_space<private>>, vector<64xf32>
    %1390 = arith.truncf %1389 : vector<64xf32> to vector<64xf16>
    vector.store %1390, %collapse_shape_12[%c0] : memref<64xf16, #gpu.address_space<private>>, vector<64xf16>
    %1391 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1392 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1393 = arith.truncf %1391 : f32 to f16
    %1394 = arith.truncf %1392 : f32 to f16
    %1395 = math.log2 %1394 : f16
    %1396 = arith.addf %1395, %1393 : f16
    %1397 = arith.mulf %1396, %cst_5 : f16
    memref.store %1397, %29[%c0] : memref<32xf16, #gpu.address_space<private>>
    %1398 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1399 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1400 = arith.truncf %1398 : f32 to f16
    %1401 = arith.truncf %1399 : f32 to f16
    %1402 = math.log2 %1401 : f16
    %1403 = arith.addf %1402, %1400 : f16
    %1404 = arith.mulf %1403, %cst_5 : f16
    memref.store %1404, %29[%c1] : memref<32xf16, #gpu.address_space<private>>
    %1405 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1406 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1407 = arith.truncf %1405 : f32 to f16
    %1408 = arith.truncf %1406 : f32 to f16
    %1409 = math.log2 %1408 : f16
    %1410 = arith.addf %1409, %1407 : f16
    %1411 = arith.mulf %1410, %cst_5 : f16
    memref.store %1411, %29[%c2] : memref<32xf16, #gpu.address_space<private>>
    %1412 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1413 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1414 = arith.truncf %1412 : f32 to f16
    %1415 = arith.truncf %1413 : f32 to f16
    %1416 = math.log2 %1415 : f16
    %1417 = arith.addf %1416, %1414 : f16
    %1418 = arith.mulf %1417, %cst_5 : f16
    memref.store %1418, %29[%c3] : memref<32xf16, #gpu.address_space<private>>
    %1419 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1420 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1421 = arith.truncf %1419 : f32 to f16
    %1422 = arith.truncf %1420 : f32 to f16
    %1423 = math.log2 %1422 : f16
    %1424 = arith.addf %1423, %1421 : f16
    %1425 = arith.mulf %1424, %cst_5 : f16
    memref.store %1425, %29[%c4] : memref<32xf16, #gpu.address_space<private>>
    %1426 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1427 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1428 = arith.truncf %1426 : f32 to f16
    %1429 = arith.truncf %1427 : f32 to f16
    %1430 = math.log2 %1429 : f16
    %1431 = arith.addf %1430, %1428 : f16
    %1432 = arith.mulf %1431, %cst_5 : f16
    memref.store %1432, %29[%c5] : memref<32xf16, #gpu.address_space<private>>
    %1433 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1434 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1435 = arith.truncf %1433 : f32 to f16
    %1436 = arith.truncf %1434 : f32 to f16
    %1437 = math.log2 %1436 : f16
    %1438 = arith.addf %1437, %1435 : f16
    %1439 = arith.mulf %1438, %cst_5 : f16
    memref.store %1439, %29[%c6] : memref<32xf16, #gpu.address_space<private>>
    %1440 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1441 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1442 = arith.truncf %1440 : f32 to f16
    %1443 = arith.truncf %1441 : f32 to f16
    %1444 = math.log2 %1443 : f16
    %1445 = arith.addf %1444, %1442 : f16
    %1446 = arith.mulf %1445, %cst_5 : f16
    memref.store %1446, %29[%c7] : memref<32xf16, #gpu.address_space<private>>
    %1447 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1448 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1449 = arith.truncf %1447 : f32 to f16
    %1450 = arith.truncf %1448 : f32 to f16
    %1451 = math.log2 %1450 : f16
    %1452 = arith.addf %1451, %1449 : f16
    %1453 = arith.mulf %1452, %cst_5 : f16
    memref.store %1453, %29[%c16] : memref<32xf16, #gpu.address_space<private>>
    %1454 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1455 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1456 = arith.truncf %1454 : f32 to f16
    %1457 = arith.truncf %1455 : f32 to f16
    %1458 = math.log2 %1457 : f16
    %1459 = arith.addf %1458, %1456 : f16
    %1460 = arith.mulf %1459, %cst_5 : f16
    memref.store %1460, %29[%c17] : memref<32xf16, #gpu.address_space<private>>
    %1461 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1462 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1463 = arith.truncf %1461 : f32 to f16
    %1464 = arith.truncf %1462 : f32 to f16
    %1465 = math.log2 %1464 : f16
    %1466 = arith.addf %1465, %1463 : f16
    %1467 = arith.mulf %1466, %cst_5 : f16
    memref.store %1467, %29[%c18] : memref<32xf16, #gpu.address_space<private>>
    %1468 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1469 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1470 = arith.truncf %1468 : f32 to f16
    %1471 = arith.truncf %1469 : f32 to f16
    %1472 = math.log2 %1471 : f16
    %1473 = arith.addf %1472, %1470 : f16
    %1474 = arith.mulf %1473, %cst_5 : f16
    memref.store %1474, %29[%c19] : memref<32xf16, #gpu.address_space<private>>
    %1475 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1476 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1477 = arith.truncf %1475 : f32 to f16
    %1478 = arith.truncf %1476 : f32 to f16
    %1479 = math.log2 %1478 : f16
    %1480 = arith.addf %1479, %1477 : f16
    %1481 = arith.mulf %1480, %cst_5 : f16
    memref.store %1481, %29[%c20] : memref<32xf16, #gpu.address_space<private>>
    %1482 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1483 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1484 = arith.truncf %1482 : f32 to f16
    %1485 = arith.truncf %1483 : f32 to f16
    %1486 = math.log2 %1485 : f16
    %1487 = arith.addf %1486, %1484 : f16
    %1488 = arith.mulf %1487, %cst_5 : f16
    memref.store %1488, %29[%c21] : memref<32xf16, #gpu.address_space<private>>
    %1489 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1490 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1491 = arith.truncf %1489 : f32 to f16
    %1492 = arith.truncf %1490 : f32 to f16
    %1493 = math.log2 %1492 : f16
    %1494 = arith.addf %1493, %1491 : f16
    %1495 = arith.mulf %1494, %cst_5 : f16
    memref.store %1495, %29[%c22] : memref<32xf16, #gpu.address_space<private>>
    %1496 = memref.load %20[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1497 = memref.load %25[%c0] : memref<2xf32, #gpu.address_space<private>>
    %1498 = arith.truncf %1496 : f32 to f16
    %1499 = arith.truncf %1497 : f32 to f16
    %1500 = math.log2 %1499 : f16
    %1501 = arith.addf %1500, %1498 : f16
    %1502 = arith.mulf %1501, %cst_5 : f16
    memref.store %1502, %29[%c23] : memref<32xf16, #gpu.address_space<private>>
    %1503 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1504 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1505 = arith.truncf %1503 : f32 to f16
    %1506 = arith.truncf %1504 : f32 to f16
    %1507 = math.log2 %1506 : f16
    %1508 = arith.addf %1507, %1505 : f16
    %1509 = arith.mulf %1508, %cst_5 : f16
    memref.store %1509, %29[%c8] : memref<32xf16, #gpu.address_space<private>>
    %1510 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1511 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1512 = arith.truncf %1510 : f32 to f16
    %1513 = arith.truncf %1511 : f32 to f16
    %1514 = math.log2 %1513 : f16
    %1515 = arith.addf %1514, %1512 : f16
    %1516 = arith.mulf %1515, %cst_5 : f16
    memref.store %1516, %29[%c9] : memref<32xf16, #gpu.address_space<private>>
    %1517 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1518 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1519 = arith.truncf %1517 : f32 to f16
    %1520 = arith.truncf %1518 : f32 to f16
    %1521 = math.log2 %1520 : f16
    %1522 = arith.addf %1521, %1519 : f16
    %1523 = arith.mulf %1522, %cst_5 : f16
    memref.store %1523, %29[%c10] : memref<32xf16, #gpu.address_space<private>>
    %1524 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1525 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1526 = arith.truncf %1524 : f32 to f16
    %1527 = arith.truncf %1525 : f32 to f16
    %1528 = math.log2 %1527 : f16
    %1529 = arith.addf %1528, %1526 : f16
    %1530 = arith.mulf %1529, %cst_5 : f16
    memref.store %1530, %29[%c11] : memref<32xf16, #gpu.address_space<private>>
    %1531 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1532 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1533 = arith.truncf %1531 : f32 to f16
    %1534 = arith.truncf %1532 : f32 to f16
    %1535 = math.log2 %1534 : f16
    %1536 = arith.addf %1535, %1533 : f16
    %1537 = arith.mulf %1536, %cst_5 : f16
    memref.store %1537, %29[%c12] : memref<32xf16, #gpu.address_space<private>>
    %1538 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1539 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1540 = arith.truncf %1538 : f32 to f16
    %1541 = arith.truncf %1539 : f32 to f16
    %1542 = math.log2 %1541 : f16
    %1543 = arith.addf %1542, %1540 : f16
    %1544 = arith.mulf %1543, %cst_5 : f16
    memref.store %1544, %29[%c13] : memref<32xf16, #gpu.address_space<private>>
    %1545 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1546 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1547 = arith.truncf %1545 : f32 to f16
    %1548 = arith.truncf %1546 : f32 to f16
    %1549 = math.log2 %1548 : f16
    %1550 = arith.addf %1549, %1547 : f16
    %1551 = arith.mulf %1550, %cst_5 : f16
    memref.store %1551, %29[%c14] : memref<32xf16, #gpu.address_space<private>>
    %1552 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1553 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1554 = arith.truncf %1552 : f32 to f16
    %1555 = arith.truncf %1553 : f32 to f16
    %1556 = math.log2 %1555 : f16
    %1557 = arith.addf %1556, %1554 : f16
    %1558 = arith.mulf %1557, %cst_5 : f16
    memref.store %1558, %29[%c15] : memref<32xf16, #gpu.address_space<private>>
    %1559 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1560 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1561 = arith.truncf %1559 : f32 to f16
    %1562 = arith.truncf %1560 : f32 to f16
    %1563 = math.log2 %1562 : f16
    %1564 = arith.addf %1563, %1561 : f16
    %1565 = arith.mulf %1564, %cst_5 : f16
    memref.store %1565, %29[%c24] : memref<32xf16, #gpu.address_space<private>>
    %1566 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1567 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1568 = arith.truncf %1566 : f32 to f16
    %1569 = arith.truncf %1567 : f32 to f16
    %1570 = math.log2 %1569 : f16
    %1571 = arith.addf %1570, %1568 : f16
    %1572 = arith.mulf %1571, %cst_5 : f16
    memref.store %1572, %29[%c25] : memref<32xf16, #gpu.address_space<private>>
    %1573 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1574 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1575 = arith.truncf %1573 : f32 to f16
    %1576 = arith.truncf %1574 : f32 to f16
    %1577 = math.log2 %1576 : f16
    %1578 = arith.addf %1577, %1575 : f16
    %1579 = arith.mulf %1578, %cst_5 : f16
    memref.store %1579, %29[%c26] : memref<32xf16, #gpu.address_space<private>>
    %1580 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1581 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1582 = arith.truncf %1580 : f32 to f16
    %1583 = arith.truncf %1581 : f32 to f16
    %1584 = math.log2 %1583 : f16
    %1585 = arith.addf %1584, %1582 : f16
    %1586 = arith.mulf %1585, %cst_5 : f16
    memref.store %1586, %29[%c27] : memref<32xf16, #gpu.address_space<private>>
    %1587 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1588 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1589 = arith.truncf %1587 : f32 to f16
    %1590 = arith.truncf %1588 : f32 to f16
    %1591 = math.log2 %1590 : f16
    %1592 = arith.addf %1591, %1589 : f16
    %1593 = arith.mulf %1592, %cst_5 : f16
    memref.store %1593, %29[%c28] : memref<32xf16, #gpu.address_space<private>>
    %1594 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1595 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1596 = arith.truncf %1594 : f32 to f16
    %1597 = arith.truncf %1595 : f32 to f16
    %1598 = math.log2 %1597 : f16
    %1599 = arith.addf %1598, %1596 : f16
    %1600 = arith.mulf %1599, %cst_5 : f16
    memref.store %1600, %29[%c29] : memref<32xf16, #gpu.address_space<private>>
    %1601 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1602 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1603 = arith.truncf %1601 : f32 to f16
    %1604 = arith.truncf %1602 : f32 to f16
    %1605 = math.log2 %1604 : f16
    %1606 = arith.addf %1605, %1603 : f16
    %1607 = arith.mulf %1606, %cst_5 : f16
    memref.store %1607, %29[%c30] : memref<32xf16, #gpu.address_space<private>>
    %1608 = memref.load %20[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1609 = memref.load %25[%c1] : memref<2xf32, #gpu.address_space<private>>
    %1610 = arith.truncf %1608 : f32 to f16
    %1611 = arith.truncf %1609 : f32 to f16
    %1612 = math.log2 %1611 : f16
    %1613 = arith.addf %1612, %1610 : f16
    %1614 = arith.mulf %1613, %cst_5 : f16
    memref.store %1614, %29[%c31] : memref<32xf16, #gpu.address_space<private>>
    %1615 = arith.divui %2, %c4 : index
    %1616 = arith.remui %2, %c4 : index
    %1617 = arith.muli %1615, %c2 overflow<nsw> : index
    %1618 = arith.addi %1617, %68 : index
    %1619 = arith.muli %1618, %c4 overflow<nsw> : index
    %1620 = arith.addi %1619, %1616 : index
    %1621 = arith.muli %1620, %c64 overflow<nsw> : index
    %1622 = arith.addi %1621, %70 : index
    %1623 = arith.cmpi ult, %68, %c2 : index
    %1624 = arith.select %1623, %1622, %c3072 : index
    %1625 = arith.index_cast %1624 : index to i32
    %1626 = vector.load %collapse_shape_12[%c0] : memref<64xf16, #gpu.address_space<private>>, vector<8xf16>
    amdgpu.raw_buffer_store %1626 -> %arg7[%1625] : vector<8xf16> -> memref<3072xf16>, i32
    %1627 = arith.addi %1622, %c16 : index
    %1628 = arith.select %1623, %1627, %c3072 : index
    %1629 = arith.index_cast %1628 : index to i32
    %1630 = vector.load %collapse_shape_12[%c16] : memref<64xf16, #gpu.address_space<private>>, vector<8xf16>
    amdgpu.raw_buffer_store %1630 -> %arg7[%1629] : vector<8xf16> -> memref<3072xf16>, i32
    %1631 = arith.addi %1622, %c32 : index
    %1632 = arith.select %1623, %1631, %c3072 : index
    %1633 = arith.index_cast %1632 : index to i32
    %1634 = vector.load %collapse_shape_12[%c32] : memref<64xf16, #gpu.address_space<private>>, vector<8xf16>
    amdgpu.raw_buffer_store %1634 -> %arg7[%1633] : vector<8xf16> -> memref<3072xf16>, i32
    %1635 = arith.addi %1622, %c48 : index
    %1636 = arith.select %1623, %1635, %c3072 : index
    %1637 = arith.index_cast %1636 : index to i32
    %1638 = vector.load %collapse_shape_12[%c48] : memref<64xf16, #gpu.address_space<private>>, vector<8xf16>
    amdgpu.raw_buffer_store %1638 -> %arg7[%1637] : vector<8xf16> -> memref<3072xf16>, i32
    %1639 = arith.select %1623, %1620, %c48 : index
    %1640 = arith.index_cast %1639 : index to i32
    %1641 = memref.load %29[%c0] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1641 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1642 = memref.load %29[%c1] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1642 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1643 = memref.load %29[%c2] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1643 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1644 = memref.load %29[%c3] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1644 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1645 = memref.load %29[%c4] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1645 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1646 = memref.load %29[%c5] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1646 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1647 = memref.load %29[%c6] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1647 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1648 = memref.load %29[%c7] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1648 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1649 = memref.load %29[%c16] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1649 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1650 = memref.load %29[%c17] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1650 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1651 = memref.load %29[%c18] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1651 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1652 = memref.load %29[%c19] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1652 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1653 = memref.load %29[%c20] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1653 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1654 = memref.load %29[%c21] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1654 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1655 = memref.load %29[%c22] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1655 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    %1656 = memref.load %29[%c23] : memref<32xf16, #gpu.address_space<private>>
    amdgpu.raw_buffer_store %1656 -> %arg6[%1640] : f16 -> memref<48xf16>, i32
    cf.br ^bb144
  ^bb144:  // 2 preds: ^bb12, ^bb143
    return
  }
  func.func @main() {
    %c32_i32 = arith.constant 32 : i32
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c17_i32 = arith.constant 17 : i32
    %c0 = arith.constant 0 : index
    %c5_i16 = arith.constant 5 : i16
    %c-5_i16 = arith.constant -5 : i16
    %c1_i32 = arith.constant 1 : i32
    call @seedRandomValues(%c1_i32) : (i32) -> ()
    %alloc = memref.alloc() : memref<768xf16>
    affine.for %arg0 = 0 to 768 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc[%arg0] : memref<768xf16>
    }
    %alloc_0 = memref.alloc() : memref<768xf16>
    memref.copy %alloc, %alloc_0 : memref<768xf16> to memref<768xf16>
    %alloc_1 = memref.alloc() : memref<147456xf16>
    affine.for %arg0 = 0 to 147456 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_1[%arg0] : memref<147456xf16>
    }
    %alloc_2 = memref.alloc() : memref<147456xf16>
    memref.copy %alloc_1, %alloc_2 : memref<147456xf16> to memref<147456xf16>
    %alloc_3 = memref.alloc() : memref<147456xf16>
    affine.for %arg0 = 0 to 147456 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_3[%arg0] : memref<147456xf16>
    }
    %alloc_4 = memref.alloc() : memref<147456xf16>
    memref.copy %alloc_3, %alloc_4 : memref<147456xf16> to memref<147456xf16>
    %alloc_5 = memref.alloc() : memref<4608xf16>
    affine.for %arg0 = 0 to 4608 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_5[%arg0] : memref<4608xf16>
    }
    %alloc_6 = memref.alloc() : memref<4608xf16>
    memref.copy %alloc_5, %alloc_6 : memref<4608xf16> to memref<4608xf16>
    %alloc_7 = memref.alloc() : memref<4608xf16>
    affine.for %arg0 = 0 to 4608 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_7[%arg0] : memref<4608xf16>
    }
    %alloc_8 = memref.alloc() : memref<4608xf16>
    memref.copy %alloc_7, %alloc_8 : memref<4608xf16> to memref<4608xf16>
    %alloc_9 = memref.alloc() : memref<3xi32>
    memref.store %c17_i32, %alloc_9[%c0] : memref<3xi32>
    memref.store %c1_i32, %alloc_9[%c1] : memref<3xi32>
    memref.store %c32_i32, %alloc_9[%c2] : memref<3xi32>
    %alloc_10 = memref.alloc() : memref<3xi32>
    memref.copy %alloc_9, %alloc_10 : memref<3xi32> to memref<3xi32>
    %alloc_11 = memref.alloc() : memref<48xf16>
    affine.for %arg0 = 0 to 48 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_11[%arg0] : memref<48xf16>
    }
    %alloc_12 = memref.alloc() : memref<48xf16>
    memref.copy %alloc_11, %alloc_12 : memref<48xf16> to memref<48xf16>
    %alloc_13 = memref.alloc() : memref<3072xf16>
    affine.for %arg0 = 0 to 3072 {
      %0 = func.call @randomIntegerValue(%c-5_i16, %c5_i16) : (i16, i16) -> f32
      %1 = arith.truncf %0 : f32 to f16
      memref.store %1, %alloc_13[%arg0] : memref<3072xf16>
    }
    %alloc_14 = memref.alloc() : memref<3072xf16>
    memref.copy %alloc_13, %alloc_14 : memref<3072xf16> to memref<3072xf16>
    call @rock_attention_gpu(%alloc, %alloc_1, %alloc_3, %alloc_5, %alloc_7, %alloc_9, %alloc_11, %alloc_13) : (memref<768xf16>, memref<147456xf16>, memref<147456xf16>, memref<4608xf16>, memref<4608xf16>, memref<3xi32>, memref<48xf16>, memref<3072xf16>) -> ()
    call @host_naive_attention(%alloc_0, %alloc_2, %alloc_4, %alloc_6, %alloc_8, %alloc_10, %alloc_12, %alloc_14) : (memref<768xf16>, memref<147456xf16>, memref<147456xf16>, memref<4608xf16>, memref<4608xf16>, memref<3xi32>, memref<48xf16>, memref<3072xf16>) -> ()
    call @rock_attention_verify7(%alloc_13, %alloc_14) : (memref<3072xf16>, memref<3072xf16>) -> ()
    memref.dealloc %alloc_0 : memref<768xf16>
    memref.dealloc %alloc_2 : memref<147456xf16>
    memref.dealloc %alloc_4 : memref<147456xf16>
    memref.dealloc %alloc_6 : memref<4608xf16>
    memref.dealloc %alloc_8 : memref<4608xf16>
    memref.dealloc %alloc_10 : memref<3xi32>
    memref.dealloc %alloc_12 : memref<48xf16>
    memref.dealloc %alloc_14 : memref<3072xf16>
    memref.dealloc %alloc : memref<768xf16>
    memref.dealloc %alloc_1 : memref<147456xf16>
    memref.dealloc %alloc_3 : memref<147456xf16>
    memref.dealloc %alloc_5 : memref<4608xf16>
    memref.dealloc %alloc_7 : memref<4608xf16>
    memref.dealloc %alloc_9 : memref<3xi32>
    memref.dealloc %alloc_11 : memref<48xf16>
    memref.dealloc %alloc_13 : memref<3072xf16>
    return
  }
  func.func private @seedRandomValues(i32)
  func.func private @randomIntegerValue(i16, i16) -> f32
  func.func @host_naive_attention(%arg0: memref<768xf16>, %arg1: memref<147456xf16>, %arg2: memref<147456xf16>, %arg3: memref<4608xf16>, %arg4: memref<4608xf16>, %arg5: memref<3xi32>, %arg6: memref<48xf16>, %arg7: memref<3072xf16>) {
    %0 = memref.get_global @__constant_384xi32 : memref<384xi32>
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0.000000e+00 : f16
    %cst_2 = arith.constant 1.000000e+00 : f32
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %cst_4 = arith.constant 0.000000e+00 : f32
    %expand_shape = memref.expand_shape %arg0 [[0, 1, 2]] output_shape [6, 2, 64] : memref<768xf16> into memref<6x2x64xf16>
    %expand_shape_5 = memref.expand_shape %arg1 [[0, 1, 2]] output_shape [6, 64, 384] : memref<147456xf16> into memref<6x64x384xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<6x2x384xf32>
    linalg.fill ins(%cst_4 : f32) outs(%alloc : memref<6x2x384xf32>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_shape, %expand_shape_5 : memref<6x2x64xf16>, memref<6x64x384xf16>) outs(%alloc : memref<6x2x384xf32>) {
    ^bb0(%in: f16, %in_38: f16, %out: f32):
      %1 = arith.addf %in_38, %cst_1 : f16
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %1 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    }
    %expand_shape_6 = memref.expand_shape %arg3 [[0, 1, 2]] output_shape [3, 4, 384] : memref<4608xf16> into memref<3x4x384xf16>
    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<3x4x384xf16>
    linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg5, %expand_shape_6 : memref<384xi32>, memref<3xi32>, memref<3x4x384xf16>) outs(%alloc_7 : memref<3x4x384xf16>) {
    ^bb0(%in: i32, %in_38: i32, %in_39: f16, %out: f16):
      %1 = arith.cmpi sgt, %in, %in_38 : i32
      %2 = arith.select %1, %cst_0, %in_39 : f16
      linalg.yield %2 : f16
    }
    %expand_shape_8 = memref.expand_shape %alloc_7 [[0], [1, 2], [3]] output_shape [3, 4, 1, 384] : memref<3x4x384xf16> into memref<3x4x1x384xf16>
    %collapse_shape = memref.collapse_shape %expand_shape_8 [[0, 1], [2], [3]] : memref<3x4x1x384xf16> into memref<12x1x384xf16>
    %expand_shape_9 = memref.expand_shape %collapse_shape [[0, 1], [2], [3]] output_shape [6, 2, 1, 384] : memref<12x1x384xf16> into memref<6x2x1x384xf16>
    %expand_shape_10 = memref.expand_shape %arg4 [[0, 1, 2]] output_shape [3, 4, 384] : memref<4608xf16> into memref<3x4x384xf16>
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<3x4x384xf16>
    linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg5, %expand_shape_10 : memref<384xi32>, memref<3xi32>, memref<3x4x384xf16>) outs(%alloc_11 : memref<3x4x384xf16>) {
    ^bb0(%in: i32, %in_38: i32, %in_39: f16, %out: f16):
      %1 = arith.cmpi sgt, %in, %in_38 : i32
      %2 = arith.select %1, %cst_1, %in_39 : f16
      linalg.yield %2 : f16
    }
    %expand_shape_12 = memref.expand_shape %alloc_11 [[0], [1, 2], [3]] output_shape [3, 4, 1, 384] : memref<3x4x384xf16> into memref<3x4x1x384xf16>
    %collapse_shape_13 = memref.collapse_shape %expand_shape_12 [[0, 1], [2], [3]] : memref<3x4x1x384xf16> into memref<12x1x384xf16>
    %expand_shape_14 = memref.expand_shape %collapse_shape_13 [[0, 1], [2], [3]] output_shape [6, 2, 1, 384] : memref<12x1x384xf16> into memref<6x2x1x384xf16>
    %collapse_shape_15 = memref.collapse_shape %expand_shape_9 [[0], [1, 2], [3]] : memref<6x2x1x384xf16> into memref<6x2x384xf16>
    %collapse_shape_16 = memref.collapse_shape %expand_shape_14 [[0], [1, 2], [3]] : memref<6x2x1x384xf16> into memref<6x2x384xf16>
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<6x2x384xf32>
    linalg.generic {indexing_maps = [#map5, #map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc, %collapse_shape_15, %collapse_shape_16 : memref<6x2x384xf32>, memref<6x2x384xf16>, memref<6x2x384xf16>) outs(%alloc_17 : memref<6x2x384xf32>) {
    ^bb0(%in: f32, %in_38: f16, %in_39: f16, %out: f32):
      %1 = arith.truncf %in : f32 to f16
      %2 = arith.mulf %1, %in_38 : f16
      %3 = arith.addf %2, %in_39 : f16
      %4 = arith.extf %3 : f16 to f32
      linalg.yield %4 : f32
    }
    %expand_shape_18 = memref.expand_shape %alloc_17 [[0], [1, 2], [3]] output_shape [6, 2, 1, 384] : memref<6x2x384xf32> into memref<6x2x1x384xf32>
    %collapse_shape_19 = memref.collapse_shape %expand_shape_18 [[0, 1], [2], [3]] : memref<6x2x1x384xf32> into memref<12x1x384xf32>
    %expand_shape_20 = memref.expand_shape %collapse_shape_19 [[0, 1], [2], [3]] output_shape [3, 4, 1, 384] : memref<12x1x384xf32> into memref<3x4x1x384xf32>
    %collapse_shape_21 = memref.collapse_shape %expand_shape_20 [[0], [1, 2], [3]] : memref<3x4x1x384xf32> into memref<3x4x384xf32>
    %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<3x4x384xf32>
    linalg.generic {indexing_maps = [#map3, #map4, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg5, %collapse_shape_21 : memref<384xi32>, memref<3xi32>, memref<3x4x384xf32>) outs(%alloc_22 : memref<3x4x384xf32>) {
    ^bb0(%in: i32, %in_38: i32, %in_39: f32, %out: f32):
      %1 = arith.cmpi sgt, %in, %in_38 : i32
      %2 = arith.select %1, %cst, %in_39 : f32
      linalg.yield %2 : f32
    }
    %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<3x4xf32>
    linalg.fill ins(%cst_3 : f32) outs(%alloc_23 : memref<3x4xf32>)
    linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_22 : memref<3x4x384xf32>) outs(%alloc_23 : memref<3x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.maximumf %in, %out : f32
      linalg.yield %1 : f32
    }
    %collapse_shape_24 = memref.collapse_shape %alloc_22 [[0, 1], [2]] : memref<3x4x384xf32> into memref<12x384xf32>
    %collapse_shape_25 = memref.collapse_shape %alloc_23 [[0, 1]] : memref<3x4xf32> into memref<12xf32>
    %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<12x384xf32>
    linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%collapse_shape_24, %collapse_shape_25 : memref<12x384xf32>, memref<12xf32>) outs(%alloc_26 : memref<12x384xf32>) {
    ^bb0(%in: f32, %in_38: f32, %out: f32):
      %1 = arith.subf %in, %in_38 : f32
      %2 = math.exp %1 : f32
      linalg.yield %2 : f32
    }
    %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<12xf32>
    linalg.fill ins(%cst_4 : f32) outs(%alloc_27 : memref<12xf32>)
    linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "reduction"]} ins(%alloc_26 : memref<12x384xf32>) outs(%alloc_27 : memref<12xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    }
    %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<12x384xf16>
    linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%alloc_26, %alloc_27 : memref<12x384xf32>, memref<12xf32>) outs(%alloc_28 : memref<12x384xf16>) {
    ^bb0(%in: f32, %in_38: f32, %out: f16):
      %1 = arith.divf %cst_2, %in_38 : f32
      %2 = arith.mulf %in, %1 : f32
      %3 = arith.truncf %2 : f32 to f16
      linalg.yield %3 : f16
    }
    %expand_shape_29 = memref.expand_shape %alloc_28 [[0, 1], [2]] output_shape [6, 2, 384] : memref<12x384xf16> into memref<6x2x384xf16>
    %expand_shape_30 = memref.expand_shape %arg2 [[0, 1, 2]] output_shape [6, 384, 64] : memref<147456xf16> into memref<6x384x64xf16>
    %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<6x2x64xf32>
    linalg.fill ins(%cst_4 : f32) outs(%alloc_31 : memref<6x2x64xf32>)
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%expand_shape_29, %expand_shape_30 : memref<6x2x384xf16>, memref<6x384x64xf16>) outs(%alloc_31 : memref<6x2x64xf32>) {
    ^bb0(%in: f16, %in_38: f16, %out: f32):
      %1 = arith.addf %in_38, %cst_1 : f16
      %2 = arith.extf %in : f16 to f32
      %3 = arith.extf %1 : f16 to f32
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %out, %4 : f32
      linalg.yield %5 : f32
    }
    %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<6x2x64xf16>
    linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_31 : memref<6x2x64xf32>) outs(%alloc_32 : memref<6x2x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %1 = arith.truncf %in : f32 to f16
      linalg.yield %1 : f16
    }
    %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<12x4xf16>
    linalg.generic {indexing_maps = [#map8, #map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%alloc_27, %collapse_shape_25 : memref<12xf32>, memref<12xf32>) outs(%alloc_33 : memref<12x4xf16>) {
    ^bb0(%in: f32, %in_38: f32, %out: f16):
      %1 = arith.truncf %in_38 : f32 to f16
      %2 = arith.truncf %in : f32 to f16
      %3 = math.log %2 : f16
      %4 = arith.addf %3, %1 : f16
      %5 = arith.addf %4, %cst_1 : f16
      linalg.yield %5 : f16
    }
    %collapse_shape_34 = memref.collapse_shape %alloc_32 [[0, 1], [2]] : memref<6x2x64xf16> into memref<12x64xf16>
    %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<12x4x64xf16>
    linalg.generic {indexing_maps = [#map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapse_shape_34 : memref<12x64xf16>) outs(%alloc_35 : memref<12x4x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %1 = arith.addf %in, %cst_1 : f16
      linalg.yield %1 : f16
    }
    %collapse_shape_36 = memref.collapse_shape %alloc_35 [[0, 1, 2]] : memref<12x4x64xf16> into memref<3072xf16>
    memref.copy %collapse_shape_36, %arg7 : memref<3072xf16> to memref<3072xf16>
    %collapse_shape_37 = memref.collapse_shape %alloc_33 [[0, 1]] : memref<12x4xf16> into memref<48xf16>
    memref.copy %collapse_shape_37, %arg6 : memref<48xf16> to memref<48xf16>
    return
  }
  func.func @rock_attention_verify7(%arg0: memref<3072xf16>, %arg1: memref<3072xf16>) {
    %false = arith.constant false
    %cst = arith.constant 1.000000e+02 : f32
    %cst_0 = arith.constant 1.000000e-03 : f32
    %c1_i8 = arith.constant 1 : i8
    %alloc = memref.alloc() : memref<3072xf32>
    call @_memcpy_f16_f32_3072(%arg0, %alloc) : (memref<3072xf16>, memref<3072xf32>) -> ()
    %cast = memref.cast %alloc : memref<3072xf32> to memref<?xf32>
    %alloc_1 = memref.alloc() : memref<3072xf32>
    call @_memcpy_f16_f32_3072(%arg1, %alloc_1) : (memref<3072xf16>, memref<3072xf32>) -> ()
    %cast_2 = memref.cast %alloc_1 : memref<3072xf32> to memref<?xf32>
    call @mcpuVerifyFloat(%cast, %cast_2, %cst_0, %cst, %cst, %c1_i8, %false) : (memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1) -> ()
    memref.dealloc %alloc : memref<3072xf32>
    memref.dealloc %alloc_1 : memref<3072xf32>
    return
  }
  func.func @_memcpy_f16_f32_3072(%arg0: memref<3072xf16>, %arg1: memref<3072xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3072 = arith.constant 3072 : index
    scf.for %arg2 = %c0 to %c3072 step %c1 {
      %0 = memref.load %arg0[%arg2] : memref<3072xf16>
      %1 = arith.extf %0 : f16 to f32
      memref.store %1, %arg1[%arg2] : memref<3072xf32>
    }
    return
  }
  func.func private @mcpuVerifyFloat(memref<?xf32>, memref<?xf32>, f32, f32, f32, i8, i1)
  func.func @rock_attention_gpu(%arg0: memref<768xf16>, %arg1: memref<147456xf16>, %arg2: memref<147456xf16>, %arg3: memref<4608xf16>, %arg4: memref<4608xf16>, %arg5: memref<3xi32>, %arg6: memref<48xf16>, %arg7: memref<3072xf16>) {
    %0 = memref.get_global @__constant_12xi32 : memref<12xi32>
    %1 = memref.get_global @__constant_4xi32 : memref<4xi32>
    %cst = arith.constant 0xFC00 : f16
    %cst_0 = arith.constant 1.000000e+00 : f16
    %cst_1 = arith.constant 0.000000e+00 : f16
    %cst_2 = arith.constant -6.550400e+04 : f16
    %memref = gpu.alloc  () : memref<768xf16>
    gpu.memcpy  %memref, %arg0 : memref<768xf16>, memref<768xf16>
    %memref_3 = gpu.alloc  () : memref<147456xf16>
    gpu.memcpy  %memref_3, %arg1 : memref<147456xf16>, memref<147456xf16>
    %memref_4 = gpu.alloc  () : memref<147456xf16>
    gpu.memcpy  %memref_4, %arg2 : memref<147456xf16>, memref<147456xf16>
    %memref_5 = gpu.alloc  () : memref<4608xf16>
    gpu.memcpy  %memref_5, %arg3 : memref<4608xf16>, memref<4608xf16>
    %memref_6 = gpu.alloc  () : memref<4608xf16>
    gpu.memcpy  %memref_6, %arg4 : memref<4608xf16>, memref<4608xf16>
    %memref_7 = gpu.alloc  () : memref<3xi32>
    gpu.memcpy  %memref_7, %arg5 : memref<3xi32>, memref<3xi32>
    %memref_8 = gpu.alloc  () : memref<48xf16>
    gpu.memcpy  %memref_8, %arg6 : memref<48xf16>, memref<48xf16>
    %memref_9 = gpu.alloc  () : memref<3072xf16>
    gpu.memcpy  %memref_9, %arg7 : memref<3072xf16>, memref<3072xf16>
    call @rock_attention(%memref, %memref_3, %memref_4, %memref_5, %memref_6, %memref_7, %memref_8, %memref_9) : (memref<768xf16>, memref<147456xf16>, memref<147456xf16>, memref<4608xf16>, memref<4608xf16>, memref<3xi32>, memref<48xf16>, memref<3072xf16>) -> ()
    gpu.memcpy  %arg0, %memref : memref<768xf16>, memref<768xf16>
    gpu.dealloc  %memref : memref<768xf16>
    gpu.memcpy  %arg1, %memref_3 : memref<147456xf16>, memref<147456xf16>
    gpu.dealloc  %memref_3 : memref<147456xf16>
    gpu.memcpy  %arg2, %memref_4 : memref<147456xf16>, memref<147456xf16>
    gpu.dealloc  %memref_4 : memref<147456xf16>
    gpu.memcpy  %arg3, %memref_5 : memref<4608xf16>, memref<4608xf16>
    gpu.dealloc  %memref_5 : memref<4608xf16>
    gpu.memcpy  %arg4, %memref_6 : memref<4608xf16>, memref<4608xf16>
    gpu.dealloc  %memref_6 : memref<4608xf16>
    gpu.memcpy  %arg5, %memref_7 : memref<3xi32>, memref<3xi32>
    gpu.dealloc  %memref_7 : memref<3xi32>
    gpu.memcpy  %arg6, %memref_8 : memref<48xf16>, memref<48xf16>
    gpu.dealloc  %memref_8 : memref<48xf16>
    gpu.memcpy  %arg7, %memref_9 : memref<3072xf16>, memref<3072xf16>
    gpu.dealloc  %memref_9 : memref<3072xf16>
    %expand_shape = memref.expand_shape %arg6 [[0, 1]] output_shape [12, 4] : memref<48xf16> into memref<12x4xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<12x4xf16>
    linalg.generic {indexing_maps = [#map10, #map8, #map7, #map7], iterator_types = ["parallel", "parallel"]} ins(%1, %0, %expand_shape : memref<4xi32>, memref<12xi32>, memref<12x4xf16>) outs(%alloc : memref<12x4xf16>) {
    ^bb0(%in: i32, %in_18: i32, %in_19: f16, %out: f16):
      %2 = arith.cmpi sge, %in, %in_18 : i32
      %3 = arith.select %2, %cst, %in_19 : f16
      linalg.yield %3 : f16
    }
    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<12xf16>
    linalg.fill ins(%cst_2 : f16) outs(%alloc_10 : memref<12xf16>)
    linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "reduction"]} ins(%alloc : memref<12x4xf16>) outs(%alloc_10 : memref<12xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.maximumf %in, %out : f16
      linalg.yield %2 : f16
    }
    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<12x4xf16>
    linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%alloc, %alloc_10 : memref<12x4xf16>, memref<12xf16>) outs(%alloc_11 : memref<12x4xf16>) {
    ^bb0(%in: f16, %in_18: f16, %out: f16):
      %2 = arith.subf %in, %in_18 : f16
      %3 = math.exp %2 : f16
      linalg.yield %3 : f16
    }
    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<12xf16>
    linalg.fill ins(%cst_1 : f16) outs(%alloc_12 : memref<12xf16>)
    linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "reduction"]} ins(%alloc_11 : memref<12x4xf16>) outs(%alloc_12 : memref<12xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.addf %in, %out : f16
      linalg.yield %2 : f16
    }
    %expand_shape_13 = memref.expand_shape %arg7 [[0, 1, 2]] output_shape [12, 4, 64] : memref<3072xf16> into memref<12x4x64xf16>
    %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<12x4x64xf16>
    linalg.generic {indexing_maps = [#map11, #map4, #map6, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1, %0, %alloc_11, %expand_shape_13 : memref<4xi32>, memref<12xi32>, memref<12x4xf16>, memref<12x4x64xf16>) outs(%alloc_14 : memref<12x4x64xf16>) {
    ^bb0(%in: i32, %in_18: i32, %in_19: f16, %in_20: f16, %out: f16):
      %2 = arith.cmpi sge, %in, %in_18 : i32
      %3 = arith.select %2, %cst, %in_20 : f16
      %4 = arith.mulf %in_19, %3 : f16
      %5 = arith.select %2, %cst_1, %4 : f16
      linalg.yield %5 : f16
    }
    %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<12x64xf16>
    linalg.fill ins(%cst_1 : f16) outs(%alloc_15 : memref<12x64xf16>)
    linalg.generic {indexing_maps = [#map5, #map9], iterator_types = ["parallel", "reduction", "parallel"]} ins(%alloc_14 : memref<12x4x64xf16>) outs(%alloc_15 : memref<12x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.addf %in, %out : f16
      linalg.yield %2 : f16
    }
    %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<12x64xf16>
    linalg.generic {indexing_maps = [#map7, #map8, #map7], iterator_types = ["parallel", "parallel"]} ins(%alloc_15, %alloc_12 : memref<12x64xf16>, memref<12xf16>) outs(%alloc_16 : memref<12x64xf16>) {
    ^bb0(%in: f16, %in_18: f16, %out: f16):
      %2 = arith.divf %cst_0, %in_18 : f16
      %3 = arith.mulf %in, %2 : f16
      linalg.yield %3 : f16
    }
    %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<12x4x64xf16>
    linalg.generic {indexing_maps = [#map9, #map5], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_16 : memref<12x64xf16>) outs(%alloc_17 : memref<12x4x64xf16>) {
    ^bb0(%in: f16, %out: f16):
      %2 = arith.addf %in, %cst_1 : f16
      linalg.yield %2 : f16
    }
    %collapse_shape = memref.collapse_shape %alloc_17 [[0, 1, 2]] : memref<12x4x64xf16> into memref<3072xf16>
    memref.copy %collapse_shape, %arg7 : memref<3072xf16> to memref<3072xf16>
    return
  }
}

