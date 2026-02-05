// RUN: rocmlir-opt -rock-gridwise-gemm-to-blockwise %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0 * 64 + d2)>
#map1 = affine_map<(d0, d1, d2) -> ((d0 * 64 + d1) * 384 + d2)>
#map2 = affine_map<(d0, d1, d2) -> ((d0 * 384 + d1) * 64 + d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0 * 384 + d2)>
#map4 = affine_map<(d0) -> (d0)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d0, 0)>
#map7 = affine_map<(d0) -> (d0 floordiv 4, d0 mod 4)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0 * 2 + d3, d1, d2)>
#map10 = affine_map<(d0, d1, d2) -> (d0, d1, 0, d2)>
#map11 = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map12 = affine_map<(d0, d1) -> (d0, d1)>
#map13 = affine_map<(d0) -> (d0, 0)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> ((d0 * 2 + d3) * 4 + d1, d2, d4)>
#map15 = affine_map<(d0, d1, d2) -> (d0 floordiv 4, d0 mod 4, 0, d1, d2)>
#map16 = affine_map<(d0, d1, d2, d3) -> ((d0 * 2 + d3) * 4 + d1, d2)>
#map17 = affine_map<(d0, d1) -> (d0 floordiv 4, d0 mod 4, 0, d1)>
#map18 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map19 = affine_map<(d0, d1) -> (d0, 0, d1)>
#transform_map = #rock.transform_map<#map by [<Unmerge{12, 64} ["g", "head_qk"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [12, 1, 64] -> [768]>
#transform_map1 = #rock.transform_map<#map1 by [<Unmerge{6, 64, 384} ["g", "head_qk", "seq_k"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [6, 64, 384] -> [147456]>
#transform_map2 = #rock.transform_map<#map2 by [<Unmerge{6, 384, 64} ["g", "seq_k", "head_v"] at [0, 1, 2] -> ["raw"] at [0]>] bounds = [6, 384, 64] -> [147456]>
#transform_map3 = #rock.transform_map<#map3 by [<Unmerge{12, 384} ["g", "seq_k"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [12, 1, 384] -> [4608]>
#transform_map4 = #rock.transform_map<#map4 by [<Unmerge{3} ["g"] at [0] -> ["raw"] at [0]>] bounds = [3] -> [3]>
#transform_map5 = #rock.transform_map<#map5 by [<Unmerge{48} ["g"] at [0] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [48, 1] -> [48]>
#transform_map6 = #rock.transform_map<#map by [<Unmerge{48, 64} ["g", "head_v"] at [0, 2] -> ["raw"] at [0]>, <AddDim{1} ["seq_q"] at [1] -> [] at []>] bounds = [48, 1, 64] -> [3072]>
#transform_map7 = #rock.transform_map<#map5 by [<AddDim{1} ["numHeadsQ"] at [1] -> [] at []>, <PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>] bounds = [3, 1] -> [3]>
#transform_map8 = #rock.transform_map<#map6 by [<Broadcast{1} ["numHeadsQ"] at [1] -> ["numHeadsQ"] at [1]>, <PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>] bounds = [3, 4] -> [3, 1]>
#transform_map9 = #rock.transform_map<#map7 by [<Merge{3, 4} ["gemmG"] at [0] -> ["gemmG", "numHeadsQ"] at [0, 1]>] bounds = [12] -> [3, 4]>
#transform_map10 = #rock.transform_map<#map8 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K", "gemm0M"] at [1, 2] -> ["gemm0K", "gemm0M"] at [2, 1]>] bounds = [12, 64, 1] -> [12, 1, 64]>
#transform_map11 = #rock.transform_map<#map9 by [<Unmerge{6, 2} ["gemmG", "numRepeats"] at [0, 3] -> ["gemmG"] at [0]>, <PassThrough ["seqLen", "headDim"] at [2, 1] -> ["seqLen", "headDim"] at [2, 1]>] bounds = [6, 64, 1, 2] -> [12, 64, 1]>
#transform_map12 = #rock.transform_map<#map10 by [<Merge{1, 2} ["seqLen"] at [2] -> ["seqLen", "numRepeats"] at [2, 3]>, <PassThrough ["gemmG", "headDim"] at [0, 1] -> ["gemmG", "headDim"] at [0, 1]>] bounds = [6, 64, 2] -> [6, 64, 1, 2]>
#transform_map13 = #rock.transform_map<#map11 by [<Unmerge{6, 2} ["gemmG", "numRepeats"] at [0, 1] -> ["gemmG"] at [0]>] bounds = [6, 2] -> [12]>
#transform_map14 = #rock.transform_map<#map12 by [<Slice{0, 1} ["numRepeats"] at [1] -> ["numRepeats"] at [1]>, <PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>] bounds = [6, 1] -> [6, 2]>
#transform_map15 = #rock.transform_map<#map13 by [<Merge{6, 1} ["seqLen"] at [0] -> ["gemmG", "numRepeats"] at [0, 1]>] bounds = [6] -> [6, 1]>
#transform_map16 = #rock.transform_map<#map14 by [<Unmerge{6, 2, 4} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>, <PassThrough ["seqLen", "headDim"] at [2, 4] -> ["seqLen", "headDim"] at [1, 2]>] bounds = [6, 4, 1, 2, 64] -> [48, 1, 64]>
#transform_map17 = #rock.transform_map<#map15 by [<Merge{1, 2} ["seqLen"] at [1] -> ["seqLen", "numRepeats"] at [2, 3]>, <Merge{6, 4} ["gemmG"] at [0] -> ["gemmG", "splitKV"] at [0, 1]>, <PassThrough ["headDim"] at [2] -> ["headDim"] at [4]>] bounds = [24, 2, 64] -> [6, 4, 1, 2, 64]>
#transform_map18 = #rock.transform_map<#map16 by [<Unmerge{6, 2, 4} ["gemmG", "numRepeats", "splitKV"] at [0, 3, 1] -> ["gemmG"] at [0]>, <PassThrough ["seqLen"] at [2] -> ["seqLen"] at [1]>] bounds = [6, 4, 1, 2] -> [48, 1]>
#transform_map19 = #rock.transform_map<#map17 by [<Merge{1, 2} ["seqLen"] at [1] -> ["seqLen", "numRepeats"] at [2, 3]>, <Merge{6, 4} ["gemmG"] at [0] -> ["gemmG", "splitKV"] at [0, 1]>] bounds = [24, 2] -> [6, 4, 1, 2]>
#transform_map20 = #rock.transform_map<#map18 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <PassThrough ["gemm0K"] at [1] -> ["gemm0K"] at [1]>, <Pad{0, 30} ["gemm0NPad"] at [2] -> ["gemm0N"] at [2]>] bounds = [6, 64, 32] -> [6, 64, 2]>
#transform_map21 = #rock.transform_map<#map18 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 30} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>, <PassThrough ["gemm1M"] at [2] -> ["gemm1M"] at [2]>] bounds = [24, 32, 64] -> [24, 2, 64]>
#transform_map22 = #rock.transform_map<#map12 by [<PassThrough ["gemmG"] at [0] -> ["gemmG"] at [0]>, <Pad{0, 30} ["gemm1NPad"] at [1] -> ["gemm1N"] at [1]>] bounds = [24, 32] -> [24, 2]>
#transform_map23 = #rock.transform_map<#map19 by [<Merge{12, 1} ["dim0"] at [0] -> ["col0", "col1"] at [0, 1]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>] bounds = [12, 384] -> [12, 1, 384]>
#transform_map24 = #rock.transform_map<#map19 by [<Merge{12} ["dim0"] at [0] -> ["exp0"] at [0]>, <PassThrough ["dim1"] at [1] -> ["dim1"] at [2]>, <ConstDim{0, 1} [] at [] -> ["unit1"] at [1]>] bounds = [12, 384] -> [12, 1, 384]>
module attributes {mhal.arch = "amdgcn-amd-amdhsa:gfx1201"} {
  memref.global "private" constant @__constant_4xi32 : memref<4xi32> = dense<[0, 1, 2, 3]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_12xi32 : memref<12xi32> = dense<[1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]> {alignment = 64 : i64}
  memref.global "private" constant @__constant_384xi32 : memref<384xi32> = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000"> {alignment = 64 : i64}
  // CHECK-LABEL: func.func @rock_attention
  func.func @rock_attention(%arg0: memref<768xf16>, %arg1: memref<147456xf16>, %arg2: memref<147456xf16>, %arg3: memref<4608xf16>, %arg4: memref<4608xf16>, %arg5: memref<3xi32>, %arg6: memref<48xf16>, %arg7: memref<3072xf16>) attributes {block_size = 32 : i32, features = #rock<GemmFeatures wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32>, grid_size = 24 : i32, kernel, mhal.arch = "amdgcn-amd-amdhsa:gfx1201"} {
    %0 = rock.transform %arg0 by #transform_map : memref<768xf16> to memref<12x1x64xf16>
    %1 = rock.transform %arg1 by #transform_map1 : memref<147456xf16> to memref<6x64x384xf16>
    %2 = rock.transform %arg2 by #transform_map2 : memref<147456xf16> to memref<6x384x64xf16>
    %3 = rock.transform %arg3 by #transform_map3 : memref<4608xf16> to memref<12x1x384xf16>
    %4 = rock.transform %arg4 by #transform_map3 : memref<4608xf16> to memref<12x1x384xf16>
    %5 = rock.transform %arg5 by #transform_map4 : memref<3xi32> to memref<3xi32>
    %6 = rock.transform %arg6 by #transform_map5 : memref<48xf16> to memref<48x1xf16>
    %7 = rock.transform %arg7 by #transform_map6 : memref<3072xf16> to memref<48x1x64xf16>
    %8 = rock.transform %5 by #transform_map7 : memref<3xi32> to memref<3x1xi32>
    %9 = rock.transform %8 by #transform_map8 : memref<3x1xi32> to memref<3x4xi32>
    %10 = rock.transform %9 by #transform_map9 : memref<3x4xi32> to memref<12xi32>
    %11 = rock.transform %0 by #transform_map10 : memref<12x1x64xf16> to memref<12x64x1xf16>
    %12 = rock.transform %11 by #transform_map11 : memref<12x64x1xf16> to memref<6x64x1x2xf16>
    %13 = rock.transform %12 by #transform_map12 : memref<6x64x1x2xf16> to memref<6x64x2xf16>
    %14 = rock.transform %10 by #transform_map13 : memref<12xi32> to memref<6x2xi32>
    %15 = rock.transform %14 by #transform_map14 : memref<6x2xi32> to memref<6x1xi32>
    %16 = rock.transform %15 by #transform_map15 : memref<6x1xi32> to memref<6xi32>
    %17 = rock.transform %7 by #transform_map16 : memref<48x1x64xf16> to memref<6x4x1x2x64xf16>
    %18 = rock.transform %17 by #transform_map17 : memref<6x4x1x2x64xf16> to memref<24x2x64xf16>
    %19 = rock.transform %6 by #transform_map18 : memref<48x1xf16> to memref<6x4x1x2xf16>
    %20 = rock.transform %19 by #transform_map19 : memref<6x4x1x2xf16> to memref<24x2xf16>
    %21 = rock.transform %13 by #transform_map20 : memref<6x64x2xf16> to memref<6x64x32xf16>
    %22 = rock.transform %18 by #transform_map21 : memref<24x2x64xf16> to memref<24x32x64xf16>
    %23 = rock.transform %20 by #transform_map22 : memref<24x2xf16> to memref<24x32xf16>
    rock.gridwise_attention_accel(%21, %1, %2, %3, %4, %16, %22, %23) features =  wmma|dot|atomic_add|atomic_add_bf16|atomic_add_f16|atomic_fmax_f32 preSoftmaxOps = {
    ^bb0(%arg8: memref<12x1x384xf16>, %arg9: memref<12x1x384xf16>, %arg10: memref<12x1x384xf16>, %arg11: memref<12x1x384xf16>):
      %24 = rock.transform %arg8 by #transform_map23 : memref<12x1x384xf16> to memref<12x384xf16>
      %25 = rock.transform %arg9 by #transform_map23 : memref<12x1x384xf16> to memref<12x384xf16>
      %26 = rock.transform %arg10 by #transform_map23 : memref<12x1x384xf16> to memref<12x384xf16>
      %alloc = memref.alloc() : memref<12x1x384xf16>
      %27 = rock.transform %alloc by #transform_map24 : memref<12x1x384xf16> to memref<12x384xf16>
      linalg.generic {indexing_maps = [#map12, #map12, #map12, #map12], iterator_types = ["parallel", "parallel"]} ins(%24, %25, %26 : memref<12x384xf16>, memref<12x384xf16>, memref<12x384xf16>) outs(%27 : memref<12x384xf16>) attrs =  {rock.majorTensorNumber = 0 : index} {
      ^bb0(%in: f16, %in_0: f16, %in_1: f16, %out: f16):
        %28 = arith.mulf %in, %in_0 : f16
        %29 = arith.addf %28, %in_1 : f16
        linalg.yield %29 : f16
      }
      memref.copy %alloc, %arg11 : memref<12x1x384xf16> to memref<12x1x384xf16>
      rock.yield
    } {blockSize = 32 : i32, firstGemmIndices = array<i64: 0>, gridSize = 24 : i32, numRepeatsGQA = 2 : index, operandSegmentSizes = array<i32: 1, 1, 1, 2, 1, 0, 1, 1>, params0 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, params1 = #rock.accel_gemm_params<kpackPerBlock = 32, mPerBlock = 32, nPerBlock = 32, kpack = 1, mPerWave = 32, nPerWave = 32, mnPerXdl = 16, splitKFactor = 1, scheduleVersion = 1, outputSwizzle = 2, wavesPerEU = 0, gridGroupSize = 0, forceUnroll = true>, prePadG0N = 2 : index, softmaxType = f32, splitKV = 4 : i32, storeMethod = #rock<StoreMethod set>} : memref<6x64x32xf16>, memref<6x64x384xf16>, memref<6x384x64xf16>, memref<12x1x384xf16>, memref<12x1x384xf16>, memref<6xi32>, memref<24x32x64xf16>, memref<24x32xf16>
    // CHECK: scf.for
    // CHECK: rock.blockwise_load_tile
    // CHECK: rock.blockwise_load_tile
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: rock.stage
    // CHECK: rock.blockwise_gemm_accel
    // CHECK: } {name = "MMA"}
    // CHECK-NEXT: rock.lds_barrier
    // CHECK-NEXT: } {pipeline = #rock.pipeline<2>}
    return
  }
}
