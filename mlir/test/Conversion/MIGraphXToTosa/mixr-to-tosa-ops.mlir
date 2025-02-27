// RUN: rocmlir-opt -split-input-file --migraphx-transform --canonicalize --migraphx-to-tosa %s -verify-diagnostics -o -| FileCheck %s

module  {
  // CHECK-LABEL: func @literal_zero
  // CHECK: %[[const:.+]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x3x7x7xf16>}> : () -> tensor<64x3x7x7xf16>
  // CHECK-NEXT: %[[reshape:.+]] = tosa.reshape %[[const]] {new_shape = array<i64: 9408>} : (tensor<64x3x7x7xf16>) -> tensor<9408xf16>
  // CHECK-NEXT: return %[[reshape]] : tensor<9408xf16>
  func.func @literal_zero() -> !migraphx.shaped<64x3x7x7xf16, 147x49x7x1> {
    %0 = migraphx.literal (dense<0.0> : tensor<64x1xf16>) : <64x3x7x7xf16, 147x49x7x1>
    return %0 : !migraphx.shaped<64x3x7x7xf16, 147x49x7x1>
  }

  // CHECK-LABEL: func @literal_dense_list
  // CHECK: %[[const:.+]] = "tosa.const"() <{value = dense<{{.*}}> : tensor<1024xi32>}> : () -> tensor<1024xi32>
  // CHECK-NEXT: return %[[const]] : tensor<1024xi32>
  func.func @literal_dense_list() -> !migraphx.shaped<1024xsi32, 1> {
    %0 = migraphx.literal(dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B0000008C0000008D0000008E0000008F000000900000009100000092000000930000009400000095000000960000009700000098000000990000009A0000009B0000009C0000009D0000009E0000009F000000A0000000A1000000A2000000A3000000A4000000A5000000A6000000A7000000A8000000A9000000AA000000AB000000AC000000AD000000AE000000AF000000B0000000B1000000B2000000B3000000B4000000B5000000B6000000B7000000B8000000B9000000BA000000BB000000BC000000BD000000BE000000BF000000C0000000C1000000C2000000C3000000C4000000C5000000C6000000C7000000C8000000C9000000CA000000CB000000CC000000CD000000CE000000CF000000D0000000D1000000D2000000D3000000D4000000D5000000D6000000D7000000D8000000D9000000DA000000DB000000DC000000DD000000DE000000DF000000E0000000E1000000E2000000E3000000E4000000E5000000E6000000E7000000E8000000E9000000EA000000EB000000EC000000ED000000EE000000EF000000F0000000F1000000F2000000F3000000F4000000F5000000F6000000F7000000F8000000F9000000FA000000FB000000FC000000FD000000FE000000FF000000000100000101000002010000030100000401000005010000060100000701000008010000090100000A0100000B0100000C0100000D0100000E0100000F010000100100001101000012010000130100001401000015010000160100001701000018010000190100001A0100001B0100001C0100001D0100001E0100001F010000200100002101000022010000230100002401000025010000260100002701000028010000290100002A0100002B0100002C0100002D0100002E0100002F010000300100003101000032010000330100003401000035010000360100003701000038010000390100003A0100003B0100003C0100003D0100003E0100003F010000400100004101000042010000430100004401000045010000460100004701000048010000490100004A0100004B0100004C0100004D0100004E0100004F010000500100005101000052010000530100005401000055010000560100005701000058010000590100005A0100005B0100005C0100005D0100005E0100005F010000600100006101000062010000630100006401000065010000660100006701000068010000690100006A0100006B0100006C0100006D0100006E0100006F010000700100007101000072010000730100007401000075010000760100007701000078010000790100007A0100007B0100007C0100007D0100007E0100007F010000800100008101000082010000830100008401000085010000860100008701000088010000890100008A0100008B0100008C0100008D0100008E0100008F010000900100009101000092010000930100009401000095010000960100009701000098010000990100009A0100009B0100009C0100009D0100009E0100009F010000A0010000A1010000A2010000A3010000A4010000A5010000A6010000A7010000A8010000A9010000AA010000AB010000AC010000AD010000AE010000AF010000B0010000B1010000B2010000B3010000B4010000B5010000B6010000B7010000B8010000B9010000BA010000BB010000BC010000BD010000BE010000BF010000C0010000C1010000C2010000C3010000C4010000C5010000C6010000C7010000C8010000C9010000CA010000CB010000CC010000CD010000CE010000CF010000D0010000D1010000D2010000D3010000D4010000D5010000D6010000D7010000D8010000D9010000DA010000DB010000DC010000DD010000DE010000DF010000E0010000E1010000E2010000E3010000E4010000E5010000E6010000E7010000E8010000E9010000EA010000EB010000EC010000ED010000EE010000EF010000F0010000F1010000F2010000F3010000F4010000F5010000F6010000F7010000F8010000F9010000FA010000FB010000FC010000FD010000FE010000FF010000000200000102000002020000030200000402000005020000060200000702000008020000090200000A0200000B0200000C0200000D0200000E0200000F020000100200001102000012020000130200001402000015020000160200001702000018020000190200001A0200001B0200001C0200001D0200001E0200001F020000200200002102000022020000230200002402000025020000260200002702000028020000290200002A0200002B0200002C0200002D0200002E0200002F020000300200003102000032020000330200003402000035020000360200003702000038020000390200003A0200003B0200003C0200003D0200003E0200003F020000400200004102000042020000430200004402000045020000460200004702000048020000490200004A0200004B0200004C0200004D0200004E0200004F020000500200005102000052020000530200005402000055020000560200005702000058020000590200005A0200005B0200005C0200005D0200005E0200005F020000600200006102000062020000630200006402000065020000660200006702000068020000690200006A0200006B0200006C0200006D0200006E0200006F020000700200007102000072020000730200007402000075020000760200007702000078020000790200007A0200007B0200007C0200007D0200007E0200007F020000800200008102000082020000830200008402000085020000860200008702000088020000890200008A0200008B0200008C0200008D0200008E0200008F020000900200009102000092020000930200009402000095020000960200009702000098020000990200009A0200009B0200009C0200009D0200009E0200009F020000A0020000A1020000A2020000A3020000A4020000A5020000A6020000A7020000A8020000A9020000AA020000AB020000AC020000AD020000AE020000AF020000B0020000B1020000B2020000B3020000B4020000B5020000B6020000B7020000B8020000B9020000BA020000BB020000BC020000BD020000BE020000BF020000C0020000C1020000C2020000C3020000C4020000C5020000C6020000C7020000C8020000C9020000CA020000CB020000CC020000CD020000CE020000CF020000D0020000D1020000D2020000D3020000D4020000D5020000D6020000D7020000D8020000D9020000DA020000DB020000DC020000DD020000DE020000DF020000E0020000E1020000E2020000E3020000E4020000E5020000E6020000E7020000E8020000E9020000EA020000EB020000EC020000ED020000EE020000EF020000F0020000F1020000F2020000F3020000F4020000F5020000F6020000F7020000F8020000F9020000FA020000FB020000FC020000FD020000FE020000FF020000000300000103000002030000030300000403000005030000060300000703000008030000090300000A0300000B0300000C0300000D0300000E0300000F030000100300001103000012030000130300001403000015030000160300001703000018030000190300001A0300001B0300001C0300001D0300001E0300001F030000200300002103000022030000230300002403000025030000260300002703000028030000290300002A0300002B0300002C0300002D0300002E0300002F030000300300003103000032030000330300003403000035030000360300003703000038030000390300003A0300003B0300003C0300003D0300003E0300003F030000400300004103000042030000430300004403000045030000460300004703000048030000490300004A0300004B0300004C0300004D0300004E0300004F030000500300005103000052030000530300005403000055030000560300005703000058030000590300005A0300005B0300005C0300005D0300005E0300005F030000600300006103000062030000630300006403000065030000660300006703000068030000690300006A0300006B0300006C0300006D0300006E0300006F030000700300007103000072030000730300007403000075030000760300007703000078030000790300007A0300007B0300007C0300007D0300007E0300007F030000800300008103000082030000830300008403000085030000860300008703000088030000890300008A0300008B0300008C0300008D0300008E0300008F030000900300009103000092030000930300009403000095030000960300009703000098030000990300009A0300009B0300009C0300009D0300009E0300009F030000A0030000A1030000A2030000A3030000A4030000A5030000A6030000A7030000A8030000A9030000AA030000AB030000AC030000AD030000AE030000AF030000B0030000B1030000B2030000B3030000B4030000B5030000B6030000B7030000B8030000B9030000BA030000BB030000BC030000BD030000BE030000BF030000C0030000C1030000C2030000C3030000C4030000C5030000C6030000C7030000C8030000C9030000CA030000CB030000CC030000CD030000CE030000CF030000D0030000D1030000D2030000D3030000D4030000D5030000D6030000D7030000D8030000D9030000DA030000DB030000DC030000DD030000DE030000DF030000E0030000E1030000E2030000E3030000E4030000E5030000E6030000E7030000E8030000E9030000EA030000EB030000EC030000ED030000EE030000EF030000F0030000F1030000F2030000F3030000F4030000F5030000F6030000F7030000F8030000F9030000FA030000FB030000FC030000FD030000FE030000FF030000"> : tensor<1024xsi32>) : <1024xsi32, 1>
    return %0 : !migraphx.shaped<1024xsi32, 1>
  }

  // CHECK-LABEL: func @dequantize_scale
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.cast
  // CHECK: tosa.mul
  func.func @dequantize_scale(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale : <1x112x112x64xi32, 802816x7168x64x1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_f32_scale
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_f32_scale(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_scale_f16
  // CHECK-NOT: tosa.sub
  // CHECK: tosa.cast{{.*}}f16
  // CHECK: tosa.mul
  func.func @dequantize_scale_f16(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>) -> !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 =  migraphx.dequantizelinear %arg, %scale : <1x112x112x64xi32, 802816x7168x64x1>, <64xf16, 1> -> <1x112x112x64xf16, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_scale_bias
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  func.func @dequantize_scale_bias(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias(%arg: !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi8, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_input
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_input(%arg: !migraphx.shaped<1x112x112x64xi32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xi32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias_fp8
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias_fp8(%arg: !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @dequantize_wide_bias_fp8_ocp
  // CHECK: tosa.cast{{.*}}f32
  // CHECK: tosa.sub{{.*}}f32
  // CHECK: tosa.mul
  func.func @dequantize_wide_bias_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.dequantizelinear %arg, %scale, %bias : <1x112x112x64xf8E4M3FN, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf32, 1> -> <1x112x112x64xf32, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i8
  // CHECK-NOT: tosa.add
  func.func @quantize_scale(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_fp8
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ
  // CHECK-NOT: tosa.add
  func.func @quantize_scale_fp8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_fp8_ocp
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FN
  // CHECK-NOT: tosa.add
  func.func @quantize_scale_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1> -> <1x112x112x64xf8E4M3FN, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f32{{.*}}i32
  // CHECK: tosa.cast{{.*}}i8{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_int = 127
  // CHECK-SAME: min_int = -128
  // CHECK: tosa.cast{{.*}}i8
  func.func @quantize_scale_bias(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_fp8
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ{{.*}}f32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_fp = 2.400000e+02
  // CHECK-SAME: min_fp = -2.400000e+02
  // CHECK: tosa.cast{{.*}}f8E4M3FNUZ
  func.func @quantize_scale_bias_fp8(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf8E4M3FNUZ, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf8E4M3FNUZ, 1> -> <1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FNUZ, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_fp8_ocp
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f8E4M3FN{{.*}}f32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK-SAME: max_fp = 4.480000e+02
  // CHECK-SAME: min_fp = -4.480000e+02
  // CHECK: tosa.cast{{.*}}f8E4M3FN
  func.func @quantize_scale_bias_fp8_ocp(%arg: !migraphx.shaped<1x112x112x64xf32, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf32, 1>, %bias: !migraphx.shaped<64xf8E4M3FN, 1>) -> !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf32, 802816x7168x64x1>, <64xf32, 1>, !migraphx.shaped<64xf8E4M3FN, 1> -> <1x112x112x64xf8E4M3FN, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xf8E4M3FN, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_bias_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}f16{{.*}}i32
  // CHECK: tosa.cast{{.*}}i8{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @quantize_scale_bias_f16(%arg: !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>, %bias: !migraphx.shaped<64xi8, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf16, 802816x7168x64x1>, <64xf16, 1>, !migraphx.shaped<64xi8, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @quantize_scale_i32_bias_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast{{.*}}i32
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @quantize_scale_i32_bias_f16(%arg: !migraphx.shaped<1x112x112x64xf16, 802816x7168x64x1>, %scale: !migraphx.shaped<64xf16, 1>, %bias: !migraphx.shaped<64xi32, 1>) -> !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quantizelinear %arg, %scale, %bias : <1x112x112x64xf16, 802816x7168x64x1>, <64xf16, 1>, !migraphx.shaped<64xi32, 1> -> <1x112x112x64xi8, 802816x7168x64x1>
    return %1 : !migraphx.shaped<1x112x112x64xi8, 802816x7168x64x1>
  }

  // CHECK-LABEL: func @conv_with_quant
  // CHECK: tosa.conv2d{{.*}} quantization_info
  // CHECK: tosa.cast
  // CHECK: tosa.cast
  // CHECK: tosa.sub
  // CHECK: tosa.mul
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  // CHECK: tosa.cast
  // CHECK: tosa.cast
  // CHECK: tosa.add
  // CHECK: tosa.clamp
  // CHECK: tosa.cast
  func.func @conv_with_quant(%arg1: !migraphx.shaped<1x3x224x224xi8, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xi8, 147x49x7x1>, %scale: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %bias: !migraphx.shaped<1x64x1x1xi32, 64x1x1x1>, %bias2: !migraphx.shaped<1x64x1x1xi8, 64x1x1x1>) -> !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1> attributes {kernel = "mixr"} {
    %1 = migraphx.quant_convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xi8, 150528x50176x224x1>, <64x3x7x7xi8, 147x49x7x1> -> <1x64x112x112xi32, 802816x12544x112x1>
    %2 = migraphx.dequantizelinear %1, %scale, %bias : <1x64x112x112xi32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xi32, 64x1x1x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %3 = migraphx.quantizelinear %2, %scale, %bias2 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x1x1xf32, 64x1x1x1>, !migraphx.shaped<1x64x1x1xi8, 64x1x1x1> -> <1x64x112x112xi8, 802816x12544x112x1>
    return %3 : !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>
  }

  // CHECK-LABEL: func.func @matmul
  // CHECK: tosa.matmul
  // CHECK-SAME: (tensor<2x256x384xf32>, tensor<2x384x768xf32>) -> tensor<2x256x768xf32>
  func.func @matmul(%arg0: !migraphx.shaped<2x256x384xf32, 98304x384x1>, %arg1: !migraphx.shaped<2x384x768xf32, 294912x768x1>) -> !migraphx.shaped<2x256x768xf32, 196608x768x1> {
    %0 = migraphx.dot %arg0, %arg1 : <2x256x384xf32, 98304x384x1>, <2x384x768xf32, 294912x768x1> -> <2x256x768xf32, 196608x768x1>
     return %0 : !migraphx.shaped<2x256x768xf32, 196608x768x1>
  }

  // CHECK-LABEL: func.func @quant_matmul
  // CHECK: tosa.matmul
  func.func @quant_matmul(%arg0: !migraphx.shaped<2x256x384xi8, 98304x384x1>, %arg1: !migraphx.shaped<2x384x768xi8, 294912x768x1>) -> !migraphx.shaped<2x256x768xi32, 196608x768x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <2x256x384xi8, 98304x384x1>, <2x384x768xi8, 294912x768x1> -> <2x256x768xi32, 196608x768x1>
     return %0 : !migraphx.shaped<2x256x768xi32, 196608x768x1>
  }

  // CHECK-LABEL: func.func @quant_matmul_fp8
  // CHECK: tosa.matmul
  func.func @quant_matmul_fp8(%arg0: !migraphx.shaped<1x12x1024x64xf8E4M3FNUZ, 786432x64x768x1>, %arg1: !migraphx.shaped<1x12x64x1024xf8E4M3FNUZ, 786432x64x1x768>) -> !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x12x1024x64xf8E4M3FNUZ, 786432x64x768x1>, <1x12x64x1024xf8E4M3FNUZ, 786432x64x1x768> -> <1x12x1024x1024xf32, 12582912x1048576x1024x1>
     return %0 : !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1>
  }

  // CHECK-LABEL: func.func @quant_matmul_fp8_ocp
  // CHECK: tosa.matmul
  func.func @quant_matmul_fp8_ocp(%arg0: !migraphx.shaped<1x12x1024x64xf8E4M3FN, 786432x64x768x1>, %arg1: !migraphx.shaped<1x12x64x1024xf8E4M3FN, 786432x64x1x768>) -> !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1> {
    %0 = migraphx.quant_dot %arg0, %arg1 : <1x12x1024x64xf8E4M3FN, 786432x64x768x1>, <1x12x64x1024xf8E4M3FN, 786432x64x1x768> -> <1x12x1024x1024xf32, 12582912x1048576x1024x1>
     return %0 : !migraphx.shaped<1x12x1024x1024xf32, 12582912x1048576x1024x1>
  }

  // CHECK-LABEL: func.func @matmul_larger_batch
  // CHECK: tosa.matmul
  func.func @matmul_larger_batch(%arg0: !migraphx.shaped<2x16x256x384xf32, 1572864x98304x384x1>, %arg1: !migraphx.shaped<2x16x384x768xf32, 4718592x294912x768x1>) -> !migraphx.shaped<2x16x256x768xf32, 3145728x196608x768x1> {
    %0 = migraphx.dot %arg0, %arg1 : <2x16x256x384xf32, 1572864x98304x384x1>, <2x16x384x768xf32, 4718592x294912x768x1> -> <2x16x256x768xf32, 3145728x196608x768x1>
     return %0 : !migraphx.shaped<2x16x256x768xf32, 3145728x196608x768x1>
  }

  // CHECK-LABEL: func.func @matmul_rank2
  // CHECK: tosa.matmul
  func.func @matmul_rank2(%arg0: !migraphx.shaped<32x72xf32, 72x1>, %arg1: !migraphx.shaped<72x64xf32, 64x1>) -> !migraphx.shaped<32x64xf32, 64x1> {
    %0 = migraphx.dot %arg0, %arg1 : <32x72xf32, 72x1>, <72x64xf32, 64x1> -> <32x64xf32, 64x1>
     return %0 : !migraphx.shaped<32x64xf32, 64x1>
  }

  // CHECK-LABEL: func.func @matmul_broadcast_op
  func.func @matmul_broadcast_op(%arg0: !migraphx.shaped<64x64x2304xf16, 147456x2304x1>, %arg1: !migraphx.shaped<64x64x768xf16, 49152x768x1>, %arg2: !migraphx.shaped<1x768x2304xf16, 1769472x2304x1>) -> !migraphx.shaped<64x64x2304xf16, 147456x2304x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 768, 2304>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 64, 64, 768>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 64, 2304>}
    // CHECK-DAG: %[[INPUT:.*]] = tosa.reshape %[[ARG2]] {new_shape = array<i64: 1, 768, 2304>}
    %0 = migraphx.broadcast %arg2 {axis = 0, out_lens = [64, 768, 2304]} : <1x768x2304xf16, 1769472x2304x1> -> <64x768x2304xf16, 0x2304x1>
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x768x2304xf16>}> : () -> tensor<64x768x2304xf16>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[INPUT]]
    %1 = migraphx.dot %arg1, %0 : <64x64x768xf16, 49152x768x1>, <64x768x2304xf16, 0x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    // CHECK-DAG: %[[MATMUL:.*]] = tosa.matmul %[[ARG1]], %[[ADD]]
    // CHECK-DAG: %[[BIASED:.*]] = tosa.add %[[MATMUL]], %[[ARG0]]
    // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[BIASED]] {new_shape = array<i64: 9437184>}
    // CHECK: return %[[RET]]
    %2 = migraphx.add %1, %arg0 : <64x64x2304xf16, 147456x2304x1>, <64x64x2304xf16, 147456x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    return %2 : !migraphx.shaped<64x64x2304xf16, 147456x2304x1>
  }

  // CHECK-LABEL: func.func @matmul_broadcast
  func.func @matmul_broadcast(%arg0: !migraphx.shaped<64x64x2304xf16, 147456x2304x1>, %arg1: !migraphx.shaped<64x64x768xf16, 49152x768x1>, %arg2: !migraphx.shaped<1x768x2304xf16, 1769472x2304x1>) -> !migraphx.shaped<64x64x2304xf16, 147456x2304x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 768, 2304>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 64, 64, 768>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 64, 2304>}
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [64, 768, 2304]} : <1x768x2304xf16, 1769472x2304x1> -> <64x768x2304xf16, 0x2304x1>
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x768x2304xf16>}> : () -> tensor<64x768x2304xf16>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG2]]
    %1 = migraphx.dot %arg1, %0 : <64x64x768xf16, 49152x768x1>, <64x768x2304xf16, 0x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    // CHECK-DAG: %[[MATMUL:.*]] = tosa.matmul %[[ARG1]], %[[ADD]]
    // CHECK-DAG: %[[BIASED:.*]] = tosa.add %[[MATMUL]], %[[ARG0]]
    // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[BIASED]] {new_shape = array<i64: 9437184>}
    // CHECK: return %[[RET]]
    %2 = migraphx.add %1, %arg0 : <64x64x2304xf16, 147456x2304x1>, <64x64x2304xf16, 147456x2304x1> -> <64x64x2304xf16, 147456x2304x1>
    return %2 : !migraphx.shaped<64x64x2304xf16, 147456x2304x1>
  }

  // CHECK-LABEL: func.func @matmul_broadcast_R5
  func.func @matmul_broadcast_R5(%arg0: !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>, %arg1: !migraphx.shaped<2x4x8x64x768xf16, 1572864x393216x49152x768x1>, %arg2: !migraphx.shaped<1x1x1x768x2304xf16, 1769472x1769472x1769472x2304x1>) -> !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 1, 1, 768, 2304>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 2, 4, 8, 64, 768>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 2, 4, 8, 64, 2304>}
    %0 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [2, 4, 8, 768, 2304]} : <1x1x1x768x2304xf16, 1769472x1769472x1769472x2304x1> -> <2x4x8x768x2304xf16, 0x0x0x2304x1>
    // CHECK-DAG: %[[RESHAPE0:.*]] = tosa.reshape %[[ARG1]] {new_shape = array<i64: 64, 64, 768>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<2x4x8x768x2304xf16>}> : () -> tensor<2x4x8x768x2304xf16>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG2]]
    // CHECK-DAG: %[[RESHAPE1:.*]] = tosa.reshape %[[ADD]] {new_shape = array<i64: 64, 768, 2304>}
    %1 = migraphx.dot %arg1, %0 : <2x4x8x64x768xf16, 1572864x393216x49152x768x1>, <2x4x8x768x2304xf16, 0x0x0x2304x1> -> <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
    // CHECK-DAG: %[[MATMUL:.*]] = tosa.matmul %[[RESHAPE0]], %[[RESHAPE1]]
    // CHECK: %[[RESHAPE2:.*]] = tosa.reshape %[[MATMUL]] {new_shape = array<i64: 2, 4, 8, 64, 2304>}
    %2 = migraphx.add %1, %arg0 : <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>, <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1> -> <2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
    return %2 : !migraphx.shaped<2x4x8x64x2304xf16, 4718592x1179648x147456x2304x1>
  }


  // broadcast ops will be lowered as implicit broadcast in tosa, passes if they're converted and legalize tosa.
  // CHECK-LABEL: func @func_mbcast
  func.func @func_mbcast(%arg0: !migraphx.shaped<1x64x1x1xf32, 64x1x1x1>, %arg1: !migraphx.shaped<1x3x224x224xf32, 150528x50176x224x1>, %arg2: !migraphx.shaped<64x3x7x7xf32, 147x49x7x1>) -> !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1> attributes {kernel = "mixr"} {
    %0 = migraphx.multibroadcast %arg0 {out_lens = [1, 64, 112, 112]} : <1x64x1x1xf32, 64x1x1x1> -> <1x64x112x112xf32, 0x1x0x0>
    %1 = migraphx.convolution %arg1, %arg2 {dilation = [1, 1], group = 1 : i64, padding = [3, 3, 3, 3], padding_mode = 0 : i64, stride = [2, 2]} : <1x3x224x224xf32, 150528x50176x224x1>, <64x3x7x7xf32, 147x49x7x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    %2 = migraphx.add %1, %0 : <1x64x112x112xf32, 802816x12544x112x1>, <1x64x112x112xf32, 0x1x0x0> -> <1x64x112x112xf32, 802816x12544x112x1>
    %3 = migraphx.relu %2 : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x112x112xf32, 802816x12544x112x1>
    return %3 : !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>
  }

  // CHECK-LABEL: func.func @mbcast_non_first_dim
  // COM: test for a bug in how mbcast was handled in this case.
  // CHECK: new_shape = array<i64: 1, 1, 5, 1>
  func.func @mbcast_non_first_dim(%arg0: !migraphx.shaped<2x3x3x5xf32, 45x15x5x1>, %arg1: !migraphx.shaped<5xf32, 1>) -> !migraphx.shaped<2x3x3x1xf32, 9x3x1x1> attributes {arch = "gfx1100", kernel = "mixr", num_cu = 48 : i64} {
    %0 = migraphx.reshape %arg1 {dims = [5, 1]} : <5xf32, 1> -> <5x1xf32, 1x1>
    %1 = migraphx.multibroadcast %0 {out_dyn_dims = [], out_lens = [2, 3, 5, 1]} : <5x1xf32, 1x1> -> <2x3x5x1xf32, 0x0x1x1>
    %2 = migraphx.dot %arg0, %1 : <2x3x3x5xf32, 45x15x5x1>, <2x3x5x1xf32, 0x0x1x1> -> <2x3x3x1xf32, 9x3x1x1>
    return %2 : !migraphx.shaped<2x3x3x1xf32, 9x3x1x1>
  }

  // CHECK-LABEL: func.func @clip_i32
  func.func @clip_i32(%arg0: !migraphx.shaped<64x64xi32, 64x1>, %arg1: !migraphx.shaped<64x64xi32, 64x1>, %arg2: !migraphx.shaped<64x64xi32, 64x1>) -> !migraphx.shaped<64x64xi32, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2
    // CHECK: %[[MAX:.*]] = tosa.maximum %[[ARG0]], %[[ARG1]]
    // CHECK: %[[MIN:.*]] = tosa.minimum %[[MAX]], %[[ARG2]]
    // CHECK: %[[RET:.*]] = tosa.reshape %[[MIN]]
    // CHECK: return %[[RET]]
    %0 = migraphx.clip %arg0, %arg1, %arg2 : <64x64xi32, 64x1>, <64x64xi32, 64x1>, <64x64xi32, 64x1> -> <64x64xi32, 64x1>
    return %0 : !migraphx.shaped<64x64xi32, 64x1>
  }

  // CHECK-LABEL: func.func @clip_broadcast
  func.func @clip_broadcast(%arg0: !migraphx.shaped<64x64xf16, 64x1>, %arg1: !migraphx.shaped<1x64xf16, 64x1>, %arg2: !migraphx.shaped<1xf16, 0>) -> !migraphx.shaped<64x64xf16, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 1, 64>}
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0.000000e+00> : tensor<64x64xf16>}> : () -> tensor<64x64xf16>
    // CHECK-DAG: %[[ADD0:.*]] = tosa.add %[[CST0]], %[[ARG1]]
    // CHECK-DAG: %[[RESHAPE:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 1, 1>}
    // CHECK-DAG: %[[ADD1:.*]] = tosa.add %[[CST0]], %[[RESHAPE]]
    // CHECK: %[[MAX:.*]] = tosa.maximum %[[ARG0]], %[[ADD0]]
    // CHECK: %[[MIN:.*]] = tosa.minimum %[[MAX]], %[[ADD1]]
    // CHECK: %[[RET:.*]] = tosa.reshape %[[MIN]] {new_shape = array<i64: 4096>}
    // CHECK: return %[[RET]]
    %0 = migraphx.multibroadcast %arg1 {out_dyn_dims = [], out_lens = [64, 64]} : <1x64xf16, 64x1> -> <64x64xf16, 0x1>
    %1 = migraphx.multibroadcast %arg2 {out_dyn_dims = [], out_lens = [64, 64]} : <1xf16, 0> -> <64x64xf16, 0x0>
    %2 = migraphx.clip %arg0, %0, %1 : <64x64xf16, 64x1>, <64x64xf16, 0x1>, <64x64xf16, 0x0> -> <64x64xf16, 64x1>
    return %2 : !migraphx.shaped<64x64xf16, 64x1>
  }

  // CHECK-LABEL: func.func @where
  func.func @where_f32(%arg0: !migraphx.shaped<64x64xi8, 64x1>, %arg1: !migraphx.shaped<64x64xf32, 64x1>, %arg2: !migraphx.shaped<64x64xf32, 64x1>) -> !migraphx.shaped<64x64xf32, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK: tosa.cast
    // CHECK: tosa.select
    %0 = migraphx.where %arg0, %arg1, %arg2 : <64x64xi8, 64x1>, <64x64xf32, 64x1>, <64x64xf32, 64x1> -> <64x64xf32, 64x1>
    return %0 : !migraphx.shaped<64x64xf32, 64x1>
  }

  // CHECK-LABEL: func.func @where_broadcast
  func.func @where_broadcast(%arg0: !migraphx.shaped<64x1xi8, 1x1>, %arg1: !migraphx.shaped<64x64xf16, 64x1>, %arg2: !migraphx.shaped<64x64xf16, 64x1>) -> !migraphx.shaped<64x64xf16, 64x1> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0 {new_shape = array<i64: 64, 1>}
    // CHECK-DAG: %[[ARG1:.*]] = tosa.reshape %arg1 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[ARG2:.*]] = tosa.reshape %arg2 {new_shape = array<i64: 64, 64>}
    // CHECK-DAG: %[[CST0:.*]] = "tosa.const"() <{value = dense<0> : tensor<64x64xi8>}> : () -> tensor<64x64xi8>
    // CHECK-DAG: %[[ADD:.*]] = tosa.add %[[CST0]], %[[ARG0]]
    // CHECK-DAG: %[[CAST:.*]] = tosa.cast %[[ADD]]
    // CHECK-DAG: tosa.select %[[CAST]], %[[ARG1]], %[[ARG2]]
    %0 = migraphx.multibroadcast %arg0 {out_dyn_dims = [], out_lens = [64, 64]} : <64x1xi8, 1x1> -> <64x64xi8, 1x0>
    %1 = migraphx.where %0, %arg1, %arg2 : <64x64xi8, 1x0>, <64x64xf16, 64x1>, <64x64xf16, 64x1> -> <64x64xf16, 64x1>
    return %1 : !migraphx.shaped<64x64xf16, 64x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf32>}> : () -> tensor<1xf32>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xf32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_f32(%arg0: !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x1x112xf32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_f16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<1.120000e+02> : tensor<1xf16>}> : () -> tensor<1xf16>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xf16>) -> tensor<1xf16>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xf16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_f16(%arg0: !migraphx.shaped<1x64x112x112xf16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xf16, 802816x12544x112x1> -> <1x64x1x112xf16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi32>}> : () -> tensor<1xi32>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi32>) -> tensor<1xi32>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi32>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i32(%arg0: !migraphx.shaped<1x64x112x112xi32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi32, 802816x12544x112x1> -> <1x64x1x112xi32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi16>}> : () -> tensor<1xi16>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi16>) -> tensor<1xi16>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi16>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i16(%arg0: !migraphx.shaped<1x64x112x112xi16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi16, 802816x12544x112x1> -> <1x64x1x112xi16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_mean_i8
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-DAG: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[N:.*]] = "tosa.const"() <{value = dense<112> : tensor<1xi8>}> : () -> tensor<1xi8>
  // CHECK-DAG: %[[NRECIP:.*]] = tosa.reciprocal %[[N]] : (tensor<1xi8>) -> tensor<1xi8>
  // CHECK-DAG: %[[MUL:.*]] = tosa.mul %[[ARG0]], %[[NRECIP]] {shift = 0 : i8} : ([[INTYPE]], tensor<1xi8>) -> [[INTYPE]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[MUL]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_mean_i8(%arg0: !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1> {
    %0 = migraphx.reduce_mean %arg0 {axes = [2 : i64]} : <1x64x112x112xi8, 802816x12544x112x1> -> <1x64x1x112xi8, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_f32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_f32(%arg0: !migraphx.shaped<1x64x112x112xf32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xf32, 802816x12544x112x1> -> <1x64x1x112xf32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_f16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_f16(%arg0: !migraphx.shaped<1x64x112x112xf16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xf16, 802816x12544x112x1> -> <1x64x1x112xf16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xf16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i32
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i32(%arg0: !migraphx.shaped<1x64x112x112xi32, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi32, 802816x12544x112x1> -> <1x64x1x112xi32, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi32, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i16
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-SAME: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i16(%arg0: !migraphx.shaped<1x64x112x112xi16, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi16, 802816x12544x112x1> -> <1x64x1x112xi16, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi16, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_reduce_sum_i8
  // CHECK-SAME: (%arg0: [[INTYPE_FLAT:.*]]) -> [[OUTTYPE_FLAT:.*]] {
  // CHECK-DAG: %[[ARG0:.*]] = tosa.reshape %arg0
  // CHECK-DAG: ([[INTYPE_FLAT]]) -> [[INTYPE:.*]]
  // CHECK-DAG: %[[REDUCE_SUM:.*]] = tosa.reduce_sum %[[ARG0]] {axis = 2 : i32} : ([[INTYPE]]) -> [[OUTTYPE:.*]]
  // CHECK-DAG: %[[RET:.*]] = tosa.reshape %[[REDUCE_SUM]]
  // CHECK-SAME: ([[OUTTYPE]]) -> [[OUTTYPE_FLAT]]
  // CHECK: return %[[RET]]
  func.func @func_reduce_sum_i8(%arg0: !migraphx.shaped<1x64x112x112xi8, 802816x12544x112x1>) -> !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1> {
    %0 = migraphx.reduce_sum %arg0 {axes = [2 : i64]} : <1x64x112x112xi8, 802816x12544x112x1> -> <1x64x1x112xi8, 7168x112x112x1>
    return %0 : !migraphx.shaped<1x64x1x112xi8, 7168x112x112x1>
  }

  // CHECK-LABEL: func.func @func_dot_mul
  // CHECK: tosa.matmul
  // CHECK: tosa.mul
  func.func @func_dot_mul(%arg0: !migraphx.shaped<1x5x4xf32, 20x4x1>, %arg1: !migraphx.shaped<1x4x3xf32, 12x3x1>, %arg2: !migraphx.shaped<1x5x3xf32, 15x3x1>) -> !migraphx.shaped<1x5x3xf32, 15x3x1> attributes{kernel, arch = ""} {
    %0 = migraphx.dot %arg0, %arg1 : <1x5x4xf32, 20x4x1>, <1x4x3xf32, 12x3x1> -> <1x5x3xf32, 15x3x1>
    %2 = migraphx.mul %0, %arg2 {} : <1x5x3xf32, 15x3x1>, <1x5x3xf32, 15x3x1> -> <1x5x3xf32, 15x3x1>
    return %2 : !migraphx.shaped<1x5x3xf32, 15x3x1>
  }

  // CHECK-LABEL: func.func @func_slice1
  // CHECK: tosa.slice
  func.func @func_slice1(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.slice %arg0 {axes = [1], ends = [12], starts = [0]} : <1x36x384x64xf32, 884736x24576x64x1> -> <1x12x384x64xf32, 294912x24576x64x1>
    return %0 : !migraphx.shaped<1x12x384x64xf32, 294912x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_slice2
  // CHECK: tosa.slice
  func.func @func_slice2(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x12x100x64xf32, 76800x6400x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.slice %arg0 {axes = [1, 2], ends = [12, 284], starts = [0, 184]} : <1x36x384x64xf32, 884736x24576x64x1> -> <1x12x100x64xf32, 76800x6400x64x1>
    return %0 : !migraphx.shaped<1x12x100x64xf32, 76800x6400x64x1>
  }
  
  // CHECK-LABEL: func.func @func_greaterorequal
  // CHECK: %[[ge:.+]] = tosa.greater_equal {{.*}} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi1>
  // CHECK-NEXT: tosa.cast %[[ge]] : (tensor<1x36x384x64xi1>) -> tensor<1x36x384x64xi32>
  func.func @func_greaterorequal(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %cst = migraphx.literal (dense<1> : tensor<1x36x384x64xi32>) : <1x36x384x64xi32, 884736x24576x64x1>
    %0 = migraphx.add %arg0, %cst : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
    %1 = migraphx.greater_or_equal %arg0, %0 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
    return %1 : !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_greaterorequal_si32
  // CHECK: %[[ge:.+]] = tosa.greater_equal {{.*}} : (tensor<1x36x384x64xi32>, tensor<1x36x384x64xi32>) -> tensor<1x36x384x64xi1>
  // CHECK-NEXT: tosa.cast %[[ge]] : (tensor<1x36x384x64xi1>) -> tensor<1x36x384x64xi32>
  func.func @func_greaterorequal_si32(%arg0: !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %cst = migraphx.literal (dense<1> : tensor<1x36x384x64xsi32>) : <1x36x384x64xsi32, 884736x24576x64x1>
    %0 = migraphx.add %arg0, %cst : <1x36x384x64xsi32, 884736x24576x64x1>, <1x36x384x64xsi32, 884736x24576x64x1> -> <1x36x384x64xsi32, 884736x24576x64x1>
    %1 = migraphx.greater_or_equal %arg0, %0 : <1x36x384x64xsi32, 884736x24576x64x1>, <1x36x384x64xsi32, 884736x24576x64x1> -> <1x36x384x64xsi32, 884736x24576x64x1>
    return %1 : !migraphx.shaped<1x36x384x64xsi32, 884736x24576x64x1>
  }
}

// -----

// Unary operations

module {
  // CHECK-LABEL: func.func @func_abs
  // CHECK: tosa.abs
  func.func @func_abs(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.abs %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_ceil
  // CHECK: tosa.ceil
  func.func @func_ceil(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.ceil %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_convert
  // CHECK: tosa.cast
  func.func @func_convert(%arg0: !migraphx.shaped<16xf16, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.convert %arg0 : <16xf16, 1> to <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_div_f32
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xf32, 884736x24576x64x1>, <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_div_f16
  // CHECK: tosa.reciprocal
  // CHECK: tosa.mul
  func.func @func_div_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xf16, 884736x24576x64x1>, <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_div_i32
  // CHECK: tosa.int_div
  func.func @func_div_i32(%arg0: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>, %arg1: !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.div %arg0, %arg1 : <1x36x384x64xi32, 884736x24576x64x1>, <1x36x384x64xi32, 884736x24576x64x1> -> <1x36x384x64xi32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xi32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_erf_f32
  // CHECK: tosa.erf
  func.func @func_erf_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.erf %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_erf_f16
  // CHECK: tosa.erf
  func.func @func_erf_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.erf %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_exp_f32
  // CHECK: tosa.exp
  func.func @func_exp_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.exp %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_exp_f16
  // CHECK: tosa.exp
  func.func @func_exp_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.exp %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_floor
  // CHECK: tosa.floor
  func.func @func_floor(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.floor %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_log_f32
  // CHECK: tosa.log
  func.func @func_log_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.log %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_log_f16
  // CHECK: tosa.log
  func.func @func_log_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.log %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_neg_f32
  // CHECK: tosa.negate
  func.func @func_neg_f32(%arg0: !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.neg %arg0 : <1x36x384x64xf32, 884736x24576x64x1> -> <1x36x384x64xf32, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf32, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_neg_f16
  // CHECK: tosa.negate
  func.func @func_neg_f16(%arg0: !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>) -> !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1> attributes{kernel, arch = ""} {
    %0 = migraphx.neg %arg0 : <1x36x384x64xf16, 884736x24576x64x1> -> <1x36x384x64xf16, 884736x24576x64x1>
    return %0 : !migraphx.shaped<1x36x384x64xf16, 884736x24576x64x1>
  }

  // CHECK-LABEL: func.func @func_power
  // CHECK: tosa.pow
  func.func @func_power(%arg0: !migraphx.shaped<16xf32, 1>, %arg1: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.pow %arg0, %arg1 : <16xf32, 1>, <16xf32, 1> -> <16xf32, 1>
    return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_recip
  // CHECK: tosa.recip
  func.func @func_recip(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.recip %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_rsqrt
  // CHECK: tosa.rsqrt
  func.func @func_rsqrt(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.rsqrt %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_sigmoid
  // CHECK: tosa.sigmoid
  func.func @func_sigmoid(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.sigmoid %arg0 : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }


  // CHECK-LABEL: func.func @func_rsqrt_opt
  // CHECK: tosa.rsqrt
  // CHECK-NOT: tosa.reciprocal
  func.func @func_rsqrt_opt(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.sqrt %arg0 : <16xf32, 1> -> <16xf32, 1>
    %1 = migraphx.recip %0 : <16xf32, 1> -> <16xf32, 1>
     return %1 : !migraphx.shaped<16xf32, 1>
  }
}

// -----

// Less trivial pointwise ops
module {
  // CHECK-LABEL: func.func @func_softmax_1d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = tosa.reduce_max [[INPUT:%[a-z0-9]+]]
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = tosa.sub [[INPUT]], [[REDUCE_MAX]]
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = tosa.exp [[SUB]]
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = tosa.reduce_sum [[EXP]]
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = tosa.reciprocal [[REDUCE_SUM]]
  // CHECK-DAG: tosa.mul [[EXP]], [[RECIPROCAL]]
  func.func @func_softmax_1d(%arg0: !migraphx.shaped<16xf32, 1>) -> !migraphx.shaped<16xf32, 1> {
    %0 = migraphx.softmax %arg0 {axis = 0 : i64} : <16xf32, 1> -> <16xf32, 1>
     return %0 : !migraphx.shaped<16xf32, 1>
  }

  // CHECK-LABEL: func.func @func_softmax_4d
  // CHECK-DAG: [[REDUCE_MAX:%[a-z0-9]+]] = tosa.reduce_max [[INPUT:%[a-z0-9]+]]
  // CHECK-DAG: [[SUB:%[a-z0-9]+]] = tosa.sub [[INPUT]], [[REDUCE_MAX]]
  // CHECK-DAG: [[EXP:%[a-z0-9]+]] = tosa.exp [[SUB]]
  // CHECK-DAG: [[REDUCE_SUM:%[a-z0-9]+]] = tosa.reduce_sum [[EXP]]
  // CHECK-DAG: [[RECIPROCAL:%[a-z0-9]+]] = tosa.reciprocal [[REDUCE_SUM]]
  // CHECK-DAG: tosa.mul [[EXP]], [[RECIPROCAL]]
  func.func @func_softmax_4d(%arg0: !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1>) -> !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1> {
    %0 = migraphx.softmax %arg0 {axis = 1 : i64} : <16x16x16x16xf32, 4096x256x16x1> -> <16x16x16x16xf32, 4096x256x16x1>
     return %0 : !migraphx.shaped<16x16x16x16xf32, 4096x256x16x1>
  }
}
