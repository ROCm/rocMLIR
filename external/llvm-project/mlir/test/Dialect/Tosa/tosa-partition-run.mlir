// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), \
// RUN: func.func(tosa-to-linalg))" \
// RUN: | mlir-opt  --tosa-to-tensor -tosa-to-arith --empty-tensor-to-alloc-tensor\
// RUN: -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" \
// RUN: --buffer-deallocation --convert-linalg-to-loops \
// RUN: -lower-affine -convert-linalg-to-llvm --convert-scf-to-cf \
// RUN: -convert-math-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
// RUN: --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext \
// RUN:   -shared-libs=%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext \
// RUN: > %t1
//
// RUN  cat %t1 | FileCheck %s
// CHECK:  Unranked Memref
// CHECK-SAME:  sizes = [5, 10, 8, 18] strides = [1440, 144, 18, 1]
// CHECK-NEXT:  [-0.399353
// CHECK-NEXT:  [-0.399353
// CHECK-NEXT:  [-0.399353
//
// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(tosa-to-linalg-named), \
// RUN: func.func(tosa-to-linalg))" \
// RUN: | mlir-opt  --tosa-to-tensor -tosa-to-arith --empty-tensor-to-alloc-tensor\
// RUN: -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" \
// RUN: --buffer-deallocation --convert-linalg-to-loops \
// RUN: -lower-affine -convert-linalg-to-llvm --convert-scf-to-cf \
// RUN: --convert-math-to-llvm --convert-memref-to-llvm --convert-func-to-llvm \
// RUN: --reconcile-unrealized-casts \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%linalg_test_lib_dir/%prefix_mlir_runner_utils%shlibext \
// RUN:   -shared-libs=%linalg_test_lib_dir/%prefix_mlir_c_runner_utils%shlibext \
// RUN: > %t2
//
// RUN: diff --ignore-matching-lines='Unranked Memref' %t1 %t2

module attributes {torch.debug_module_name = "Conv2dNoPaddingModule"} {
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func private @printNewline()
  func.func @main() {
    %0 = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %1 = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %2 = "tosa.const"() {value = dense<0.000000e+00> : tensor<10xf32>} : () -> tensor<10xf32>
    %3 = "tosa.const"() {value = dense<1> : tensor<5x2x10x20xi32>} : () -> tensor<5x2x10x20xi32>
    %4 = "tosa.const"() {value = dense<"0x2E4CE7BABE79013E42A646BE27A031BEC9EBB9BDC771813DE50699BB015F3F3E7D5AABBC74777F3D3EE291BD7AC53DBD029566BE0BD91FBED4FCC6BD880D0F3CE4D5BE3D2BD2103E94A023BEB234D2BDDF54AF3DF56B483EDFAF46BDA39C343EBF9C1BBD4750CC3C818B5A3ED6E65FBED9F117BE3B6A74BDDE29BCBDD388503EAE711CBE0936DEBD8D9F28BE2B0C62BE1EE40CBEC9784F3ED265D73DD5F5E93D62184B3C4F7BF7BD4A56233D115B61BEE0652EBE11DBF8BD5A48183E94830D3E4915D6BD4C570BBC225D1A3E6EF16F3E0F95BF3D6F6C023DF7D3213EE11C0EBE40E7333D86203BBE5C4827BE435DF9BDC06ADA3D5221C23D3DF80EBE25D5913DD97F043EAEB5F3BC8A72133C39B25F3DCABB153E69C0673E3DFF39BED1E6B0BD8AB6BD3D5BFA473E5008523E00F7543ECA22403DB7E151BE36A0B13CDDFE16BE85EF60BE1D72563E0F85373E31C370BEE6B3343D2FA322BD93DF1EBDB4F7DCBDDDA1B93D36F50EBED8F5B03D511DF43D3AC82C3E657EB43DB2E16EBEA4911CBE4907F13D31114A3DC7473CBEB2FA0ABEF40E633ED0A1223E2F7AD2BD55FC72BD11EB65BEB7D28ABBF4C135BE892C3ABEDAC754BC9CF4103D0FB0C5BD93370F3E60E012BE86005B3E4367253EB6884BBE314870BD8A402E3CB5DB0C3D08FA643DAA6EBD3DB851673C3888EBBDBE6AE43D028667BE1F0E0FBE75AD71BD7523EBBDF6DEA8BDBBD245BEEB5C4DBD475E4E3DB93C1DBEAD2E46BCF2C62C3E857EC6BC27A7D63B0493A6BCE862433D3577193E63A0643ECB46193E4D26653ED8A48BBC6ED158BE81D8E4BDBA57243E0345C8BAEFEEEFBDD2F438BE5DE061BE86B54BBE74D343BDE05C043E187D023E41C668BE348E163EA4DD3CBE6B1A4CBDA2BAC3BD33F539BD718E3DBD669558BE0B6650BE2D1217BD60C3473B6A49DBBDEED6B53DBF3C59BE2D4F82BC8341543E9BE5C4BDB2F2593E77D1AE3D32D159BE10B5183EE3CFDEBC1D7DD7BDED00413E890A43BE"> : tensor<10x2x3x3xf32>} : () -> tensor<10x2x3x3xf32>
    %5 = "tosa.cast"(%3) : (tensor<5x2x10x20xi32>) -> tensor<5x2x10x20xf32>
    %6 = "tosa.transpose"(%5, %1) : (tensor<5x2x10x20xf32>, tensor<4xi32>) -> tensor<5x10x20x2xf32>
    %7 = "tosa.transpose"(%4, %1) : (tensor<10x2x3x3xf32>, tensor<4xi32>) -> tensor<10x3x3x2xf32>
    %8 = "tosa.conv2d"(%6, %7, %2) {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<5x10x20x2xf32>, tensor<10x3x3x2xf32>, tensor<10xf32>) -> tensor<5x8x18x10xf32>
    %9 = "tosa.transpose"(%8, %0) : (tensor<5x8x18x10xf32>, tensor<4xi32>) -> tensor<5x10x8x18xf32>
    %10 = bufferization.alloc_tensor() copy(%9) : tensor<5x10x8x18xf32>
    %11 = tensor.cast %10 : tensor<5x10x8x18xf32> to tensor<*xf32>
    call @printMemrefF32(%11) : (tensor<*xf32>) -> ()
    call @printNewline() : () -> ()
    return
  }
}
