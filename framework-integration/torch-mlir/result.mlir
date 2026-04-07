#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
module {
  func.func @main(%arg0: tensor<20x10xf32>) -> tensor<20x10xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant dense_resource<torch_tensor_10_10_torch.float32> : tensor<10x10xf32>
    %cst_1 = arith.constant dense_resource<torch_tensor_10_torch.float32> : tensor<10xf32>
    %0 = tensor.empty() : tensor<10x10xf32>
    %transposed = linalg.transpose ins(%cst_0 : tensor<10x10xf32>) outs(%0 : tensor<10x10xf32>) permutation = [1, 0] 
    %1 = tensor.empty() : tensor<20x10xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<20x10xf32>) -> tensor<20x10xf32>
    %3 = linalg.matmul ins(%arg0, %transposed : tensor<20x10xf32>, tensor<10x10xf32>) outs(%2 : tensor<20x10xf32>) -> tensor<20x10xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %cst_1 : tensor<20x10xf32>, tensor<10xf32>) outs(%1 : tensor<20x10xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<20x10xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %arg0 : tensor<20x10xf32>, tensor<20x10xf32>) outs(%1 : tensor<20x10xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %6 = arith.addf %in, %in_2 : f32
      linalg.yield %6 : f32
    } -> tensor<20x10xf32>
    return %5 : tensor<20x10xf32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      torch_tensor_10_10_torch.float32: "0x0400000019C28ABE851BF3BC61CADD3C6C1840BC4DF9953E417C3FBEDFB857BD51E3B93DD6D289BD9687283A179E103D6F6FFFBD9687623D3229B23D8437B1BD9502943EDA0680BEE03A5FBEA40D973E95343E3E14A0223E18DA0D3D93826ABC4C7787BE6C579CBEC0AF673E2F6DB2BD85254FBE3D6D653E1B213E3D77B0113E932A0E3E7F02DF3DEBCA1FBEE946B73DB43D343E2EAF2C3EE12F11BE37A4B03D6330F23D812B9BBC28376CBE11A333BE879CCDBD511F96BE6B365F3EECB92CBE95F59D3E7B5C183E9FEDC73D04DF20BDAF13A43D2480CCB9A5D16ABE272A79BEC43F423C66B74C3D01F43F3E3A6801BE7EC9DA3D3F5B0DBEEA423EBEC17189BD7E5A92BEDB291A3CE1DB21BD49F0583E6F7B6CBC5AAA98BDB7D072BED818DB3CFE179D3EA73854BEFCF2D5BDB2C21FBEA07C963DF5DE0A3EBC1DD6BC8A2C66BDB444E83D689F74BEC20034BEAF3DE83D7640E3BD4E221DBE531AA7BDE11EBC3CDACB97BE544E5ABE52D5B2BC8B410BBEAE1482BE8902DCBD1639053E0CCE323E20A593BEB8D835BD30C371BE37185FBEE2BC3D3E",
      torch_tensor_10_torch.float32: "0x040000006D8581BEC69643BE4A65FB3DD8424B3DE032C53D44A64EBEC7E7F83D14F6133C5F8C223ED542273E"
    }
  }
#-}
