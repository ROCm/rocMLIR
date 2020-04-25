#map0 = (d0, d1) -> (d0 + d1)

#map1_0 = (d0, d1) -> (d1)
#map1_1 = (d0, d1) -> (d0 floordiv 9)
#map1_2 = (d0, d1) -> ((d0 floordiv 9) floordiv 3)
#map1_3 = (d0, d1) -> ((d0 floordiv 9) mod 3)

module {
  func @test(%y : index, %x : index) {
    %0 = affine.apply #map0 (%y, %x)

    return
  }

  func @test2(%y : index) {
    %c1 = constant 1: index
    %0 = affine.apply #map0 (%y, %c1)

    return
  }

  func @test3(%offset : index) {
    %c3 = constant 13 : index
    %c0 = constant 0 : index
    %0 = affine.apply #map1_0 (%c3, %c0)
    %1 = affine.apply #map1_1 (%c3, %c0)
    %2 = affine.apply #map1_2 (%c3, %c0)
    %3 = affine.apply #map1_3 (%c3, %c0)

    return
  }
}
