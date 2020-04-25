module {
  func @test_alloc() {
    %buffer = alloc() : memref<128xf32, 5>
    return
  }
}
