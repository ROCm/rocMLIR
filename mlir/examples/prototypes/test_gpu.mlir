module attributes {gpu.kernel_module} {
  func @test_gpu() attributes {gpu.kernel} {
    %tid = "gpu.thread_id"() {dimension = "x"} : () -> index
    %bid = "gpu.block_id"() {dimension = "x"} : () -> index

    return
  }
}
