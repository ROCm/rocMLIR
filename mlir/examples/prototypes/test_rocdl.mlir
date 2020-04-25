module {
  func @test_rocdl() {
    %0 = rocdl.workitem.id.x : !llvm.i32
    %1 = rocdl.workgroup.id.x : !llvm.i32
    return
  }
}
