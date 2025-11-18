// RUN: rocmlir-opt %s --rock-add-alias-info -split-input-file | FileCheck %s

// Test that llvm.load operations with LDS address space get the appropriate alias scope
// CHECK-DAG: #[[$DOMAIN:.+]] = #llvm.alias_scope_domain<id = "amdgpu.LoadsScope", description = "Domain to hold alias scopes to specify aliasing information for operations that load directly from global memory to LDS">
// CHECK-DAG: #[[$LOCAL_LOAD_SCOPE:.+]] = #llvm.alias_scope<id = "amdgpu.LocalLoads", domain = #[[$DOMAIN]], description = "Scope containing all LocalLoad ops">
// CHECK-DAG: #[[$DIRECT_TO_LDS_SCOPE:.+]] = #llvm.alias_scope<id = "amdgpu.DirectToLDSLoads", domain = #[[$DOMAIN]], description = "Scope containing all operations that perform direct global-to-LDS loads">

gpu.module @test_module {
  // Test function with local load and local store.
  llvm.func @test_local_load(%global_ptr: !llvm.ptr<1>, %lds_addr: !llvm.ptr<3>) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %lds_ptr = llvm.getelementptr %lds_addr[%c0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32

    // CHECK: llvm.load %{{.*}} {alias_scopes = [#[[$LOCAL_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    %val = llvm.load %lds_ptr : !llvm.ptr<3> -> i32
    llvm.return %val : i32
  }

   llvm.func @test_local_store(%global_ptr: !llvm.ptr<1>, %value: i32, %lds_addr: !llvm.ptr<3>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %lds_ptr = llvm.getelementptr %lds_addr[%c0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32

    // CHECK: llvm.store %{{.*}}, %{{.*}} {alias_scopes = [#[[$LOCAL_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    llvm.store %value, %lds_ptr : i32, !llvm.ptr<3>
    llvm.return
  }
}

