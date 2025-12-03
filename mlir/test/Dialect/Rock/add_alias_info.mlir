// RUN: rocmlir-opt %s --rock-add-direct-to-lds-alias-info -split-input-file | FileCheck %s

// Test that llvm.load operations with LDS address space get the appropriate alias scope
// CHECK-DAG: #[[$DOMAIN:.+]] = #llvm.alias_scope_domain<id = "amdgpu.LoadsScope", description = "{{.*}}">
// CHECK-DAG: #[[$LDS_LOAD_SCOPE:.+]] = #llvm.alias_scope<id = "amdgpu.LDSLoads", domain = #[[$DOMAIN]], description = "{{.*}}">
// CHECK-DAG: #[[$DIRECT_TO_LDS_SCOPE:.+]] = #llvm.alias_scope<id = "amdgpu.DirectToLDSLoads", domain = #[[$DOMAIN]], description = "{{.*}}">
gpu.module @test_module {
  // Test function with local load and local store.
  llvm.func @test_local_load(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<3>) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.getelementptr %arg1[%0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK: llvm.load %{{.*}} {alias_scopes = [#[[$LDS_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    %2 = llvm.load %1 : !llvm.ptr<3> -> i32
    llvm.return %2 : i32
  }

  llvm.func @test_local_store(%arg0: !llvm.ptr<1>, %arg1: i32, %arg2: !llvm.ptr<3>) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.getelementptr %arg2[%0] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK: llvm.store %{{.*}}, %{{.*}} {alias_scopes = [#[[$LDS_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    llvm.store %arg1, %1 : i32, !llvm.ptr<3>
    llvm.return

  }

  llvm.func @test_load_to_lds(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<3>) {
    %0 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: rocdl.load.to.lds %{{.*}}, %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    rocdl.load.to.lds %arg0, %arg1, 0, 0, 0 : <1>
    llvm.return
  }

  llvm.func @test_flatbuffer_load_lds(%arg0: !llvm.ptr<8>, %arg1: !llvm.ptr<3>) {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(128 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: rocdl.raw.ptr.buffer.load.lds %{{.*}}, %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    rocdl.raw.ptr.buffer.load.lds %arg0, %arg1, %0, %1, %2, %3, %4
    llvm.return
  }

  llvm.func @test_combined(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<3>) {
    // First, load from LDS - should have LocalLoadScope and noalias with DirectToLDS
    // CHECK: llvm.load %{{.*}} {alias_scopes = [#[[$LDS_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    %0 = llvm.load %arg1 : !llvm.ptr<3> -> i32

    // Then, perform a direct-to-LDS load - should have DirectToLDSLoadScope and noalias with LocalLoad    
    // CHECK: rocdl.load.to.lds %{{.*}}, %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    rocdl.load.to.lds %arg0, %arg1, 0, 0, 0 : <1>
    
    llvm.return
  }

  llvm.func @test_both_loads(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<3>) {        
    // CHECK: rocdl.load.to.lds %{{.*}}, %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    rocdl.load.to.lds %arg0, %arg1, 0, 0, 0 : <1>
    
    // CHECK: llvm.load %{{.*}} {alias_scopes = [#[[$LDS_LOAD_SCOPE]]], noalias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]]}
    %0 = llvm.load %arg1 : !llvm.ptr<3> -> i32
    
    llvm.return
  }

  llvm.func @test_negative_direct_to_lds(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<3>) {        
    // CHECK-NOT: llvm.load %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    %0 = llvm.load %arg0 : !llvm.ptr<1> -> i32

    // CHECK-NOT: llvm.store %{{.*}}, %{{.*}} {alias_scopes = [#[[$DIRECT_TO_LDS_SCOPE]]], noalias_scopes = [#[[$LDS_LOAD_SCOPE]]]}
    llvm.store %0, %arg1 : i32, !llvm.ptr<3>
    
    llvm.return
  }
}
