gpu.module @gpu_module{
llvm.func @rocdl.rawbuf(%rsrc : vector<4xi32>,
                        %offset : i32, %soffset : i32,
                        %vdata1 : vector<1xi32>,
                        %vdata2 : vector<2xi32>, 
                        %vdata4 : vector<4xi32>) {
  // CHECK-LABEL: rocdl.rawbuf
  %aux = llvm.mlir.constant(0 : i32) : i32
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v1f32(<1 x i32> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}})
  rocdl.raw.buffer.store %vdata1, %rsrc, %offset, %soffset, %aux : vector<1xi32>
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v2f32(<2 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}})
  rocdl.raw.buffer.store %vdata2, %rsrc, %offset, %soffset, %aux : vector<2xi32>
  // CHECK: call void @llvm.amdgcn.raw.buffer.store.v4f32(<4 x float> %{{.*}}, <4 x i32> %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 {{.*}})
  rocdl.raw.buffer.store %vdata4, %rsrc, %offset, %soffset, %aux : vector<4xi32>

  llvm.return
}

}
