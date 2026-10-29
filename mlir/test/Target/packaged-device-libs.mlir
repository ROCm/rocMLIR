// RUN: rocmlir-opt %s --rocdl-attach-target=chip=gfx942 --gpu-module-to-binary 2>&1 \
// RUN: | FileCheck %s --implicit-check-not="Linking two modules"

// The packaged device libraries can be built by a newer LLVM than ours, with a
// different triple spelling and data layout. Linking them must not warn.

// CHECK: gpu.binary @kernels
module attributes {gpu.container_module} {
  gpu.module @kernels {
    llvm.func @__ocml_exp_f32(f32) -> f32
    llvm.func @kernel(%arg0: f32, %arg1: !llvm.ptr<1>) attributes {rocdl.kernel} {
      %0 = llvm.call @__ocml_exp_f32(%arg0) : (f32) -> f32
      llvm.store %0, %arg1 : f32, !llvm.ptr<1>
      llvm.return
    }
  }
}
