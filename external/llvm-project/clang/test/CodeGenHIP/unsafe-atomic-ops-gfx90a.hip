// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -munsafe-fp-atomics -target-cpu gfx90a -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a -x hip -emit-llvm -fcuda-is-device -o - %s | FileCheck %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

// CHECK-LABEL: @_Z15unsafeAtomicAddPff(ptr %addr, float %value
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
  // CHECK: %[[ADDR_ADDR:.*]] = alloca ptr, align 8, addrspace(5)
  // CHECK: %[[ADDR_ADDR_ASCAST:.*]] = addrspacecast ptr addrspace(5) %[[ADDR_ADDR]] to ptr 
  // CHECK: %[[ADDR_PTR:.*]] = load ptr, ptr %[[ADDR_ADDR_ASCAST]], align 8
  // CHECK: %[[ADDR:.*]] = addrspacecast ptr %[[ADDR_PTR]] to ptr addrspace(3)
  // CHECK: call contract float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %[[ADDR]]
  return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
}

__global__ void test_global_atomic_add_f32(float *val){
  float *rtn;
  *rtn = unsafeAtomicAdd(val, 1.0);
}
