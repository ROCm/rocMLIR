// REQUIRES: amdgpu-registered-target
// REQUIRES: x86-registered-target
// XFAIL: *
// FIXME: With -no-opaque-pointers, compiler aorts.
// FIXME: without, we need to update expected results

// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c -triple amdgcn-amd-amdhsa -fopenmp-is-device -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-host-ir-file-path %t-host.bc -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK
// expected-no-diagnostics

extern int printf(const char *, ...);


// CHECK-DAG: [[CMA_PRINTF_ARG_TY:%[a-zA-Z0-9_.]+]] = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }
// CHECK-DAG: [[CMA_ARG_STR:@[a-zA-Z0-9_.]+]] = private unnamed_addr addrspace(4) constant [9 x i8] c"%s %d %s\00", align 1
// CHECK-DAG: [[CMA_ARG_STR_1:@[a-zA-Z0-9_.]+]] = private unnamed_addr addrspace(4) constant [8 x i8] c"testing\00", align 1
// CHECK-DAG: define weak_odr protected amdgpu_kernel void @__omp_offloading{{.+}}CheckMultipleArgs{{.+}}
int CheckMultipleArgs(int a) {
  char *test = "testing";
  char *t;
#pragma omp target private(t)
  {
    // .master block

    // to make sure compiler is not able to calculate size of t
    t = test + a;

    // CHECK: [[CMA_T_STR_LEN:%[a-zA-Z0-9_]+]] = call i32 @__strlen_max(ptr [[CMA_T_STR:%[0-9]+]], i32 1024)
    // CHECK: [[TOTAL_STR_LEN:%[a-zA-Z0-9_]+]] = add i32 0, [[CMA_T_STR_LEN]]
    // CHECK: [[TOTAL_BUFFER_SIZE:%[a-zA-Z0-9_]+]] = add i32 [[TOTAL_STR_LEN]], 57
    // CHECK: [[CMA_ALLOC:%[a-zA-Z0-9_]+]] = call ptr @printf_allocate(i32 [[TOTAL_BUFFER_SIZE]])
    // CHECK: [[CMA_PRINTF_ARGS_CASTED:%[a-zA-Z0-9_]+]] = addrspacecast ptr [[CMA_ALLOC]] to ptr addrspace(1)
    // CHECK: [[CMA_ARG0:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 0
    // CHECK: store i32 40, ptr addrspace(1) [[CMA_ARG0]], align 4
    // CHECK: [[CMA_ARG_NUM:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 1
    // CHECK: store i32 4, ptr addrspace(1) [[CMA_ARG_NUM]], align 4
    // CHECK: [[CMA_ARGT1:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 2
    // CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) [[CMA_ARGT1]], align 4
    // CHECK: [[CMA_ARGT2:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 3
    // CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) [[CMA_ARGT2]], align 4
    // CHECK: [[CMA_ARGT3:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 4
    // CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) [[CMA_ARGT3]], align 4
    // CHECK: [[CMA_ARGT4:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 5
    // CHECK: store i32 {{[0-9]+}}, ptr addrspace(1) [[CMA_ARGT4]], align 4
    // CHECK: [[CMA_ARG1:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 6

    // CHECK: store i32 9, ptr addrspace(1) [[CMA_ARG1]], align 4
    // CHECK: [[CMA_ARG2:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 7
    // CHECK: store i32 [[CMA_T_STR_LEN]], ptr addrspace(1) [[CMA_ARG2]], align 4
    // CHECK: [[CMA_ARG3:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 8
    // CHECK: store i32 21, ptr addrspace(1) [[CMA_ARG3]], align 4
    // CHECK: [[CMA_ARG4:%[a-zA-Z0-9_]+]] = getelementptr inbounds [[CMA_PRINTF_ARG_TY]], ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i32 0, i32 9
    // CHECK: store i32 8, ptr addrspace(1) [[CMA_ARG4]], align 4
    // CHECK: [[CMA_NEXT_COPY:%[a-zA-Z0-9_]+]] = getelementptr inbounds i8, ptr addrspace(1) [[CMA_PRINTF_ARGS_CASTED]], i64 40
    // CHECK: call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 1 [[CMA_NEXT_COPY]], ptr align 1 addrspacecast (ptr addrspace(4) [[CMA_ARG_STR]] to ptr), i64 9, i1 false)
    // CHECK: [[CMA_NEXT_COPY1:%[a-zA-Z0-9_]+]] = getelementptr inbounds i8, ptr addrspace(1) [[CMA_NEXT_COPY]], i64 9
    // CHECK: call void @llvm.memcpy.p1.p0.i32(ptr addrspace(1) align 1 [[CMA_NEXT_COPY1]], ptr align 1 [[CMA_T_STR]], i32 [[CMA_T_STR_LEN]], i1 false)
    // CHECK: [[CMA_NEXT_COPY2:%[a-zA-Z0-9_]+]] = getelementptr i8, ptr addrspace(1) [[CMA_NEXT_COPY1]], i32 [[CMA_T_STR_LEN]]
    // CHECK: call void @llvm.memcpy.p1.p0.i64(ptr addrspace(1) align 1 [[CMA_NEXT_COPY2]], ptr align 1 addrspacecast (ptr addrspace(4) [[CMA_ARG_STR_1]] to ptr), i64 8, i1 false)
    // CHECK: {{.*}} = call i32 @printf_execute(ptr [[CMA_ALLOC]], i32 [[TOTAL_BUFFER_SIZE]])
    printf("%s %d %s", t, 21, test);
  }

  return 0;
}
