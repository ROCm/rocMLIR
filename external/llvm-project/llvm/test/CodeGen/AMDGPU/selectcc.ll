; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefixes=SI -check-prefix=FUNC %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=VI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}selectcc_i64:
; SI: v_cmp_eq_u64
; VI: s_cmp_eq_u64
; GCN: s_cselect_b32
define amdgpu_kernel void @selectcc_i64(i64 addrspace(1) * %out, i64 %lhs, i64 %rhs, i64 %true, i64 %false) {
entry:
  %0 = icmp eq i64 %lhs, %rhs
  %1 = select i1 %0, i64 %true, i64 %false
  store i64 %1, ptr addrspace(1) %out
  ret void
}
