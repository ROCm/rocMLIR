//===-- solver_test.cpp - Unit tests for WinogradSolver -------------------===//
//
// Part of the rocMLIR Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

// RUN: winograd-solver-test | FileCheck %s

#include "mlir/Dialect/Rock/Winograd/WinogradArgLayout.h"
#include "mlir/Dialect/Rock/Winograd/WinogradConvProblem.h"
#include "mlir/Dialect/Rock/Winograd/WinogradSolver.h"
#include <cstdio>

using namespace mlir::rock::winograd;

static WinogradConvProblem
makeBasicProblem(const char *arch, int64_t N, int64_t C, int64_t H, int64_t W,
                 int64_t K, int64_t R, int64_t S, bool fp16, bool fp32,
                 bool bf16, int64_t numCU = 120, int64_t stride = 1,
                 int64_t dilation = 1, int64_t group = 1) {
  WinogradConvProblem p;
  p.arch = arch;
  p.N = N;
  p.C = C;
  p.H = H;
  p.W = W;
  p.K = K;
  p.R = R;
  p.S = S;
  p.padH = R / 2;
  p.padW = S / 2;
  p.strideH = stride;
  p.strideW = stride;
  p.dilationH = dilation;
  p.dilationW = dilation;
  p.outH = (H + 2 * p.padH - (dilation * (R - 1) + 1)) / stride + 1;
  p.outW = (W + 2 * p.padW - (dilation * (S - 1) + 1)) / stride + 1;
  p.groupCount = group;
  p.numCU = numCU;
  p.isFp16 = fp16;
  p.isFp32 = fp32;
  p.isBf16 = bf16;
  p.isXnackEnabled = false;
  p.direction = WinogradDirection::Forward;
  return p;
}

int main() {
  // ============================================
  // POSITIVE TESTS - should be applicable
  // ============================================

  // CHECK: PASS: 3x3_fp32_gfx942
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx942\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp16_gfx942
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, true, false,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp16_gfx942\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_bf16_gfx942
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, false,
                              true, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_bf16_gfx942\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp32_gfx900
  {
    auto p = makeBasicProblem("gfx900", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 60);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx900\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp32_gfx906
  {
    auto p = makeBasicProblem("gfx906", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 60);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx906\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp32_gfx908
  {
    auto p = makeBasicProblem("gfx908", 1, 256, 14, 14, 256, 3, 3, false, true,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx908\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp16_gfx1100
  {
    auto p = makeBasicProblem("gfx1100", 1, 128, 28, 28, 128, 3, 3, true,
                              false, false, 96);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp16_gfx1100\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp32_gfx1200
  {
    auto p = makeBasicProblem("gfx1200", 1, 512, 7, 7, 512, 3, 3, false, true,
                              false, 96);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx1200\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_large_batch
  {
    auto p = makeBasicProblem("gfx942", 64, 128, 28, 28, 128, 3, 3, false,
                              true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_large_batch\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_large_channels
  {
    auto p = makeBasicProblem("gfx942", 1, 1024, 14, 14, 1024, 3, 3, false,
                              true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_large_channels\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp16_gfx942_stride2
  // V30 accepts stride 1-2 on gfx9; Rage rejects stride != 1 but V30 covers it
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, true, false,
                              false, 120, 2);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp16_gfx942_stride2\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: 3x3_fp32_gfx942_full_arch_string
  // Verify that "amdgcn-amd-amdhsa:gfx942" is handled identically to "gfx942"
  {
    auto p = makeBasicProblem("amdgcn-amd-amdhsa:gfx942", 1, 64, 56, 56, 64, 3,
                              3, false, true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 3x3_fp32_gfx942_full_arch_string\n", ok ? "PASS" : "FAIL");
  }

  // ============================================
  // NEGATIVE TESTS - should NOT be applicable
  // ============================================

  // CHECK: PASS_NEG: 5x5_filter
  // R=5 > 3: V30 rejects (R!=3), Rage rejects (R>3)
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 5, 5, false, true,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 5x5_filter\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: 7x7_filter
  // R=7 > 3: all families reject
  {
    auto p = makeBasicProblem("gfx942", 1, 3, 224, 224, 64, 7, 7, false, true,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 7x7_filter\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: 1x1_filter_gfx908
  // R=1,S=1: V30 rejects (R!=3). On gfx908 Rage/Fury are not applicable.
  // Note: gfx942 would accept 1x1 through Rage_V4_9 (which allows R<=3).
  {
    auto p = makeBasicProblem("gfx908", 1, 64, 56, 56, 64, 1, 1, false, true,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: 1x1_filter_gfx908\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: dilation_2
  // All families require dilation == 1
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120, 1, 2);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: dilation_2\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: stride_3
  // stride > 2: V30 rejects (stride must be 1-2), Rage rejects (stride != 1)
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120, 3);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: stride_3\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: grouped_conv
  // All families require groupCount == 1
  {
    auto p = makeBasicProblem("gfx908", 1, 128, 28, 28, 128, 3, 3, false, true,
                              false, 120, 1, 1, 4);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: grouped_conv\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: unsupported_arch
  // gfx803 is not supported by any Winograd family
  {
    auto p = makeBasicProblem("gfx803", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 60);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: unsupported_arch\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: backward_weight
  // V30 only accepts Forward/BackwardData; Rage only accepts Forward
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    p.direction = WinogradDirection::BackwardWeight;
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: backward_weight\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: unsupported_dtype
  // No dtype flag set (simulates int8 or other non-Winograd type)
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, false,
                              false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: unsupported_dtype\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // CHECK: PASS_NEG: huge_dimensions
  // H=70000 overflows the 16-bit shader constraint in all families
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 70000, 70000, 64, 3, 3, false,
                              true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: huge_dimensions\n", !ok ? "PASS_NEG" : "FAIL_NEG");
  }

  // ============================================
  // SELECTION TESTS
  // ============================================

  // CHECK: SELECT: best_for_gfx942
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel.has_value()) {
      printf(
          "SELECT: best_for_gfx942 family=%d nGroups=%lld wti=%.3f file=%s\n",
          (int)sel->family, (long long)sel->nGroups, sel->wti,
          sel->kernelFile.c_str());
    } else {
      printf("FAIL: no selection for gfx942\n");
    }
  }

  // CHECK: SELECT: best_for_gfx1100_fp16
  {
    auto p = makeBasicProblem("gfx1100", 1, 128, 28, 28, 128, 3, 3, true,
                              false, false, 96);
    auto sel = WinogradSolver::selectBest(p);
    if (sel.has_value()) {
      printf("SELECT: best_for_gfx1100_fp16 family=%d nGroups=%lld file=%s\n",
             (int)sel->family, (long long)sel->nGroups,
             sel->kernelFile.c_str());
    } else {
      printf("FAIL: no selection for gfx1100 fp16\n");
    }
  }

  // CHECK: SELECT: no_selection_for_5x5
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 5, 5, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (!sel.has_value()) {
      printf("SELECT: no_selection_for_5x5\n");
    } else {
      printf("FAIL: unexpected selection for 5x5\n");
    }
  }

  // ============================================
  // PERF CONFIG SERIALIZATION TESTS
  // ============================================

  // CHECK: PERFCONFIG: roundtrip config=winograd:v1,
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel.has_value()) {
      std::string cfg = WinogradSolver::toPerfConfigStr(*sel);
      printf("PERFCONFIG: roundtrip config=%s\n", cfg.c_str());

      // CHECK: PERFCONFIG: resolved family=
      auto resolved = WinogradSolver::resolveFromPerfConfig(p, cfg);
      if (resolved.has_value()) {
        printf("PERFCONFIG: resolved family=%d nGroups=%lld\n",
               (int)resolved->family, (long long)resolved->nGroups);
      } else {
        printf("FAIL: roundtrip resolve failed\n");
      }
    }
  }

  // CHECK: PERFCONFIG: reject_bad_prefix
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved =
        WinogradSolver::resolveFromPerfConfig(p, "gemm:v3,32,256");
    if (!resolved.has_value()) {
      printf("PERFCONFIG: reject_bad_prefix\n");
    } else {
      printf("FAIL: accepted bad prefix\n");
    }
  }

  // CHECK: PERFCONFIG: reject_truncated
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved = WinogradSolver::resolveFromPerfConfig(p, "winograd:v1,V30");
    if (!resolved.has_value()) {
      printf("PERFCONFIG: reject_truncated\n");
    } else {
      printf("FAIL: accepted truncated config\n");
    }
  }

  // CHECK: PERFCONFIG: reject_bad_family
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved = WinogradSolver::resolveFromPerfConfig(
        p, "winograd:v1,BadFamily,64,default,fp32");
    if (!resolved.has_value()) {
      printf("PERFCONFIG: reject_bad_family\n");
    } else {
      printf("FAIL: accepted bad family name\n");
    }
  }

  // ============================================
  // XNACK REJECTION TESTS
  // ============================================

  // CHECK: PASS: xnack_rejects_v30_gfx942
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    p.isXnackEnabled = true;
    auto sel = WinogradSolver::selectBest(p);
    // V30 on gfx942 requires xnack off; Rage should still work
    printf("PASS: xnack_rejects_v30_gfx942\n");
  }

  // CHECK: PASS: xnack_v21_rejected
  {
    auto p = makeBasicProblem("gfx906", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 60);
    p.isXnackEnabled = true;
    bool ok = WinogradSolver::isApplicable(p);
    // V21 on gfx906 with xnack should be rejected
    printf("%s: xnack_v21_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // ============================================
  // NCHW LAYOUT REJECTION TESTS
  // ============================================

  // CHECK: PASS: nhwc_rejected
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    p.isNCHW = false;
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: nhwc_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // ============================================
  // GFX10 ARCHITECTURE TESTS
  // ============================================

  // CHECK: PASS: gfx1030_fp32_v30
  {
    auto p = makeBasicProblem("gfx1030", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 36);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: gfx1030_fp32_v30\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: gfx1030_fp16_v30
  {
    auto p = makeBasicProblem("gfx1030", 1, 64, 56, 56, 64, 3, 3, true, false,
                              false, 36);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: gfx1030_fp16_v30\n", ok ? "PASS" : "FAIL");
  }

  // ============================================
  // GFX12 ARCHITECTURE TESTS
  // ============================================

  // CHECK: PASS: gfx1200_fp16_rage_v4_9
  {
    auto p = makeBasicProblem("gfx1200", 1, 64, 56, 56, 64, 3, 3, true, false,
                              false, 60);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: gfx1200_fp16_rage_v4_9\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: gfx1200_fp32_rejected_by_rage
  {
    // gfx12 + fp32: Rage_V4_9 rejects fp32 on gfx12 (only fp16 on gfx12)
    // V40 should still accept fp32 on gfx12
    auto p = makeBasicProblem("gfx1200", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 60);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: gfx1200_fp32_rejected_by_rage\n", ok ? "PASS" : "FAIL");
  }

  // ============================================
  // BF16 TESTS
  // ============================================

  // CHECK: PASS: bf16_gfx942_rage_applicable
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, false,
                              true, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: bf16_gfx942_rage_applicable\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: bf16_gfx942_selection_name
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, false,
                              true, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel && sel->kernelFile.find("bf16") != std::string::npos) {
      printf("PASS: bf16_gfx942_selection_name file=%s\n",
             sel->kernelFile.c_str());
    } else {
      printf("FAIL: bf16 selection missing or wrong filename\n");
    }
  }

  // CHECK: PASS: bf16_gfx908_rejected
  {
    // bf16 not supported on gfx908 (only fp16/fp32 via V30)
    auto p = makeBasicProblem("gfx908", 1, 64, 56, 56, 64, 3, 3, false, false,
                              true, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: bf16_gfx908_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // ============================================
  // BACKWARD DATA DIRECTION TESTS
  // ============================================

  // CHECK: PASS: backward_data_gfx942_applicable
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    p.direction = WinogradDirection::BackwardData;
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: backward_data_gfx942_applicable\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: backward_weight_rejected
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    p.direction = WinogradDirection::BackwardWeight;
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: backward_weight_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // ============================================
  // WTI VALUE TESTS
  // ============================================

  // CHECK: PASS: wti_positive_value
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel && sel->wti > 0.0f) {
      printf("PASS: wti_positive_value wti=%.4f\n", sel->wti);
    } else {
      printf("FAIL: wti not positive\n");
    }
  }

  // CHECK: PASS: wti_larger_spatial_better
  {
    auto p1 = makeBasicProblem("gfx942", 1, 64, 14, 14, 64, 3, 3, false, true,
                               false, 120);
    auto p2 = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                               false, 120);
    auto s1 = WinogradSolver::selectBest(p1);
    auto s2 = WinogradSolver::selectBest(p2);
    if (s1 && s2 && s2->wti >= s1->wti) {
      printf("PASS: wti_larger_spatial_better 14x14=%.4f 56x56=%.4f\n",
             s1->wti, s2->wti);
    } else {
      printf("FAIL: expected larger spatial to have >= WTI\n");
    }
  }

  // ============================================
  // DIMENSION OVERFLOW TESTS
  // ============================================

  // CHECK: PASS: huge_H_rejected
  {
    // H > 2^16 should fail fitsInBits for Rage
    auto p = makeBasicProblem("gfx942", 1, 64, 70000, 70000, 64, 3, 3, false,
                              true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: huge_H_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: huge_batch_product_rejected
  {
    // N*C*H*W > 2^31 should fail product4FitsInBits
    auto p = makeBasicProblem("gfx942", 256, 1024, 256, 256, 64, 3, 3, false,
                              true, false, 120);
    bool ok = WinogradSolver::isApplicable(p);
    printf("%s: huge_batch_product_rejected\n", !ok ? "PASS" : "FAIL");
  }

  // ============================================
  // PERF CONFIG EDGE CASES
  // ============================================

  // CHECK: PERFCONFIG: reject_negative_ngroups
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved = WinogradSolver::resolveFromPerfConfig(
        p, "winograd:v1,RageV4_9,-5,default,fp32");
    printf("PERFCONFIG: %s\n",
           !resolved.has_value() ? "reject_negative_ngroups"
                                 : "FAIL_accepted_negative");
  }

  // CHECK: PERFCONFIG: reject_nonnumeric_ngroups
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved = WinogradSolver::resolveFromPerfConfig(
        p, "winograd:v1,RageV4_9,abc,default,fp32");
    printf("PERFCONFIG: %s\n",
           !resolved.has_value() ? "reject_nonnumeric_ngroups"
                                 : "FAIL_accepted_nonnumeric");
  }

  // CHECK: PERFCONFIG: reject_wrong_version
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto resolved = WinogradSolver::resolveFromPerfConfig(
        p, "winograd:v99,RageV4_9,120,default,fp32");
    printf("PERFCONFIG: %s\n",
           !resolved.has_value() ? "reject_wrong_version"
                                 : "FAIL_accepted_wrong_version");
  }

  // CHECK: PERFCONFIG: bf16_roundtrip
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, false,
                              true, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel) {
      std::string cfg = WinogradSolver::toPerfConfigStr(*sel);
      auto resolved = WinogradSolver::resolveFromPerfConfig(p, cfg);
      if (resolved && resolved->kernelFile.find("bf16") != std::string::npos) {
        printf("PERFCONFIG: bf16_roundtrip config=%s\n", cfg.c_str());
      } else {
        printf("FAIL: bf16 roundtrip failed\n");
      }
    } else {
      printf("FAIL: bf16 selection failed\n");
    }
  }

  // ============================================
  // SELECTION DETAILS TESTS
  // ============================================

  // CHECK: PASS: rage_v4_9_block_size_768
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel && sel->family == WinogradFamily::Rage_V4_9 &&
        sel->blockSize == 768) {
      printf("PASS: rage_v4_9_block_size_768\n");
    } else if (sel) {
      printf("FAIL: unexpected blockSize=%lld family=%d\n",
             (long long)sel->blockSize, (int)sel->family);
    } else {
      printf("FAIL: no selection\n");
    }
  }

  // CHECK: PASS: rage_v4_9_abi_v2
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, true, false,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel && sel->abiVersion == 2) {
      printf("PASS: rage_v4_9_abi_v2\n");
    } else {
      printf("FAIL: expected abiVersion=2\n");
    }
  }

  // CHECK: PASS: v30_gfx908_abi_v1
  {
    auto p = makeBasicProblem("gfx908", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto sel = WinogradSolver::selectBest(p);
    if (sel && sel->abiVersion == 1) {
      printf("PASS: v30_gfx908_abi_v1\n");
    } else {
      printf("FAIL: expected abiVersion=1\n");
    }
  }

  // ============================================
  // WINOGRAD ARG LAYOUT TESTS
  // ============================================

  // CHECK: PASS: v1_layout_size_248
  {
    auto layout = WinogradArgLayout::createV1();
    printf("%s: v1_layout_size_248 (got %lld)\n",
           layout.getTotalSize() == 248 ? "PASS" : "FAIL",
           (long long)layout.getTotalSize());
  }

  // CHECK: PASS: v2_layout_size_232
  {
    auto layout = WinogradArgLayout::createV2();
    printf("%s: v2_layout_size_232 (got %lld)\n",
           layout.getTotalSize() == 232 ? "PASS" : "FAIL",
           (long long)layout.getTotalSize());
  }

  // CHECK: PASS: v1_has_3_pointer_slots
  {
    auto layout = WinogradArgLayout::createV1();
    auto slots = layout.getPointerSlots();
    printf("%s: v1_has_3_pointer_slots (got %zu)\n",
           slots.size() == 3 ? "PASS" : "FAIL", slots.size());
  }

  // CHECK: PASS: v2_has_3_pointer_slots
  {
    auto layout = WinogradArgLayout::createV2();
    auto slots = layout.getPointerSlots();
    printf("%s: v2_has_3_pointer_slots (got %zu)\n",
           slots.size() == 3 ? "PASS" : "FAIL", slots.size());
  }

  // CHECK: PASS: v2_pointer_offsets_32_40_48
  {
    auto layout = WinogradArgLayout::createV2();
    auto slots = layout.getPointerSlots();
    bool ok = slots.size() == 3 && slots[0].offset == 32 &&
              slots[1].offset == 40 && slots[2].offset == 48;
    printf("%s: v2_pointer_offsets_32_40_48\n", ok ? "PASS" : "FAIL");
  }

  // CHECK: PASS: strides_nchw_correct
  {
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    auto st = WinogradArgLayout::computeStrides(p);
    // NCHW strides in elements: d_H=W=56, d_C=H*W=3136, d_N=C*H*W=200704
    bool ok = st.d_H == 56 && st.d_C == 3136 && st.d_N == 200704 &&
              st.f_R == 3 && st.f_C == 9 && st.f_K == 576 &&
              st.o_H == 56 && st.o_K == 3136 && st.o_N == 200704;
    printf("%s: strides_nchw_correct d_H=%u d_C=%u d_N=%u\n",
           ok ? "PASS" : "FAIL", st.d_H, st.d_C, st.d_N);
  }

  // CHECK: PASS: flags_v1_forward
  {
    uint32_t flags = WinogradArgLayout::computeFlagsV1(true);
    // Forward: no REVERSE bits set, but NKCHR_STRIDES + TENSOR_OFFSETS
    bool ok = (flags & (1 << 9)) != 0; // NKCHR_STRIDES
    printf("%s: flags_v1_forward flags=0x%x\n", ok ? "PASS" : "FAIL", flags);
  }

  // CHECK: PASS: flags_v1_backward
  {
    uint32_t flags = WinogradArgLayout::computeFlagsV1(false);
    // Backward: REVERSE_R | REVERSE_S | FLIP_K_C set
    bool hasReverse = (flags & ((1 << 6) | (1 << 7) | (1 << 8))) != 0;
    printf("%s: flags_v1_backward flags=0x%x\n",
           hasReverse ? "PASS" : "FAIL", flags);
  }

  // CHECK: PASS: flags_v2_forward
  {
    uint64_t flags = WinogradArgLayout::computeFlagsV2(true, false, false, false);
    bool ok = (flags & (1ULL << 15)) != 0; // F_USE_EXTENDED_FLAGS_64
    printf("%s: flags_v2_forward flags=0x%llx\n", ok ? "PASS" : "FAIL",
           (unsigned long long)flags);
  }

  // CHECK: PASS: flags_v2_bias
  {
    uint64_t flags = WinogradArgLayout::computeFlagsV2(true, true, false, false);
    bool hasBias = (flags & (1ULL << 4)) != 0;
    printf("%s: flags_v2_bias flags=0x%llx\n", hasBias ? "PASS" : "FAIL",
           (unsigned long long)flags);
  }

  // CHECK: PASS: buildtemplate_v2_232_bytes
  {
    auto layout = WinogradArgLayout::createV2();
    auto p = makeBasicProblem("gfx942", 1, 64, 56, 56, 64, 3, 3, false, true,
                              false, 120);
    uint32_t flags = 0;
    auto buf = layout.buildTemplate(p, 120, flags);
    printf("%s: buildtemplate_v2_232_bytes (got %zu)\n",
           buf.size() == 232 ? "PASS" : "FAIL", buf.size());
  }

  return 0;
}
