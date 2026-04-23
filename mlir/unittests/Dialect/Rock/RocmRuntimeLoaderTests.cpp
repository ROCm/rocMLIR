//===- RocmRuntimeLoaderTests.cpp - tests for the ROCm runtime loader ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These tests document the cross-process coordination contract of
// `mlir::rocm_loader::loadRocmLibrary`: any second consumer of HIP in
// the same process must observe the same handle as `RocmSystemDetect`,
// otherwise KFD's "one HSA session per process" rule produces
// `hipErrorNoDevice` for one of them.
//
// The tests are runtime-conditional: when no HIP runtime is installed
// (CI without `libamdhip64.so` on the loader path), they GTEST_SKIP
// instead of failing.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/RocmRuntimeLoader.h"
#include "mlir/ExecutionEngine/RocmSystemDetect.h"

#include "gtest/gtest.h"

using mlir::rocm_loader::CoordinationPolicy;
using mlir::rocm_loader::Library;
using mlir::rocm_loader::LoadedLibrary;
using mlir::rocm_loader::loadRocmLibrary;
using mlir::rocm_loader::resolveRocmSymbol;

namespace {

// Touching `RocmSystemDetect::get()` forces it (the canonical owner) to
// load HIP in its own link-map namespace and publishes
// `mlirRocmSystemDetectGetHipHandle` for subsequent loaders to find.
//
// We do this in a SetUp helper rather than in the constructor so the
// test can GTEST_SKIP cleanly when HIP is unavailable.
class RocmRuntimeLoaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    (void)mlir::RocmSystemDetect::get();
    sharedHandle = mlirRocmSystemDetectGetHipHandle();
    if (!sharedHandle)
      GTEST_SKIP() << "no HIP runtime available; skipping loader contract "
                      "tests";
  }

  void *sharedHandle = nullptr;
};

// `CoordinationPolicy::Auto` for `Library::Hip` MUST return the same
// handle as `RocmSystemDetect`. If a future change accidentally opens a
// second HIP namespace via `dlmopen`, KFD will start handing out
// `hipErrorNoDevice` and large parts of the JIT path will break.
TEST_F(RocmRuntimeLoaderTest, AutoPolicyReusesSystemDetectHandle) {
  LoadedLibrary lib = loadRocmLibrary(Library::Hip);
  ASSERT_NE(nullptr, lib.handle)
      << "loadRocmLibrary(Hip) returned a null handle even though "
         "RocmSystemDetect successfully loaded HIP";
  EXPECT_EQ(sharedHandle, lib.handle)
      << "Auto policy must reuse RocmSystemDetect's HIP handle to keep "
         "the per-process KFD session count at 1";
}

// `CoordinationPolicy::Owned` is reserved for `RocmSystemDetect` itself
// (to break recursion at first load). When called from anywhere else
// it MAY return a fresh handle, but MUST still produce a usable one.
TEST_F(RocmRuntimeLoaderTest, OwnedPolicyAlwaysReturnsAUsableHandle) {
  LoadedLibrary lib = loadRocmLibrary(Library::Hip, /*relatedHandle=*/nullptr,
                                      CoordinationPolicy::Owned);
  ASSERT_NE(nullptr, lib.handle)
      << "loadRocmLibrary(Hip, Owned) failed even though HIP is present";
  // We do NOT assert handle equality here: under glibc, a fresh
  // dlmopen call produces a distinct handle by design. The test only
  // proves that Owned does not regress to a null result.
  void *sym = resolveRocmSymbol(lib, "hipGetDeviceCount");
  EXPECT_NE(nullptr, sym)
      << "An owned HIP handle must be able to resolve hipGetDeviceCount";
}

// Symbol resolution against a null handle must be a soft failure
// (return null), never a crash. This protects all the `if (!fn) return
// false;` fallbacks in the wrapper translation units.
TEST(RocmRuntimeLoaderUnit, ResolveAgainstNullHandleReturnsNullSafely) {
  LoadedLibrary lib;
  EXPECT_EQ(nullptr, resolveRocmSymbol(lib, "anything"));
}

} // namespace
