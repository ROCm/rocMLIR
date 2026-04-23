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

// Pin the version-agnosticism contract: if HIP is available at all on
// the host, the loader MUST find it without any compile-time knowledge
// of which ROCm major version is installed. We use the `Auto` policy
// (the path downstream consumers actually take) so we avoid bumping
// into KFD's "one HSA session per process" limit, which would mask a
// true loader failure under a spurious `dlmopen` failure from
// repeated `Owned` calls. Success here means the loader resolved a
// SONAME -- either via the cross-library coordination handle or via
// the bare-name / numeric-fallback enumeration. A future ROCm release
// that bumps the major version should keep this test green without
// code changes (so long as AMD stays at or below
// `kMaxProbedRocmMajor`).
class RocmRuntimeLoaderVersionContract : public ::testing::Test {
protected:
  void SetUp() override {
    (void)mlir::RocmSystemDetect::get();
    if (!mlirRocmSystemDetectGetHipHandle())
      GTEST_SKIP() << "no HIP runtime available on this host";
  }
};

TEST_F(RocmRuntimeLoaderVersionContract, FindsHipWithoutVersionHardcoding) {
  LoadedLibrary lib = loadRocmLibrary(Library::Hip);
  ASSERT_NE(nullptr, lib.handle)
      << "loader could not resolve HIP through its candidate list "
         "(bare name + numeric `.so.<MAJOR>` fallback). Either ROCm is "
         "missing from the dynamic-loader path, or AMD has shipped a "
         "major version above `kMaxProbedRocmMajor` (in which case "
         "bump that constant in RocmRuntimeLoader.cpp)";
  // `hipGetDeviceCount` has been part of the HIP C ABI since ROCm 1.x
  // and is therefore present on every supported major version, which
  // makes it the right symbol to pin "the loaded library is in fact
  // HIP, not some other library that happened to match the SONAME
  // pattern".
  EXPECT_NE(nullptr, resolveRocmSymbol(lib, "hipGetDeviceCount"));
}

} // namespace
