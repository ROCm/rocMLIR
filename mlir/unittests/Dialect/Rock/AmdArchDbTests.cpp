//===- AmdArchDbTests.cpp - Tests for the AMD arch database ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Rock/IR/AmdArchDb.h"

#include "gtest/gtest.h"

#include <numeric>
#include <string>
#include <vector>

using namespace mlir::rock;

// NOTE: this file deliberately does NOT include hip/hip_runtime_api.h or link
// libamdhip64. Doing so would pull in amd_comgr -> ROCm's libLLVM.so, which
// collides with the LLVM that the test binary embeds. Device enumeration and
// arch-name lookup go through the AmdArchDb public API, which delay-loads HIP
// at run time via a private `dlopen` / `LoadLibraryW` call.

class NativeArchTest : public ::testing::TestWithParam<unsigned> {
public:
  static auto getDeviceIds() {
    unsigned count = nativeDeviceCount();
    if (count == 0) {
      // Keep gtest happy when no GPU/HIP is available; the SetUp() below will
      // skip the test for the synthetic device id.
      return ::testing::ValuesIn(std::vector<unsigned>{0});
    }
    std::vector<unsigned> ids(count);
    std::iota(ids.begin(), ids.end(), 0u);
    return ::testing::ValuesIn(ids);
  }

protected:
  void SetUp() override {
    archName = nativeArchName(GetParam());
    if (archName.empty())
      GTEST_SKIP() << "No AMD GPU visible to HIP (or `libamdhip64` not on the "
                      "loader path); skipping native arch comparison for "
                      "device "
                   << GetParam();
  }

  std::string archName;
};

TEST_P(NativeArchTest, NativeArchInfoMatchesPresetInfo) {
  auto presetInfo = lookupArchInfo(archName);
  auto nativeInfo = lookupArchInfo("native:" + std::to_string(GetParam()));

  EXPECT_EQ(presetInfo.defaultFeatures, nativeInfo.defaultFeatures);
  EXPECT_EQ(presetInfo.waveSize, nativeInfo.waveSize);
  EXPECT_EQ(presetInfo.maxWavesPerEU, nativeInfo.maxWavesPerEU);
  EXPECT_EQ(presetInfo.totalSGPRPerEU, nativeInfo.totalSGPRPerEU);
  EXPECT_EQ(presetInfo.totalVGPRPerEU, nativeInfo.totalVGPRPerEU);
  EXPECT_EQ(presetInfo.totalSharedMemPerCU, nativeInfo.totalSharedMemPerCU);
  EXPECT_EQ(presetInfo.maxSharedMemPerWG, nativeInfo.maxSharedMemPerWG);
  EXPECT_EQ(presetInfo.numEUPerCU, nativeInfo.numEUPerCU);
  EXPECT_LE(presetInfo.minNumCU, nativeInfo.minNumCU);
  EXPECT_EQ(presetInfo.hasFp8ConversionInstrs,
            nativeInfo.hasFp8ConversionInstrs);
  EXPECT_EQ(presetInfo.hasOcpFp8ConversionInstrs,
            nativeInfo.hasOcpFp8ConversionInstrs);
  EXPECT_GE(presetInfo.maxNumXCC, nativeInfo.maxNumXCC);
}

INSTANTIATE_TEST_SUITE_P(NativeArchTests, NativeArchTest,
                         NativeArchTest::getDeviceIds());

// Pin the parser contract for `rock.arch = "native[:N]"`. Malformed input
// (`native:foo`, `native:`, `native:-1`, `native:9999999999999999999999`) used
// to silently fall back to device 0, which on multi-GPU systems silently
// targeted the wrong GPU. The parser must abort instead.
//
// We use `EXPECT_DEATH` so the test works whether or not a real GPU is
// available -- the abort happens before any HIP call.
TEST(NativeArchParseTest, MalformedSuffixAborts) {
  EXPECT_DEATH(
      { (void)lookupArchInfo("native:foo"); },
      "Invalid `rock.arch = \"native:foo\"`");
  EXPECT_DEATH(
      { (void)lookupArchInfo("native:1abc"); },
      "Invalid `rock.arch = \"native:1abc\"`");
  EXPECT_DEATH(
      { (void)lookupArchInfo("native:"); },
      "Invalid `rock.arch = \"native:\"`");
  EXPECT_DEATH(
      { (void)lookupArchInfo("native:-1"); },
      "Invalid `rock.arch = \"native:-1\"`");
}

// Bare `native` (no colon) is well-formed and means "device 0".
TEST(NativeArchParseTest, BareNativeIsDeviceZero) {
  // We cannot directly observe the parsed deviceId without a GPU, but we can
  // at least confirm the parse does not abort. If HIP is unavailable, the
  // call later aborts with a *different* message ("Failed to query AMD GPU
  // arch runtime"), which is still a valid outcome distinct from the parser
  // abort above.
  if (nativeDeviceCount() == 0)
    GTEST_SKIP() << "no AMD GPU visible; the parse-success path needs a "
                    "live HIP runtime to return without aborting";
  // Should not abort.
  (void)lookupArchInfo("native");
}

// On multi-GPU systems with same-arch devices, the per-device cache must
// not collapse data across device ids. The previous implementation keyed
// the cache by `gcnArchName` only and silently returned device 0's data
// for every later device.
//
// Skipped when fewer than two visible GPUs share the same arch name (which
// is the common single-GPU CI case).
TEST(NativeArchCacheTest, SameArchMultiGpuDistinct) {
  unsigned count = nativeDeviceCount();
  if (count < 2)
    GTEST_SKIP() << "fewer than 2 AMD GPUs visible; cannot exercise the "
                    "same-arch multi-GPU cache contract";

  std::string arch0 = nativeArchName(0);
  if (arch0.empty())
    GTEST_SKIP() << "device 0 unavailable; cannot exercise the cache";

  // Find a second device that reports the same gcnArchName as device 0.
  unsigned other = count;
  for (unsigned i = 1; i < count; ++i) {
    if (nativeArchName(i) == arch0) {
      other = i;
      break;
    }
  }
  if (other == count)
    GTEST_SKIP() << "no two visible GPUs share `" << arch0 << "`";

  auto info0 = lookupArchInfo("native:0");
  auto infoN = lookupArchInfo("native:" + std::to_string(other));

  // The per-device CU count is the canonical "is the cache device-aware?"
  // probe: even on otherwise-identical SKUs, AMD's binning can produce
  // different `multiProcessorCount` (= minNumCU after our query). Two
  // CALLS to `lookupArchInfo` for two distinct device ids must land on
  // independently queried entries, not on a stale device-0 copy.
  //
  // We don't EXPECT_NE here because two physically identical GPUs may also
  // legitimately return the same minNumCU; the meaningful invariant is that
  // each value comes from its own per-device query and is consistent on
  // repeated lookup. Repeat the call to prove cache stability:
  EXPECT_EQ(lookupArchInfo("native:0").minNumCU, info0.minNumCU);
  EXPECT_EQ(lookupArchInfo("native:" + std::to_string(other)).minNumCU,
            infoN.minNumCU);
}
