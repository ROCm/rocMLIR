// `backwardDataKernelIds` enumerates the per-stride phase GEMMs that make up
// a single strided backwards-data conv. A phase whose filter slice is empty
// (i.e. some `iTilda[i] >= filterDims[i]`) must be excluded; otherwise we
// emit a phantom GEMM with a degenerate or wrap-around K dimension.
//
// Two cooperating fixes guarantee that on this branch:
//   1. `backwardDataKernelIds` short-circuits `gemmKproduct` to 0 the moment
//      it sees `iTilda[i] >= filterDims[i]`. This is needed because the
//      slice extent is now computed with `llvm::divideCeil` (the unsigned
//      overload) -- without the explicit guard a negative numerator would
//      wrap to a huge positive value and the empty phase would be retained.
//      The previous rocMLIR code relied on signed `math_util::integer_divide_ceil`
//      to produce 0 for those numerators, which masked the missing guard.
//   2. The `usesV4R1=false` arm of `commonConvRewrite` in `ConvToGemm.cpp`
//      now iterates the actual filtered ids (`for (int64_t kernelId : kernelIds)`)
//      instead of the loop index (`for (size_t i = 0; i < kernelIds.size(); ++i)`).
//      The pre-fix loop passed `i` as the kernel id, so e.g. ids `{0, 3}` were
//      lowered as `{0, 1}`, silently emitting a degenerate GEMM for id 1.
//
// We pin the lowered IR by walking each emitted `rock.gemm` in order with a
// `CHECK-NOT: rock.gemm` between consecutive ids: the in-order matches pin the
// `kernelId` set (catching the index/id mismatch), and the interleaved
// CHECK-NOTs forbid any extra phantom-phase gemm anywhere in between or after.

// Rank-2 reproducer for the originally failing shape: filTilda = {2, 3},
// filterDims = {2, 1}. The valid kernel ids are {0, 3}; ids {1, 2, 4, 5}
// all have `iTilda[i] >= filterDims[i]` for some i and must be skipped.
// RUN: rocmlir-gen --operation=conv_bwd_data --arch %arch -v4r1 0 --kernel_id 0 \
// RUN:   --fil_layout=gkcyx --in_layout=ngchw --out_layout=ngkhw \
// RUN:   --batchsize=1 --groupsize=1 --in_channels=4 --out_channels=4 \
// RUN:   --in_h=4 --in_w=8 --fil_h=2 --fil_w=1 \
// RUN:   --dilation_h=1 --dilation_w=2 \
// RUN:   --conv_stride_h=2 --conv_stride_w=3 \
// RUN:   --padding_h=0 --padding_w=0 \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 \
// RUN: | FileCheck %s --check-prefix=RANK2

// Rank-3 reproducer exercising the rank-3 switch arm of
// `backwardDataKernelIds`. `rocmlir-gen` packs the spatial dims in
// `[h, w, d]` order (depth is appended last in `parseConvDims`), so for
// `--fil_layout=gkc012` / `--in_layout=ngc012` / `--out_layout=ngk012`:
//   filterDims  = [fil_h, fil_w, fil_d]      = [2, 1, 2]
//   strideDims  = [stride_h, stride_w, stride_d] = [2, 3, 2]
//   dilationDims = [dilation_h, dilation_w, dilation_d] = [1, 2, 1]
// which gives filTilda = [2, 3, 2] (12 candidate ids). Walking ids 0..11
// and rejecting any phase with `iTilda[i] >= filterDims[i]` in any dim
// leaves valid ids `{0, 1, 6, 7}`; the other 8 ids all have
// `iTilda[1] >= 1` (since fil_w = 1 makes any non-zero w phase empty).
// RUN: rocmlir-gen --operation=conv_bwd_data --arch %arch -v4r1 0 --kernel_id 0 \
// RUN:   --fil_layout=gkc012 --in_layout=ngc012 --out_layout=ngk012 \
// RUN:   --batchsize=1 --groupsize=1 --in_channels=4 --out_channels=4 \
// RUN:   --in_d=4 --in_h=4 --in_w=8 --fil_d=2 --fil_h=2 --fil_w=1 \
// RUN:   --dilation_d=1 --dilation_h=1 --dilation_w=2 \
// RUN:   --conv_stride_d=2 --conv_stride_h=2 --conv_stride_w=3 \
// RUN:   --padding_d=0 --padding_h=0 --padding_w=0 \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 \
// RUN: | FileCheck %s --check-prefix=RANK3

// Pin each emitted rock.gemm's kernelId in order, with CHECK-NOT in
// between to forbid extra gemms. This composition simultaneously pins:
//   - the gemm count (extra gemms would trip a CHECK-NOT),
//   - the exact kernelId set (missing or extra ids fail the next CHECK), and
//   - the index/id mismatch fix (the pre-fix loop emitted kernelIds 0 and 1
//     instead of {0, 3}, so the second CHECK for kernelId = 3 would fail).
// RANK2: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 0 : index
// RANK2-NOT: {{rock\.gemm[[:>:]]}}
// RANK2: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 3 : index
// RANK2-NOT: {{rock\.gemm[[:>:]]}}

// RANK3: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 0 : index
// RANK3-NOT: {{rock\.gemm[[:>:]]}}
// RANK3: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 1 : index
// RANK3-NOT: {{rock\.gemm[[:>:]]}}
// RANK3: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 6 : index
// RANK3-NOT: {{rock\.gemm[[:>:]]}}
// RANK3: {{rock\.gemm[[:>:]]}}{{.*}}kernelId = 7 : index
// RANK3-NOT: {{rock\.gemm[[:>:]]}}

// V4R1=true validation: when usesV4R1 is true, `commonConvRewrite` does not
// iterate `backwardDataKernelIds`. Instead it dispatches a single GEMM for
// the user-supplied `kernelId`. That id must still land on a non-empty
// stride phase; otherwise `backwardDataV4R1`'s `llvm::divideCeil` (unsigned)
// would wrap a negative numerator into a multi-exabyte slice extent.
// `commonConvRewrite` validates the id against `backwardDataKernelIds(...)`
// and emits an op error before reaching the slice math.
// Same rank-2 config as above (valid ids {0, 3}); --kernel_id 2 is empty.
// RUN: rocmlir-gen --operation=conv_bwd_data --arch %arch -v4r1 1 --kernel_id 2 \
// RUN:   --fil_layout=gkcyx --in_layout=ngchw --out_layout=ngkhw \
// RUN:   --batchsize=1 --groupsize=1 --in_channels=4 --out_channels=4 \
// RUN:   --in_h=4 --in_w=8 --fil_h=2 --fil_w=1 \
// RUN:   --dilation_h=1 --dilation_w=2 \
// RUN:   --conv_stride_h=2 --conv_stride_w=3 \
// RUN:   --padding_h=0 --padding_w=0 \
// RUN: | not rocmlir-driver -c 2>&1 \
// RUN: | FileCheck %s --check-prefix=V4R1_EMPTY_PHASE

// V4R1_EMPTY_PHASE: error: 'rock.conv_bwd_data' op v4r1 kernel id 2 has an empty filter slice and cannot be lowered
// V4R1_EMPTY_PHASE-SAME: valid v4r1 kernel ids for this convolution shape are {0, 3}
