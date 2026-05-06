// backwardDataKernelIds used to emit phantom kernel IDs for stride phases whose
// filter slice is empty.

// The pre-fix implementation expressed the slice extent as
// `divideCeil(filterDims[i] - iTilda[i], filTilda[i])`. LLVM's default
// `divideCeil` is the unsigned-converting overload, so a negative numerator wrapped to a huge positive value -- making an empty phase look like real GEMM work and emitting a bogus per-phase rock.gemm.
// The fix detects iTilda[i] >= filterDims[i] and short-circuits gemmKproduct to 0 so the phase is correctly excluded.
// We pin the count of `rock.gemm` ops emitted by `rock-conv-to-gemm`, since that pass calls backwardDataKernelIds once per ConvBwdDataOp and emits one rock.gemm per returned kernel ID.

// Rank-2 reproducer for the originally failing shape: filTilda = {2, 3},
// filterDims = {2, 1}. The fix prunes iTilda[1] in {1, 2}, leaving the
// kernel-ID set {0, 3}. Without the fix iTilda[1] == 2 (kernel IDs 2, 5)
// would also slip through with a wrapped slice extent, giving 4 gemms.
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
// backwardDataKernelIds:
// filTilda = {2, 2, 3}, filterDims = {2, 2, 1}, so only iTilda[2]
// (the dim with filterDims[i] < filTilda[i]) can index out of bounds.
// The fix prunes iTilda[2] in {1, 2}, leaving kernel IDs
// {0, 3, 6, 9} == 4 gemms. Without the fix iTilda[2] == 2 would also
// slip through (kernel IDs 2, 5, 8, 11), giving 8 gemms.
// RUN: rocmlir-gen --operation=conv_bwd_data --arch %arch -v4r1 0 --kernel_id 0 \
// RUN:   --fil_layout=gkc012 --in_layout=ngc012 --out_layout=ngk012 \
// RUN:   --batchsize=1 --groupsize=1 --in_channels=4 --out_channels=4 \
// RUN:   --in_d=4 --in_h=4 --in_w=8 --fil_d=2 --fil_h=2 --fil_w=1 \
// RUN:   --dilation_d=1 --dilation_h=1 --dilation_w=2 \
// RUN:   --conv_stride_d=2 --conv_stride_h=2 --conv_stride_w=3 \
// RUN:   --padding_d=0 --padding_h=0 --padding_w=0 \
// RUN: | rocmlir-driver -c --mlir-print-ir-after=rock-conv-to-gemm 2>&1 \
// RUN: | FileCheck %s --check-prefix=RANK3

// RANK2-COUNT-2: {{rock\.gemm[[:>:]]}}
// RANK2-NOT: {{rock\.gemm[[:>:]]}}

// RANK3-COUNT-4: {{rock\.gemm[[:>:]]}}
// RANK3-NOT: {{rock\.gemm[[:>:]]}}
