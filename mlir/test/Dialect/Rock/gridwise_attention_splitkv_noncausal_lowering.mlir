// Non-causal / non-KV-cache split-KV attention lowering.
//
// This locks in the per-split M-loop iteration math produced by
// GridwiseGemmToBlockwise::getMLoopInfo() for the non-causal split-KV branch.
// The bounds must use ceil-division (itersPerSplit) together with a minui
// clamp to gemm0MBlocks so that:
//   * trailing splits never iterate past the available gemm0M blocks, and
//   * configurations with gemm0MBlocks < splitKV still get >= 1 iteration per
//     split (truncating division would give 0, skipping softmax for every
//     split and producing 0/0 NaNs in scaleFinalOutput).
//
// For this config gemm0MBlocks = 16 and splitKV = 8, so itersPerSplit = 2.
//
// RUN: rocmlir-gen --arch gfx908 --operation attention -seq_len_q 1 -seq_len_k 512 -head_dim_qk 32 -head_dim_v 32 -t f16 -return_lse -split_kv 8 -g 1 | rocmlir-driver -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise | rocmlir-opt -canonicalize | FileCheck %s

// CHECK-LABEL: @rock_attention
// CHECK-DAG: %[[cBlocks:.+]] = arith.constant 16 : index
// CHECK-DAG: %[[cIters:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[cSplit:.+]] = arith.constant 8 : index
// CHECK-DAG: %[[cOne:.+]] = arith.constant 1 : index
// splitBlock = workgroup_id % splitKV
// CHECK: %[[block:.+]] = arith.remui %{{.+}}, %[[cSplit]] : index
// start = splitBlock * itersPerSplit
// CHECK: %[[start:.+]] = arith.muli %[[block]], %[[cIters]] : index
// end = min((splitBlock + 1) * itersPerSplit, gemm0MBlocks)
// CHECK: %[[blockP1:.+]] = arith.addi %[[block]], %[[cOne]] : index
// CHECK: %[[endRaw:.+]] = arith.muli %[[blockP1]], %[[cIters]] : index
// CHECK: %[[end:.+]] = arith.minui %[[endRaw]], %[[cBlocks]] : index
// CHECK: scf.for %{{.+}} = %[[start]] to %[[end]] step %[[cOne]]
