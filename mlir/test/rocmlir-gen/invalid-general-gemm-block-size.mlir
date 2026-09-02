// RUN: rocmlir-gen --arch gfx1150 --operation=conv -t f32 \
// RUN:   --batchsize=1 --in_channels=256 --out_channels=16 \
// RUN:   --in_h=20 --in_w=20 --fil_h=3 --fil_w=3 \
// RUN:   --padding_h=1 --padding_w=1 --num_cu=6 \
// RUN:   --perf_config=v3:256,32,128,4,2,2,1,1,2 \
// RUN: | not rocmlir-opt -rock-affix-params >/dev/null

// The A tile contains only 4 * 32 = 128 elements for 256 threads, so
// aCopyPerThread would be zero. The config must be rejected while affixing
// tuning parameters, before it reaches GridwiseGemmToBlockwise.

// This file intentionally contains only the command-line regression above.
