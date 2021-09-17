// XFAIL: *
// RUN: mlir-miopen-driver -c --batchsize 128 --in_channels 256 --in_h 256 \
// RUN: --in_w 256 --out_channels 4 --fil_w 3 --fil_h 3
