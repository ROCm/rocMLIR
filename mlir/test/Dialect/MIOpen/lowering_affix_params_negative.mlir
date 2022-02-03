// XFAIL: *
// REQUIRES: miopen-driver
// RUN: miopen-gen -p -x2 -t f16 --perf_config "128,128,2,64,64,2,1,1" | mlir-miopen-driver -miopen-affix-params
// RUN: miopen-gen -p -x2 -t f16 --perf_config "128,128,2,64,64,16,1,1" | mlir-miopen-driver -miopen-affix-params

// Negative test to deliberately pass incorrect tuning parameters.
