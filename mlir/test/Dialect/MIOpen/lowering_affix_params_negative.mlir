// Negative test to deliberately pass incorrect tuning parameters.

// RUN: miopen-gen -p -x2 -t f16 --perf_config "128,128,2,64,64,2,1,1" | miopen-opt -miopen-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 2}}

// RUN: miopen-gen -p -x2 -t f16 --perf_config "128,128,2,64,64,16,1,1" | miopen-opt -miopen-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 16}}

// RUN: miopen-gen -p -x2 -t f32 --perf_config "128,128,8,64,64,8,1,1" | miopen-opt -miopen-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 8}}
