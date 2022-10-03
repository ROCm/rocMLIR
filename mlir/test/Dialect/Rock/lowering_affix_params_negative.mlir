// Negative test to deliberately pass incorrect tuning parameters.

// RUN: rocmlir-gen --arch %arch -p -mfma=on -t f16 --perf_config "128,128,2,64,64,2,1,1" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 2}}

// RUN: rocmlir-gen --arch %arch -p -mfma=on -t f16 --perf_config "128,128,2,64,64,16,1,1" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 16}}

// RUN: rocmlir-gen --arch %arch -p -mfma=on -t f32 --perf_config "128,128,8,64,64,8,1,1" | rocmlir-opt -rock-affix-params -verify-diagnostics
// expected-error{{Incorrect KPACK tuning parameter: 8}}
