# End-to-end smoke test for tuningRunner.py: tune a tiny f32 GEMM with
# ``--tuning-space=quick``. Drives rocmlir-gen | rocmlir-tuning-driver and
# verifies the winning perf-config against the CPU reference, exercising
# every Python-side step (config parsing, subprocess orchestration, result
# parsing) end-to-end.
#
# Remove any stale tuning DB / state file from a previous run; the new
# tuningRunner caches already-tuned and previously-failed configs by test
# vector, so a leftover tsv or `<tsv>.state` file would short-circuit the
# run and produce empty output for FileCheck.
# RUN: rm -f %t.tsv %t.tsv.state
# RUN: tuningRunner.py --op gemm --tuning-space=quick \
# RUN:     --config='-g 1 -m 64 -n 64 -k 64 -t f32 -out_datatype f32 -transA 0 -transB 0' \
# RUN:     -q -o %t.tsv 2>&1 | FileCheck %s
#
# CHECK: Tuned and verified
# CHECK-SAME: gemm:v2:
#
# Same tiny GEMM, now with ``--debug-quick-tune-data``. This emits a `.debug`
# TSV of the per-config table entries (PerfConfig + TFlops) but, unlike the
# full ``--debug`` flag, omits the heavy per-iteration ``MeasurementsMs``
# arrays. Verify the debug file is produced and that the measurements column
# is absent.
# RUN: rm -f %t2.tsv %t2.tsv.state %t2.tsv.debug
# RUN: tuningRunner.py --op gemm --tuning-space=quick --debug-quick-tune-data \
# RUN:     --config='-g 1 -m 64 -n 64 -k 64 -t f32 -out_datatype f32 -transA 0 -transB 0' \
# RUN:     -q -o %t2.tsv 2>&1 | FileCheck %s
# RUN: FileCheck %s --check-prefix=DEBUG --implicit-check-not=MeasurementsMs < %t2.tsv.debug
#
# DEBUG: PerfConfig{{.*}}TFlops
# DEBUG: gemm:v2:
