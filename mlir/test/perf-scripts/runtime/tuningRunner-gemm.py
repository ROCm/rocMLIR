# Using a tiny GEMM with --debug-quick-tune-data. This emits a `.debug`
# TSV of the per-config table entries (PerfConfig + TFlops) but, unlike the
# full ``--debug`` flag, omits the heavy per-iteration ``MeasurementsMs``
# arrays. Verify the debug file is produced, has the expected header and
# per-config rows, and that the measurements column is absent.
#
# tuningRunner.py drives real GPU tuning, so it needs the ROCm runner / GPU
# runtime.
# REQUIRES: rocm-runner
# RUN: rm -f %t2.tsv %t2.tsv.state %t2.tsv.debug
# RUN: tuningRunner.py --op gemm --tuning-space=quick --debug-quick-tune-data \
# RUN:     --config='-g 1 -m 64 -n 64 -k 64 -t f32 -out_datatype f32 -transA 0 -transB 0' \
# RUN:     -q -o %t2.tsv
# RUN: FileCheck %s --check-prefix=DEBUG --implicit-check-not=MeasurementsMs < %t2.tsv.debug
#
# DEBUG: PerfConfig{{.*}}TFlops
# DEBUG: {{v[0-9]+:}}
