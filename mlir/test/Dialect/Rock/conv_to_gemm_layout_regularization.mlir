// RUN: rocmlir-gen --arch gfx908 -p -operation conv -fil_layout gkyxc -in_layout ngchw -out_layout hwgkn \
// RUN: | rocmlir-opt --rock-affix-params --rock-conv-to-gemm --mlir-print-local-scope \
// RUN: | FileCheck %s --check-prefix=FWD

// FWD: <affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)> by
// FWD-SAME: <PassThrough ["g", "k", "0", "1", "c"] at [0, 1, 3, 4, 2] -> ["g", "k", "0", "1", "c"] at [0, 1, 2, 3, 4]>
// FWD-NEXT: <affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d2, d3, d0)> by
// FWD-SAME: <PassThrough ["0o", "1o", "go", "ko", "no"] at [1, 4, 2, 3, 0] -> ["0o", "1o", "go", "ko", "no"] at [0, 1, 2, 3, 4]>
