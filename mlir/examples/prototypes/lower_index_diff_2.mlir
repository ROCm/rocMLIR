func @miopen_lower_index_diff_some_dynamic(%arg0 : index, %arg1 : index, %arg2 : index) -> (index, index, index, index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index

  %u0, %l0, %l1, %l2 = miopen.lower_index_diff(%c1, %c3, %c4, %arg0, %arg1, %arg2) { map = affine_map<(d0) -> (d0 floordiv 9, (d0 mod 9) floordiv 3, (d0 mod 9) mod 3)>, bound = [1024, 3, 3] } : index, index, index, index, index, index

  return %u0, %l0, %l1, %l2 : index, index, index, index
}
