// RUN: rocmlir-opt -convert-rock-to-gpu %s -split-input-file -verify-diagnostics

// expected-error@+1 {{kernel func op 'grid_size_invalid' has an invalid grid size}}
func.func public @grid_size_invalid() attributes {kernel, arch="gfx942", block_size = 256 : i32, grid_size = -100 : i32} {
  return
}
func.func @main() {
  call @grid_size_invalid() : () -> ()
  return
}


// -----

// expected-error@+1 {{kernel func op 'block_size_invalid' has an invalid block size}}
func.func public @block_size_invalid() attributes {kernel, arch="gfx942", block_size = 0 : i32, grid_size = 256 : i32} {
  return
}
func.func @main() {
  call @block_size_invalid() : () -> ()
  return
}

// -----

// expected-error@+1 {{kernel func op 'block_size_missing' is missing the block_size attribute}}
func.func public @block_size_missing() attributes {kernel, arch="gfx942", grid_size = 256 : i32} {
  return
}
func.func @main() {
  call @block_size_missing() : () -> ()
  return
}

// -----

// expected-error@+1 {{kernel func op 'grid_size_missing' is missing the grid_size attribute}}
func.func public @grid_size_missing() attributes {kernel, arch="gfx942", block_size = 256 : i32} {
  return
}
func.func @main() {
  call @grid_size_missing() : () -> ()
  return
}
