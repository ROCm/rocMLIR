// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --linalg-to-rock -verify-diagnostics --split-input-file

// expected-error @+1 {{func op does not have the kernel attribute for linalg-to-rock lowering}}
func.func @no_kernel_attribute_test() {
  func.return
}
