// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-opt --tosa-to-rock -verify-diagnostics --split-input-file

// expected-error @+1 {{func op does not have the kernel attribute}}
func.func @no_kernel_attribute_test() {
  func.return
}
