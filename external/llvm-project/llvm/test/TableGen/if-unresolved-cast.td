// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A { int x; }

// This is a workaround for not having imported all the tablegen changes
// and so getting a different error message
// The actual test is: // CHECK[colon] Undefined reference to record: ''
// CHECK: error: attempting to loop over '!if(!cast<A>("").x, [1], [])', expected a list
if !cast<A>("").x then
  def x;
