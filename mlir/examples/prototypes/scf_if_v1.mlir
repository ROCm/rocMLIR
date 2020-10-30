func @test(%carry_or_borrow: i1, %arg0: index, %arg1: index, %arg2: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c1024 = constant 1024 : index

  %idx_low_new_0 = addi %arg0, %c0 : index
  %idx_low_new_1 = addi %arg1, %c0 : index
  %idx_low_new_2 = addi %arg2, %c1 : index

  scf.if %carry_or_borrow {
    // carry logic.

    // 2nd digit
    %carry_2 = constant 0 : i1
    %idx_low_new_2_carried = scf.if %carry_2 -> (index) {
      %carried = addi %idx_low_new_2, %c1 : index
      scf.yield %carried : index
    } else {
      scf.yield %idx_low_new_2 : index
    }

    %carry_1 = cmpi "sgt", %idx_low_new_2_carried, %c3 : index
    %idx_low_new_2_updated = scf.if %carry_1 -> (index) {
      %updated = subi %idx_low_new_2_carried, %c3 : index
      scf.yield %updated : index
    } else {
      scf.yield %idx_low_new_2_carried : index
    }

    // 1st digit
    %idx_low_new_1_carried = scf.if %carry_1 -> (index) {
      %carried = addi %idx_low_new_1, %c1 : index
      scf.yield %carried : index
    } else {
      scf.yield %idx_low_new_1 : index
    }

    %carry_0 = cmpi "sgt", %idx_low_new_1_carried, %c3 : index
    %idx_low_new_1_updated = scf.if %carry_0 -> (index) {
      %updated = subi %idx_low_new_1_carried, %c3 : index
      scf.yield %updated : index
    } else {
      scf.yield %idx_low_new_1_carried : index
    }

    // 0th digit
    %idx_low_new_0_carried = scf.if %carry_0 -> (index) {
      %carried = addi %idx_low_new_0, %c1 : index
      scf.yield %carried : index
    } else {
      scf.yield %idx_low_new_0 : index
    }

    // no need to emit carry logic for the 0th digit
    // %carry_m1 = cmpi "sgt", %idx_low_new_0_carried, %c1024 : index
    // %idx_low_new_0_updated = scf.if %carry_m1 -> (index) {
    //   %updated = subi %idx_low_new_0_carried, %c1024 : index
    //   scf.yield %updated : index
    // } else {
    //   scf.yield %idx_low_new_0_carried : index
    // }
    %idx_low_new_0_updated = subi %idx_low_new_0, %c0 : index
  } else {
    // borrow logic.

    // 2nd digit
    %borrow_2 = constant 0 : i1
    %idx_low_new_2_borrowed = scf.if %borrow_2 -> (index) {
      %borrowed = subi %idx_low_new_2, %c1 : index
      scf.yield %borrowed : index
    } else {
      scf.yield %idx_low_new_2 : index
    }

    %borrow_1 = cmpi "slt", %idx_low_new_2_borrowed, %c0 : index
    %idx_low_new_2_updated = scf.if %borrow_1 -> (index) {
      %updated = addi %idx_low_new_2_borrowed, %c3 : index
      scf.yield %updated : index
    } else {
      scf.yield %idx_low_new_2_borrowed : index
    }

    // 1st digit
    %idx_low_new_1_borrowed = scf.if %borrow_1 -> (index) {
      %borrowed = subi %idx_low_new_1, %c1 : index
      scf.yield %borrowed : index
    } else {
      scf.yield %idx_low_new_1 : index
    }

    %borrow_0 = cmpi "slt", %idx_low_new_1_borrowed, %c0 : index
    %idx_low_new_1_updated = scf.if %borrow_0 -> (index) {
      %updated = addi %idx_low_new_1_borrowed, %c3 : index
      scf.yield %updated : index
    } else {
      scf.yield %idx_low_new_1_borrowed : index
    }

    // 0th digit
    %idx_low_new_0_borrowed = scf.if %borrow_0 -> (index) {
      %borrowed = subi %idx_low_new_0, %c1 : index
      scf.yield %borrowed : index
    } else {
      scf.yield %idx_low_new_0 : index
    }

    // no need to emit borrow logic for the 0th digit
    // %borrow_m1 = cmpi "slt", %idx_low_new_0_borrowed, %c0 : index
    // %idx_low_new_0_updated = scf.if %borrow_m1 -> (index) {
    //   %updated = addi %idx_low_new_0_borrowed, %c1024 : index
    //   scf.yield %updated : index
    // } else {
    //   scf.yield %idx_low_new_0_carried : index
    // }
    %idx_low_new_0_updated = addi %idx_low_new_0, %c0 : index
  }

  return
}
