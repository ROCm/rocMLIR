// RUN: rocmlir-opt %s --rock-to-rocdl
func.func @async_wait_lowering() {
  // The waitcnt stores all counters in one i32 bits 15:14 and 3:0 store the vmcnt we have to wait on
  // CHECK: rocdl.s.waitcnt -49168
  // CHECK: rocdl.s.barrier
  rock.async_wait {numInst =  0 : i32}
  // CHECK: rocdl.s.waitcnt -49167
  // CHECK: rocdl.s.barrier
  rock.async_wait {numInst =  1 : i32}
  // CHECK: rocdl.s.waitcnt -2
  // CHECK: rocdl.s.barrier
  rock.async_wait {numInst =  62 : i32}
  // CHECK: rocdl.s.waitcnt -1
  // CHECK: rocdl.s.barrier
  rock.async_wait {numInst =  63 : i32}
  // Check that we clamp values > 63
  // CHECK: rocdl.s.waitcnt -1
  // CHECK: rocdl.s.barrier
  rock.async_wait {numInst =  64 : i32}
  return  
}
