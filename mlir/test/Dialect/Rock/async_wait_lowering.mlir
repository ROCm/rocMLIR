// RUN: rocmlir-opt %s --rock-to-rocdl=chipset=gfx950 | FileCheck %s -check-prefix=GFX950
// RUN: rocmlir-opt %s --rock-to-rocdl=chipset=gfx1250 | FileCheck %s -check-prefix=GFX1250
llvm.func @async_wait_lowering() {
  // The waitcnt stores all counters in one i32 bits 15:14 and 3:0 store the vmcnt we have to wait on
  // GFX950: rocdl.s.waitcnt -49168
  // GFX950: rocdl.s.barrier
  // GFX1250: rocdl.s.wait.asynccnt 0
  rock.async_wait {numInst =  0 : i32}
  // GFX950: rocdl.s.waitcnt -49167
  // GFX950: rocdl.s.barrier
  // GFX1250: rocdl.s.wait.asynccnt 0
  rock.async_wait {numInst =  1 : i32}
  // GFX950: rocdl.s.waitcnt -2
  // GFX950: rocdl.s.barrier
  // GFX1250: rocdl.s.wait.asynccnt 0
  rock.async_wait {numInst =  62 : i32}
  // GFX950: rocdl.s.waitcnt -1
  // GFX950: rocdl.s.barrier
  // GFX1250: rocdl.s.wait.asynccnt 0
  rock.async_wait {numInst =  63 : i32}
  // Check that we clamp values > 63
  // GFX950: rocdl.s.waitcnt -1
  // GFX950: rocdl.s.barrier
  // GFX1250: rocdl.s.wait.asynccnt 0
  rock.async_wait {numInst =  64 : i32}
  llvm.return
}
