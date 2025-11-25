import sys
import types
from pathlib import Path

# Ensure the performance utilities are importable as top-level modules.
PERFORMANCE_DIR = Path(__file__).resolve().parent.parent
if str(PERFORMANCE_DIR) not in sys.path:
    sys.path.insert(0, str(PERFORMANCE_DIR))


# Provide a light-weight stub for the optional HIP dependency so imports succeed
# on hosts without a GPU runtime.
if "hip" not in sys.modules:
    hip_module = types.ModuleType("hip")
    fake_hip = types.SimpleNamespace()

    class FakeHipError(int):
        hipSuccess = 0

    class FakeDeviceProp:
        def __init__(self):
            self.gcnArchName = b"gfx000"
            self.computeUnit = 0

    fake_hip.hipError_t = FakeHipError
    fake_hip.hipDeviceProp_t = FakeDeviceProp
    fake_hip.hipGetDeviceCount = lambda: (FakeHipError(0), 0)
    fake_hip.hipGetDeviceProperties = lambda props, device: (FakeHipError(0),)

    hip_module.hip = fake_hip
    sys.modules["hip"] = hip_module
