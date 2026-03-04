"""Inject mock 'hip' module so perfRunner can be imported without ROCm (e.g. in CI)."""
import sys
import types

class _HipErrorT:
    hipSuccess = 0

class _HipDevicePropT:
    def __init__(self):
        self.gcnArchName = b"gfx900"

def _mock_get_device_count():
    return (_HipErrorT.hipSuccess, 1)

def _mock_get_device_properties(props, device):
    props.gcnArchName = b"gfx900"
    return (_HipErrorT.hipSuccess,)

class _MockHip:
    hipError_t = _HipErrorT
    hipDeviceProp_t = _HipDevicePropT
    hipGetDeviceCount = staticmethod(_mock_get_device_count)
    hipGetDeviceProperties = staticmethod(_mock_get_device_properties)

if "hip" not in sys.modules:
    hip_pkg = types.ModuleType("hip")
    hip_pkg.hip = _MockHip()
    sys.modules["hip"] = hip_pkg
