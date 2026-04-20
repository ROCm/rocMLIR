"""Inject mock 'hip' and 'amd_arch_db' modules so perfRunner can be imported without ROCm."""
import sys
import types


class _HipErrorT:
    hipSuccess = 0  # noqa: N815 - mirrors real HIP API


class _HipDevicePropT:

    def __init__(self):
        self.gcnArchName = b"gfx900"


def _mock_get_device_count():
    return (_HipErrorT.hipSuccess, 1)


def _mock_get_device_properties(props, device):
    props.gcnArchName = b"gfx900"
    return (_HipErrorT.hipSuccess,)


class _MockHip:
    hipError_t = _HipErrorT  # noqa: N815 - mirrors real HIP API
    hipDeviceProp_t = _HipDevicePropT  # noqa: N815 - mirrors real HIP API
    hipGetDeviceCount = staticmethod(_mock_get_device_count)  # noqa: N815
    hipGetDeviceProperties = staticmethod(_mock_get_device_properties)  # noqa: N815


if "hip" not in sys.modules:
    hip_pkg = types.ModuleType("hip")
    hip_pkg.hip = _MockHip()
    sys.modules["hip"] = hip_pkg

# --- Mock amd_arch_db (compiled C++ extension, unavailable in CI) ---
# Keep bit positions in sync with GemmFeatures in
# mlir/include/mlir/Dialect/Rock/IR/RockAttrDefs.td


class _MockGemmFeatures:
    """Minimal mock of the GemmFeatures enum with arithmetic support."""

    def __init__(self, value=0):
        self._value = int(value)

    def __int__(self):
        return self._value

    def __and__(self, other):
        return _MockGemmFeatures(self._value & int(other))

    def __or__(self, other):
        return _MockGemmFeatures(self._value | int(other))

    def __bool__(self):
        return self._value != 0


_MockGemmFeatures.NONE = _MockGemmFeatures(0)
_MockGemmFeatures.MFMA = _MockGemmFeatures(1 << 0)
_MockGemmFeatures.WMMA = _MockGemmFeatures(1 << 1)
_MockGemmFeatures.DOT = _MockGemmFeatures(1 << 2)
_MockGemmFeatures.ATOMIC_ADD = _MockGemmFeatures(1 << 3)
_MockGemmFeatures.ATOMIC_ADD_BF16 = _MockGemmFeatures(1 << 4)
_MockGemmFeatures.ATOMIC_ADD_F16 = _MockGemmFeatures(1 << 5)
_MockGemmFeatures.ATOMIC_FMAX_F32 = _MockGemmFeatures(1 << 6)
_MockGemmFeatures.DIRECT_TO_LDS_32B = _MockGemmFeatures(1 << 7)
_MockGemmFeatures.DIRECT_TO_LDS_128B = _MockGemmFeatures(1 << 8)


class _MockAmdArchInfo:

    def __init__(self, **kwargs):
        self.default_features = kwargs.get("default_features", _MockGemmFeatures(0))
        self.wave_size = kwargs.get("wave_size", 64)
        self.max_waves_per_eu = kwargs.get("max_waves_per_eu", 10)
        self.total_sgpr_per_eu = kwargs.get("total_sgpr_per_eu", 512)
        self.total_vgpr_per_eu = kwargs.get("total_vgpr_per_eu", 256)
        self.total_shared_mem_per_cu = kwargs.get("total_shared_mem_per_cu", 65536)
        self.max_shared_mem_per_wg = kwargs.get("max_shared_mem_per_wg", 65536)
        self.num_eu_per_cu = kwargs.get("num_eu_per_cu", 4)
        self.min_num_cu = kwargs.get("min_num_cu", 64)
        self.has_fp8_conversion_instrs = kwargs.get("has_fp8_conversion_instrs", False)
        self.has_ocp_fp8_conversion_instrs = kwargs.get("has_ocp_fp8_conversion_instrs", False)
        self.has_fp4 = kwargs.get("has_fp4", False)
        self.has_scaled_gemm = kwargs.get("has_scaled_gemm", False)
        self.max_num_xcc = kwargs.get("max_num_xcc", 1)
        self.has_lds_transpose_load = kwargs.get("has_lds_transpose_load", False)


_DEFAULT_MOCK_INFO = _MockAmdArchInfo()


def _mock_lookup_arch_info(arch):
    return _DEFAULT_MOCK_INFO


if "amd_arch_db" not in sys.modules:
    amd_arch_db_mod = types.ModuleType("amd_arch_db")
    amd_arch_db_mod.GemmFeatures = _MockGemmFeatures
    amd_arch_db_mod.lookup_arch_info = _mock_lookup_arch_info
    sys.modules["amd_arch_db"] = amd_arch_db_mod
